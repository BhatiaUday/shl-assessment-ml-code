#!/usr/bin/env python3
"""
Grammar Scoring Model V4 - Comparative Learning (Siamese Architecture)
Revolutionary approach: Learn by comparing pairs of texts

Key Innovation:
- 409 samples → 83,436 training pairs (204x data expansion!)
- Learn relative quality (easier than absolute scoring)
- Siamese network: same encoder for both texts
- Predict score difference instead of absolute scores

Training: Compare pairs, learn that "text A is better than text B by X points"
Inference: Score single text by encoding it once

Expected: 0.45-0.50 RMSE (significant improvement over V2's 0.5380)
"""

import os
import gc
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNet
import pickle
import itertools

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

import spacy
import language_tool_python
import textstat
import lightgbm as lgb
from faster_whisper import WhisperModel

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths - NEW FOLDER FOR V4
    DATA_DIR = Path('/home/azureuser/shl2/dataset/csvs')
    AUDIO_DIR = Path('/home/azureuser/shl2/dataset/audios')
    CACHE_DIR = Path('/home/azureuser/shl2/cache')
    V4_DIR = Path('/home/azureuser/shl2/v4_comparative')  # NEW FOLDER
    MODEL_DIR = Path('/home/azureuser/shl2/v4_comparative/models')
    PAIRS_DIR = Path('/home/azureuser/shl2/v4_comparative/pairs')
    
    # Model settings - Same as V2
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LEN = 512
    BATCH_SIZE = 8  # Smaller because we process pairs
    EPOCHS = 10  # More epochs since we have more data (pairs)
    LR = 1e-4
    DROPOUT = 0.3
    
    # LoRA settings - Same as V2
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_LAYERS = 6
    
    # Comparative learning settings
    PAIR_SAMPLING = 'all'  # 'all', 'stratified', 'hard_negatives'
    MARGIN = 0.5  # Margin for contrastive loss
    
    # Loss weights
    W_PAIR = 0.7  # Weight for pairwise comparison loss
    W_ABSOLUTE = 0.3  # Weight for absolute score prediction
    
    # CV settings
    N_FOLDS = 5
    SEED = 42
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
Config.V4_DIR.mkdir(exist_ok=True, parents=True)
Config.MODEL_DIR.mkdir(exist_ok=True, parents=True)
Config.PAIRS_DIR.mkdir(exist_ok=True, parents=True)
Config.CACHE_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# PAIR DATASET
# ============================================================================

class PairDataset(Dataset):
    """Dataset that yields pairs of texts for comparative learning"""
    def __init__(self, texts, labels, pairs, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.pairs = pairs  # List of (idx1, idx2)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        
        text1 = str(self.texts[idx1])
        text2 = str(self.texts[idx2])
        label1 = self.labels[idx1]
        label2 = self.labels[idx2]
        
        # Encode both texts
        encoding1 = self.tokenizer(
            text1,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            text2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids1': encoding1['input_ids'].squeeze(0),
            'attention_mask1': encoding1['attention_mask'].squeeze(0),
            'input_ids2': encoding2['input_ids'].squeeze(0),
            'attention_mask2': encoding2['attention_mask'].squeeze(0),
            'label1': torch.tensor(label1, dtype=torch.float32),
            'label2': torch.tensor(label2, dtype=torch.float32),
            'label_diff': torch.tensor(label1 - label2, dtype=torch.float32)
        }

class SingleDataset(Dataset):
    """Standard dataset for inference"""
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }

# ============================================================================
# MODEL
# ============================================================================

class MeanPool(nn.Module):
    """Mean pooling with attention mask"""
    def forward(self, last_hidden_state, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1.0)

class ComparativeGrammarModel(nn.Module):
    """Siamese network for comparative grammar scoring"""
    def __init__(self, model_name=Config.MODEL_NAME, dropout=Config.DROPOUT):
        super().__init__()
        
        # Shared encoder (Siamese)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Attach LoRA
        self._attach_lora_top_layers(last_n_layers=Config.LORA_LAYERS)
        
        # Freeze non-LoRA parameters
        for n, p in self.encoder.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        
        hidden_size = self.encoder.config.hidden_size
        
        # Pooling
        self.pool = MeanPool()
        
        # Absolute score head (for single text scoring)
        self.absolute_head = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        
        # Comparative head (for score difference prediction)
        self.comparative_head = nn.Sequential(
            nn.Linear(hidden_size * 3, 1024),  # concat(emb1, emb2, abs_diff)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def _attach_lora_top_layers(self, last_n_layers=6):
        n_layers = len(self.encoder.encoder.layer)
        keep_layers = set(range(n_layers - last_n_layers, n_layers))
        
        target_modules = []
        for i in keep_layers:
            target_modules.extend([
                f"encoder.layer.{i}.attention.self.query_proj",
                f"encoder.layer.{i}.attention.self.key_proj",
                f"encoder.layer.{i}.attention.self.value_proj"
            ])
        
        cfg = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            target_modules=target_modules,
            modules_to_save=[]
        )
        
        self.encoder = get_peft_model(self.encoder, cfg)
    
    def encode(self, input_ids, attention_mask):
        """Encode a single text into embedding"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        return pooled
    
    def forward_single(self, input_ids, attention_mask):
        """Forward pass for single text (inference)"""
        emb = self.encode(input_ids, attention_mask)
        return self.absolute_head(emb).squeeze(-1)
    
    def forward_pair(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        """Forward pass for text pair (training)"""
        # Encode both texts
        emb1 = self.encode(input_ids1, attention_mask1)
        emb2 = self.encode(input_ids2, attention_mask2)
        
        # Absolute scores for each text
        score1 = self.absolute_head(emb1).squeeze(-1)
        score2 = self.absolute_head(emb2).squeeze(-1)
        
        # Comparative features: concatenate embeddings and their difference
        comparative_features = torch.cat([
            emb1, 
            emb2, 
            torch.abs(emb1 - emb2)
        ], dim=-1)
        
        # Predict score difference
        score_diff = self.comparative_head(comparative_features).squeeze(-1)
        
        return score1, score2, score_diff

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def comparative_loss(score1, score2, score_diff_pred, label1, label2, margin=0.5):
    """
    Comparative loss that learns relative quality
    - Predicts score difference
    - Margin-based ranking loss
    """
    label_diff = label1 - label2
    
    # MSE on score difference
    diff_loss = nn.MSELoss()(score_diff_pred, label_diff)
    
    # Ranking loss with margin
    # If label1 > label2, then score1 should be > score2 + margin
    ranking_loss = torch.relu(margin - (score1 - score2) * torch.sign(label_diff)).mean()
    
    return diff_loss + 0.5 * ranking_loss

def combined_loss(score1, score2, score_diff_pred, label1, label2):
    """Combined loss: comparative + absolute"""
    # Absolute MSE loss
    absolute_loss = nn.MSELoss()(score1, label1) + nn.MSELoss()(score2, label2)
    
    # Comparative loss
    comp_loss = comparative_loss(score1, score2, score_diff_pred, label1, label2, margin=Config.MARGIN)
    
    # Weighted combination
    total_loss = Config.W_ABSOLUTE * absolute_loss + Config.W_PAIR * comp_loss
    
    return total_loss, {
        'absolute': absolute_loss.item(),
        'comparative': comp_loss.item()
    }

# ============================================================================
# PAIR GENERATION
# ============================================================================

def generate_pairs(labels, fold_indices, strategy='stratified', max_pairs_per_sample=200):
    """
    Generate training pairs with different strategies
    
    Args:
        labels: All labels
        fold_indices: Indices in current fold
        strategy: 'all', 'stratified', 'hard_negatives'
        max_pairs_per_sample: Limit pairs per sample to avoid memory issues
    
    Returns:
        List of (idx1, idx2) pairs
    """
    fold_labels = labels[fold_indices]
    n = len(fold_indices)
    
    print(f"Generating pairs for {n} samples...")
    
    if strategy == 'all':
        # All possible pairs (can be huge)
        all_pairs = list(itertools.combinations(range(n), 2))
        
        # Limit pairs per sample
        if len(all_pairs) > n * max_pairs_per_sample:
            print(f"  Too many pairs ({len(all_pairs)}), sampling...")
            np.random.shuffle(all_pairs)
            all_pairs = all_pairs[:n * max_pairs_per_sample]
        
        pairs = all_pairs
    
    elif strategy == 'stratified':
        # Sample pairs from similar score ranges and different ranges
        pairs = []
        
        # Bin labels into ranges
        bins = np.linspace(fold_labels.min(), fold_labels.max(), 6)
        label_bins = np.digitize(fold_labels, bins)
        
        for i in range(n):
            # Sample pairs within same bin (fine-grained comparison)
            same_bin = np.where(label_bins == label_bins[i])[0]
            same_bin = same_bin[same_bin != i]
            if len(same_bin) > 0:
                n_same = min(50, len(same_bin))
                sampled_same = np.random.choice(same_bin, size=n_same, replace=False)
                pairs.extend([(i, j) for j in sampled_same])
            
            # Sample pairs from different bins (coarse comparison)
            diff_bin = np.where(label_bins != label_bins[i])[0]
            if len(diff_bin) > 0:
                n_diff = min(50, len(diff_bin))
                sampled_diff = np.random.choice(diff_bin, size=n_diff, replace=False)
                pairs.extend([(i, j) for j in sampled_diff])
    
    elif strategy == 'hard_negatives':
        # Focus on pairs with similar labels (hard to distinguish)
        pairs = []
        for i in range(n):
            # Find samples with similar labels
            label_diffs = np.abs(fold_labels - fold_labels[i])
            # Samples within 0.5 score difference
            similar_indices = np.where((label_diffs > 0) & (label_diffs < 0.5))[0]
            
            if len(similar_indices) > 0:
                n_similar = min(100, len(similar_indices))
                sampled = np.random.choice(similar_indices, size=n_similar, replace=False)
                pairs.extend([(i, j) for j in sampled])
            
            # Also add some easy pairs (different labels)
            diff_indices = np.where(label_diffs > 1.0)[0]
            if len(diff_indices) > 0:
                n_diff = min(50, len(diff_indices))
                sampled = np.random.choice(diff_indices, size=n_diff, replace=False)
                pairs.extend([(i, j) for j in sampled])
    
    print(f"  Generated {len(pairs)} pairs ({len(pairs)/n:.1f} pairs per sample)")
    return pairs

# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        input_ids1 = batch['input_ids1'].to(Config.DEVICE)
        attention_mask1 = batch['attention_mask1'].to(Config.DEVICE)
        input_ids2 = batch['input_ids2'].to(Config.DEVICE)
        attention_mask2 = batch['attention_mask2'].to(Config.DEVICE)
        label1 = batch['label1'].to(Config.DEVICE)
        label2 = batch['label2'].to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        with autocast(dtype=torch.bfloat16):
            score1, score2, score_diff = model.forward_pair(
                input_ids1, attention_mask1,
                input_ids2, attention_mask2
            )
            loss, loss_dict = combined_loss(score1, score2, score_diff, label1, label2)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(loader)

def validate(model, loader):
    """Validate on single texts"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            preds = model.forward_single(input_ids, attention_mask)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    pearson_corr = pearsonr(preds, labels)[0]
    
    return rmse, pearson_corr, preds

# ============================================================================
# GRAMMAR FEATURES
# ============================================================================

def extract_grammar_features(text, nlp, grammar_checker):
    doc = nlp(text)
    
    num_tokens = len(doc)
    num_sentences = len(list(doc.sents))
    avg_sentence_len = num_tokens / max(num_sentences, 1)
    avg_word_len = np.mean([len(token.text) for token in doc if not token.is_punct])
    
    matches = grammar_checker.check(text)
    num_grammar_errors = len(matches)
    
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    smog_index = textstat.smog_index(text)
    automated_readability_index = textstat.automated_readability_index(text)
    coleman_liau_index = textstat.coleman_liau_index(text)
    
    return [
        num_tokens, num_sentences, avg_sentence_len, avg_word_len,
        num_grammar_errors, flesch_reading_ease, flesch_kincaid_grade,
        gunning_fog, smog_index, automated_readability_index, coleman_liau_index
    ]

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("GRAMMAR SCORING MODEL V4 - COMPARATIVE LEARNING")
    print("=" * 80)
    print(f"\nDevice: {Config.DEVICE}")
    print(f"\nRevolutionary Approach:")
    print("  - Siamese network: learns by comparing pairs")
    print("  - 409 samples → ~83,000 training pairs!")
    print("  - Predicts relative quality (easier than absolute)")
    print(f"\nV2 Baseline: 0.5380")
    print(f"V4 Target: 0.45-0.50 (10-15% improvement)")
    
    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    train_df = pd.read_csv(Config.DATA_DIR / 'train.csv')
    print(f"Train samples: {len(train_df)}")
    
    # Transcribe audio
    cache_path = Config.CACHE_DIR / 'train_transcripts.csv'
    if cache_path.exists():
        print("Loading cached transcripts...")
        train_df = pd.read_csv(cache_path)
    else:
        print("Transcribing audio files...")
        whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")
        transcripts = []
        
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
            audio_path = Config.AUDIO_DIR / 'train' / f"{row['filename']}.wav"
            segments, _ = whisper.transcribe(str(audio_path), beam_size=5, language="en")
            text = " ".join([seg.text for seg in segments]).strip()
            transcripts.append(text)
        
        train_df['transcript'] = transcripts
        train_df.to_csv(cache_path, index=False)
        del whisper
        gc.collect()
    
    # Extract grammar features
    print("\nExtracting grammar features...")
    nlp = spacy.load('en_core_web_sm')
    grammar_checker = language_tool_python.LanguageTool('en-US')
    
    features = []
    for text in tqdm(train_df['transcript']):
        features.append(extract_grammar_features(text, nlp, grammar_checker))
    
    grammar_feats = pd.DataFrame(features, columns=[
        'num_tokens', 'num_sentences', 'avg_sentence_len', 'avg_word_len',
        'num_grammar_errors', 'flesch_reading_ease', 'flesch_kincaid_grade',
        'gunning_fog', 'smog_index', 'automated_readability_index', 'coleman_liau_index'
    ])
    
    # Create folds
    print("\nCreating stratified folds...")
    train_df['label_bin'] = pd.cut(train_df['label'], bins=10, labels=False)
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Train with comparative learning
    print("\n" + "=" * 80)
    print("TRAINING COMPARATIVE MODEL")
    print("=" * 80)
    
    oof_preds = np.zeros(len(train_df))
    oof_labels = train_df['label'].values
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label_bin'])):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold}")
        print('=' * 60)
        
        # Generate pairs for training
        print("\nGenerating training pairs...")
        train_pairs = generate_pairs(
            train_df['label'].values,
            train_idx,
            strategy='stratified',
            max_pairs_per_sample=200
        )
        
        # Save pairs for analysis
        pairs_file = Config.PAIRS_DIR / f'train_pairs_fold{fold}.pkl'
        with open(pairs_file, 'wb') as f:
            pickle.dump(train_pairs, f)
        
        # Create datasets
        train_pair_dataset = PairDataset(
            train_df.iloc[train_idx]['transcript'].values,
            train_df.iloc[train_idx]['label'].values,
            train_pairs,
            tokenizer,
            Config.MAX_LEN
        )
        
        val_dataset = SingleDataset(
            train_df.iloc[val_idx]['transcript'].values,
            train_df.iloc[val_idx]['label'].values,
            tokenizer,
            Config.MAX_LEN
        )
        
        train_loader = DataLoader(train_pair_dataset, batch_size=Config.BATCH_SIZE, 
                                  shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE * 2,
                                shuffle=False, num_workers=2)
        
        # Create model
        model = ComparativeGrammarModel().to(Config.DEVICE)
        
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Training pairs: {len(train_pairs):,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=0.01)
        scaler = GradScaler()
        
        best_rmse = float('inf')
        
        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch + 1)
            val_rmse, val_pearson, val_preds = validate(model, val_loader)
            
            print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val RMSE: {val_rmse:.4f}, Pearson: {val_pearson:.4f}")
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), Config.MODEL_DIR / f'model_fold{fold}_best.pth')
                print(f"  → Best model saved! (RMSE: {best_rmse:.4f})")
        
        # Get final predictions
        print("\nGenerating OOF predictions...")
        model.load_state_dict(torch.load(Config.MODEL_DIR / f'model_fold{fold}_best.pth'))
        _, _, oof_preds[val_idx] = validate(model, val_loader)
        
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
    
    # Calculate OOF metrics
    print("\n" + "=" * 80)
    print("OUT-OF-FOLD RESULTS (Comparative Model)")
    print("=" * 80)
    oof_rmse = np.sqrt(np.mean((oof_preds - oof_labels) ** 2))
    oof_pearson = pearsonr(oof_preds, oof_labels)[0]
    print(f"OOF RMSE: {oof_rmse:.4f}")
    print(f"OOF Pearson: {oof_pearson:.4f}")
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'filename': train_df['filename'],
        'true_label': oof_labels,
        'predicted_label': oof_preds
    })
    oof_df.to_csv(Config.V4_DIR / 'oof_predictions.csv', index=False)
    
    # Train LightGBM
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM")
    print("=" * 80)
    
    lgb_oof_preds = np.zeros(len(train_df))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label_bin'])):
        print(f"\nFold {fold}...")
        
        X_train = grammar_feats.iloc[train_idx]
        y_train = train_df.iloc[train_idx]['label'].values
        X_val = grammar_feats.iloc[val_idx]
        y_val = train_df.iloc[val_idx]['label'].values
        
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=500,
            random_state=Config.SEED,
            verbose=-1
        )
        
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        lgb_oof_preds[val_idx] = lgb_model.predict(X_val)
        
        with open(Config.MODEL_DIR / f'lgb_fold{fold}.pkl', 'wb') as f:
            pickle.dump(lgb_model, f)
    
    lgb_rmse = np.sqrt(np.mean((lgb_oof_preds - oof_labels) ** 2))
    lgb_pearson = pearsonr(lgb_oof_preds, oof_labels)[0]
    print(f"\nLightGBM OOF RMSE: {lgb_rmse:.4f}")
    print(f"LightGBM OOF Pearson: {lgb_pearson:.4f}")
    
    # Ensemble
    print("\n" + "=" * 80)
    print("ENSEMBLE FUSION")
    print("=" * 80)
    
    ensemble_X = np.column_stack([oof_preds, lgb_oof_preds])
    ensemble_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=Config.SEED)
    ensemble_model.fit(ensemble_X, oof_labels)
    
    ensemble_preds = ensemble_model.predict(ensemble_X)
    ensemble_rmse = np.sqrt(np.mean((ensemble_preds - oof_labels) ** 2))
    ensemble_pearson = pearsonr(ensemble_preds, oof_labels)[0]
    
    print(f"\nFinal Ensemble OOF RMSE: {ensemble_rmse:.4f}")
    print(f"Final Ensemble OOF Pearson: {ensemble_pearson:.4f}")
    
    with open(Config.MODEL_DIR / 'ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINES")
    print("=" * 80)
    print(f"V2 (baseline): 0.5380")
    print(f"V2_H1 (multi-pool): 0.5624")
    print(f"Ensemble (3 models): 0.5323")
    print(f"V4 (comparative): {ensemble_rmse:.4f}")
    
    if ensemble_rmse < 0.5380:
        improvement = ((0.5380 - ensemble_rmse) / 0.5380) * 100
        print(f"\n✓ SUCCESS: {improvement:.1f}% better than V2!")
    elif ensemble_rmse < 0.5323:
        improvement = ((0.5323 - ensemble_rmse) / 0.5323) * 100
        print(f"\n✓ BEST MODEL: {improvement:.1f}% better than ensemble!")
    else:
        degradation = ((ensemble_rmse - 0.5323) / 0.5323) * 100
        print(f"\n✗ WORSE: {degradation:.1f}% worse than best (ensemble)")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nAll V4 data saved to: {Config.V4_DIR}")
    print(f"Models saved to: {Config.MODEL_DIR}")
    print(f"Pairs saved to: {Config.PAIRS_DIR}")

if __name__ == '__main__':
    main()
