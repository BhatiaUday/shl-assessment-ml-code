#!/usr/bin/env python3
"""
Improved Grammar Scoring Model Training (Version 2)
Addresses central prediction bias with:
1. Stronger variance penalty to encourage spread
2. Quantile loss to better handle distribution
3. Increased model capacity
4. Better regularization balance
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
    # Paths
    DATA_DIR = Path('/home/azureuser/shl2/dataset/csvs')
    AUDIO_DIR = Path('/home/azureuser/shl2/dataset/audios')
    CACHE_DIR = Path('/home/azureuser/shl2/cache')
    MODEL_DIR = Path('/home/azureuser/shl2/models_v2')
    
    # Model settings
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LEN = 512
    BATCH_SIZE = 16
    EPOCHS = 8  # Increased from 6
    LR = 1e-4  # Reduced learning rate for stability
    DROPOUT = 0.3  # Increased dropout
    
    # LoRA settings
    LORA_R = 16  # Increased from 8
    LORA_ALPHA = 32  # Increased from 16
    LORA_DROPOUT = 0.1
    LORA_LAYERS = 6  # Increased from 4
    
    # Loss weights - ADJUSTED FOR BETTER SPREAD
    W_MSE = 0.6  # Reduced
    W_PEARSON = 0.2  # Same
    W_RANKING = 0.1  # Same
    W_VARIANCE = 0.1  # Increased from 0.05
    
    # New: Quantile loss weight
    W_QUANTILE = 0.1
    
    # CV settings
    N_FOLDS = 5
    SEED = 42
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Config.CACHE_DIR.mkdir(exist_ok=True, parents=True)
Config.MODEL_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
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
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MeanPool(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1.0)

class TextHead(nn.Module):
    """Enhanced regression head with 3 layers for more capacity"""
    def __init__(self, hidden_size, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 2048),  # Increased from 1024
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),  # Additional layer
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

class TextModel(nn.Module):
    def __init__(self, model_name=Config.MODEL_NAME, dropout=Config.DROPOUT):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name)
        
        # Attach LoRA to more layers
        self._attach_lora_top_layers(last_n_layers=Config.LORA_LAYERS)
        
        # Enable gradients only for LoRA parameters
        for n, p in self.enc.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        
        hid = self.enc.config.hidden_size
        self.pool = MeanPool()
        self.reg_head = TextHead(hid, dropout=dropout)
    
    def _attach_lora_top_layers(self, last_n_layers=6):
        n_layers = len(self.enc.encoder.layer)
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
        
        self.enc = get_peft_model(self.enc, cfg)
    
    def forward(self, batch):
        out = self.enc(**batch)
        pooled = self.pool(out.last_hidden_state, batch['attention_mask'])
        return self.reg_head(pooled)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def pearson_loss(pred, target):
    """Negative Pearson correlation as loss"""
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    
    num = (pred_centered * target_centered).sum()
    den = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
    
    return -num / (den + 1e-8)

def ranking_loss(pred, target, margin=0.3):
    """Pairwise ranking loss"""
    n = pred.size(0)
    if n < 2:
        return torch.tensor(0.0, device=pred.device)
    
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    
    target_sign = torch.sign(target_diff)
    
    loss = torch.clamp(margin - target_sign * pred_diff, min=0)
    return loss.mean()

def variance_penalty(pred, target, target_std_scale=1.2):
    """STRONGER penalty to encourage predictions to spread out"""
    pred_std = pred.std()
    target_std = target.std()
    
    # Encourage pred_std to be at least target_std_scale * target_std
    desired_std = target_std * target_std_scale
    
    if pred_std < desired_std:
        # Quadratic penalty when under-dispersed
        return ((desired_std - pred_std) / target_std) ** 2
    else:
        return torch.tensor(0.0, device=pred.device)

def quantile_loss(pred, target, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """Quantile regression loss to better match distribution"""
    loss = 0.0
    for q in quantiles:
        errors = target - pred
        loss += torch.max((q - 1) * errors, q * errors).mean()
    return loss / len(quantiles)

def combined_loss(pred, target):
    """Multi-objective loss with enhanced variance penalty"""
    mse = nn.MSELoss()(pred, target)
    pearson = pearson_loss(pred, target)
    ranking = ranking_loss(pred, target)
    variance = variance_penalty(pred, target, target_std_scale=1.3)  # Encourage 1.3x spread
    quantile = quantile_loss(pred, target)
    
    total = (
        Config.W_MSE * mse + 
        Config.W_PEARSON * pearson + 
        Config.W_RANKING * ranking + 
        Config.W_VARIANCE * variance +
        Config.W_QUANTILE * quantile
    )
    
    return total, {
        'mse': mse.item(),
        'pearson': pearson.item(),
        'ranking': ranking.item(),
        'variance': variance.item(),
        'quantile': quantile.item(),
        'total': total.item()
    }

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training')
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(Config.DEVICE)
        attention_mask = batch['attention_mask'].to(Config.DEVICE)
        labels = batch['label'].to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        with autocast(dtype=torch.bfloat16):
            preds = model({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
            loss, loss_dict = combined_loss(preds, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mse': f"{loss_dict['mse']:.4f}",
            'var': f"{loss_dict['variance']:.4f}"
        })
    
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    preds_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            with autocast(dtype=torch.bfloat16):
                preds = model({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })
            
            preds_list.append(preds.float().cpu())
            labels_list.append(labels.cpu())
    
    preds = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy()
    
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    pearson, _ = pearsonr(preds, labels)
    
    # Check variance ratio
    pred_std = preds.std()
    label_std = labels.std()
    var_ratio = pred_std / label_std
    
    return rmse, pearson, preds, var_ratio

# ============================================================================
# GRAMMAR FEATURES
# ============================================================================

def extract_grammar_features(text, nlp, grammar_checker):
    if not text or len(text.strip()) == 0:
        return [0] * 11
    
    doc = nlp(text)
    
    num_tokens = len([t for t in doc if not t.is_space])
    num_sentences = len(list(doc.sents))
    avg_sentence_len = num_tokens / max(num_sentences, 1)
    avg_word_len = np.mean([len(t.text) for t in doc if not t.is_space]) if num_tokens > 0 else 0
    
    try:
        errors = grammar_checker.check(text)
        num_grammar_errors = len(errors)
    except:
        num_grammar_errors = 0
    
    try:
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        gunning_fog = textstat.gunning_fog(text)
        smog_index = textstat.smog_index(text)
        automated_readability_index = textstat.automated_readability_index(text)
        coleman_liau_index = textstat.coleman_liau_index(text)
    except:
        flesch_reading_ease = flesch_kincaid_grade = gunning_fog = 0
        smog_index = automated_readability_index = coleman_liau_index = 0
    
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
    print("IMPROVED GRAMMAR SCORING MODEL TRAINING (V2)")
    print("=" * 80)
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"LoRA: r={Config.LORA_R}, alpha={Config.LORA_ALPHA}, layers={Config.LORA_LAYERS}")
    print(f"Loss weights: MSE={Config.W_MSE}, Pearson={Config.W_PEARSON}, Ranking={Config.W_RANKING}, Variance={Config.W_VARIANCE}, Quantile={Config.W_QUANTILE}")
    
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
        torch.cuda.empty_cache()
    
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
    
    oof_text_preds = np.zeros(len(train_df))
    oof_grammar_preds = np.zeros(len(train_df))
    calibrations = {'text': [], 'grammar': []}
    
    # Train models
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label_bin'])):
        print("\n" + "=" * 80)
        print(f"FOLD {fold}")
        print("=" * 80)
        
        # TEXT MODEL
        print("\n[Text Model]")
        model = TextModel().to(Config.DEVICE)
        
        train_dataset = TextDataset(
            train_df.iloc[train_idx]['transcript'].values,
            train_df.iloc[train_idx]['label'].values,
            model.tok,
            Config.MAX_LEN
        )
        val_dataset = TextDataset(
            train_df.iloc[val_idx]['transcript'].values,
            train_df.iloc[val_idx]['label'].values,
            model.tok,
            Config.MAX_LEN
        )
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
        scaler = GradScaler()
        
        best_rmse = float('inf')
        for epoch in range(Config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            train_loss = train_epoch(model, train_loader, optimizer, scaler)
            val_rmse, val_pearson, val_preds, var_ratio = validate(model, val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val RMSE: {val_rmse:.4f}, Pearson: {val_pearson:.4f}, Var Ratio: {var_ratio:.3f}")
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), Config.MODEL_DIR / f'text_fold{fold}.pt')
                print(f"✓ Saved (RMSE: {val_rmse:.4f})")
        
        # Load best and get OOF predictions
        model.load_state_dict(torch.load(Config.MODEL_DIR / f'text_fold{fold}.pt'))
        _, _, val_preds, _ = validate(model, val_loader)
        oof_text_preds[val_idx] = val_preds
        
        # Calibrate
        from sklearn.linear_model import LinearRegression
        cal = LinearRegression().fit(val_preds.reshape(-1, 1), train_df.iloc[val_idx]['label'].values)
        calibrations['text'].append((cal.coef_[0], cal.intercept_))
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # GRAMMAR MODEL
        print("\n[Grammar Model]")
        X_train = grammar_feats.iloc[train_idx]
        y_train = train_df.iloc[train_idx]['label'].values
        X_val = grammar_feats.iloc[val_idx]
        y_val = train_df.iloc[val_idx]['label'].values
        
        gbm = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=Config.SEED
        )
        gbm.fit(X_train, y_train)
        gbm.booster_.save_model(str(Config.MODEL_DIR / f'grammar_fold{fold}.txt'))
        
        val_preds_gbm = gbm.predict(X_val)
        oof_grammar_preds[val_idx] = val_preds_gbm
        
        rmse_gbm = np.sqrt(np.mean((val_preds_gbm - y_val) ** 2))
        print(f"Grammar RMSE: {rmse_gbm:.4f}")
        
        # Calibrate
        cal = LinearRegression().fit(val_preds_gbm.reshape(-1, 1), y_val)
        calibrations['grammar'].append((cal.coef_[0], cal.intercept_))
    
    # Calibrate OOF predictions
    print("\n" + "=" * 80)
    print("CALIBRATING OOF PREDICTIONS")
    print("=" * 80)
    
    oof_text_calibrated = np.zeros_like(oof_text_preds)
    oof_grammar_calibrated = np.zeros_like(oof_grammar_preds)
    
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df['label_bin'])):
        a_t, b_t = calibrations['text'][fold]
        a_g, b_g = calibrations['grammar'][fold]
        oof_text_calibrated[val_idx] = a_t * oof_text_preds[val_idx] + b_t
        oof_grammar_calibrated[val_idx] = a_g * oof_grammar_preds[val_idx] + b_g
    
    # Fusion model
    print("\n" + "=" * 80)
    print("TRAINING FUSION MODEL")
    print("=" * 80)
    
    X_fusion = np.column_stack([oof_text_calibrated, oof_grammar_calibrated])
    y_fusion = train_df['label'].values
    
    fusion_model = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=Config.SEED)
    fusion_model.fit(X_fusion, y_fusion)
    
    oof_final = fusion_model.predict(X_fusion)
    
    # Final metrics
    rmse_final = np.sqrt(np.mean((oof_final - y_fusion) ** 2))
    pearson_final, _ = pearsonr(oof_final, y_fusion)
    pred_std = oof_final.std()
    label_std = y_fusion.std()
    var_ratio_final = pred_std / label_std
    
    print("\n" + "=" * 80)
    print("FINAL OOF RESULTS")
    print("=" * 80)
    print(f"RMSE:              {rmse_final:.4f}")
    print(f"Pearson:           {pearson_final:.4f}")
    print(f"Prediction Std:    {pred_std:.4f}")
    print(f"Label Std:         {label_std:.4f}")
    print(f"Variance Ratio:    {var_ratio_final:.3f} (target: >1.0)")
    print("=" * 80)
    
    # Save fusion and calibrations
    with open(Config.MODEL_DIR / 'fusion_enet.pkl', 'wb') as f:
        pickle.dump(fusion_model, f)
    
    with open(Config.MODEL_DIR / 'calibrations.pkl', 'wb') as f:
        pickle.dump(calibrations, f)
    
    print("\n✅ Training complete!")
    print(f"Models saved to: {Config.MODEL_DIR}")

if __name__ == '__main__':
    main()
