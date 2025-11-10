#!/usr/bin/env python3
"""
Grammar Scoring Model V5 - Knowledge Distillation from V2 to V4

Strategy: Use V2 as teacher (proven 0.533 test RMSE) to guide V4 student
- V2's predictions = soft targets with correct distribution
- V4's architecture = powerful comparative learning
- Distillation = transfer V2's generalization to V4's capacity

Expected: Best of both worlds!
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
    # Paths
    DATA_DIR = Path('/home/azureuser/shl2/dataset/csvs')
    AUDIO_DIR = Path('/home/azureuser/shl2/dataset/audios')
    CACHE_DIR = Path('/home/azureuser/shl2/cache')
    V2_MODEL_DIR = Path('/home/azureuser/shl2/models_v2')  # Teacher models
    V5_DIR = Path('/home/azureuser/shl2/v5_distillation')  # NEW FOLDER
    MODEL_DIR = Path('/home/azureuser/shl2/v5_distillation/models')
    
    # Model settings (same as V4)
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LEN = 512
    BATCH_SIZE = 8
    EPOCHS = 12  # More epochs for distillation
    LR = 5e-5  # Lower LR for distillation
    DROPOUT = 0.3
    
    # LoRA settings
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_LAYERS = 6
    
    # Distillation settings
    TEMPERATURE = 3.0  # Temperature for soft targets
    ALPHA_DISTILL = 0.5  # Weight for distillation loss
    ALPHA_HARD = 0.3  # Weight for hard label loss
    ALPHA_COMPARATIVE = 0.2  # Weight for comparative loss
    
    # CV settings
    N_FOLDS = 5
    SEED = 42
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
Config.V5_DIR.mkdir(exist_ok=True, parents=True)
Config.MODEL_DIR.mkdir(exist_ok=True, parents=True)
Config.CACHE_DIR.mkdir(exist_ok=True, parents=True)

print(f"""
{'='*80}
GRAMMAR SCORING MODEL V5 - KNOWLEDGE DISTILLATION
{'='*80}

Strategy: V2 (Teacher) → V5 (Student)
  - V2 has correct test distribution (0.533 RMSE)
  - V4 architecture is powerful but overfit
  - V5 learns from both: V2's wisdom + V4's capacity

Device: {Config.DEVICE}

Distillation Parameters:
  - Temperature: {Config.TEMPERATURE}
  - Distillation weight: {Config.ALPHA_DISTILL}
  - Hard label weight: {Config.ALPHA_HARD}
  - Comparative weight: {Config.ALPHA_COMPARATIVE}

Expected: 0.51-0.52 RMSE (better than V2's 0.533!)
{'='*80}
""")

# ============================================================================
# LOAD V2 TEACHER MODELS
# ============================================================================

class TextHead(nn.Module):
    """V2's regression head structure"""
    def __init__(self, hidden_size, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

class V2TextModel(nn.Module):
    """V2 Teacher Model Architecture"""
    def __init__(self, model_name='microsoft/deberta-v3-large', dropout=0.3):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        self._attach_lora_top_layers(last_n_layers=6)
        
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
            r=16, lora_alpha=32, lora_dropout=0.1,
            bias="none", target_modules=target_modules, modules_to_save=[]
        )
        self.enc = get_peft_model(self.enc, cfg)
    
    def forward(self, batch):
        out = self.enc(**batch)
        pooled = self.pool(out.last_hidden_state, batch['attention_mask'])
        return self.reg_head(pooled)

class MeanPool(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1.0)

print("Loading V2 teacher models...")
teacher_models = []
for fold in range(5):
    model = V2TextModel()
    state_dict = torch.load(Config.V2_MODEL_DIR / f'text_fold{fold}.pt', map_location=Config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE)
    model.eval()
    teacher_models.append(model)
    print(f"  ✓ V2 Teacher Fold {fold} loaded")

print("✅ All V2 teacher models loaded\n")

# ============================================================================
# STUDENT MODEL (V4 ARCHITECTURE)
# ============================================================================

class V5StudentModel(nn.Module):
    """V5 Student: V4 architecture learning from V2 teacher"""
    def __init__(self, model_name=Config.MODEL_NAME, dropout=Config.DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self._attach_lora_top_layers(last_n_layers=Config.LORA_LAYERS)
        
        for n, p in self.encoder.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        
        hidden_size = self.encoder.config.hidden_size
        self.pool = MeanPool()
        
        # Absolute score head
        self.absolute_head = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        
        # Comparative head
        self.comparative_head = nn.Sequential(
            nn.Linear(hidden_size * 3, 1024),
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
            r=Config.LORA_R, lora_alpha=Config.LORA_ALPHA, 
            lora_dropout=Config.LORA_DROPOUT,
            bias="none", target_modules=target_modules, modules_to_save=[]
        )
        self.encoder = get_peft_model(self.encoder, cfg)
    
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        return pooled
    
    def forward_single(self, input_ids, attention_mask):
        emb = self.encode(input_ids, attention_mask)
        return self.absolute_head(emb).squeeze(-1)
    
    def forward_pair(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        emb1 = self.encode(input_ids1, attention_mask1)
        emb2 = self.encode(input_ids2, attention_mask2)
        
        score1 = self.absolute_head(emb1).squeeze(-1)
        score2 = self.absolute_head(emb2).squeeze(-1)
        
        comparative_features = torch.cat([emb1, emb2, torch.abs(emb1 - emb2)], dim=-1)
        score_diff = self.comparative_head(comparative_features).squeeze(-1)
        
        return score1, score2, score_diff

# ============================================================================
# DISTILLATION LOSS
# ============================================================================

def distillation_loss(student_pred, teacher_pred, true_label, temperature=Config.TEMPERATURE):
    """Knowledge distillation loss with temperature scaling"""
    # Soft targets from teacher (with temperature)
    soft_loss = nn.MSELoss()(student_pred / temperature, teacher_pred / temperature)
    
    # Hard targets (true labels)
    hard_loss = nn.MSELoss()(student_pred, true_label)
    
    return soft_loss, hard_loss

def combined_distillation_loss(student_pred, teacher_pred, true_label, 
                               score1=None, score2=None, score_diff=None,
                               label1=None, label2=None, teacher_diff=None):
    """Combined loss: distillation + hard labels + comparative"""
    
    # Distillation loss
    soft_loss, hard_loss = distillation_loss(student_pred, teacher_pred, true_label)
    
    # Comparative loss (if pairs provided)
    comp_loss = 0.0
    if score1 is not None and score2 is not None:
        # Learn from teacher's predicted differences
        true_diff = label1 - label2 if teacher_diff is None else teacher_diff
        comp_loss = nn.MSELoss()(score_diff, true_diff)
        
        # Ranking loss (prefer correct ordering)
        ranking_loss = torch.relu(0.5 - torch.sign(true_diff) * (score1 - score2))
        comp_loss = comp_loss + ranking_loss.mean()
    
    # Combined weighted loss
    total_loss = (Config.ALPHA_DISTILL * soft_loss + 
                  Config.ALPHA_HARD * hard_loss + 
                  Config.ALPHA_COMPARATIVE * comp_loss)
    
    return total_loss, soft_loss, hard_loss, comp_loss

# ============================================================================
# DATASET
# ============================================================================

class DistillationDataset(Dataset):
    """Dataset with teacher predictions as soft targets"""
    def __init__(self, texts, labels, teacher_preds, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.teacher_preds = teacher_preds
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        teacher_pred = self.teacher_preds[idx]
        
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32),
            'teacher_pred': torch.tensor(teacher_pred, dtype=torch.float32)
        }

class PairDistillationDataset(Dataset):
    """Pair dataset with teacher predictions"""
    def __init__(self, texts, labels, teacher_preds, pairs, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.teacher_preds = teacher_preds
        self.pairs = pairs
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
        teacher1 = self.teacher_preds[idx1]
        teacher2 = self.teacher_preds[idx2]
        
        enc1 = self.tokenizer(text1, add_special_tokens=True, max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        enc2 = self.tokenizer(text2, add_special_tokens=True, max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        
        return {
            'input_ids1': enc1['input_ids'].squeeze(0),
            'attention_mask1': enc1['attention_mask'].squeeze(0),
            'input_ids2': enc2['input_ids'].squeeze(0),
            'attention_mask2': enc2['attention_mask'].squeeze(0),
            'label1': torch.tensor(label1, dtype=torch.float32),
            'label2': torch.tensor(label2, dtype=torch.float32),
            'teacher1': torch.tensor(teacher1, dtype=torch.float32),
            'teacher2': torch.tensor(teacher2, dtype=torch.float32),
            'label_diff': torch.tensor(label1 - label2, dtype=torch.float32),
            'teacher_diff': torch.tensor(teacher1 - teacher2, dtype=torch.float32)
        }

# ============================================================================
# TRAINING
# ============================================================================

def generate_teacher_predictions(teacher_models, train_df, tokenizer, fold_idx):
    """Generate V2 teacher predictions for distillation"""
    print(f"\nGenerating V2 teacher predictions for fold {fold_idx}...")
    
    # Use the OTHER folds as teachers (out-of-fold style)
    teacher_preds = []
    
    dataset = DistillationDataset(
        train_df['transcript'].values,
        train_df['label'].values,
        np.zeros(len(train_df)),  # Placeholder
        tokenizer
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Average predictions from all teacher folds except current
    for t_fold, teacher in enumerate(teacher_models):
        if t_fold == fold_idx:
            continue  # Skip same fold
        
        fold_preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attn_mask = batch['attention_mask'].to(Config.DEVICE)
                
                with autocast(dtype=torch.bfloat16):
                    preds = teacher({'input_ids': input_ids, 'attention_mask': attn_mask})
                
                fold_preds.append(preds.float().cpu().numpy())
        
        teacher_preds.append(np.concatenate(fold_preds))
    
    # Average teacher predictions
    teacher_preds = np.mean(teacher_preds, axis=0)
    print(f"  ✓ Teacher predictions: mean={teacher_preds.mean():.3f}, std={teacher_preds.std():.3f}")
    
    return teacher_preds

def generate_pairs(labels, indices, strategy='stratified', max_pairs_per_sample=200):
    """Generate training pairs"""
    labels_subset = labels[indices]
    n = len(indices)
    
    if strategy == 'all':
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    elif strategy == 'stratified':
        # Balance by score difference
        pairs = []
        for i in range(n):
            diffs = np.abs(labels_subset - labels_subset[i])
            # Sample from different difficulty levels
            easy = np.where(diffs > 2.0)[0]
            medium = np.where((diffs > 1.0) & (diffs <= 2.0))[0]
            hard = np.where((diffs > 0.1) & (diffs <= 1.0))[0]
            
            n_per_level = max_pairs_per_sample // 3
            selected = []
            if len(easy) > 0:
                selected.extend(np.random.choice(easy, min(n_per_level, len(easy)), replace=False))
            if len(medium) > 0:
                selected.extend(np.random.choice(medium, min(n_per_level, len(medium)), replace=False))
            if len(hard) > 0:
                selected.extend(np.random.choice(hard, min(n_per_level, len(hard)), replace=False))
            
            for j in selected:
                if i < j:
                    pairs.append((i, j))
    
    return pairs[:len(indices) * max_pairs_per_sample]

def train_epoch_single(model, loader, optimizer, scaler, epoch):
    """Train on single samples with distillation"""
    model.train()
    losses = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} (Single)")
    for batch in pbar:
        input_ids = batch['input_ids'].to(Config.DEVICE)
        attn_mask = batch['attention_mask'].to(Config.DEVICE)
        labels = batch['label'].to(Config.DEVICE)
        teacher_preds = batch['teacher_pred'].to(Config.DEVICE)
        
        with autocast(dtype=torch.bfloat16):
            preds = model.forward_single(input_ids, attn_mask)
            soft_loss, hard_loss = distillation_loss(preds, teacher_preds, labels)
            loss = Config.ALPHA_DISTILL * soft_loss + Config.ALPHA_HARD * hard_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{np.mean(losses):.4f}'})
    
    return np.mean(losses)

def train_epoch_pairs(model, loader, optimizer, scaler, epoch):
    """Train on pairs with distillation"""
    model.train()
    losses = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} (Pairs)")
    for batch in pbar:
        input_ids1 = batch['input_ids1'].to(Config.DEVICE)
        attn_mask1 = batch['attention_mask1'].to(Config.DEVICE)
        input_ids2 = batch['input_ids2'].to(Config.DEVICE)
        attn_mask2 = batch['attention_mask2'].to(Config.DEVICE)
        label1 = batch['label1'].to(Config.DEVICE)
        label2 = batch['label2'].to(Config.DEVICE)
        teacher1 = batch['teacher1'].to(Config.DEVICE)
        teacher2 = batch['teacher2'].to(Config.DEVICE)
        teacher_diff = batch['teacher_diff'].to(Config.DEVICE)
        
        with autocast(dtype=torch.bfloat16):
            score1, score2, score_diff = model.forward_pair(
                input_ids1, attn_mask1, input_ids2, attn_mask2
            )
            
            # Distillation on both scores
            soft_loss1, hard_loss1 = distillation_loss(score1, teacher1, label1)
            soft_loss2, hard_loss2 = distillation_loss(score2, teacher2, label2)
            
            # Comparative loss with teacher differences
            comp_loss = nn.MSELoss()(score_diff, teacher_diff)
            
            loss = (Config.ALPHA_DISTILL * (soft_loss1 + soft_loss2) / 2 +
                   Config.ALPHA_HARD * (hard_loss1 + hard_loss2) / 2 +
                   Config.ALPHA_COMPARATIVE * comp_loss)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{np.mean(losses):.4f}'})
    
    return np.mean(losses)

def validate(model, loader):
    """Validation"""
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attn_mask = batch['attention_mask'].to(Config.DEVICE)
            
            with autocast(dtype=torch.bfloat16):
                pred = model.forward_single(input_ids, attn_mask)
            
            preds.append(pred.float().cpu().numpy())
            labels.append(batch['label'].numpy())
    
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    pearson = pearsonr(preds, labels)[0]
    
    return rmse, pearson, preds

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    train_df = pd.read_csv(Config.DATA_DIR / 'train.csv')
    print(f"Train samples: {len(train_df)}")
    
    # Load cached transcripts
    cache_path = Config.CACHE_DIR / 'train_transcripts.csv'
    if cache_path.exists():
        print("Loading cached transcripts...")
        cached = pd.read_csv(cache_path)
        train_df['transcript'] = cached['transcript']
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # 5-fold CV
    train_df['label_bin'] = pd.cut(train_df['label'], bins=5, labels=False)
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    oof_preds = np.zeros(len(train_df))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label_bin'])):
        print(f"\n{'='*80}")
        print(f"FOLD {fold}")
        print(f"{'='*80}")
        
        # Generate teacher predictions (out-of-fold)
        teacher_preds = generate_teacher_predictions(teacher_models, train_df, tokenizer, fold)
        
        # Create model
        model = V5StudentModel()
        model.to(Config.DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=0.01)
        scaler = GradScaler()
        
        # Datasets
        train_dataset = DistillationDataset(
            train_df.iloc[train_idx]['transcript'].values,
            train_df.iloc[train_idx]['label'].values,
            teacher_preds[train_idx],
            tokenizer
        )
        
        val_dataset = DistillationDataset(
            train_df.iloc[val_idx]['transcript'].values,
            train_df.iloc[val_idx]['label'].values,
            teacher_preds[val_idx],
            tokenizer
        )
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Generate pairs
        pairs = generate_pairs(train_df['label'].values, train_idx, strategy='stratified', max_pairs_per_sample=150)
        pair_dataset = PairDistillationDataset(
            train_df['transcript'].values,
            train_df['label'].values,
            teacher_preds,
            pairs,
            tokenizer
        )
        pair_loader = DataLoader(pair_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Pairs: {len(pairs)}")
        
        # Training
        best_rmse = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(1, Config.EPOCHS + 1):
            # Alternate between single and pair training
            if epoch % 2 == 1:
                train_loss = train_epoch_single(model, train_loader, optimizer, scaler, epoch)
            else:
                train_loss = train_epoch_pairs(model, pair_loader, optimizer, scaler, epoch)
            
            val_rmse, val_pearson, val_preds = validate(model, val_loader)
            
            print(f"Epoch {epoch}/{Config.EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val RMSE: {val_rmse:.4f}, Pearson: {val_pearson:.4f}")
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), Config.MODEL_DIR / f'model_fold{fold}_best.pth')
                print(f"  → Best model saved! (RMSE: {best_rmse:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        
        # Load best and predict OOF
        model.load_state_dict(torch.load(Config.MODEL_DIR / f'model_fold{fold}_best.pth'))
        _, _, oof_preds[val_idx] = validate(model, val_loader)
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Final OOF
    oof_rmse = np.sqrt(np.mean((oof_preds - train_df['label'].values) ** 2))
    oof_pearson = pearsonr(oof_preds, train_df['label'].values)[0]
    
    print(f"\n{'='*80}")
    print("V5 DISTILLATION RESULTS")
    print(f"{'='*80}")
    print(f"OOF RMSE: {oof_rmse:.4f}")
    print(f"OOF Pearson: {oof_pearson:.4f}")
    print(f"\nComparison:")
    print(f"  V2 (teacher): 0.5380")
    print(f"  V4 (student arch): 0.5106")
    print(f"  V5 (distillation): {oof_rmse:.4f}")
    print(f"\nModels saved to: {Config.MODEL_DIR}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
