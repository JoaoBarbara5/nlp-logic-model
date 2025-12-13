# train_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from config import CONFIG
from dagn_model import DAGN
from data_processor import DAGNProcessor, collate_dagn

class LogicDataset(Dataset):
    def __init__(self, df, processor, is_test=False):
        self.df = df
        self.processor = processor
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        processed_opts = self.processor.process_sample(row)
        label = int(row['label']) if not self.is_test else -1
        return processed_opts, label, row['id']

def train_cross_validation(train_df, vocab, device):
    print(f"\n[Starting Cross-Validation] k={CONFIG['k_folds']}")
    kf = KFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    fold_accuracies = []
    
    processor = DAGNProcessor(vocab)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n--- Fold {fold+1} ---")
        train_sub = train_df.iloc[train_idx]
        val_sub = train_df.iloc[val_idx]
        
        train_ds = LogicDataset(train_sub, processor)
        val_ds = LogicDataset(val_sub, processor)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], 
                                  shuffle=True, collate_fn=collate_dagn)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], 
                                shuffle=False, collate_fn=collate_dagn)
        
        # Initialize DAGN Model
        model = DAGN(
            vocab_size=len(vocab.stoi),
            embed_dim=CONFIG['embed_dim'],
            hidden_size=CONFIG['hidden_dim'],
            dropout_prob=CONFIG['dropout'],
            gcn_steps=CONFIG['gcn_steps']
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(CONFIG['epochs']):
            model.train()
            train_loss = 0
            
            for _, batch_data, y in train_loader:
                y = y.to(device)
                # Unpack batch_data dict to device
                inputs = {k: v.to(device) for k, v in batch_data.items()}
                
                optimizer.zero_grad()
                
                # Forward Pass: Returns [Batch*4, 1]
                logits_flat = model(**inputs)
                
                # Reshape to [Batch, 4] for CrossEntropy
                logits = logits_flat.view(-1, 4)
                
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _, batch_data, y in val_loader:
                    y = y.to(device)
                    inputs = {k: v.to(device) for k, v in batch_data.items()}
                    
                    logits_flat = model(**inputs)
                    logits = logits_flat.view(-1, 4)
                    
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            
            val_acc = correct / total
            print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        fold_accuracies.append(val_acc)
        
    print(f"\nCV Complete. Avg Acc: {np.mean(fold_accuracies):.4f}")

def generate_submission(train_df, test_df, vocab, device):
    print("\n[Training Final Model & Generating Submission]")
    processor = DAGNProcessor(vocab)
    full_ds = LogicDataset(train_df, processor)
    test_ds = LogicDataset(test_df, processor, is_test=True)
    
    train_loader = DataLoader(full_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_dagn)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_dagn)
    
    model = DAGN(len(vocab.stoi), CONFIG['embed_dim'], CONFIG['hidden_dim'], 
                 CONFIG['dropout'], CONFIG['gcn_steps']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    
    # Train
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        for _, batch_data, y in train_loader:
            y = y.to(device)
            inputs = {k: v.to(device) for k, v in batch_data.items()}
            optimizer.zero_grad()
            logits = model(**inputs).view(-1, 4)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")
        
    # Predict
    model.eval()
    all_preds = []
    all_ids = []
    with torch.no_grad():
        for ids, batch_data, _ in test_loader:
            inputs = {k: v.to(device) for k, v in batch_data.items()}
            logits = model(**inputs).view(-1, 4)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_ids.extend(ids)
            
    sub = pd.DataFrame({'id': all_ids, 'label': all_preds})
    sub.to_csv('submission.csv', index=False)
    print("Submission saved.")