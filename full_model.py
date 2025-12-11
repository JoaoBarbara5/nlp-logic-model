import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import ast
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from transformers import AutoTokenizer
from tqdm import tqdm 
import itertools
import copy
import numpy as np


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, H):
        """
        H: Output of BiLSTM [Batch, Seq_Len, Hidden_Dim]
        """
        M = torch.tanh(H) 
        
        scores = torch.matmul(M, self.w)  
        alpha = F.softmax(scores, dim=1)  
        
        r = torch.sum(H * alpha.unsqueeze(-1), dim=1) 
        
        h_star = torch.tanh(r)
        
        return h_star

class AttBiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout_rate):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout_emb = nn.Dropout(dropout_rate)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout_lstm = nn.Dropout(dropout_rate)
        
        self.attention = AttentionLayer(hidden_dim * 2)

    def forward(self, x):
        
        emb = self.dropout_emb(self.embedding(x))
        
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout_lstm(lstm_out)
        
        sentence_vector = self.attention(lstm_out)
        
        return sentence_vector

class ReasoningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=100, dropout=0.3):
        super().__init__()
        
        self.encoder = AttBiLSTMEncoder(vocab_size, embed_dim, hidden_dim, dropout)
        
        self.classifier = nn.Linear(hidden_dim * 2 * 2, 1) 

    def forward(self, ctx_input, hyp_input):
        """
        Input Shapes: [Batch, 4_Options, Seq_Len]
        """
        batch_size, num_options, seq_len = ctx_input.shape
        
        flat_ctx = ctx_input.view(-1, seq_len)
        flat_hyp = hyp_input.view(-1, seq_len)
        
        vec_ctx = self.encoder(flat_ctx)
        vec_hyp = self.encoder(flat_hyp)
        
        combined = torch.cat((vec_ctx, vec_hyp), dim=1) 
        
        logits = self.classifier(combined)
        
        return logits.view(batch_size, num_options)



class TextPipeline:
    def __init__(self, max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def prepare_dataset(self, csv_path):
        print(f"Reading {csv_path}...")
        df = pd.read_csv(csv_path)
        
        all_ctx = []
        all_hyp = []
        all_labels = []

        for _, row in df.iterrows():
            context = row['context']
            question = row['question']
            label = row['label'] 
            
            try:
                options = ast.literal_eval(row['answers'])
            except:
                options = row['answers'] 
            
            
            q_ctx = []
            q_hyp = []
            
            for option in options:
                q_ctx.append(context)
                q_hyp.append(f"{question} {option}")
                
            all_ctx.append(q_ctx)  
            all_hyp.append(q_hyp)   
            all_labels.append(label)

        flat_ctx = [item for sublist in all_ctx for item in sublist]
        flat_hyp = [item for sublist in all_hyp for item in sublist]
        
        print("Tokenizing... (This may take a moment)")
        encoded_ctx = self.tokenizer(flat_ctx, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")['input_ids']
        encoded_hyp = self.tokenizer(flat_hyp, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")['input_ids']
        
        num_q = len(df)
        tensor_ctx = encoded_ctx.view(num_q, 4, -1)
        tensor_hyp = encoded_hyp.view(num_q, 4, -1)
        tensor_lbl = torch.tensor(all_labels)
        
        return TensorDataset(tensor_ctx, tensor_hyp, tensor_lbl), self.tokenizer.vocab_size



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = r"aml-2025-read-between-the-lines/train.csv" 

print("Initializing Pipeline...")
pipeline = TextPipeline() 
full_dataset, vocab_size = pipeline.prepare_dataset(CSV_PATH)

print(f"Vocab Size: {vocab_size}")
print(f"Dataset Size: {len(full_dataset)}")


param_grid = {
    'batch_size': [32],        
    'hidden_dim': [64, 100],      
    'dropout': [0.3, 0.5],         
    'lr': [1.0],                   
    'epochs': [10]                  
}

keys, values = zip(*param_grid.items())
grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Total configurations to test: {len(grid_combinations)}")

def train_evaluate(config, dataset, vocab_size):
    print(f"\nTesting Config: {config}")
    
    val_size = 800
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42) 
    )
    
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
    

    model = ReasoningModel(
        vocab_size=vocab_size,
        embed_dim=100, 
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    ).to(DEVICE)
    
   
    optimizer = torch.optim.Adadelta(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=1e-5 
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(config['epochs']):
        model.train()
        for ctx, hyp, lbl in train_loader:
            ctx, hyp, lbl = ctx.to(DEVICE), hyp.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ctx, hyp)
            loss = criterion(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for ctx, hyp, lbl in val_loader:
                ctx, hyp, lbl = ctx.to(DEVICE), hyp.to(DEVICE), lbl.to(DEVICE)
                logits = model(ctx, hyp)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == lbl).sum().item()
                total += lbl.size(0)
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    print(f" -> Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc

results = []

for config in grid_combinations:
    try:
        acc = train_evaluate(config, full_dataset, vocab_size)
        results.append((acc, config))
    except RuntimeError as e:
        print(f"Skipping config {config} due to OOM or Error: {e}")

# --- REPORT RESULTS ---
results.sort(key=lambda x: x[0], reverse=True) 

print("\n" + "="*30)
print("GRID SEARCH RESULTS")
print("="*30)
print(f"Top Performer: {results[0][0]*100:.2f}% accuracy")
best_config = results[0][1] # Grab the dictionary of the best params
print(f"Best Parameters: {best_config}")
print("-" * 30)

# --- RETRAIN FINAL MODEL (CRITICAL FIX) ---
# We must re-create the model using the best parameters to save the weights
print("\nRetraining final model with best parameters...")

# 1. Setup Data again (Standard Split)
val_size = 800
train_size = len(full_dataset) - val_size
train_subset, val_subset = random_split(
    full_dataset, [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_subset, batch_size=best_config['batch_size'], shuffle=True)
val_loader = DataLoader(val_subset, batch_size=best_config['batch_size'], shuffle=False)

# 2. Re-Initialize Model with WINNING Hyperparameters
final_model = ReasoningModel(
    vocab_size=vocab_size, 
    embed_dim=100, 
    hidden_dim=best_config['hidden_dim'], 
    dropout=best_config['dropout']
).to(DEVICE)


final_optimizer = torch.optim.Adadelta(
    final_model.parameters(), 
    lr=best_config['lr'], 
    weight_decay=1e-5 
)
criterion = nn.CrossEntropyLoss()

FINAL_EPOCHS = 10 

for epoch in range(FINAL_EPOCHS):
    final_model.train()
    total_loss = 0
    
    progress = tqdm(train_loader, desc=f"Final Train Epoch {epoch+1}")
    
    for ctx, hyp, lbl in progress:
        ctx, hyp, lbl = ctx.to(DEVICE), hyp.to(DEVICE), lbl.to(DEVICE)
        
        final_optimizer.zero_grad()
        logits = final_model(ctx, hyp)
        loss = criterion(logits, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 5.0)
        final_optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})

SAVE_PATH = "best_att_bilstm_model.pth"

checkpoint = {
    'epoch': FINAL_EPOCHS,
    'model_state_dict': final_model.state_dict(),       
    'optimizer_state_dict': final_optimizer.state_dict(), 
    'config': {
        'vocab_size': vocab_size,
        'embed_dim': 100,
        'hidden_dim': best_config['hidden_dim'],
        'dropout': best_config['dropout'],
        'batch_size': best_config['batch_size']
    },
    'best_val_accuracy': results[0][0]
}

torch.save(checkpoint, SAVE_PATH)
print(f"Final model trained and saved to {SAVE_PATH} successfully.")