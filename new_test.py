import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import ast
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm 

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

CSV_PATH = r"aml-2025-read-between-the-lines/train.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16          
LEARNING_RATE = 1.0      
L2_REG = 1e-5            
DROPOUT = 0.3            
NUM_EPOCHS = 10          

EMBED_DIM = 100
HIDDEN_DIM = 100

print("Initializing Pipeline...")
pipeline = TextPipeline() 
full_dataset, vocab_size = pipeline.prepare_dataset(CSV_PATH)

val_size = 800 
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Data Loaded: {len(train_dataset)} Training, {len(val_dataset)} Validation")

model = ReasoningModel(
    vocab_size=vocab_size, 
    embed_dim=EMBED_DIM, 
    hidden_dim=HIDDEN_DIM, 
    dropout=DROPOUT
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adadelta(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=L2_REG 
)

def calculate_accuracy(preds, labels):
    predicted_classes = torch.argmax(preds, dim=1)
    correct = (predicted_classes == labels).sum().item()
    return correct

print("\n--- Starting Training ---")

for epoch in range(NUM_EPOCHS):
    
    model.train()
    total_train_loss = 0
    train_correct = 0
    total_train_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    
    for ctx_batch, hyp_batch, label_batch in progress_bar:
        ctx_batch = ctx_batch.to(DEVICE)
        hyp_batch = hyp_batch.to(DEVICE)
        label_batch = label_batch.to(DEVICE)
        
        optimizer.zero_grad() 
        logits = model(ctx_batch, hyp_batch) 
        
        loss = criterion(logits, label_batch)
        
        loss.backward()
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_train_loss += loss.item()
        train_correct += calculate_accuracy(logits, label_batch)
        total_train_samples += label_batch.size(0)
        
        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = train_correct / total_train_samples

    model.eval()
    total_val_loss = 0
    val_correct = 0
    total_val_samples = 0
    
    with torch.no_grad():
        for ctx_batch, hyp_batch, label_batch in val_loader:
            ctx_batch = ctx_batch.to(DEVICE)
            hyp_batch = hyp_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            
            logits = model(ctx_batch, hyp_batch)
            loss = criterion(logits, label_batch)
            
            total_val_loss += loss.item()
            val_correct += calculate_accuracy(logits, label_batch)
            total_val_samples += label_batch.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = val_correct / total_val_samples

    print(f"\nEpoch {epoch+1} Results:")
    print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
    
    # Save best model (Optional)
    # if val_acc > best_val_acc:
    #     torch.save(model.state_dict(), "best_model.pth")
    
    print("-" * 30)

print("Training Complete.")
