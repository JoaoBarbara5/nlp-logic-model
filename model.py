import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
import ast

class Tokenizer: # Removed (AutoTokenizer) inheritance, cleaner to use composition
    def __init__(self, max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def tokenize_pair(self, context_list, hypothesis_list):
        ctx_ids = self.tokenizer(
            context_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )['input_ids']

        hyp_ids = self.tokenizer(
            hypothesis_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )['input_ids']

        return ctx_ids, hyp_ids

    def process_csv(self, df):
        all_contexts = []
        all_hypotheses = []
        all_labels = []

        for index, row in df.iterrows():
            context = row['context']
            question = row['question']
            options = ast.literal_eval(row['answers'])
            label = row['label'] 
            
            for i, option in enumerate(options):
                all_contexts.append(context)
                hypothesis = f"{question} {option}"
                all_hypotheses.append(hypothesis)
            
            all_labels.append(label)

        ctx_ids_flat, hyp_ids_flat = self.tokenize_pair(all_contexts, all_hypotheses)
        
        num_questions = len(df)
        
        # [Batch_Size, 4_Options, Seq_Len]
        ctx_tensor = ctx_ids_flat.view(num_questions, 4, -1)
        hyp_tensor = hyp_ids_flat.view(num_questions, 4, -1)
        label_tensor = torch.tensor(all_labels)

        return ctx_tensor, hyp_tensor, label_tensor

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout_rate=0.3):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embed_dim,
            padding_idx=0
        )
        # The paper mentions using dropout on the embedding layer 
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        # 1. Embed the words
        x = self.embedding(input_ids)
        # 2. Apply dropout
        x = self.dropout(x)
        return x


class BiLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout_rate=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,    # 100 in your case
            hidden_size=hidden_dim,  # The size of the "memory" vector
            num_layers=num_layers,   
            batch_first=True,        # We use [Batch, Seq, Feature]
            bidirectional=True       # This makes it a BLSTM 
        )
        
        # The paper applies dropout to the LSTM layer as well 
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: [Batch_Size, Seq_Len, Embed_Dim]
        
        # output contains the hidden states for EVERY word in the sequence
        # (h_n, c_n) contains only the FINAL state (we usually ignore this for Attention models)
        lstm_output, (h_n, c_n) = self.lstm(x)
        
        # Apply dropout to the output features
        return self.dropout(lstm_output)




# --- LOAD DATA ---
print("Loading Data...")
# Ensure path is correct
tokenizer = Tokenizer()
df = pd.read_csv(r"aml-2025-read-between-the-lines\train.csv", sep=",")
df_batch = df.iloc[0:32,:]
ctx_tensor, hyp_tensor, label_tensor = tokenizer.process_csv(df_batch)

# --- CONFIGURATION ---
VOCAB_SIZE = tokenizer.tokenizer.vocab_size  
EMBED_DIM = 100          # From paper/your setup
HIDDEN_DIM = 100         # Size of LSTM memory
DROPOUT = 0.3            # From paper [cite: 151]

# --- INITIALIZE LAYERS ---
print("Loading Embedder...")
embed_layer = EmbeddingLayer(VOCAB_SIZE, EMBED_DIM, dropout_rate=DROPOUT)
print("Loading BLTSM...")
lstm_layer = BiLSTMLayer(EMBED_DIM, HIDDEN_DIM, dropout_rate=DROPOUT)

# --- EXECUTION ---

# 1. Flatten the inputs (Batch and Options together) 
# We treat every option as a separate sentence for now
batch_size, num_options, seq_len = ctx_tensor.shape
flat_ctx = ctx_tensor.view(-1, seq_len) # Shape: [Batch*4, 128]

# 2. Pass through Embedding
# Input: [Batch*4, 128] -> Output: [Batch*4, 128, 100]
embedded_vectors = embed_layer(flat_ctx) 

# 3. Pass through BiLSTM
# Input: [Batch*4, 128, 100] -> Output: [Batch*4, 128, 200]
# Note: Output dim is 200 because it is 100 (Forward) + 100 (Backward)
lstm_features = lstm_layer(embedded_vectors)

print("\n--- STATUS CHECK ---")
print(f"Original Input: {flat_ctx.shape}")
print(f"Embedding Out:  {embedded_vectors.shape}")
print(f"LSTM Output:    {lstm_features.shape}")