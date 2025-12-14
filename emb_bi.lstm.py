import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ast
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

CONFIG = {
    'embedding_dim': 100,      
    'hidden_dim': 128,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 20,
    'dropout': 0.4,
    'vocab_min_freq': 2
}

def download_nltk_resources():
    resources = ['punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}')
        except LookupError:
            nltk.download(r, quiet=True)

download_nltk_resources()

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens

class Vocabulary:
    def __init__(self, min_freq=1):
        self.stoi = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
        self.itos = {0: "<PAD>", 1: "<UNK>", 2: "<SEP>"}
        self.min_freq = min_freq
        
    def build(self, sentence_list):
        freqs = {}
        for sentence in sentence_list:
            for word in sentence:
                freqs[word] = freqs.get(word, 0) + 1
        
        idx = 3
        for word, freq in freqs.items():
            if freq >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
                
    def numericalize(self, tokens):
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]
    
    def __len__(self): return len(self.stoi)

class MCQA_Dataset(Dataset):
    def __init__(self, df, vocab, preprocessor, max_len=128):
        self.df = df
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.max_len = max_len
        self.data = self._process_data()

    def _process_data(self):
        processed = []
        for _, row in self.df.iterrows():
            context = self.preprocessor.preprocess(row['context'])
            question = self.preprocessor.preprocess(row['question'])
            try: answers = ast.literal_eval(row['answers'])
            except: answers = ["", "", "", ""]
            label = int(row['label'])
            
            ans_indices = []
            for ans in answers:
                combined = context + ["<SEP>"] + question + ["<SEP>"] + self.preprocessor.preprocess(ans)
                indices = self.vocab.numericalize(combined)
                if len(indices) < self.max_len:
                    indices += [self.vocab.stoi["<PAD>"]] * (self.max_len - len(indices))
                else:
                    indices = indices[:self.max_len]
                ans_indices.append(indices)
            processed.append((ans_indices, label))
        return processed

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0], dtype=torch.long), torch.tensor(self.data[idx][1], dtype=torch.long)

class MCQA_TestDataset(Dataset):
    def __init__(self, df, vocab, preprocessor, max_len=128):
        self.df = df
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.max_len = max_len
        self.data = self._process_data()

    def _process_data(self):
        processed = []
        for _, row in self.df.iterrows():
            row_id = row['id']
            context = self.preprocessor.preprocess(row['context'])
            question = self.preprocessor.preprocess(row['question'])
            try: answers = ast.literal_eval(row['answers'])
            except: answers = ["", "", "", ""]
            
            ans_indices = []
            for ans in answers:
                combined = context + ["<SEP>"] + question + ["<SEP>"] + self.preprocessor.preprocess(ans)
                indices = self.vocab.numericalize(combined)
                if len(indices) < self.max_len:
                    indices += [self.vocab.stoi["<PAD>"]] * (self.max_len - len(indices))
                else:
                    indices = indices[:self.max_len]
                ans_indices.append(indices)
            processed.append((ans_indices, row_id))
        return processed

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0], dtype=torch.long), self.data[idx][1]
    def __len__(self): return len(self.data)

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_embeddings=None, dropout_rate=0.5):
        super(BiLSTMAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True 
        else:
            nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
            
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention_linear = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, num_choices, seq_len = x.size()
        x_flat = x.view(-1, seq_len)
        
        embedded = self.embedding(x_flat)
        embedded = self.dropout(embedded)
        
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        
        attn_scores = self.attention_linear(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        context_vector = self.dropout(context_vector)
        
        logits = self.classifier(context_vector)
        logits = logits.view(batch_size, num_choices)
        
        return logits

class FullTrainPipeline:
    def __init__(self, train_path, test_path, config):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.vocab = Vocabulary(min_freq=config['vocab_min_freq'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_matrix = None
        self.model = None
        
    def prepare_resources(self):
        print("1. Preparing Resources (Training Data ONLY)...")
        
        train_sentences = []
        
        for _, row in self.train_df.iterrows():
            train_sentences.append(self.preprocessor.preprocess(row['context']))
            train_sentences.append(self.preprocessor.preprocess(row['question']))
            try:
                for a in ast.literal_eval(row['answers']):
                    train_sentences.append(self.preprocessor.preprocess(a))
            except: pass
        
        self.vocab.build(train_sentences)
        print(f"   Vocabulary Size: {len(self.vocab)}")
        
        print("   Training Word2Vec on training corpus...")
        w2v_model = Word2Vec(sentences=train_sentences, 
                             vector_size=self.config['embedding_dim'], 
                             window=5, min_count=1, workers=4)
        
        print("   Initializing Embedding Matrix...")
        vocab_size = len(self.vocab)
        emb_dim = self.config['embedding_dim']
        self.embedding_matrix = np.zeros((vocab_size, emb_dim))
        
        hits = 0
        misses = 0
        for word, idx in self.vocab.stoi.items():
            if word in w2v_model.wv:
                self.embedding_matrix[idx] = w2v_model.wv[word]
                hits += 1
            else:
                self.embedding_matrix[idx] = np.random.normal(0, 0.1, emb_dim)
                misses += 1
                
        print(f"   Embeddings: {hits} hits, {misses} misses.")
        
    def train(self):
        print("\n2. Starting Full Training...")
        
        train_dataset = MCQA_Dataset(self.train_df, self.vocab, self.preprocessor)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        self.model = BiLSTMAttention(len(self.vocab), 
                                     self.config['embedding_dim'], 
                                     self.config['hidden_dim'],
                                     pretrained_embeddings=self.embedding_matrix, 
                                     dropout_rate=self.config['dropout'])
        self.model = self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            print(f"   Epoch {epoch+1}/{self.config['epochs']} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        torch.save(self.model.state_dict(), 'final_model.pth')
        print("   Training Complete. Model saved to 'final_model.pth'")

    def predict(self):
        print("\n3. Generating Predictions on Test Set...")
        
        self.model.eval()
        
        test_dataset = MCQA_TestDataset(self.test_df, self.vocab, self.preprocessor)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        ids = []
        predictions = []
        
        with torch.no_grad():
            for inputs, batch_ids in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                _, preds = torch.max(outputs, 1)
                
                ids.extend(batch_ids)
                predictions.extend(preds.cpu().numpy())
                
        submission = pd.DataFrame({'id': ids, 'prediction': predictions})
        submission.to_csv('submission.csv', index=False)
        print("   Predictions saved to 'submission.csv'")

if __name__ == "__main__":
    pipeline = FullTrainPipeline('train.csv', 'test.csv', CONFIG)
    
    pipeline.prepare_resources() 
    pipeline.train()             
    pipeline.predict()