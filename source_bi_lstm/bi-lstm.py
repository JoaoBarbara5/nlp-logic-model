import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import ast
import re
import itertools
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

def download_nltk_resources():
    resources = ['punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}')
        except LookupError:
            nltk.download(r, quiet=True)

download_nltk_resources()

PARAM_GRID = {
    'embedding_dim': [100],          
    'hidden_dim': [64, 128],         
    'learning_rate': [0.001],
    'batch_size': [16],
    'epochs': [5],
    'k_folds': [5],
    'vocab_min_freq': [2],
    'dropout': [0.2, 0.5]  
}

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
    
    def __len__(self):
        return len(self.stoi)

class MCQA_Dataset(Dataset):
    def __init__(self, df, vocab, preprocessor, max_len=128):
        self.df = df
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.max_len = max_len
        self.data = self._process_data()

    def _process_data(self):
        processed_data = []
        for i, row in self.df.iterrows():
            context = self.preprocessor.preprocess(row['context'])
            question = self.preprocessor.preprocess(row['question'])
            try: answers = ast.literal_eval(row['answers'])
            except: answers = ["", "", "", ""]
            label = int(row['label'])
            
            answer_indices = []
            for ans in answers:
                combined = context + ["<SEP>"] + question + ["<SEP>"] + self.preprocessor.preprocess(ans)
                indices = self.vocab.numericalize(combined)
                if len(indices) < self.max_len:
                    indices += [self.vocab.stoi["<PAD>"]] * (self.max_len - len(indices))
                else:
                    indices = indices[:self.max_len]
                answer_indices.append(indices)
            processed_data.append((answer_indices, label))
        return processed_data

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        inputs, label = self.data[idx]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate=0.5):
        super(BiLSTMAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
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

class GridSearchPipeline:
    def __init__(self, csv_path, param_grid):
        self.param_grid = param_grid
        self.df = pd.read_csv(csv_path)
        self.preprocessor = TextPreprocessor()
        self.vocab = Vocabulary(min_freq=2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self):
        print("Building Vocabulary...")
        all_sentences = []
        for _, row in self.df.iterrows():
            all_sentences.append(self.preprocessor.preprocess(row['context']))
            all_sentences.append(self.preprocessor.preprocess(row['question']))
            try:
                for a in ast.literal_eval(row['answers']):
                    all_sentences.append(self.preprocessor.preprocess(a))
            except: pass
        self.vocab.build(all_sentences)
        self.dataset = MCQA_Dataset(self.df, self.vocab, self.preprocessor)

    def run(self):
        self.prepare_data()
        
        keys, values = zip(*self.param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        best_acc = 0.0
        best_params = None
        
        print(f"\nStarting Grid Search over {len(param_combinations)} combinations...")
        
        for i, params in enumerate(param_combinations):
            print(f"\n--- Combination {i+1}: {params} ---")
            avg_acc = self.run_kfold(params)
            print(f"  -> Avg Validation Accuracy: {avg_acc:.2f}%")
            
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_params = params
                
        print("\n=============================================")
        print(f"Best Accuracy: {best_acc:.2f}%")
        print(f"Best Parameters: {best_params}")
        self.best_params = best_params
        
        if best_params:
            self.save_best_model(best_params)
        print("=============================================")

    def run_kfold(self, params):
        kfold = KFold(n_splits=params['k_folds'], shuffle=True, random_state=42)
        fold_accuracies = []
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.dataset)):
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)
            train_loader = DataLoader(self.dataset, batch_size=params['batch_size'], sampler=train_subsampler)
            val_loader = DataLoader(self.dataset, batch_size=params['batch_size'], sampler=val_subsampler)
            
            model = BiLSTMAttention(len(self.vocab), params['embedding_dim'], 
                                    params['hidden_dim'], dropout_rate=params['dropout'])
            model = model.to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            for epoch in range(params['epochs']):
                self.train_epoch(model, train_loader, criterion, optimizer)
            
            fold_accuracies.append(self.evaluate(model, val_loader))
            
        return np.mean(fold_accuracies)

    def save_best_model(self, params):
        print("\nRetraining final model on FULL dataset with best parameters...")
        
        final_model = BiLSTMAttention(len(self.vocab), params['embedding_dim'], 
                                      params['hidden_dim'], dropout_rate=params['dropout'])
        final_model = final_model.to(self.device)
        
        full_loader = DataLoader(self.dataset, batch_size=params['batch_size'], shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(final_model.parameters(), lr=params['learning_rate'])
        
        for epoch in range(params['epochs']):
            loss = self.train_epoch(final_model, full_loader, criterion, optimizer)
            print(f"  Epoch {epoch+1}/{params['epochs']} Loss: {loss:.4f}")
            
        save_path = "best_model.pth"
        torch.save(final_model.state_dict(), save_path)
        print(f"Model saved to '{save_path}'")

    def train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        total_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(self, model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

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

if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'assignment_data', 'train.csv')
    pipeline = GridSearchPipeline(csv_path, PARAM_GRID)
    pipeline.run()
    
    def predict_submission(pipeline, test_csv_path, model_path='best_model.pth', output_file='submission.csv'):
        print(f"\n--- Generating Predictions for {test_csv_path} ---")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, output_file)
        
        if not hasattr(pipeline, 'best_params') or pipeline.best_params is None:
            print("Error: pipeline.best_params not found. Please modify GridSearchPipeline.run() to store it.")
            return

        params = pipeline.best_params
        device = pipeline.device
        
        print(f"Loading model from '{model_path}' with params: {params}...")
        model = BiLSTMAttention(len(pipeline.vocab), 
                                params['embedding_dim'], 
                                params['hidden_dim'], 
                                dropout_rate=params['dropout'])
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() 
        
        
        test_df = pd.read_csv(test_csv_path)
        test_dataset = MCQA_TestDataset(test_df, pipeline.vocab, pipeline.preprocessor)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        
        ids = []
        predictions = []
        
        with torch.no_grad():
            for inputs, batch_ids in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                
                _, preds = torch.max(outputs, 1)
                
                ids.extend(batch_ids)
                predictions.extend(preds.cpu().numpy())
                
       
        submission = pd.DataFrame({'id': ids, 'label': predictions})
        submission.to_csv(output_file, index=False)
        print(f"Predictions saved to '{output_file}'")
    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', 'assignment_data', 'test.csv')
    predict_submission(pipeline, csv_path)