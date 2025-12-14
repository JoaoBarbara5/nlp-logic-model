import torch
import pandas as pd
import numpy as np
import re
import ast
from gensim.models import FastText

# ---------------------------------------------------------
# 1. Exact Cleaning Logic (Matches your tokenizer.py)
# ---------------------------------------------------------
class TextCleaner:
    def __init__(self):
        # Maps copied from your tokenizer.py
        self.tag_map = {
            '<b>': ' startbold ', '</b>': ' endbold ',
            '<br>': ' ', '<i>': ' ', '</i>': ' ', '<u>': ' ', '</u>': ' '
        }
        self.symbol_map = {'%': ' percent ', '$': ' dollar ', '&': ' and '}
        
        # Regex to remove non-alphanumeric (except hyphen)
        self.cleanup_pattern = re.compile(r"[^\w\s-]")

    def clean_text(self, text):
        if not isinstance(text, str): 
            return []
        
        # 1. Structural Tags
        for tag, replacement in self.tag_map.items():
            text = text.replace(tag, replacement)
            
        # 2. Safety Net for other HTML
        text = re.sub(r'<[^>]+>', ' ', text)

        # 3. Semantic Symbols
        for symbol, replacement in self.symbol_map.items():
            text = text.replace(symbol, replacement)

        # 4. Apostrophes
        text = text.replace("'", "")

        # 5. General Cleanup (removes punctuation like . , ! ?)
        text = self.cleanup_pattern.sub(' ', text)

        # 6. Lowercase and Split
        return text.lower().split()

# ---------------------------------------------------------
# 2. The Expansion Logic
# ---------------------------------------------------------
def build_expanded_embeddings(model_path, csv_paths):
    """
    Scans all CSVs, asks Gensim for vectors (using n-grams if needed),
    and appends special tokens <SEP>, <PAD>, <UNK>.
    """
    print(f"Loading Gensim model: {model_path}...")
    # This automatically loads the .npy n-grams file if present
    ft_model = FastText.load(model_path)
    embed_dim = ft_model.vector_size
    
    cleaner = TextCleaner()
    unique_words = set()
    
    # --- Step A: Scan ALL data to find every word ---
    print("Scanning datasets to build master vocabulary...")
    for csv_file in csv_paths:
        print(f"  - Processing {csv_file}...")
        try:
            df = pd.read_csv(csv_file)
            
            # Process Context and Question
            for text in df['context'].fillna(""):
                unique_words.update(cleaner.clean_text(text))
            
            for text in df['question'].fillna(""):
                unique_words.update(cleaner.clean_text(text))
                
            # Process Answers (stored as string representation of list)
            for ans_str in df['answers'].fillna("[]"):
                try:
                    # Convert "['ans1', 'ans2']" -> list ["ans1", "ans2"]
                    ans_list = ast.literal_eval(ans_str)
                    for ans in ans_list:
                        unique_words.update(cleaner.clean_text(ans))
                except:
                    pass # Skip malformed answer rows
                    
        except Exception as e:
            print(f"    Warning: Could not read {csv_file}. Error: {e}")

    # Convert to sorted list for deterministic indexing
    vocab_list = sorted(list(unique_words))
    vocab_size = len(vocab_list)
    print(f"Found {vocab_size} unique words.")

    # --- Step B: Build the Matrix ---
    # Size = Words + 3 Special Tokens
    matrix_tensor = torch.zeros((vocab_size + 3, embed_dim))
    word_to_idx = {}

    print("Generating vectors (using FastText n-grams for OOVs)...")
    
    # 1. Fill standard words
    for i, word in enumerate(vocab_list):
        # ft_model.wv[word] is the MAGIC line.
        # If 'word' is in training: returns exact vector.
        # If 'word' is OOV: constructs vector from n-grams.
        vector = ft_model.wv[word]
        matrix_tensor[i] = torch.tensor(vector)
        word_to_idx[word] = i

    # 2. Append Special Tokens
    # Indices:
    idx_pad = vocab_size
    idx_sep = vocab_size + 1
    idx_unk = vocab_size + 2
    
    # Initialize <PAD> as zeros (already done by torch.zeros, but being explicit)
    matrix_tensor[idx_pad] = torch.zeros(embed_dim)
    
    # Initialize <SEP> and <UNK> as random noise so they can be learned
    matrix_tensor[idx_sep] = torch.normal(mean=0.0, std=0.1, size=(embed_dim,))
    matrix_tensor[idx_unk] = torch.normal(mean=0.0, std=0.1, size=(embed_dim,))
    
    # Update Dictionary
    word_to_idx["<PAD>"] = idx_pad
    word_to_idx["<SEP>"] = idx_sep
    word_to_idx["<UNK>"] = idx_unk

    print("Special tokens added:")
    print(f"  <PAD>: {idx_pad}")
    print(f"  <SEP>: {idx_sep}")
    print(f"  <UNK>: {idx_unk}")
    
    return matrix_tensor, word_to_idx

# ---------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Add ALL your data files here (Train, Test, Validation)
    # It is safe to look at test data here because we aren't learning labels, 
    # we are just defining the vocabulary.
    my_csvs = ["train.csv"] # Add "test.csv" if you have it!
    model_file = "logic_fasttext.model"
    
    emb_matrix, vocab_dict = build_expanded_embeddings(model_file, my_csvs)
    
    # Save the results for your Logic Model to use
    torch.save(emb_matrix, "final_embeddings.pt")
    torch.save(vocab_dict, "final_vocab.pt")
    
    print("\nSUCCESS! Saved 'final_embeddings.pt' and 'final_vocab.pt'")
    print(f"Final Matrix Shape: {emb_matrix.shape}")