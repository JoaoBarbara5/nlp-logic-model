# main.py
import pandas as pd
import torch
from config import CONFIG
from data_processor import Vocabulary
from train_utils import train_cross_validation, generate_submission

def main():
    # 1. Load Data
    try:
        train_df = pd.read_csv(CONFIG['train_path'])
        test_df = pd.read_csv(CONFIG['test_path'])
    except FileNotFoundError:
        print(f"Error: Could not find data at {CONFIG['train_path']}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    
    # 2. Build Vocabulary
    print("Building Vocabulary...")
    all_text = pd.concat([
        train_df['context'], 
        train_df['question'],
        train_df['answers'].apply(lambda x: " ".join(eval(x)) if isinstance(x, str) else "")
    ]).astype(str).tolist()
    
    vocab = Vocabulary(min_freq=CONFIG['min_freq'])
    vocab.build_vocab(all_text)
    print(f"Vocab size: {len(vocab.stoi)}")
    
    # 3. Pipeline
    # A. Cross Validation
    train_cross_validation(train_df, vocab, device)
    
    # B. Generate Submission (Train on full data -> Predict Test)
    generate_submission(train_df, test_df, vocab, device)

if __name__ == "__main__":
    main()