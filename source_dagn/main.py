# main.py
import pandas as pd
import torch
from data_utils import Vocabulary
from train_utils import train_cross_validation, train_final_model, generate_submission
from config import CONFIG

def main():
    # 1. Setup
    try:
        train_df = pd.read_csv(CONFIG['train_path'])
        test_df = pd.read_csv(CONFIG['test_path'])
    except FileNotFoundError:
        print("Error: train.csv or test.csv not found in the given directory.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # 2. Build Vocabulary
    print("Building Vocabulary...")
    all_text = pd.concat([train_df['context'], train_df['question'], 
                          test_df['context'], test_df['question']]).astype(str).tolist()
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocab(all_text)
    print(f"Vocab size: {len(vocab.stoi)}")
    
    # 3. Execution Pipeline
    # A. Cross-Validation to check performance
    train_cross_validation(train_df, vocab, device)
    
    # B. Train final model on all data
    #final_model = train_final_model(train_df, vocab, device)
    
    # C. Generate submission file
    #generate_submission(final_model, test_df, vocab, device)

if __name__ == "__main__":
    main()