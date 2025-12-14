import pandas as pd
import json
import ast
import os
import shutil
from sklearn.model_selection import train_test_split
import subprocess
import sys

def convert_to_reclor_format(df, output_path, split_name):
    """
    Convert dataframe to Reclor JSON format and save.
    """
    data = []
    for idx, row in df.iterrows():
        try:
            answers = ast.literal_eval(row['answers'])
        except:
            # Fallback if already list or different format, though csv usually reads as str
            answers = row['answers']
            
        entry = {
            "id_string": str(row['id']),
            "context": row['context'],
            "question": row['question'],
            "answers": answers,
            "label": int(row['label']) if 'label' in row else 0
        }
        data.append(entry)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def run_full_training(train_df, test_df):
    run_name = "full_run"
    data_dir = f"cv_data/{run_name}"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save datasets
    # we use the full training set
    convert_to_reclor_format(train_df, os.path.join(data_dir, "train.json"), "train")
    # we use the test set
    convert_to_reclor_format(test_df, os.path.join(data_dir, "test.json"), "test")
    
    # Create dummy dev set to satisfy potential checks (though we won't run eval)
    convert_to_reclor_format(train_df.head(10), os.path.join(data_dir, "val.json"), "dev")
    
    # Create dummy 100_train.json and 100_val.json
    convert_to_reclor_format(train_df.head(10), os.path.join(data_dir, "100_train.json"), "train")
    convert_to_reclor_format(train_df.head(10), os.path.join(data_dir, "100_val.json"), "dev")
    
    output_dir = f"checkpoints/{run_name}"
    
    # Construct command
    cmd = [
        "python3", "run_multiple_choice.py",
        "--task_name", "reclor",
        "--model_type", "DAGN",
        "--token_encoder_type", "random",
        "--model_name_or_path", "roberta-large", 
        "--init_weights",
        "--do_train",
        "--do_predict", 
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--max_seq_length", "128",
        "--per_device_train_batch_size", "4",
        "--per_device_eval_batch_size", "4",
        "--gradient_accumulation_steps", "4",
        "--num_train_epochs", "5", 
        "--learning_rate", "1e-4", 
        "--roberta_lr", "1e-4", 
        "--gcn_lr", "1e-4",
        "--proj_lr", "1e-4",
        "--overwrite_output_dir"
    ]
    # Add DAGN specific args
    cmd.extend([
        "--graph_building_block_version", "2",
        "--data_processing_version", "2",
        "--merge_type", "1",
        "--gnn_version", "GCN",
        "--use_gcn",
        "--use_pool",
        "--gcn_steps", "1"
    ])
    
    print(f"Running Training on Full Dataset...")
    print(" ".join(cmd))
    
    # Run command
    subprocess.check_call(cmd)
    
    return output_dir

def main():
    # Load data
    if not os.path.exists("train.csv"):
        print("Error: train.csv not found.")
        sys.exit(1)
    if not os.path.exists("test.csv"):
        print("Error: test.csv not found.")
        sys.exit(1)
        
    print("Loading datasets...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    output_dir = run_full_training(train_df, test_df)
    
    # Post-processing predictions
    print("Processing predictions...")
    pred_file = os.path.join(output_dir, "predictions.npy")
    if os.path.exists(pred_file):
        import numpy as np
        
        preds = np.load(pred_file)
        # preds is an array of label indices (0-3) from run_multiple_choice.py (pred_ids)
        
        # Ensure alignment
        if len(preds) != len(test_df):
            print(f"Warning: Number of predictions ({len(preds)}) does not match test set size ({len(test_df)}).")
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'label': preds
        })
        
        # Save submission
        submission_file = "submission.csv"
        submission_df.to_csv(submission_file, index=False)
        print(f"Predictions saved to {submission_file}")
    else:
        print(f"Error: Predictions file not found at {pred_file}")

if __name__ == "__main__":
    main()
