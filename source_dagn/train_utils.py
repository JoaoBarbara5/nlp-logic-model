import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

from config import CONFIG
from model import DAGNRelational
from data_utils import LogicalReasoningDataset, collate_fn

def train_cross_validation(train_df, vocab, device):
    """Trains a DAGNRelational model using K-Fold cross-validation.

    This function performs K-Fold cross-validation on the training data to evaluate
    the model's performance and provide a more robust estimate of its generalization
    capability. It trains a new model for each fold and reports the average accuracy.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training data, including
            features and labels.
        vocab (object): A vocabulary object (e.g., an instance of `Vocab`) that
            maps tokens to numerical indices.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to
            perform training and evaluation.

    Returns:
        float: The average validation accuracy across all folds.
    """
    print(f"\n[Starting Cross-Validation] k={CONFIG['k_folds']}")
    kf = KFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n--- Fold {fold+1} ---")
        train_sub = train_df.iloc[train_idx]
        val_sub = train_df.iloc[val_idx]
        
        train_loader = DataLoader(LogicalReasoningDataset(train_sub, vocab), 
                                  batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(LogicalReasoningDataset(val_sub, vocab), 
                                batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
        
        model = DAGNRelational(len(vocab.stoi), CONFIG['embed_dim'], CONFIG['hidden_dim'], 
                               CONFIG['dropout'], CONFIG['num_relations']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(CONFIG['epochs']):
            model.train()
            train_loss = 0
            for _, x, adjs, graph_counts, y in train_loader:
                x, graph_counts, y = x.to(device), graph_counts.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x, adjs, graph_counts)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for _, x, adjs, graph_counts, y in val_loader:
                    x, graph_counts, y = x.to(device), graph_counts.to(device), y.to(device)
                    logits = model(x, adjs, graph_counts)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            
            val_acc = correct / total
            print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        fold_accuracies.append(val_acc)
    
    avg_acc = np.mean(fold_accuracies)
    print(f"\nCV Complete. Average Accuracy: {avg_acc:.4f} (+/- {np.std(fold_accuracies):.4f})")
    return avg_acc

def train_final_model(train_df, vocab, device):
    """Trains the final DAGNRelational model on the entire training dataset.

    After cross-validation, this function trains a single model using all available
    training data. This model is then typically used for making predictions on
    unseen test data.

    Args:
        train_df (pd.DataFrame): DataFrame containing the full training data.
        vocab (object): A vocabulary object that maps tokens to numerical indices.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to
            perform training.

    Returns:
        torch.nn.Module: The trained DAGNRelational model.
    """
    print("\n[Training Final Model on Full Dataset]")
    full_ds = LogicalReasoningDataset(train_df, vocab)
    full_loader = DataLoader(full_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    model = DAGNRelational(len(vocab.stoi), CONFIG['embed_dim'], CONFIG['hidden_dim'], 
                           CONFIG['dropout'], CONFIG['num_relations']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        for _, x, adjs, graph_counts, y in full_loader:
            x, graph_counts, y = x.to(device), graph_counts.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, adjs, graph_counts)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {total_loss/len(full_loader):.4f}")
        
    return model

def generate_submission(model, test_df, vocab, device):
    """Generates a submission file with predictions for the test dataset.

    This function takes a trained model, processes the test data, makes predictions,
    and saves them to a CSV file named 'submission.csv' in the format required
    for submission.

    Args:
        model (torch.nn.Module): The trained DAGNRelational model.
        test_df (pd.DataFrame): DataFrame containing the test data, including
            unique identifiers for each sample.
        vocab (object): A vocabulary object that maps tokens to numerical indices.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to
            perform inference.

    Returns:
        None: The function saves the submission file directly to disk.
    """
    print("\n[Generating Submission]")
    test_ds = LogicalReasoningDataset(test_df, vocab, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    model.eval()
    predictions = []
    ids_list = []
    
    with torch.no_grad():
        for batch_ids, x, adjs, graph_counts, _ in test_loader:
            x, graph_counts = x.to(device), graph_counts.to(device)
            logits = model(x, adjs, graph_counts)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            ids_list.extend(batch_ids)
            
    submission = pd.DataFrame({'id': ids_list, 'label': predictions})
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to 'submission.csv'")