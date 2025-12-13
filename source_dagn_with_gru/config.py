# config.py

CONFIG = {
    'embed_dim': 128,
    'hidden_dim': 128,      # This serves as both LSTM hidden size and Node dimension
    'gcn_steps': 1,         # Number of GCN iterations (DAGN paper uses 1 or 2)
    'dropout': 0.2,
    'lr': 0.001,
    'batch_size': 16,
    'epochs': 15,
    'k_folds': 5,
    'min_freq': 2,          # Vocab min frequency
    
    # Paths - Update these to your local paths
    'train_path': '/Users/malakkhan/Desktop/UVA/Semester 1/AML/nlp-logic-model/assignment_data/train.csv',
    'test_path': '/Users/malakkhan/Desktop/UVA/Semester 1/AML/nlp-logic-model/assignment_data/test.csv',
}

# Discourse Markers (Simplified for whitespace tokenization)
# In a pro-setup with BPE, you would map multi-word phrases to token spans.
# Here we list common single-word markers or handle them in the processor.
DISCOURSE_MARKERS = {
    'because', 'since', 'therefore', 'thus', 'hence', 'consequently', 
    'however', 'but', 'although', 'though', 'while', 'if', 'unless', 
    'except', 'instead', 'meanwhile', 'then', 'so', 'and', 'or'
}

PUNCTUATION_MARKERS = {'.', ',', ';', '?', '!'}