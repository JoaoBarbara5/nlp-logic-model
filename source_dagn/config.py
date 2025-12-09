# config.py

CONFIG = {
    'embed_dim': 128,
    'hidden_dim': 128,
    'gnn_layers': 2,
    'dropout': 0.3,
    'lr': 0.001,
    'batch_size': 32,
    'epochs': 15,
    'k_folds': 5,
    'num_relations': 5, 
    'train_path': '/Users/malakkhan/Desktop/UVA/Semester 1/AML/nlp-logic-model/assignment_data/train.csv',
    'test_path': '/Users/malakkhan/Desktop/UVA/Semester 1/AML/nlp-logic-model/assignment_data/test.csv',
}

# Discourse Markers from Huang et al. (2021) Table 4 + Standard PDTB markers
# Sorted by length (descending) to ensure phrases like "on the contrary" are matched before "on"
DISCOURSE_MARKERS = [
    'on the one hand', 'on the other hand', 'as an alternative', 'on the contrary',
    'in other words', 'simultaneously', 'consequently', 'specifically', 'in the end',
    'accordingly', 'by contrast', 'furthermore', 'nevertheless', 'regardless',
    'insofar as', 'conversely', 'when and if', 'as soon as', 'meanwhile', 
    'thereafter', 'therefore', 'likewise', 'meantime', 'although', 'in short', 
    'however', 'earlier', 'neither', 'because', 'instead', 'whereas', 'unless', 
    'except', 'in sum', 'by then', 'as well', 'much as', 'though', 'before', 
    'after', 'since', 'while', 'hence', 'least', 'plus', 'then', 'else', 
    'thus', 'lest', 'yet', 'but', 'and', 'for', 'if', 'or', 'so'
]

# Punctuation delimiters
PUNCTUATION_MARKERS = [
    '.', ',', ';', '?', '!'
]

# Mapping markers to relations (Simplified taxonomy)
# 0: Contrast, 1: Cause, 2: Condition, 3: Temporal, 4: Expansion (Default)
RELATION_MAP = {
    # Contrast
    'but': 0, 'however': 0, 'although': 0, 'yet': 0, 'nevertheless': 0,
    'whereas': 0, 'on the contrary': 0, 'by contrast': 0, 'conversely': 0,
    'instead': 0, 'regardless': 0, 'on the one hand': 0, 'on the other hand': 0,

    # Cause/Effect
    'because': 1, 'since': 1, 'therefore': 1, 'thus': 1, 'hence': 1,
    'consequently': 1, 'accordingly': 1, 'for': 1, 'so': 1,

    # Condition
    'if': 2, 'unless': 2, 'lest': 2, 'when and if': 2, 'insofar as': 2,

    # Temporal
    'after': 3, 'before': 3, 'then': 3, 'meanwhile': 3,
    'thereafter': 3, 'earlier': 3, 'by then': 3, 'simultaneously': 3,
    'as soon as': 3, 'while': 3,

    # Everything else falls to Expansion (4)
}

# Add remaining discourse markers and all punctuation markers to Expansion (4)
for marker in DISCOURSE_MARKERS:
    if marker not in RELATION_MAP:
        RELATION_MAP[marker] = 4

for punc in PUNCTUATION_MARKERS:
    RELATION_MAP[punc] = 4