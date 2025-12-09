# data_utils.py
import torch
from torch.utils.data import Dataset
import numpy as np
import re
from collections import Counter
from config import CONFIG, DISCOURSE_MARKERS, PUNCTUATION_MARKERS, RELATION_MAP

class Vocabulary:
    def __init__(self, min_freq=1):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.min_freq = min_freq

    def build_vocab(self, text_list):
        freqs = Counter()
        for text in text_list:
            tokens = self.tokenize(text)
            freqs.update(tokens)
        
        idx = 2
        for word, freq in freqs.items():
            if freq >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
                
    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

class HeuristicDiscourseParser:
    """
    Revised Parser: Splits text at BOTH punctuation AND discourse markers.
    """
    def __init__(self):
        # Create a regex pattern that matches discourse markers OR punctuation
        # We escape markers to handle special regex characters
        # Markers are sorted by length in config.py to ensure longest match first
        marker_pattern = '|'.join(map(re.escape, DISCOURSE_MARKERS))
        punct_pattern = '|'.join(map(re.escape, PUNCTUATION_MARKERS))
        
        # Capture the delimiter so we can keep it
        # Pattern: (Word_Boundary + Marker + Word_Boundary) OR (Punctuation)
        # Note: \b is important for words like 'and' to not match 'hand'
        self.split_pattern = re.compile(
            r'(\b(?:' + marker_pattern + r')\b|[' + punct_pattern + r'])', 
            re.IGNORECASE
        )

    def parse(self, text):
        # 1. Split text keeping the delimiters
        # split() with capturing parenthesis returns [text, delimiter, text, delimiter...]
        parts = self.split_pattern.split(text)
        
        edus = []
        current_edu = ""
        
        # 2. Reconstruct EDUs
        # Logic: 
        # - Discourse Marker: Starts a NEW EDU (marker is included in the new EDU)
        # - Punctuation: Ends the CURRENT EDU (punctuation is included in the old EDU)
        
        for part in parts:
            if not part: continue
            
            clean_part = part.strip()
            lower_part = clean_part.lower()

            if lower_part in DISCOURSE_MARKERS:
                # Marker: Start new EDU. Push current one if exists.
                if current_edu.strip():
                    edus.append(current_edu.strip())
                current_edu = clean_part + " " # Start new with marker
                
            elif lower_part in PUNCTUATION_MARKERS:
                # Punctuation: Append to current, then close it.
                current_edu += clean_part
                edus.append(current_edu.strip())
                current_edu = ""
                
            else:
                # Regular text
                current_edu += part
                
        if current_edu.strip():
            edus.append(current_edu.strip())
            
        # Fallback if no splits
        if not edus: edus = [text]

        # 3. Build Adjacency Matrix
        num_nodes = len(edus)
        adj = np.zeros((CONFIG['num_relations'], num_nodes, num_nodes), dtype=np.float32)
        
        for i in range(num_nodes - 1):
            # Check the start of the NEXT EDU for the relation type
            next_edu_text = edus[i+1].lower()
            relation = 4 # Default: Expansion
            
            # Check if EDU starts with a known marker
            for marker, rel_id in RELATION_MAP.items():
                # We check "startswith" to match the marker at the beginning
                if next_edu_text.startswith(marker):
                    relation = rel_id
                    break
            
            adj[relation, i, i+1] = 1.0

        # Self-loops
        for i in range(num_nodes):
            adj[4, i, i] = 1.0

        return edus, adj

parser = HeuristicDiscourseParser()

class LogicalReasoningDataset(Dataset):
    def __init__(self, df, vocab, is_test=False):
        self.data = df
        self.vocab = vocab
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        context = row['context']
        question = row['question']
        
        try:
            answers = eval(row['answers'])
        except:
            answers = ["", "", "", ""] 
            
        processed_options = []
        option_adjs = []
        
        for opt in answers:
            full_text = f"{context} {question} {opt}"
            edus, adj = parser.parse(full_text)
            
            edu_indices = [self.vocab.encode(edu) for edu in edus]
            processed_options.append(edu_indices)
            option_adjs.append(adj)

        label = int(row['label']) if not self.is_test else -1
        return processed_options, option_adjs, label, row['id']

def collate_fn(batch):
    batch_ids = []
    batch_labels = []
    
    flat_edus = [] 
    graph_node_counts = []
    flat_adjs = []
    
    for processed_opts, option_adjs, label, sample_id in batch:
        batch_ids.append(sample_id)
        batch_labels.append(label)
        
        for i in range(4): 
            edus = processed_opts[i]
            adj = option_adjs[i]
            
            count = len(edus)
            graph_node_counts.append(count)
            flat_adjs.append(adj)
            
            for edu in edus:
                flat_edus.append(torch.tensor(edu, dtype=torch.long))
    
    padded_edus = torch.nn.utils.rnn.pad_sequence(flat_edus, batch_first=True, padding_value=0)
    
    return batch_ids, padded_edus, flat_adjs, torch.tensor(graph_node_counts), torch.tensor(batch_labels, dtype=torch.long)