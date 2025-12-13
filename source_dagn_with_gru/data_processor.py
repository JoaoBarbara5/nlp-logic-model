# data_processor.py
import torch
import numpy as np
import re
from collections import Counter
from config import CONFIG, DISCOURSE_MARKERS, PUNCTUATION_MARKERS


class Vocabulary:
    """Manages a vocabulary of words, including mapping between words and their numerical indices.

    Attributes:
        itos (dict): A dictionary mapping numerical indices to tokens (e.g., {0: "<PAD>", 1: "<UNK>", ...}).
        stoi (dict): A dictionary mapping tokens to their numerical indices (e.g., {"<PAD>": 0, "<UNK>": 1, ...}).
        min_freq (int): The minimum frequency a token must have in the corpus to be included in the vocabulary.
    """
    def __init__(self, min_freq=1):
        """Initializes the Vocabulary object.

        Args:
            min_freq (int, optional): The minimum frequency a token must have to be included in the vocabulary.
                Tokens with frequency below this threshold will be treated as unknown. Defaults to 1.
        """
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "UNK": 1} # Fix UNK key here. It should be "<UNK>"
        self.min_freq = min_freq

    def build_vocab(self, text_list):
        """Builds the vocabulary from a list of text documents.

        This method tokenizes each text, counts token frequencies, and populates the
        `stoi` and `itos` dictionaries based on the `min_freq` threshold.

        Args:
            text_list (list of str): A list of text strings from which to build the vocabulary.
        """
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
        """Tokenizes a text string into a list of tokens."""
        # Simple whitespace + punctuation split
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text):
        """Encodes a text string into a list of token IDs."""
        tokens = self.tokenize(text)
        return [self.stoi.get(t, 1) for t in tokens], tokens

class DAGNProcessor:
    """Processes a sample (context, question, and 4 options) into a list of 4 dictionaries (one per option).
    
    Args:
        vocab (Vocabulary): The vocabulary object to use for tokenization and encoding.
    """
    def __init__(self, vocab):
        self.vocab = vocab

    def process_sample(self, row):
        """
        Processes one row (Context, Question, 4 Options) into tensors.
        
        Args:
            row (pandas.Series): A row from the dataset containing 'context', 'question', and 'answers'.
        
        Returns:
            list of dict: A list of 4 dictionaries (one per option), each containing token IDs, masks, and split IDs.
        """
        context = row['context']
        question = row['question']
        try:
            answers = eval(row['answers'])
        except:
            answers = ["", "", "", ""] # Fallback

        processed_options = []
        
        for opt in answers:
            # 1. Tokenize & Encode
            # Structure: Context tokens | Question tokens | Option tokens
            c_ids, c_toks = self.vocab.encode(context)
            q_ids, q_toks = self.vocab.encode(question)
            o_ids, o_toks = self.vocab.encode(opt)
            
            full_ids = c_ids + q_ids + o_ids
            full_toks = c_toks + q_toks + o_toks
            
            seq_len = len(full_ids)
            
            # 2. Masks
            attention_mask = [1] * seq_len
            
            # Passage Mask: 1 on context, 0 elsewhere
            passage_mask = [1] * len(c_ids) + [0] * (len(q_ids) + len(o_ids))
            
            # Question Mask: 1 on question, 0 elsewhere
            question_mask = [0] * len(c_ids) + [1] * len(q_ids) + [0] * len(o_ids)
            
            # 3. Split IDs (The Logic Graph Markers)
            # 0: None, 1: Argument, 2: Punctuation
            split_ids = [0] * seq_len
            
            for i, token in enumerate(full_toks):
                if token in DISCOURSE_MARKERS:
                    split_ids[i] = 1
                elif token in PUNCTUATION_MARKERS:
                    split_ids[i] = 2
                    
            processed_options.append({
                'input_ids': torch.tensor(full_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'split_ids': torch.tensor(split_ids, dtype=torch.long),
                'passage_mask': torch.tensor(passage_mask, dtype=torch.long),
                'question_mask': torch.tensor(question_mask, dtype=torch.long)
            })
            
        return processed_options

def collate_dagn(batch):
    """
    Collates a list of samples.
    Batch: List of tuples (processed_options_list, label, id)
    """
    # Flatten: [Sample1_OptA, Sample1_OptB, ..., Sample2_OptA...]
    flat_data = [opt for sample in batch for opt in sample[0]]
    labels = [sample[1] for sample in batch]
    ids = [sample[2] for sample in batch]
    
    batch_out = {}
    keys = ['input_ids', 'attention_mask', 'split_ids', 'passage_mask', 'question_mask']
    
    for k in keys:
        tensors = [d[k] for d in flat_data]
        # Pad sequences to max length in batch
        batch_out[k] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
        
    return ids, batch_out, torch.tensor(labels, dtype=torch.long)