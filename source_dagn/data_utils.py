# data_utils.py
import torch
from torch.utils.data import Dataset
import numpy as np
import re
from collections import Counter
from config import CONFIG, DISCOURSE_MARKERS, PUNCTUATION_MARKERS, RELATION_MAP

class Vocabulary:
    """Manages the vocabulary for tokenizing and encoding text.

    This class builds a vocabulary from a list of texts, mapping words to unique
    integer IDs and vice-versa. It handles padding and unknown tokens.

    Attributes:
        itos (dict): Integer-to-string mapping (ID to word).
        stoi (dict): String-to-integer mapping (word to ID).
        min_freq (int): The minimum frequency a word must have to be included
                        in the vocabulary. Words with frequency below this
                        threshold will be treated as unknown tokens.
    """
    def __init__(self, min_freq=1):
        """Initializes the Vocabulary with special tokens.

        Args:
            min_freq (int, optional): The minimum frequency a word must have
                                      to be included in the vocabulary. Defaults to 1.
        """
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.min_freq = min_freq

    def build_vocab(self, text_list):
        """Builds the vocabulary from a list of texts.

        Tokens are extracted from each text, and their frequencies are counted.
        Only words appearing at least `self.min_freq` times are added to the
        vocabulary.

        Args:
            text_list (list of str): A list of text strings from which to build
                                     the vocabulary.
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
        """Tokenizes a given text string.

        Splits the text into words and punctuation marks, converting all to lowercase.

        Args:
            text (str): The input text string to be tokenized.

        Returns:
            list of str: A list of tokens extracted from the text.
        """
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text):
        """Encodes a text string into a list of integer IDs.

        Each token in the text is mapped to its corresponding ID in the vocabulary.
        Tokens not found in the vocabulary (or below `min_freq`) are mapped to
        the ID for the unknown token (`<UNK>`).

        Args:
            text (str): The input text string to be encoded.

        Returns:
            list of int: A list of integer IDs representing the encoded text.
        """
        tokens = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

class HeuristicDiscourseParser:
    """Revised Parser: Splits text at BOTH punctuation AND discourse markers.

    This parser identifies Elementary Discourse Units (EDUs) within a given text
    based on predefined discourse markers and punctuation. It also constructs
    an adjacency matrix representing heuristic discourse relations between these EDUs.

    Attributes:
        split_pattern (re.Pattern): A compiled regular expression used to split
                                     text based on discourse markers and punctuation.
    """
    def __init__(self):
        """Initializes the HeuristicDiscourseParser.

        Constructs a regex pattern to identify discourse markers and punctuation
        for splitting text. Markers are escaped to handle special regex characters
        and sorted by length (assumed from `config.py`) to ensure longest match first.
        """
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
        """Parses a text into Elementary Discourse Units (EDUs) and an adjacency matrix.

        The text is split based on discourse markers and punctuation. Discourse markers
        typically initiate a new EDU, while punctuation typically concludes the current EDU.
        An adjacency matrix is constructed where edges represent heuristic discourse
        relations between consecutive EDUs, primarily based on the starting word of the
        subsequent EDU.

        Args:
            text (str): The input text to be parsed.

        Returns:
            tuple: A tuple containing:
                - edus (list of str): A list of extracted Elementary Discourse Units.
                - adj (np.ndarray): A 3D NumPy array representing the adjacency matrix
                                    of discourse relations. Dimensions are
                                    (num_relations, num_nodes, num_nodes).
                                    `num_relations` is defined in `CONFIG`.
                                    Self-loops are added for a default relation (Expansion).
        """
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
    """LogicalReasoningDataset for logical reasoning tasks.

    This dataset class is responsible for loading and preprocessing data for logical
    reasoning tasks. It takes a DataFrame containing context, questions, answer options,
    and labels, then processes each entry by parsing the combined text of context,
    question, and each answer option into Elementary Discourse Units (EDUs) using a
    heuristic discourse parser. It also generates an adjacency matrix representing
    discourse relations between EDUs for each option and encodes the EDUs using a
    provided vocabulary.

    Assumptions:
        - The input DataFrame `df` contains columns 'context', 'question', 'answers',
          'label', and 'id'.
        - The 'answers' column contains a string representation of a list of answer
          options, which can be safely evaluated using `eval()`.
        - The 'label' column contains an integer representing the correct answer index.
        - The `vocab` object is an instance of the `Vocabulary` class with a
          `tokenize` and `encode` method.
        - The `parser` object (HeuristicDiscourseParser) is globally available and
          correctly initialized to parse text into EDUs and adjacency matrices.

    Example (from assignment_data/train.csv):
        Input DataFrame Row:
            'context': "The cat sat on the mat. It was a fluffy cat."
            'question': "Where did the cat sit?"
            'answers': '["The cat sat on the mat.", "The cat sat on the chair."]'
            'label': 0
            'id': "sample_id_1"

        Processing for Answer Option 0 ("The cat sat on the mat."):
            1. Combined Text: "The cat sat on the mat. It was a fluffy cat. Where did the cat sit? The cat sat on the mat."
            2. Heuristic Discourse Parsing (using `parser`):
                - `edus` (list of strings):
                    ["The cat sat on the mat.", "It was a fluffy cat.", "Where did the cat sit?", "The cat sat on the mat."]
                - `adj` (numpy array of shape `(num_relations, num_nodes, num_nodes)`):
                    A 3D matrix where `adj[relation_id, i, j] = 1.0` if there's a relation
                    of type `relation_id` from EDU `i` to EDU `j`.
                    E.g., `adj[4, 0, 1] = 1.0` (Expansion relation from EDU 0 to EDU 1),
                    and self-loops `adj[4, i, i] = 1.0` for all `i`.
            3. EDU Encoding (using `vocab`):
                - `encoded_edus` (list of lists of integers):
                    Tokenized and numerically encoded version of `edus`.
                    E.g., `[[<token_id for "The">, ..., <token_id for "mat.">], ...]`.

        The `__getitem__` method would return a dictionary containing these processed
        items for each answer option, along with the original question, label, and id.
    """
    def __init__(self, df, vocab, is_test=False):
        """Initializes the LogicalReasoningDataset.

        Args:
            df (pd.DataFrame): The input DataFrame containing the dataset.
                               Expected columns: 'context', 'question', 'answers',
                               'label', and 'id'.
            vocab (Vocabulary): An initialized Vocabulary object for tokenizing and
                                encoding text.
            is_test (bool, optional): A flag indicating if the dataset is for testing.
                                      If True, the label will be -1. Defaults to False.
        """
        self.data = df
        self.vocab = vocab
        self.is_test = is_test

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: The number of rows in the underlying DataFrame.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves a single sample from the dataset at the specified index.

        For each sample, it combines context, question, and each answer option,
        parses them into EDUs and an adjacency matrix, and encodes the EDUs.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - processed_options (list of list of list of int): A list of 4
                                    elements, where each element corresponds to an
                                    answer option. Each option is a list of EDUs,
                                    and each EDU is a list of encoded token IDs.
                - option_adjs (list of np.ndarray): A list of 4 NumPy arrays,
                                    where each array is the adjacency matrix for
                                    the corresponding answer option.
                - label (int): The integer label for the correct answer (0-3),
                               or -1 if `is_test` is True.
                - sample_id (str or int): The unique identifier for the sample.
        """
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
    """Collates a list of samples into a batch for the DataLoader.

    This function takes a batch of samples (as returned by `__getitem__`) and
    processes them into tensors suitable for model input. It flattens the EDUs
    across all options and samples, pads them to a uniform length, and collects
    adjacency matrices and node counts.

    Args:
        batch (list of tuple): A list of samples, where each sample is a tuple
                               (processed_options, option_adjs, label, sample_id)
                               as returned by `LogicalReasoningDataset.__getitem__`.

    Returns:
        tuple: A tuple containing:
            - batch_ids (list of str or int): A list of unique identifiers for
                                              each sample in the batch.
            - padded_edus (torch.Tensor): A 2D tensor of shape (total_edus_in_batch, max_edu_len)
                                          containing all EDUs from all options and samples,
                                          padded with 0s.
            - flat_adjs (list of np.ndarray): A flattened list of all adjacency matrices
                                              from all options and samples in the batch.
            - graph_node_counts (torch.Tensor): A 1D tensor of shape (total_options_in_batch,)
                                                indicating the number of EDUs (nodes) for
                                                each graph (answer option).
            - batch_labels (torch.Tensor): A 1D tensor of shape (batch_size,) containing
                                           the labels for each sample.
    """
    batch_ids = []
    batch_labels = []
    
    flat_edus = [] 
    graph_node_counts = []
    flat_adjs = []
    
    for processed_opts, option_adjs, label, sample_id in batch:
        batch_ids.append(sample_id)
        batch_labels.append(label)
        
        for i in range(4): # Assuming 4 answer options per sample
            edus = processed_opts[i]
            adj = option_adjs[i]
            
            count = len(edus)
            graph_node_counts.append(count)
            flat_adjs.append(adj)
            
            for edu in edus:
                flat_edus.append(torch.tensor(edu, dtype=torch.long))
    
    padded_edus = torch.nn.utils.rnn.pad_sequence(flat_edus, batch_first=True, padding_value=0)
    
    return batch_ids, padded_edus, flat_adjs, torch.tensor(graph_node_counts), torch.tensor(batch_labels, dtype=torch.long)