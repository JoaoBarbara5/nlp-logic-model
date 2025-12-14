import pandas as pd
import re
import ast
import random

class FastTextTokenizer:
    def __init__(self):
        # 1. Structural Tags (Critical for ~75 questions)
        # We explicitly map these to tokens before regex cleaning
        self.tag_map = {
            '<b>': ' startbold ',
            '</b>': ' endbold ',
            '<br>': ' ',
            '<i>': ' ',
            '</i>': ' ',
            '<u>': ' ',
            '</u>': ' '
        }
        
        # 2. Semantic Symbols
        # Map symbols to words so they aren't deleted
        self.symbol_map = {
            '%': ' percent ',
            '$': ' dollar ',
            '&': ' and '
        }
        
        # 3. Cleanup Regex
        # Matches anything that is NOT: Alphanumeric, Whitespace, or Hyphen
        # These will be replaced by SPACE to prevent word merging
        self.cleanup_pattern = re.compile(r"[^\w\s-]")
        
        # 4. Sentence Splitter
        self.sentence_split_pattern = re.compile(r'(?<=[.!?])\s+')

    def prepare_embedder_data(self, train_df):

        corpus = []
        for _, row in train_df.iterrows():
            extracted_sentences = self._extract_raw_sentences(row)
            for sentence in extracted_sentences:
                tokens = self._tokenize_sentence(sentence)
                if tokens:
                    corpus.append(tokens)
                    
        return corpus

    def _extract_raw_sentences(self, row):
        raw_sentences = []
        
        context_text = str(row['context'])
        split_ctx = self.sentence_split_pattern.split(context_text)
        raw_sentences.extend(split_ctx)
        
        raw_sentences.append(str(row['question']))
        
        try:
            answers_list = ast.literal_eval(row['answers'])
        except (ValueError, SyntaxError):
            answers_list = [str(row['answers'])] 
            
        raw_sentences.extend(answers_list)
        
        return raw_sentences

    def _tokenize_sentence(self, text):
        # 1. Lowercase
        text = text.lower()
        
        # 2. Handle HTML Tags (Map Bold, Delete others)
        for tag, replacement in self.tag_map.items():
            text = text.replace(tag, replacement)
        
        # [NEW SAFETY NET] Remove any remaining HTML tags we missed
        text = re.sub(r'<[^>]+>', ' ', text)

        # 3. Handle Semantic Symbols ($, %, &)
        for symbol, replacement in self.symbol_map.items():
            text = text.replace(symbol, replacement)
        
        # 4. Handle Apostrophes (Join Strategy)
        # "body's" -> "bodys"
        text = text.replace("'", "")
        
        # 5. General Cleanup (Split Strategy)
        # Replace remaining punctuation (/, [, ], (, ), :) with SPACE
        text = self.cleanup_pattern.sub(' ', text)
        
        # 6. Final Tokenization
        tokens = text.split()
        
        return tokens


if __name__ == "__main__":
    csv_path = r"aml-2025-read-between-the-lines/train.csv"
    df = pd.read_csv(csv_path)
    
    ids = df.index.tolist()
    random.seed(42)
    random.shuffle(ids)
    
    split_idx = int(len(ids) * (0.8))
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]
    
    train_df = df.loc[train_ids]
    val_df = df.loc[val_ids]

    tokenizer = FastTextTokenizer()
    train_corpus = tokenizer.prepare_embedder_data(train_df)
    """
    with open("tokenized_output.txt", "w", encoding="utf-8") as f:
        f.write('\n'.join(str(sentence) for sentence in train_corpus))

    with open("tokenized_output.txt", "r", encoding="utf-8") as f:
        loaded_sentences = [ast.literal_eval(line) for line in f]
    """

    

    