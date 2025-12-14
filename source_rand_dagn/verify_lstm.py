
import torch
import sys
import os

# Add local directory to path so we can import dagn
sys.path.append(os.getcwd())

from dagn import DAGN
from transformers import AutoConfig

def test_lstm_forward():
    print("Testing Bi-LSTM DAGN...")
    config = AutoConfig.from_pretrained("roberta-large")
    # Small hidden size for speed if we were training, but here we just init
    # But to match BiLSTMEncoder logic (embedding -> hidden), we should keep standard or ensure dims match.
    # roberta-large has hidden_size=1024.
    
    model = DAGN(
        config=config,
        init_weights=False,
        max_rel_id=5,
        hidden_size=config.hidden_size,
        dropout_prob=0.1,
        merge_type=1,
        token_encoder_type="lstm",
        gnn_version="GCN",
        use_pool=True,
        use_gcn=True
    )
    
    bsz = 2
    num_choices = 4
    seq_len = 20
    
    input_ids = torch.randint(0, config.vocab_size, (bsz, num_choices, seq_len))
    attention_mask = torch.ones((bsz, num_choices, seq_len), dtype=torch.long)
    passage_mask = torch.ones((bsz, num_choices, seq_len), dtype=torch.long)
    question_mask = torch.ones((bsz, num_choices, seq_len), dtype=torch.long)
    argument_bpe_ids = torch.zeros((bsz, num_choices, seq_len)).long()
    # Set a few random argument ids to 1 to ensure graph construction works
    argument_bpe_ids[:, :, 1::5] = 1 
    domain_bpe_ids = torch.zeros((bsz, num_choices, seq_len)).long()
    punct_bpe_ids = torch.zeros((bsz, num_choices, seq_len)).long()
    labels = torch.randint(0, num_choices, (bsz,))
    
    print("Running forward pass...")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        passage_mask=passage_mask,
        question_mask=question_mask,
        argument_bpe_ids=argument_bpe_ids,
        domain_bpe_ids=domain_bpe_ids,
        punct_bpe_ids=punct_bpe_ids,
        labels=labels
    )
    
    loss = outputs[0]
    logits = outputs[1]
    
    print(f"Loss: {loss.item()}")
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (bsz, num_choices)
    print("Bi-LSTM Test Passed!")

def test_roberta_forward():
    print("\nTesting Roberta DAGN (Regression Check)...")
    config = AutoConfig.from_pretrained("roberta-large")
    
    # We won't load pretrained weights to save time/bandwidth, just init structure
    model = DAGN(
        config=config,
        init_weights=False,
        max_rel_id=5,
        hidden_size=config.hidden_size,
        dropout_prob=0.1,
        merge_type=1,
        token_encoder_type="roberta",
        gnn_version="GCN",
        use_pool=True,
        use_gcn=True
    )
    
    bsz = 2
    num_choices = 4
    seq_len = 20
    
    input_ids = torch.randint(0, config.vocab_size, (bsz, num_choices, seq_len))
    attention_mask = torch.ones((bsz, num_choices, seq_len))
    passage_mask = torch.ones((bsz, num_choices, seq_len))
    question_mask = torch.ones((bsz, num_choices, seq_len))
    argument_bpe_ids = torch.zeros((bsz, num_choices, seq_len)).long()
    domain_bpe_ids = torch.zeros((bsz, num_choices, seq_len)).long()
    punct_bpe_ids = torch.zeros((bsz, num_choices, seq_len)).long()
    labels = torch.randint(0, num_choices, (bsz,))
    
    print("Running forward pass...")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        passage_mask=passage_mask,
        question_mask=question_mask,
        argument_bpe_ids=argument_bpe_ids,
        domain_bpe_ids=domain_bpe_ids,
        punct_bpe_ids=punct_bpe_ids,
        labels=labels
    )
    
    loss = outputs[0]
    logits = outputs[1]
    
    print(f"Loss: {loss.item()}")
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (bsz, num_choices)
    print("Roberta Test Passed!")

if __name__ == "__main__":
    test_lstm_forward()
    # test_roberta_forward() # Optional, can uncomment if needed
