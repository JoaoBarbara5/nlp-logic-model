
import torch
from dagn import DAGN
from transformers import RobertaConfig

def test_random_encoder():
    print("Testing RandomEmbeddingEncoder...")
    
    # 1. Setup Config
    config = RobertaConfig.from_pretrained("roberta-large")
    # Decrease header count/size for speed if desired, but good to match large
    
    # 2. Initialize Model with random encoder
    model = DAGN(
        config=config,
        init_weights=True,
        max_rel_id=5,
        hidden_size=config.hidden_size,
        dropout_prob=0.1,
        merge_type=1,
        token_encoder_type="random",  # <--- Testing this
        gnn_version="GCN",
        use_pool=True,
        use_gcn=True
    )
    
    # Check if correct encoder initialized
    from dagn import RandomEmbeddingEncoder
    assert isinstance(model.roberta, RandomEmbeddingEncoder), "Model should use RandomEmbeddingEncoder"
    print("Model initialized with RandomEmbeddingEncoder.")

    # 3. Create dummy inputs
    batch_size = 2
    num_choices = 4
    seq_len = 32
    vocab_size = config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, num_choices, seq_len))
    attention_mask = torch.ones((batch_size, num_choices, seq_len), dtype=torch.long)
    
    # DAGN specific inputs
    passage_mask = torch.ones((batch_size, num_choices, seq_len), dtype=torch.long)
    question_mask = torch.ones((batch_size, num_choices, seq_len), dtype=torch.long)
    
    # -1 padding, 0 non-arg. Just fill 0s
    argument_bpe_ids = torch.zeros((batch_size, num_choices, seq_len), dtype=torch.long)
    argument_bpe_ids[:, :, 5:10] = 1 # Make some arguments
    
    domain_bpe_ids = torch.zeros((batch_size, num_choices, seq_len), dtype=torch.long)
    punct_bpe_ids = torch.zeros((batch_size, num_choices, seq_len), dtype=torch.long)
    
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    # 4. Forward Pass
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
    
    # 5. Check Output
    # Output should be (loss, logits) because labels provided
    loss = outputs[0]
    logits = outputs[1]
    
    print(f"Loss: {loss.item()}")
    print(f"Logits shape: {logits.shape}")
    
    assert logits.shape == (batch_size, num_choices)
    print("Test Passed!")

if __name__ == "__main__":
    test_random_encoder()
