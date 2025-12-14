from gensim.models import FastText
import torch
import torch.nn as nn
import numpy as np

def load_and_patch_embeddings(model_path="logic_fasttext.model"):
    print("Loading Gensim model...")
    # Load the model
    ft_model = FastText.load(model_path)
    
    # 1. Extract the existing weights and vocabulary
    # Gensim 4.x stores the vectors for the vocab in .vectors
    # shape: (vocab_size, vector_size)
    existing_weights = torch.tensor(ft_model.wv.vectors)
    
    # dict: {word: index}
    word_to_idx = ft_model.wv.key_to_index.copy()
    
    print(f"Original Vocab Size: {len(word_to_idx)}")
    print(f"Original Matrix Shape: {existing_weights.shape}")

    # 2. Define special tokens
    # <PAD> is usually index 0 in many setups, but appending is safer to avoid shifting all indices
    special_tokens = ["<PAD>", "<SEP>", "<UNK>"]
    
    # 3. Create random vectors for the new tokens
    # Using small random noise 
    embed_dim = existing_weights.shape[1]
    new_vectors = torch.normal(mean=0.0, std=0.1, size=(len(special_tokens), embed_dim))
    
    # 4. Concatenate
    final_weights = torch.cat((existing_weights, new_vectors), dim=0)
    
    # 5. Update the dictionary
    start_index = len(word_to_idx)
    for i, token in enumerate(special_tokens):
        word_to_idx[token] = start_index + i
        
    print(f"New Vocab Size: {len(word_to_idx)}")
    print(f"New Matrix Shape: {final_weights.shape}")
    print(f"Added tokens at indices: {list(range(start_index, start_index + 3))}")
    
    return final_weights, word_to_idx

# --- USAGE ---
# Run the function
final_embedding_weights, vocab_dict = load_and_patch_embeddings()

# Save them so you can load them easily in your training loop without Gensim
torch.save(final_embedding_weights, "patched_embeddings.pt")
torch.save(vocab_dict, "vocab_dict.pt")
print("Saved patched_embeddings.pt and vocab_dict.pt")