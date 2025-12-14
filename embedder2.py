import ast
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    """Callback to print progress after each epoch."""
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        # Optional: Print when starting
        print(f"Epoch #{self.epoch + 1} start...", end=" ", flush=True)

    def on_epoch_end(self, model):
        # Print when finished
        print(f"finished.")
        self.epoch += 1

# 1. Define a data loader that reads your file line-by-line
#    This is memory efficient; it doesn't load the whole file into RAM.
class MyCorpus:
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                # ast.literal_eval safely converts the string "['word', 'word']" 
                # back into a real Python list
                try:
                    yield ast.literal_eval(line)
                except ValueError:
                    continue  # Skip empty or malformed lines

# 2. Configure the training parameters
#    min_count=1:  We want to keep ALL words, even rare ones (crucial for your small logic dataset)
#    window=5:     Look at 5 words to the left and 5 to the right
#    vector_size=100: A standard size. If you have very little data, try 50.
#    sg=1:         Use Skip-Gram (better for rare words) instead of CBOW
#    workers=4:    Use 4 CPU cores to train faster
tokenized_data = MyCorpus("tokenized_output.txt")

print("Training FastText model... (this may take a moment)")
# ... (previous imports and MyCorpus class) ...

# Initialize and train
epoch_logger = EpochLogger()
model = FastText(vector_size=100, 
                 window=8, 
                 min_count=2,    # discard words appearing only once
                 workers=4, 
                 sg=1,           # Skip-gram
                 bucket=200000,  # Optimized: Reduced from 2M to 200k for your dataset size
                 epochs=100,
                 alpha=0.02)      # Increased epochs for better learning on small data

print("Building vocabulary...")
model.build_vocab(tokenized_data)

print("Training model...")
model.train(tokenized_data, total_examples=model.corpus_count, epochs=model.epochs, callbacks=[epoch_logger])

# Save
model.save("logic_fasttext.model")
print("Training complete! Model saved as 'logic_fasttext.model'")

# --- QUICK TEST ---
# Let's check if it learned a logical relationship
test_word = "strengthens"
if test_word in model.wv:
    print(f"\nWords most similar to '{test_word}':")
    print(model.wv.most_similar(test_word))
else:
    print(f"'{test_word}' not found in vocabulary.")