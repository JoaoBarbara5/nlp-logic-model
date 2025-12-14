import fasttext
import os

class EmbeddingModelTrainer:
    def __init__(self, output_dir="models"):
        self.output_dir = output_dir
        self.model = None
        # The optimal hyperparameters we discussed
        self.hyperparams = {
            'model': 'skipgram',
            'dim': 100, # up to 300
            'ws': 5,
            'epoch': 20,
            'minCount': 2,
            'neg': 10,
            'wordNgrams': 2,  # Critical: Handles sub-phrases automatically
            'lr': 0.05,
            'thread': 4
        }
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fit(self, input_data_path):
        """
        Input: Path to the file containing pre-processed spaced sentences.
        """
        print(f"Training FastText model with: {self.hyperparams}")
        # FastText accepts the file path directly - most efficient method
        self.model = fasttext.train_unsupervised(
            input=input_data_path,
            **self.hyperparams
        )

    def save(self, model_name):
        if self.model is None:
            raise ValueError("Model not trained.")
        
        output_path = os.path.join(self.output_dir, model_name)
        # Saves as .bin (standard format)
        self.model.save_model(output_path)
        print(f"Model saved to {output_path}")