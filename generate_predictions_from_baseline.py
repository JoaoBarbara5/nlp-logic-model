import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ast import literal_eval
from tqdm import tqdm

def generate_predictions():
    # 1. Load the Test Data
    # We use literal_eval to parse the stringified list of answers (e.g., "['opt1', 'opt2']")
    test_df = pd.read_csv("test.csv")
    test_df['answers'] = test_df['answers'].apply(literal_eval)
    
    print(f"Loaded {len(test_df)} test examples.")

    # 2. Load Model and Tokenizer
    # The model is a Sequence Classifier (NLI). Label 1 usually corresponds to Entailment/True.
    model_name = "qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-3"
    
    print("Loading model... (this may take a while for XXLarge)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    predictions = []

    # 3. Inference Loop
    # Iterate through each question in the test set
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        context = row['context']
        question = row['question']
        options = row['answers']
        
        # Construct the premise (Context + Question)
        # We ensure a space exists between context and question
        premise = f"{context} {question}"
        
        # Prepare inputs for all options for this single question
        # We create a batch where each item is a pair: (Premise, Option)
        encoded_inputs = tokenizer(
            [premise] * len(options),  # Repeat premise for each option
            options,                   # The list of options (hypotheses)
            padding=True,
            truncation=True,
            max_length=512,            # DeBERTa limit
            return_tensors="pt"
        )
        
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        
        # Run the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # Shape: (num_options, 2)
        
        # 4. Select the Best Answer
        # The model outputs 2 scores per pair: [Logits_False, Logits_True]
        # We want the option with the highest score for 'True' (Index 1)
        
        # Extract the column for label 1 (Entailment/Equivalence)
        entailment_scores = logits[:, 1]
        
        # Find the index of the option with the highest entailment score
        best_option_idx = torch.argmax(entailment_scores).item()
        predictions.append(best_option_idx)

    # 5. Save Predictions
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    
    output_filename = "submission.csv"
    submission_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")
    print(submission_df.head())

if __name__ == "__main__":
    generate_predictions()