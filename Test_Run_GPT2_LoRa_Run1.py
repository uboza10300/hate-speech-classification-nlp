# Import required libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import kagglehub

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("uboza10300/finetuned-gpt2-hatexplainV2")
model = AutoModelForSequenceClassification.from_pretrained("uboza10300/finetuned-gpt2-hatexplainV2")

# Load the new dataset
import kagglehub
path = kagglehub.dataset_download("mrmorj/hate-speech-and-offensive-language-dataset")
print("Path to dataset files:", path)

# Assuming the dataset is a CSV file
file_path = path + "/labeled_data.csv"  # Adjust if the filename differs
df = pd.read_csv(file_path)

# Select the first 200 rows for testing
test_data = df.head(200)

# Function to preprocess and predict
def classify_text(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=1)
    return predictions, probabilities

# Classify the text from the dataset
texts = test_data["tweet"].tolist()  # Assuming the text column is named "tweet"
predictions, probabilities = classify_text(texts, tokenizer, model)

# Add predictions to the DataFrame
test_data["predicted_label"] = predictions.numpy()
test_data["probabilities"] = probabilities.numpy().tolist()

# Save the predictions to a new file
output_path = "predicted_results.csv"
test_data.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")
