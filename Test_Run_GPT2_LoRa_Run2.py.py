import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("uboza10300/finetuned-gpt2-hatexplainV2")
model = AutoModelForSequenceClassification.from_pretrained("uboza10300/finetuned-gpt2-hatexplainV2")

# Load the dataset
dataset_path = "labeled_data.csv"
data = pd.read_csv(dataset_path)
print(f"Columns in the dataset: {data.columns}")

# Select only the first 200 rows
data = data.iloc[:200]

# Preprocessing
texts = data["tweet"].tolist()
true_labels = data["class"].tolist()

# Tokenize the data
inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=128)

# Predict with the model
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1).numpy()
probabilities = torch.softmax(outputs.logits, dim=-1).detach().numpy()

# Save predictions and probabilities
data["predicted_label"] = predictions
data["probabilities"] = probabilities.tolist()

# Save misclassified examples
misclassified = data[data["class"] != data["predicted_label"]]
misclassified.to_csv("misclassified_examples.csv", index=False)

# Metrics calculation
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions, target_names=["Hate Speech", "Offensive", "Neutral"], output_dict=True)

# Generate and save confusion matrix
cm = confusion_matrix(true_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Hate Speech", "Offensive", "Neutral"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Precision-Recall Curve for class of interest
class_of_interest = 1  # Change this to 0 or 2 for other classes
probs = [prob[class_of_interest] for prob in probabilities]

precision, recall, _ = precision_recall_curve(np.array(true_labels) == class_of_interest, probs)
plt.plot(recall, precision, lw=2, color="b", label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve for Class {class_of_interest}")
plt.legend(loc="upper right")
plt.savefig(f"precision_recall_curve_class_{class_of_interest}.png")
plt.show()

# Save predictions and evaluation metrics
data.to_csv("predicted_results.csv", index=False)

# Print summary metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{classification_report(true_labels, predictions, target_names=['Hate Speech', 'Offensive', 'Neutral'])}")
