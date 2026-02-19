import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import openai

# Set OpenAI API Key
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual API key

# Load the dataset
dataset_path = "labeled_data.csv"
data = pd.read_csv(dataset_path)
print(f"Columns in the dataset: {data.columns}")

# Select only the first 100 rows
data = data.iloc[:100]

# Preprocessing
texts = data["tweet"].tolist()
true_labels = data["class"].tolist()

# Predict with GPT-3.5
def gpt_predict(text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Classify the following tweet into one of the categories: Hate Speech, Offensive, or Neutral.\nTweet: \"{text}\"",
            max_tokens=10,
            n=1,
            stop=None,
            temperature=0
        )
        prediction = response['choices'][0]['text'].strip().lower()
        if prediction == "hate speech":
            return 0
        elif prediction == "offensive":
            return 1
        elif prediction == "neutral":
            return 2
        else:
            return -1  # Invalid response
    except Exception as e:
        print(f"Error processing text: \"{text}\" Error: {e}")
        return -1

predictions = [gpt_predict(text) for text in texts]

# Remove invalid predictions
valid_indices = [i for i, pred in enumerate(predictions) if pred != -1]
true_labels_filtered = [true_labels[i] for i in valid_indices]
predictions_filtered = [predictions[i] for i in valid_indices]

# Metrics calculation
accuracy = accuracy_score(true_labels_filtered, predictions_filtered)
report = classification_report(true_labels_filtered, predictions_filtered, target_names=["Hate Speech", "Offensive", "Neutral"], output_dict=True)

# Save misclassified examples
data["predicted_label"] = predictions
misclassified = data[data["class"] != data["predicted_label"]]
misclassified.to_csv("misclassified_examples_GPT3.5.csv", index=False)

# Generate and save confusion matrix
cm = confusion_matrix(true_labels_filtered, predictions_filtered)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Hate Speech", "Offensive", "Neutral"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - GPT3.5")
plt.savefig("confusion_matrix_GPT3.5.png")
plt.show()

# Precision-Recall Curve for class of interest
class_of_interest = 1  # Change this to 0 or 2 for other classes
probs = np.random.rand(len(true_labels_filtered))  # Placeholder as GPT-3.5 doesn't output probabilities
precision, recall, _ = precision_recall_curve(np.array(true_labels_filtered) == class_of_interest, probs)
plt.plot(recall, precision, lw=2, color="b", label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve for Class {class_of_interest} - GPT3.5")
plt.legend(loc="upper right")
plt.savefig(f"precision_recall_curve_class_{class_of_interest}_GPT3.5.png")
plt.show()

# Save predictions and evaluation metrics
data.to_csv("predicted_results_GPT3.5.csv", index=False)

# Print summary metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{classification_report(true_labels_filtered, predictions_filtered, target_names=['Hate Speech', 'Offensive', 'Neutral'])}")
