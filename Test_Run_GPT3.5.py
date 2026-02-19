import pandas as pd
import openai
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt

# Configure OpenAI API
openai.api_key = "your_openai_api_key_here"  # Replace with your actual OpenAI API key

# Dataset path
dataset_path = "labeled_data.csv"
data = pd.read_csv(dataset_path)

# Limit to the first 100 rows for testing
data = data.iloc[:200]

# Define class mapping for output readability
class_mapping = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}

# Prepare the dataset
texts = data["tweet"].tolist()
true_labels = data["class"].tolist()

# Function to get predictions from GPT-3.5-turbo
def get_predictions(texts):
    predictions = []
    for text in texts:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Classify the following text into one of these categories: Hate Speech, Offensive, Neutral."},
                    {"role": "user", "content": text},
                ]
            )
            output = response["choices"][0]["message"]["content"].strip().lower()
            if "hate speech" in output:
                predictions.append(0)
            elif "offensive" in output:
                predictions.append(1)
            elif "neutral" in output:
                predictions.append(2)
            else:
                predictions.append(-1)  # Mark as invalid
        except Exception as e:
            print(f"Error processing text: \"{text}\" Error: {e}")
            predictions.append(-1)  # Mark as invalid for errors
    return predictions

# Get predictions
predicted_labels = get_predictions(texts)

# Save raw predictions to the dataset
data["predicted_label_GPT3.5"] = predicted_labels
data.to_csv("predicted_results_GPT3.5.csv", index=False)

# Filter valid predictions
filtered_indices = [i for i, label in enumerate(predicted_labels) if label in [0, 1, 2]]
true_labels_filtered = [true_labels[i] for i in filtered_indices]
predicted_labels_filtered = [predicted_labels[i] for i in filtered_indices]

# Check for no valid predictions
if len(true_labels_filtered) == 0 or len(predicted_labels_filtered) == 0:
    print("No valid predictions available for evaluation. Check the predictions or API usage.")
else:
    # Metrics calculation
    accuracy = accuracy_score(true_labels_filtered, predicted_labels_filtered)
    report = classification_report(true_labels_filtered, predicted_labels_filtered, target_names=["Hate Speech", "Offensive", "Neutral"])
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(true_labels_filtered, predicted_labels_filtered)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Hate Speech", "Offensive", "Neutral"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - GPT-3.5-turbo")
    plt.savefig("confusion_matrix_GPT3.5.png")
    plt.show()

    # Generate precision-recall curves
    true_labels_one_hot = pd.get_dummies(true_labels_filtered).values
    predicted_probs = pd.get_dummies(predicted_labels_filtered).values

    for i, class_name in enumerate(["Hate Speech", "Offensive", "Neutral"]):
        precision, recall, _ = precision_recall_curve(true_labels_one_hot[:, i], predicted_probs[:, i])
        display = PrecisionRecallDisplay(precision=precision, recall=recall)
        display.plot()
        plt.title(f"Precision-Recall Curve - {class_name} (GPT-3.5-turbo)")
        plt.savefig(f"precision_recall_curve_{class_name.replace(' ', '_')}_GPT3.5.png")
        plt.show()
