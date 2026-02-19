import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from huggingface_hub import login

# Step 1: Verify GPU Availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 2: Login to Hugging Face
login(token="Your Token HERE")  # Replace with your token

# Step 3: Load and Preprocess HateXplain Dataset
dataset = load_dataset("hatexplain", trust_remote_code=True)

label2id = {"normal": 0, "hatespeech": 1, "offensive": 2}
id2label = {v: k for k, v in label2id.items()}

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    texts = [" ".join(tokens) for tokens in examples["post_tokens"]]
    labels = [max(set(annotator["label"]), key=annotator["label"].count) for annotator in examples["annotators"]]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = labels
    return tokenized

train_dataset = dataset["train"].map(preprocess_function, batched=True)
validation_dataset = dataset["validation"].map(preprocess_function, batched=True)
test_dataset = dataset["test"].map(preprocess_function, batched=True)

# Step 4: Load Pre-trained DistilBERT Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)
model.to(device)

# Step 5: Define Evaluation Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Step 6: Set Up Optimized Training Arguments
training_args = TrainingArguments(
    output_dir="/tmp/distilbert-hatexplain",  # Use a temporary directory
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,  # Learning rate optimized for DistilBERT
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,  # Adjusted for faster convergence
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=True,
    hub_model_id="uboza10300/finetuned-distilbert-hatexplainV2",
)

# Step 7: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Step 8: Fine-Tune the Model
trainer.train()

# Step 9: Evaluate on Test Set
test_results = trainer.evaluate(test_dataset)
print("Test Set Metrics:", test_results)

# Step 10: Push Model to Hugging Face Hub
trainer.push_to_hub()
print("Model successfully pushed to Hugging Face Hub!")