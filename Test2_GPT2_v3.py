import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

# Load the dataset
dataset = load_dataset("hatexplain")

# Preprocessing function
def preprocess_function(examples):
    # Join tokens into a single string
    posts = [" ".join(post) for post in examples["post_tokens"]]
    return tokenizer(
        posts,
        padding="max_length",
        truncation=True,
        max_length=128,
    )

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)

# Tokenizer adjustments for GPT-2
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Preprocess datasets
encoded_dataset = dataset.map(preprocess_function, batched=True)

print("HereeeeeeeeeeeeeHereeeeeeeeeeeeeHereeeeeeeeeeeeeHereeeeeeeeeeeeeHereeeeeeeeeeeee")
# Print the training dataset for debugging
print(encoded_dataset["train"])
print("HereeeeeeeeeeeeeHereeeeeeeeeeeeeHereeeeeeeeeeeeeHereeeeeeeeeeeeeHereeeeeeeeeeeee")

# Extract and flatten labels
train_labels = []
for annotators in encoded_dataset["train"]["annotators"]:
    for label in annotators["label"]:
        train_labels.append(label)

# Class weights calculation
class_counts = [train_labels.count(i) for i in range(3)]
class_weights = torch.tensor(class_counts).float()
class_weights = 1.0 / class_weights
class_weights = class_weights / class_weights.sum()



# Define custom trainer class for weighted loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-hatexplain",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
    hub_model_id="uboza10300/Test2_GPT2_v3",
    report_to="none",
    load_best_model_at_end=True,  # Added for EarlyStoppingCallback
)


# Metrics computation
def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Initialize Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train the model
trainer.train()

# Evaluate the model
test_results = trainer.evaluate(encoded_dataset["test"])
print("Test Results:", test_results)

# Push model to Hugging Face Hub
trainer.push_to_hub()
print("Model successfully pushed to Hugging Face Hub!")
