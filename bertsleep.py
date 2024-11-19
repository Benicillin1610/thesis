from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModel, BertTokenizer, DataCollatorWithPadding, 
    BertForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
)
#LlamaForSequenceClassification
#BertForSequenceClassification
#LlamaTokenizer
#BertTokenizer


import evaluate
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt  # Import for plotting
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Use the Longformer model
name = "google-bert/bert-base-uncased"
matrix_name="confusion_matrix_bert_5hours_new_addnormaltokens.png"
loss_epoch="loss_comparison_epoch_bert_5hours_new_addnormaltokens.png"
accuracy_epoch="accuracy_epoch_bert_5hours_new_addnormaltokens.png"
output_dir = "bertsleep_5hours_new_addnormaltokens"


# Load the data from the new files
data_healthy_train = []
data_healthy_val = []
data_healthy_test = []
data_unhealthy_train = []
data_unhealthy_val = []
data_unhealthy_test = []

# Read and parse healthy train, val, test data
with open("healthy_labelled_new_train_single_10W_500_restored.txt", 'r') as txt_file:
    for line in txt_file:
        data_healthy_train.append(ast.literal_eval(line))

with open("healthy_labelled_new_val_single_10W_500_restored.txt", 'r') as txt_file:
    for line in txt_file:
        data_healthy_val.append(ast.literal_eval(line))

with open("healthy_labelled_new_test_single_10W_500_restored.txt", 'r') as txt_file:
    for line in txt_file:
        data_healthy_test.append(ast.literal_eval(line))

# Read and parse unhealthy train, val, test data
with open("unhealthy_labelled_new_train_single_10W_5500_restored.txt", 'r') as txt_file:
    for line in txt_file:
        data_unhealthy_train.append(ast.literal_eval(line))

with open("unhealthy_labelled_new_val_single_10W_500_restored.txt", 'r') as txt_file:
    for line in txt_file:
        data_unhealthy_val.append(ast.literal_eval(line))

with open("unhealthy_labelled_new_test_single_10W_500_restored.txt", 'r') as txt_file:
    for line in txt_file:
        data_unhealthy_test.append(ast.literal_eval(line))

# Convert lists to Hugging Face Datasets
dataset_healthy_train = Dataset.from_list(data_healthy_train)
dataset_healthy_val = Dataset.from_list(data_healthy_val)
dataset_healthy_test = Dataset.from_list(data_healthy_test)

dataset_unhealthy_train = Dataset.from_list(data_unhealthy_train)
dataset_unhealthy_val = Dataset.from_list(data_unhealthy_val)
dataset_unhealthy_test = Dataset.from_list(data_unhealthy_test)

# Combine healthy and unhealthy datasets for each split (train, validation, test)
combined_train = Dataset.from_dict({
    key: dataset_healthy_train[key] + dataset_unhealthy_train[key] for key in dataset_healthy_train.column_names
})

combined_val = Dataset.from_dict({
    key: dataset_healthy_val[key] + dataset_unhealthy_val[key] for key in dataset_healthy_val.column_names
})

combined_test = Dataset.from_dict({
    key: dataset_healthy_test[key] + dataset_unhealthy_test[key] for key in dataset_healthy_test.column_names
})

# Create a DatasetDict to hold all the splits
dataset = DatasetDict({
    'train': combined_train,
    'validation': combined_val,
    'test': combined_test
})
print(dataset)

# Initialize the Bert tokenizer
tokenizer = BertTokenizer.from_pretrained(name)

#special_tokens_dict = {'additional_special_tokens': }
#tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.add_tokens(['N1', 'N2', 'N3', 'R', 'W'], special_tokens=True)

def preprocess_function(examples):
    return tokenizer(examples["stages"], truncation=True, padding="max_length", max_length=512)

# Tokenize the dataset
tokenized_sleep = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load multiple metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Compute all metrics
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")
    prec = precision.compute(predictions=predictions, references=labels, average="weighted")
    rec = recall.compute(predictions=predictions, references=labels, average="weighted")
    
    # Combine all metrics in a dictionary
    return {**acc, **f1_score, **prec, **rec}

id2label = {0: "Healthy", 1: "Unhealthy"}
label2id = {"Healthy": 0, "Unhealthy": 1}

# Initialize the Bert model
model = BertForSequenceClassification.from_pretrained(
   name, num_labels=2, id2label=id2label, label2id=label2id
)

## new the encoder layers (all layers except the classification head)
#for param in model.bert.parameters():
#    param.requires_grad = False
#
## Now only the classification head is trainable
#for param in model.classifier.parameters():
#    param.requires_grad = True


embedding_size = model.get_input_embeddings().weight.shape[0]
print(embedding_size)
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))


training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=3e-5,
    per_device_train_batch_size=2, 
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=64,
    num_train_epochs=10
    weight_decay=0.01,
    eval_strategy="epoch",  # For evaluation during training
    logging_strategy = 'epoch',
    save_strategy="epoch",
    logging_dir=f"./logs",  # Log directory
    logging_steps=10,  # Log every 10 steps
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_sleep["train"],
    eval_dataset=tokenized_sleep["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model and capture the logs
train_output = trainer.train()

# Evaluate the model (this will load the best model if `load_best_model_at_end=True`)
metrics = trainer.evaluate()
print(f"Final evaluation: {metrics}")

# Print the directory of the best model checkpoint
print(f"Best model is saved at: {trainer.state.best_model_checkpoint}")

# Capture the loss and metrics vs step from the log history
log_history = trainer.state.log_history

# Convert log history to a DataFrame
log_df = pd.DataFrame(log_history)

# Save the DataFrame to a CSV file
log_df.to_csv('training_log.csv', index=False)


# Initialize lists to store training and validation losses
train_losses = []
eval_losses = []
train_steps = []
eval_steps = []
eval_accuracies = []

# Loop through the log history to capture the relevant losses
for log in log_history:
    if 'loss' in log:
        train_losses.append(log['loss'])
        train_steps.append(log['epoch'])
    if 'eval_loss' in log:
        eval_losses.append(log['eval_loss'])
        eval_steps.append(log['epoch'])
    if 'eval_accuracy' in log:
        eval_accuracies.append(log['eval_accuracy'])


#Plot training and validation loss vs steps
plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_losses, label="Training Loss", color='blue')
plt.plot(eval_steps, eval_losses, label="Validation Loss", color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig(loss_epoch)  # Save the loss comparison plot

#Plot training and validation loss vs steps
plt.figure(figsize=(10, 6))
plt.plot(eval_steps, eval_accuracies, label="Validation Accuracy", color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.savefig(accuracy_epoch)  # Save the loss comparison plot

# Load trained model
model_path = trainer.state.best_model_checkpoint
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
model.eval()
# Run predictions on the test set
test_results = trainer.predict(tokenized_sleep['test'])
preds = np.argmax(test_results.predictions, axis=1)  
labels = test_results.label_ids
recordings = tokenized_sleep['test']['recording']

label_mask_ids_triples = list(zip(recordings, preds, labels))

# Print the triples
for recording, pred, label in label_mask_ids_triples:
    print(f"Recording: {recording}, Predicted: {pred}, Actual: {label}")


cm = confusion_matrix(labels, preds)
print(cm)

# Step 3: Visualize the confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Unhealthy"])
disp.plot(cmap=plt.cm.Blues)  # Optional: change the color map

# Save the confusion matrix plot
plt.savefig(matrix_name)  # Specify file path and name
plt.close()  # Close the plot after saving
