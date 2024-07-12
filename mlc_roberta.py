import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
import numpy as np
import joblib

def compute_metrics(p):
    pred, labels = p
    pred = np.where(pred > 0.5, 1, 0)
    accuracy = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average='micro')
    precision = precision_score(labels, pred, average='micro')
    recall = recall_score(labels, pred, average='micro')
    # Calculate class-wise F1 score
    f1_score_classwise = f1_score(labels, pred, average=None)
    print("Class-wise F1 Scores:")
    for i, score in enumerate(f1_score_classwise):
        print(f"\tClass {class_mapping[i]}: {score}")

        #print("\tClass", i+1,":", score)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

# Load your data
data = pd.read_csv('original_label.csv')

# Assuming your labels are in a column named 'labels' and are comma-separated
data['label'] = data['label'].apply(lambda x: x.strip("[]").replace("'", "").split(', '))
# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'], data['label'], test_size=0.2)

# Encode the labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_labels)
val_labels = mlb.transform(val_labels)

# Access the classes_ attribute to see the mapping
class_mapping = mlb.classes_


#print("Class mapping:", class_mapping)

# Save the label mapping
joblib.dump(mlb.classes_, 'class_mapping.pkl')


# Load pre-trained BERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize the text
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

# Create a Dataset object
class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MultiLabelDataset(train_encodings, train_labels)
val_dataset = MultiLabelDataset(val_encodings, val_labels)

# Load pre-trained BERT model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(mlb.classes_), problem_type="multi_label_classification")


training_args = TrainingArguments(
    output_dir='./originalrresults',
    num_train_epochs=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)
# training_args = TrainingArguments(
#     output_dir='./resultszephyr',
#     num_train_epochs=50,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     evaluation_strategy="steps",
#     eval_steps=1000,  # Evaluate every 1000 steps
#     save_strategy="steps",
#     save_steps=500,  # Save checkpoint every 500 steps
#     save_total_limit=3  # Keep only the last 3 checkpoints
# )
# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate(eval_dataset=val_dataset)
print(metrics)  
