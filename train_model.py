import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from snorkel.labeling.model import LabelModel
from snorkel.labeling import labeling_function, PandasLFApplier

# Define labels
ABSTAIN = -1
POSITIVE = 0
NEGATIVE = 1
NEUTRAL = 2

# Load dataset
df = pd.read_csv('Reviews.csv')

# 🔹 Ensure correct column name
TEXT_COLUMN = "Text"  # Change this to match your dataset's text column

# Define labeling functions (LFs)
@labeling_function()
def lf_contains_good(row):
    text = row[TEXT_COLUMN] if isinstance(row[TEXT_COLUMN], str) else ""
    return POSITIVE if "good" in text.lower() else ABSTAIN

@labeling_function()
def lf_contains_bad(row):
    text = row[TEXT_COLUMN] if isinstance(row[TEXT_COLUMN], str) else ""
    return NEGATIVE if "bad" in text.lower() else ABSTAIN

@labeling_function()
def lf_contains_okay(row):
    text = row[TEXT_COLUMN] if isinstance(row[TEXT_COLUMN], str) else ""
    return NEUTRAL if "okay" in text.lower() else ABSTAIN

# Apply labeling functions
lfs = [lf_contains_good, lf_contains_bad, lf_contains_okay]
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df)  # ✅ Fixed: Now applies to DataFrame rows correctly

# Train Snorkel Label Model
label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train, n_epochs=500, log_freq=100)

# Assign Snorkel Labels
df["snorkel_label"] = label_model.predict(L_train)

# Remove low-confidence labels
df_filtered = df[df["snorkel_label"] != ABSTAIN]

# 🔹 Fix potential column name issue here as well
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_filtered[TEXT_COLUMN].tolist(), df_filtered["snorkel_label"].tolist(), test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize data
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=128)
val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=128)

# Create dataset class
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
model.save_pretrained("./bert_model")
tokenizer.save_pretrained("./bert_model")
