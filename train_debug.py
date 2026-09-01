import json
import re
import random
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# =====================================================
# HYPERPARAMETER
# =====================================================

RANDOM_STATE = 42
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
EPOCHS = 100
model_name = "model_debug.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# =====================================================
# LOAD DATASET
# =====================================================

with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

sentences = []
labels = []

for intent in intents["intents"]:
    tag = intent["tag"]

    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(tag)

print("Jumlah data :", len(sentences))
print("Jumlah intent :", len(set(labels)))

# =====================================================
# PREPROCESSING
# =====================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


def tokenize(text):

    return clean_text(text).split()


tokenized_sentences = [tokenize(sentence) for sentence in sentences]

# =====================================================
# BUILD VOCAB
# =====================================================

counter = Counter()

for sentence in tokenized_sentences:
    counter.update(sentence)

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word, freq in counter.items():

    if freq >= 1:
        vocab[word] = len(vocab)

print("Vocabulary :", len(vocab))

# =====================================================
# LABEL ENCODER
# =====================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)

print("Jumlah kelas :", num_classes)

# =====================================================
# CONVERT TO INDEX
# =====================================================

def numericalize(tokens):

    return [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]


sequences = [
    numericalize(sentence)
    for sentence in tokenized_sentences
]

max_length = max(len(sequence) for sequence in sequences)

print("Max Length :", max_length)

# =====================================================
# DATASET
# =====================================================

class ChatDataset(Dataset):

    def __init__(self, sequences, labels):

        self.sequences = sequences
        self.labels = labels

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, index):

        sequence = torch.tensor(
            self.sequences[index],
            dtype=torch.long
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return sequence, label


# =====================================================
# COLLATE FUNCTION
# =====================================================

def collate_fn(batch):

    sequences = [item[0] for item in batch]

    labels = torch.stack(
        [item[1] for item in batch]
    )

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=vocab["<PAD>"]
    )

    return padded, labels


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    sequences,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

train_dataset = ChatDataset(
    X_train,
    y_train
)

test_dataset = ChatDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print("Train :", len(train_dataset))
print("Test :", len(test_dataset))

# =====================================================
# MODEL LSTM
# =====================================================

class ChatLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_classes,
        num_layers,
        dropout
    ):
        super(ChatLSTM, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=vocab["<PAD>"]
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded)

        hidden = hidden[-1]

        hidden = self.dropout(hidden)

        output = self.fc(hidden)

        return output


# =====================================================
# INISIALISASI MODEL
# =====================================================

model = ChatLSTM(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_classes=num_classes,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print(model)

# =====================================================
# EVALUASI
# =====================================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, targets in loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == targets).sum().item()

            total += targets.size(0)

    accuracy = 100 * correct / total

    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy


# =====================================================
# TRAINING
# =====================================================

best_accuracy = 0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        correct += (predicted == targets).sum().item()

        total += targets.size(0)

    train_accuracy = 100 * correct / total

    train_loss = running_loss / len(train_loader)

    val_loss, val_accuracy = evaluate(
        model,
        test_loader
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # Simpan model terbaik
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(model.state_dict(), model_name)

        print("Model terbaik disimpan.")

print("\nTraining selesai.")

print(f"Akurasi terbaik : {best_accuracy:.2f}%")

config = {
    "vocab": vocab,
    "label_encoder": label_encoder,
    "max_length": max_length
}

with open("config.pkl", "wb") as f:
    pickle.dump(config, f)

torch.save(model.state_dict(), "model.pth")
# =====================================================
# LOAD MODEL UNTUK TEST
# =====================================================

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

with open("config.pkl", "rb") as f:
    config = pickle.load(f)

vocab = config["vocab"]
label_encoder = config["label_encoder"]
max_length = config["max_length"]

labels = list(label_encoder.classes_)

model = ChatLSTM(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_size=256,
    num_classes=len(labels),
    num_layers=2,
    dropout=0.5
).to(device)

classes = []

for intent in intents["intents"]:
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

model.load_state_dict(torch.load(model_name, map_location=device))
model.eval()

print("\nModel berhasil dimuat.")
print("Jumlah Vocabulary :", len(vocab))
print("Jumlah Label :", len(labels))
print("Jumlah Class :", len(classes))

all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)
precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)
print("="*40)

print(f"Accuracy  : {test_accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("="*40)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=classes,
    output_dict=True,
    zero_division=0
)

# Ambil hanya setiap intent
rows = []

for intent in classes:
    rows.append({
        "Intent": intent,
        "Precision": round(report[intent]["precision"], 4),
        "Recall": round(report[intent]["recall"], 4),
        "F1-Score": round(report[intent]["f1-score"], 4),
        "Support": int(report[intent]["support"])
    })

df = pd.DataFrame(rows)

print("\n===== Evaluasi Per Intent =====")
print(df.to_string(index=False))

#with pd.ExcelWriter('doc/hasil_evaluasi_per_intent.xlsx', mode='a', if_sheet_exists='new') as writer:
df.to_excel("hasil_evaluasi_per_intent.xlsx", index=False)