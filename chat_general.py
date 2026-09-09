import json
import random
import re
import pickle
import os
from collections import Counter
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

SETTINGS_FILE = "runtime_settings.json"
DEFAULT_SETTINGS = {
    "model": "model.pth",
    "intent": "intents.json",
    "config": "config.pkl"
}
file_model = DEFAULT_SETTINGS["model"]
file_intent = DEFAULT_SETTINGS["intent"]
file_config = DEFAULT_SETTINGS["config"]

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# PREPROCESSING
# =====================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text):
    return clean_text(text).split()

# =====================================================
# BUILD VOCAB & LABELS
# =====================================================

def load_config():

    with open(file_config, "rb") as f:
        config = pickle.load(f)

    vocab = config["vocab"]
    label_encoder = config["label_encoder"]
    max_length = config["max_length"]

    labels = list(label_encoder.classes_)

    return vocab, label_encoder, max_length, labels

# =====================================================
# MODEL
# =====================================================

EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.5
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
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
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
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# =====================================================
# LOAD MODEL
# =====================================================

def build_model(config_values, model_file):
    vocab_value, label_encoder_value, max_length_value, labels_value = config_values
    model = ChatLSTM(
        vocab_size=len(vocab_value),
        embedding_dim=128,
        hidden_size=256,
        num_classes=len(labels_value),
        num_layers=2,
        dropout=0.5
    )

    model.load_state_dict(
        torch.load(
            model_file,
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model

def load_model():
    return build_model(load_config(), file_model)


def load_config_file(config_file):
    with open(config_file, "rb") as config_handle:
        config = pickle.load(config_handle)
    label_encoder_value = config["label_encoder"]
    return (
        config["vocab"],
        label_encoder_value,
        config["max_length"],
        list(label_encoder_value.classes_)
    )


def get_runtime_settings():
    if not os.path.exists(SETTINGS_FILE):
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
        return {
            key: settings.get(key, DEFAULT_SETTINGS[key])
            for key in DEFAULT_SETTINGS
        }
    except (OSError, json.JSONDecodeError):
        return DEFAULT_SETTINGS.copy()


def get_available_files():
    return {
        "models": sorted(
            name for name in os.listdir(".")
            if name.endswith(".pth") and os.path.isfile(name)
        ),
        "intents": sorted(
            name for name in os.listdir(".")
            if name.endswith(".json") and name != SETTINGS_FILE
            and os.path.isfile(name)
        ),
        "configs": sorted(
            name for name in os.listdir(".")
            if name.endswith(".pkl") and os.path.isfile(name)
        )
    }


def configure_runtime(model_file, intent_file, config_file):
    selected = {
        "model": os.path.basename(model_file),
        "intent": os.path.basename(intent_file),
        "config": os.path.basename(config_file)
    }
    available = get_available_files()
    if (selected["model"] not in available["models"] or
            selected["intent"] not in available["intents"] or
            selected["config"] not in available["configs"]):
        raise ValueError("Model, intents, atau config tidak tersedia.")

    with open(selected["intent"], "r", encoding="utf-8") as intent_handle:
        intents_value = json.load(intent_handle)
    if not isinstance(intents_value.get("intents"), list):
        raise ValueError("File intents harus memiliki properti 'intents'.")

    config_values = load_config_file(selected["config"])
    model_value = build_model(config_values, selected["model"])

    global file_model, file_intent, file_config
    global intents, vocab, label_encoder, max_length, labels, model
    file_model = selected["model"]
    file_intent = selected["intent"]
    file_config = selected["config"]
    intents = intents_value
    vocab, label_encoder, max_length, labels = config_values
    model = model_value

    with open(SETTINGS_FILE, "w", encoding="utf-8") as settings_file:
        json.dump(selected, settings_file, indent=4)
    return selected


def initialize_runtime():
    settings = get_runtime_settings()
    try:
        configure_runtime(
            settings["model"],
            settings["intent"],
            settings["config"]
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
        configure_runtime(
            DEFAULT_SETTINGS["model"],
            DEFAULT_SETTINGS["intent"],
            DEFAULT_SETTINGS["config"]
        )


initialize_runtime()

# =====================================================
# UTILITIES
# =====================================================

def numericalize(tokens):
    return [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]
def pad(sequence):
    if len(sequence) < max_length:
        sequence += [0] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

# =====================================================
# CHAT
# =====================================================

def format_response(text):

    # ==========================================
    # CEK APAKAH RESPONSE MEMILIKI LIST
    # ==========================================

    if "\n-" not in text:
        return f"<p>{text}</p>"

    lines = text.splitlines()

    html_parts = []

    paragraph = []
    list_items = []

    def flush_paragraph():
        nonlocal paragraph

        if paragraph:

            text_paragraph = " ".join(
                paragraph
            )

            html_parts.append(
                f"<p>{text_paragraph}</p>"
            )

            paragraph = []

    def flush_list():
        nonlocal list_items

        if list_items:

            html = (
                "<ol "
                "style='margin-left:10px;"
                "padding-left:10px;'>"
            )

            for item in list_items:

                html += (
                    "<li style='margin-bottom:6px;'>"
                    f"{item}"
                    "</li>"
                )

            html += "</ol>"

            html_parts.append(html)

            list_items = []

    # ==========================================
    # PROSES SETIAP BARIS
    # ==========================================

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # ======================================
        # ITEM LIST
        # ======================================

        if line.startswith("-"):

            # Tutup paragraf sebelumnya
            flush_paragraph()

            item = line[1:].strip()

            list_items.append(item)

        # ======================================
        # TEKS BIASA
        # ======================================

        else:

            # Jika sebelumnya sedang membuat list,
            # tutup list terlebih dahulu
            flush_list()

            paragraph.append(line)

    # ==========================================
    # TUTUP BAGIAN TERAKHIR
    # ==========================================

    flush_paragraph()
    flush_list()

    return "".join(html_parts)

def detect_region(text):
    text = clean_text(text)
    # Cek DKJ
    for keyword in DKJ_KEYWORDS:
        if keyword in text:
            return "dkj"
    # Cek wilayah lainnya
    for keyword in NON_DKJ_KEYWORDS:
        if keyword in text:
            return "non_dkj"
    # Tidak ditemukan wilayah
    return None

def get_response(message, threshold=0.75):
    tokens = tokenize(message)
    sequence = numericalize(tokens)
    sequence = pad(sequence)
    x = torch.tensor(
        [sequence],
        dtype=torch.long
    ).to(device)
    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )
    confidence = confidence.item()
    tag = labels[predicted.item()]
    if confidence < threshold:
        return {
        "response": "Maaf, saya belum memahami pertanyaan tersebut.",
        "intent": "unknown",
        "confidence": confidence
    }
    
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return {
                "response": format_response(response),
                "intent": tag,
                "confidence": confidence
            }

            return {
                "response": "Terjadi kesalahan.",
                "intent": "error",
                "confidence": confidence
            }