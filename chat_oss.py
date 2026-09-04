import json
import random
import re
import pickle
from collections import Counter
from flask import session
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

file_model = "model_oss_generated.pth"
file_intent = "intents_oss_generated.json"
file_config = "config_oss_generated.pkl"

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD INTENTS
# =====================================================

with open(file_intent, "r", encoding="utf-8") as f:
    intents = json.load(f)

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

vocab, label_encoder, max_length, labels = load_config()

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

def load_model():

    global vocab
    global label_encoder
    global max_length
    global labels

    vocab, label_encoder, max_length, labels = load_config()

    model = ChatLSTM(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_size=256,
        num_classes=len(labels),
        num_layers=2,
        dropout=0.5
    )

    model.load_state_dict(
        torch.load(
            file_model,
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model

model = load_model()

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

#==========================
#Buat Klarifikasi Ambiguitas PBG Padel DKJ dan PU
#==========================
DKJ_KEYWORDS = [
    "dkj",
    "jakarta",
    "daerah khusus jakarta",
    "dki",
    "dki jakarta"
]
NON_DKJ_KEYWORDS = [
    "indonesia",
    "nasional",
    "luar jakarta",
    "selain jakarta",
    "daerah lain",
    "wilayah lain",
    "bekasi",
    "depok",
    "tangerang",
    "bogor",
    "bandung",
    
]

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

    # ==========================================
    # CEK APAKAH SEDANG MENUNGGU KLARIFIKASI
    # ==========================================

    if session.get("waiting_clarification") == "pbg_region":
        region = detect_region(message)
        if region == "dkj":
            session.pop(
                "waiting_clarification",
                None
            )
            return get_response(
                "syarat_izin_PBG_Padel_SIMBG_DKJ"
            )
        elif region == "non_dkj":
            session.pop(
                "waiting_clarification",
                None
            )
            return get_response(
                "syarat_izin_PBG_Padel_SIMBG_PU"
            )
        else:
            session.pop(
                "waiting_clarification",
                None
            )
        
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

    #===========================================
    # Deteksi Wilayah DKJ atau Non-DKJ untuk Intent Syarat PBG Padel
    #===========================================

    region = detect_region(message)

    # ==========================================
    # CLARIFICATION PBG
    # ==========================================

    if tag == "syarat_izin_PBG_Padel_SIMBG_DKJ":
        if region is None:
            session["waiting_clarification"] = "pbg_region"
            return {
                "response":
                    "PBG lapangan padel yang dimaksud "
                    "untuk wilayah Jakarta atau wilayah lainnya?",
                "intent": "clarification_pbg",
                "confidence": confidence
            }
    if tag == "syarat_izin_PBG_Padel_SIMBG_PU":
        if region is None:
            session["waiting_clarification"] = "pbg_region"
            return {
                "response":
                    "PBG lapangan padel yang dimaksud "
                    "untuk wilayah Jakarta atau wilayah lainnya?",
                "intent": "clarification_pbg",
                "confidence": confidence
            }
        
    #=========================================
    # Akhir Klarifikasi PBG
    #=========================================
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