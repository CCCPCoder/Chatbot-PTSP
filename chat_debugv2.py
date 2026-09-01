import json
import pickle
import random
import re

import torch
import torch.nn as nn

#=====================================================
#DEVICE
#=====================================================

device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

#=====================================================
#LOAD CONFIG
#=====================================================

with open("config.pkl","rb") as f:
    config=pickle.load(f)

vocab=config["vocab"]
label_encoder=config["label_encoder"]
max_length=config["max_length"]

labels=list(label_encoder.classes_)

#=====================================================
#LOAD INTENTS
#=====================================================

with open(
    "intents.json",
    "r",
    encoding="utf-8"
) as f:
    intents=json.load(f)

#=====================================================
#PREPROCESSING
#=====================================================

def clean_text(text):

    text=text.lower()

    text=re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text=re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

def tokenize(text):
    return clean_text(text).split()

def numericalize(tokens):

    return[
        vocab.get(
            token,
            vocab["<UNK>"]
        )
        for token in tokens
    ]

def pad(sequence):

    if len(sequence)<max_length:

        sequence=sequence+[
            vocab["<PAD>"]
        ]*(
            max_length-len(sequence)
        )

    else:

        sequence=sequence[:max_length]

    return sequence

#=====================================================
#MODEL
#=====================================================

EMBEDDING_DIM=128
HIDDEN_SIZE=256
NUM_LAYERS=2
DROPOUT=0.5

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

        self.embedding=nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm=nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout=nn.Dropout(dropout)

        self.fc=nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self,x):

        embedding=self.embedding(x)

        output,(hidden,cell)=self.lstm(
            embedding
        )

        hidden_last=hidden[-1]

        hidden_last=self.dropout(
            hidden_last
        )

        logits=self.fc(hidden_last)

        return{
            "embedding":embedding,
            "output":output,
            "hidden":hidden,
            "cell":cell,
            "logits":logits
        }

model=ChatLSTM(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_classes=len(labels),
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

#=====================================================
#AMBIL BOBOT LSTM LAYER 1
#=====================================================

weight_ih=model.lstm.weight_ih_l0.detach()
weight_hh=model.lstm.weight_hh_l0.detach()

bias_ih=model.lstm.bias_ih_l0.detach()
bias_hh=model.lstm.bias_hh_l0.detach()

hidden_size=model.lstm.hidden_size

#=====================================================
#PEMISAHAN GATE SESUAI PYTORCH
#URUTAN:
#INPUT
#FORGET
#CANDIDATE
#OUTPUT
#=====================================================

Wii,Wif,Wig,Wio=torch.split(
    weight_ih,
    hidden_size,
    dim=0
)

Whi,Whf,Whg,Who=torch.split(
    weight_hh,
    hidden_size,
    dim=0
)

Bii,Bif,Big,Bio=torch.split(
    bias_ih,
    hidden_size
)

Bhi,Bhf,Bhg,Bho=torch.split(
    bias_hh,
    hidden_size
)

#=====================================================
#AMBIL NEURON PERTAMA
#=====================================================

Wii=Wii[0]
Wif=Wif[0]
Wig=Wig[0]
Wio=Wio[0]

Whi=Whi[0]
Whf=Whf[0]
Whg=Whg[0]
Who=Who[0]

Bii=Bii[0]
Bif=Bif[0]
Big=Big[0]
Bio=Bio[0]

Bhi=Bhi[0]
Bhf=Bhf[0]
Bhg=Bhg[0]
Bho=Bho[0]

print("="*70)
print("MODEL BERHASIL DIMUAT")
print("="*70)
print("Embedding Dimension :",EMBEDDING_DIM)
print("Hidden Size :",HIDDEN_SIZE)
print("Jumlah Intent :",len(labels))
print("="*70)
print("DEBUG MENGGUNAKAN NEURON KE-1")
print("="*70)

