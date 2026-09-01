import json
import random
import re
import pickle
import torch
import torch.nn as nn

#=====================================================
#DEVICE
#=====================================================

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

with open("intents.json","r",encoding="utf-8") as f:
    intents=json.load(f)

#=====================================================
#PREPROCESSING
#=====================================================

def case_folding(text):
    return text.lower()

def cleaning(text):
    text=re.sub(r"[^a-zA-Z0-9\s]"," ",text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

def tokenize(text):
    return text.split()

def numericalize(tokens):
    return[
        vocab.get(token,vocab["<UNK>"])
        for token in tokens
    ]

def padding(sequence):
    if len(sequence)<max_length:
        sequence=sequence+[vocab["<PAD>"]]*(max_length-len(sequence))
    else:
        sequence=sequence[:max_length]
    return sequence

#=====================================================
#MODEL PARAMETER
#=====================================================

EMBEDDING_DIM=128
HIDDEN_SIZE=256
NUM_LAYERS=2
DROPOUT=0.5
NUM_CLASSES=len(labels)

print("="*70)
print("Device :",device)
print("Vocabulary :",len(vocab))
print("Jumlah Intent :",NUM_CLASSES)
print("Max Length :",max_length)
print("="*70)
#=====================================================
#MODEL LSTM
#=====================================================

class ChatLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_classes,
        num_layers=2,
        dropout=0.5
    ):
        super().__init__()

        self.embedding=nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm=nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
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
        lstm_output,(hidden_state,cell_state)=self.lstm(embedding)
        hidden_last=hidden_state[-1]
        dropout_output=self.dropout(hidden_last)
        logits=self.fc(dropout_output)
        return{
            "embedding":embedding,
            "lstm_output":lstm_output,
            "hidden_state":hidden_state,
            "cell_state":cell_state,
            "dropout":dropout_output,
            "logits":logits
        }

#=====================================================
#MEMBUAT MODEL
#=====================================================

model=ChatLSTM(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_CLASSES,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)
#=====================================================
#LOAD MODEL
#=====================================================

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model berhasil dimuat.")
print("="*70)

#=====================================================
#DISPLAY FUNCTION
#=====================================================

def show_tensor(title,tensor,max_element=128):

    print("\n"+"="*70)
    print(title)
    print("="*70)

    print("Shape :",tuple(tensor.shape))

    flat=tensor.detach().cpu().reshape(-1)

    if len(flat)<=max_element:
        print(flat.numpy())
    else:
        print(flat[:max_element].numpy())
        print("...")
        print("Total Element :",len(flat))

def show_softmax(probabilities):

    print("\n"+"="*70)
    print("SOFTMAX")
    print("="*70)

    probs=probabilities.squeeze().detach().cpu().numpy()

    for label,prob in zip(labels,probs):
        print(f"{label:<45}{prob*100:.4f}%")

def show_top5(probabilities):

    print("\n"+"="*70)
    print("TOP 5 CONFIDENCE")
    print("="*70)

    values,index=torch.topk(probabilities,5)

    values=values.squeeze().cpu().numpy()
    index=index.squeeze().cpu().numpy()

    for idx,val in zip(index,values):
        print(f"{labels[idx]:<45}{val*100:.4f}%")

print("Model siap digunakan.")
print("="*70)

#=====================================================
#FUNGSI MENGHITUNG GATE LSTM
#=====================================================

def sigmoid(x):
    return torch.sigmoid(x)

def debug_lstm_gate(model, embedding):

    lstm = model.lstm

    weight_ih = lstm.weight_ih_l0
    weight_hh = lstm.weight_hh_l0

    bias_ih = lstm.bias_ih_l0
    bias_hh = lstm.bias_hh_l0

    hidden_size = lstm.hidden_size

    h = torch.zeros(1, hidden_size).to(embedding.device)
    c = torch.zeros(1, hidden_size).to(embedding.device)

    for t in range(embedding.size(1)):

        x = embedding[:, t, :]

        gates = (
            torch.matmul(x, weight_ih.t())
            + bias_ih
            + torch.matmul(h, weight_hh.t())
            + bias_hh
        )

        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c + i * g

        h = o * torch.tanh(c)
        if(t == 0):
            print("="*70)
            print("TIME STEP :", t+1)

            print("\nForget Gate")
            print(f)

            print("\nInput Gate")
            print(i)

            print("\nCandidate Cell")
            print(g)

            print("\nCell State")
            print(c)

            print("\nOutput Gate")
            print(o)

            print("\nHidden State")
            print(h)

#=====================================================
#GET RESPONSE
#=====================================================

def get_response(message,threshold=0.75):

    print("\n"+"="*70)
    print("INPUT TEXT")
    print("="*70)
    print(message)

    casefold=case_folding(message)

    print("\n"+"="*70)
    print("CASE FOLDING")
    print("="*70)
    print(casefold)

    cleaned=cleaning(casefold)

    print("\n"+"="*70)
    print("CLEANING")
    print("="*70)
    print(cleaned)

    tokens=tokenize(cleaned)

    print("\n"+"="*70)
    print("TOKENIZING")
    print("="*70)
    print(tokens)

    word_index=numericalize(tokens)

    print("\n"+"="*70)
    print("WORD INDEX")
    print("="*70)
    print(word_index)

    padded=padding(word_index)

    print("\n"+"="*70)
    print("PADDING")
    print("="*70)
    print(padded)

    x=torch.tensor(
        [padded],
        dtype=torch.long
    ).to(device)

    print("\n"+"="*70)
    print("INPUT TENSOR")
    print("="*70)
    print("Shape :",tuple(x.shape))
    print(x)

    with torch.no_grad():

        outputs=model(x)

        embedding=outputs["embedding"]
        
        lstm_output=outputs["lstm_output"]
        hidden_state=outputs["hidden_state"]
        cell_state=outputs["cell_state"]
        dropout_output=outputs["dropout"]
        logits=outputs["logits"]

        probabilities=torch.softmax(
            logits,
            dim=1
        )

        confidence,predicted=torch.max(
            probabilities,
            dim=1
        )

        confidence=confidence.item()

        predicted_index=predicted.item()

        predicted_tag=labels[predicted_index]
        show_tensor(
            "EMBEDDING LAYER",
            embedding
        )
        debug_lstm_gate(
            model,
            embedding
        )
        show_tensor(
            "LSTM OUTPUT",
            lstm_output
        )

        show_tensor(
            "HIDDEN STATE",
            hidden_state
        )

        show_tensor(
            "CELL STATE",
            cell_state
        )

        show_tensor(
            "DROPOUT OUTPUT",
            dropout_output
        )

        show_tensor(
            "DENSE LAYER (LOGITS)",
            logits
        )

        show_softmax(
            probabilities
        )

        show_top5(
            probabilities
        )

        print("\n"+"="*70)
        print("PREDICTED INTENT")
        print("="*70)
        print(predicted_tag)

        print("\n"+"="*70)
        print("CONFIDENCE")
        print("="*70)
        print(f"{confidence*100:.2f}%")
        if confidence<threshold:

            print("\n"+"="*70)
            print("RESPONSE")
            print("="*70)
            print("Maaf, saya belum memahami pertanyaan tersebut.")

            return{
                "response":"Maaf, saya belum memahami pertanyaan tersebut.",
                "intent":"unknown",
                "confidence":confidence
            }

        for intent in intents["intents"]:

            if intent["tag"]==predicted_tag:

                response=random.choice(intent["responses"])

                print("\n"+"="*70)
                print("RESPONSE")
                print("="*70)
                print(response)

                return{
                    "response":response,
                    "intent":predicted_tag,
                    "confidence":confidence
                }

        print("\n"+"="*70)
        print("RESPONSE")
        print("="*70)
        print("Terjadi kesalahan.")

        return{
            "response":"Terjadi kesalahan.",
            "intent":"error",
            "confidence":confidence
        }

#=====================================================
#MAIN PROGRAM
#=====================================================

print("\n")
print("="*70)
print("CHATBOT DEBUG LSTM")
print("="*70)
print("Ketik 'exit' atau 'quit' untuk keluar.")
print("="*70)

while True:

    message=input("\nAnda : ")

    if message.lower() in ["exit","quit"]:
        print("\nProgram selesai.")
        break

    result=get_response(message)

    print("\n"+"="*70)
    print("HASIL AKHIR")
    print("="*70)
    print("Intent      :",result["intent"])
    print("Confidence  : {:.2f}%".format(result["confidence"]*100))
    print("Chatbot     :",result["response"])
    print("="*70)