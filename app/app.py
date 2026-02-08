import torch
import torch.nn as nn
from flask import Flask, render_template, request
from model import Encoder, Decoder, Attention, Seq2Seq
import os

app = Flask(__name__)

class SimpleVocab:
    def __init__(self, counter, min_freq=2):
        self.itos = []
        self.stoi = {}
    def __len__(self): 
        return len(self.itos)
    def encode(self, tokens):
        return [self.stoi.get(tok, 0) for tok in tokens]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Vocabularies
en_vocab = torch.load('en_vocab.pt', map_location=device, weights_only = False)
ne_vocab = torch.load('ne_vocab.pt', map_location=device, weights_only = False)

# 2. Initialize Model Parameters 
INPUT_DIM = len(en_vocab)
OUTPUT_DIM = len(ne_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(HID_DIM, attn_type='additive')
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, attn)
model = Seq2Seq(enc, dec, device).to(device)

# 3. Load trained weights
model.load_state_dict(torch.load('additive_model.pt', map_location=device, weights_only=False))
model.eval()

def tokenize(text):
    return text.lower().split()

def translate_sentence(sentence, max_len=50):
    tokens = tokenize(sentence)
    # <sos> = 2, <eos> = 3
    src_idxs = [2] + en_vocab.encode(tokens) + [3]
    src_tensor = torch.tensor(src_idxs).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    trg_idxs = [2] # Start with <sos>
    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_idxs[-1]]).to(device)
        with torch.no_grad():
            output, hidden, _ = model.decoder(trg_tensor, hidden, encoder_outputs)
        
        pred_token = output.argmax(1).item()
        trg_idxs.append(pred_token)
        if pred_token == 3: # <eos>
            break

    # Convert indices to words, skip <sos> and <eos>
    translated_tokens = [ne_vocab.itos[i] for i in trg_idxs if i not in [0, 1, 2, 3]]
    return " ".join(translated_tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    translation = ""
    original = ""
    if request.method == 'POST':
        original = request.form.get('text_input')
        if original:
            translation = translate_sentence(original)
            
    return render_template('index.html', original=original, translation=translation)

if __name__ == '__main__':
    app.run(debug=True)