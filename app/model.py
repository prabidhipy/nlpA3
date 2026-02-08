import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hid_dim, attn_type='additive'):
        super().__init__()
        self.attn_type = attn_type
        if attn_type == 'additive':
            self.W = nn.Linear(hid_dim * 3, hid_dim)
            self.v = nn.Linear(hid_dim, 1, bias=False)
        else:
            self.W = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        h = encoder_outputs.permute(1, 0, 2)
        s = hidden.unsqueeze(1).repeat(1, src_len, 1)

        if self.attn_type == 'additive':
            energy = torch.tanh(self.W(torch.cat((s, h), dim=2)))
            attn = self.v(energy).squeeze(2)
        else:
            attn = torch.bmm(
                hidden.unsqueeze(1),
                self.W(h).permute(0, 2, 1)
            ).squeeze(1)

        return F.softmax(attn, dim=1)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))
        )
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        h = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, h).permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        prediction = self.fc_out(
            torch.cat(
                (output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)),
                dim=1
            )
        )

        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
