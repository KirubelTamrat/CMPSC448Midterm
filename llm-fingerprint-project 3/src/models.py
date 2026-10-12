"""Small text CNN and bidirectional LSTM classifiers written in PyTorch."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextCNN(nn.Module):
    def __init__(self, vocab, classes, embed=64, channels=64, dropout=.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed, channels, k, padding=k//2) for k in (3, 5, 7)])
        self.drop = nn.Dropout(dropout)
        self.output = nn.Linear(3 * channels, classes)

    def forward(self, token_ids, lengths):
        x = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            y = torch.relu(conv(x))
            valid = torch.arange(y.size(-1), device=y.device)[None, :] < lengths[:, None]
            y = y.masked_fill(~valid.unsqueeze(1), -1e4)
            pooled.append(y.amax(-1))
        return self.output(self.drop(torch.cat(pooled, dim=1)))


class TextLSTM(nn.Module):
    def __init__(self, vocab, classes, embed=64, hidden=48, dropout=.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed, padding_idx=0)
        self.rnn = nn.LSTM(embed, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.output = nn.Linear(hidden * 2, classes)

    def forward(self, token_ids, lengths):
        emb = self.embedding(token_ids)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        return self.output(self.drop(torch.cat((h[-2], h[-1]), dim=-1)))
