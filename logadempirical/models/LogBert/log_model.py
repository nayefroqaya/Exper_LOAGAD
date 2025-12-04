import pdb

import torch.nn as nn
import torch
from .bert import BERT
from typing import Optional
from logadempirical.models.utils import ModelOutput
from torch.nn import LogSoftmax
from torch.nn.utils.rnn import pad_sequence


class BERTLog(nn.Module):
    """
    BERT Log Model
    """

    def __init__(self, bert: BERT, vocab_size, criterion: Optional[nn.Module] = None, hidden_size: int = 128,
                 n_class: int = 1, is_bilstm: bool = True):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLogModel(self.bert.hidden, vocab_size)
        self.time_lm = TimeLogModel(self.bert.hidden)
        # self.fnn_cls = LinearCLS(self.bert.hidden)
        # self.cls_lm = LogClassifier(self.bert.hidden)
        self.fc = nn.Linear(self.bert.hidden, vocab_size)
        self.num_directions = 2 if is_bilstm else 1
        self.fc2 = nn.Linear(vocab_size, n_class)  # sửa đổi so với gốc
        self.criterion = nn.NLLLoss()
        # self.result = {"logkey_output": None, "cls_output": None, }

    def forward(self, batch, time_info=None, device="cpu"):
        """
           Robust LogBERT forward pass.
           Handles variable-length sequences, segment_info, time_info, and optional labels.
           """
        # --- Ensure batch is a dict ---
        if isinstance(batch, (list, tuple)):
            batch_dict = {"sequential": batch[0]}
            if len(batch) > 1:
                batch_dict["label"] = batch[1]
            if len(batch) > 2:
                batch_dict["segment_info"] = batch[2]
            batch = batch_dict

        # --- Process sequences ---
        x = batch["sequential"]
        if isinstance(x, list):
            sequences = [torch.tensor(seq, dtype=torch.long, device=device) for seq in x]
            x = pad_sequence(sequences, batch_first=True, padding_value=0)
        elif isinstance(x, torch.Tensor):
            x = x.to(device)
        else:
            raise TypeError(f"Unsupported type for batch['sequential']: {type(x)}")

        batch_size, seq_len = x.size()

        # --- Process segment_info ---
        if "segment_info" in batch and batch["segment_info"] is not None:
            segs = [torch.tensor(s, dtype=torch.long, device=device) for s in batch["segment_info"]]
            segment_info = pad_sequence(segs, batch_first=True, padding_value=0)
            # Match x's length
            diff = seq_len - segment_info.size(1)
            if diff > 0:
                segment_info = torch.cat(
                    [segment_info, torch.zeros(segment_info.size(0), diff, device=device, dtype=torch.long)], dim=1)
            else:
                segment_info = segment_info[:, :seq_len]
        else:
            segment_info = torch.zeros_like(x, dtype=torch.long)

        # --- Process time_info ---
        if time_info is not None:
            if isinstance(time_info, list):
                times = [torch.tensor(t, dtype=torch.float, device=device) for t in time_info]
                time_info = pad_sequence(times, batch_first=True, padding_value=0.0)
            elif isinstance(time_info, torch.Tensor):
                time_info = time_info.to(device)
            else:
                raise TypeError(f"Unsupported type for time_info: {type(time_info)}")
            # Match x's length
            diff = seq_len - time_info.size(1)
            if diff > 0:
                time_info = torch.cat([time_info, torch.zeros(time_info.size(0), diff, device=device)], dim=1)
            else:
                time_info = time_info[:, :seq_len]
        else:
            time_info = torch.zeros_like(x, dtype=torch.float)

        # --- Process labels ---
        y = batch.get('label', None)
        if y is not None:
            if isinstance(y, list):
                y = torch.tensor(y, dtype=torch.long, device=device)
            elif isinstance(y, torch.Tensor):
                y = y.to(device)

        # --- Forward pass through BERT ---
        x = self.bert(x, segment_info=segment_info, time_info=time_info)
        x = self.mask_lm(x)
        logits = self.fc2(x)
        probabilities = torch.softmax(x, dim=-1)

        # --- Loss calculation ---
        loss = None
        if y is not None and self.criterion is not None:
            loss = self.criterion(x.transpose(1, 2).type(torch.FloatTensor), y.type(torch.LongTensor))

        # --- Return output ---
        return ModelOutput(logits=logits, probabilities=probabilities, loss=loss, embeddings=x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def predict(self, src, device="cpu"):
        del src['label']
        return self.forward(src, device=device).probabilities

    def predict_class(self, src, top_k=1, device="cpu"):
        del src['label']
        return torch.topk(self.forward(src, device=device).probabilities, k=top_k, dim=1).indices


class MaskedLogModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class TimeLogModel(nn.Module):
    def __init__(self, hidden, time_size=1):
        super().__init__()
        self.linear = nn.Linear(hidden, time_size)

    def forward(self, x):
        return self.linear(x)


class LogClassifier(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, cls):
        return self.linear(cls)


class LinearCLS(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.linear(x)
