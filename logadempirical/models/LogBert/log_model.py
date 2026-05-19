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
    '''
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
    '''

    def forward(self, batch, time_info=None, device="cpu"):
        """
        Robust LogBERT forward pass.
        Auto-detects the correct label tensor from tuple/list batches.
        """

        original_batch = batch

        # --- Ensure batch is a dict ---
        if isinstance(batch, (list, tuple)):
            batch_dict = {"sequential": batch[0]}

            # Temporary label; if wrong, we auto-detect the correct one later
            if len(batch) > 1:
                batch_dict["label"] = batch[1]

            batch = batch_dict

        # --- Process sequences ---
        x = batch["sequential"]

        if isinstance(x, list):
            sequences = [torch.tensor(seq, dtype=torch.long, device=device) for seq in x]
            x = pad_sequence(sequences, batch_first=True, padding_value=0)

        elif isinstance(x, torch.Tensor):
            x = x.to(device).long()

        else:
            raise TypeError(f"Unsupported type for batch['sequential']: {type(x)}")

        # Fix shape if input is [seq_len, batch] instead of [batch, seq_len]
        if x.dim() == 2 and x.shape[0] < x.shape[1]:
            x = x.transpose(0, 1).contiguous()

        batch_size, seq_len = x.size()

        # --- Disable segment_info ---
        segment_info = None

        # --- Process time_info ---
        if time_info is not None:
            if isinstance(time_info, list):
                times = [torch.tensor(t, dtype=torch.long, device=device) for t in time_info]
                time_info = pad_sequence(times, batch_first=True, padding_value=0)

            elif isinstance(time_info, torch.Tensor):
                time_info = time_info.to(device).long()

            else:
                raise TypeError(f"Unsupported type for time_info: {type(time_info)}")

            if time_info.dim() == 2 and time_info.shape[0] < time_info.shape[1]:
                time_info = time_info.transpose(0, 1).contiguous()

            if time_info.size(1) < seq_len:
                diff = seq_len - time_info.size(1)
                time_info = torch.cat(
                    [time_info, torch.zeros(time_info.size(0), diff, device=device, dtype=torch.long)], dim=1)
            elif time_info.size(1) > seq_len:
                time_info = time_info[:, :seq_len]

        else:
            time_info = None

        # --- Process initial labels ---
        y = batch.get("label", None)

        if y is not None:
            if isinstance(y, list):
                y = torch.tensor(y, dtype=torch.long, device=device)
            elif isinstance(y, torch.Tensor):
                y = y.to(device).long()
            else:
                y = torch.tensor(y, dtype=torch.long, device=device)

            if y.dim() == 2:
                if y.shape[0] < y.shape[1]:
                    y = y.transpose(0, 1).contiguous()
                y = y[:, -1]

        # --- Forward pass through BERT ---
        bert_out = self.bert(x, segment_info=segment_info, time_info=time_info)

        # token_scores: [batch_size, seq_len, vocab_size]
        token_scores = self.mask_lm(bert_out)

        # Since y is one label per sequence, use only the last timestep
        if token_scores.dim() == 3:
            final_scores = token_scores[:, -1, :]  # [batch_size, vocab_size]
        else:
            final_scores = token_scores

        logits = final_scores
        probabilities = torch.softmax(final_scores, dim=-1)

        # --- Helper: check if a batch item can be the label tensor ---
        def normalize_possible_label(t):
            if not isinstance(t, torch.Tensor):
                return None

            tt = t.to(device).long()

            if tt.dim() == 2:
                if tt.shape[0] < tt.shape[1]:
                    tt = tt.transpose(0, 1).contiguous()
                tt = tt[:, -1]

            if tt.dim() != 1:
                return None

            if tt.numel() != final_scores.size(0):
                return None

            return tt

        # --- Loss calculation ---
        loss = None
        if self.criterion is not None:
            n_classes = final_scores.size(-1)

            y_is_valid = (y is not None and y.dim() == 1 and y.numel() == final_scores.size(
                0) and y.min().item() >= 0 and y.max().item() < n_classes)

            # If current y is invalid, search original tuple/list batch
            if not y_is_valid:
                found_y = None

                if isinstance(original_batch, (list, tuple)):
                    for i, item in enumerate(original_batch):
                        candidate = normalize_possible_label(item)

                        if candidate is None:
                            continue

                        if candidate.min().item() >= 0 and candidate.max().item() < n_classes:
                            found_y = candidate
                            print(f"Using batch[{i}] as label")
                            break

                if found_y is None:
                    print("BAD LOSS TARGET FOUND")
                    print("final_scores shape:", final_scores.shape)
                    print("n_classes:", n_classes)

                    if y is not None:
                        print("y shape:", y.shape)
                        print("y min:", y.min().item())
                        print("y max:", y.max().item())
                        print("bad y values:", y[(y < 0) | (y >= n_classes)][:50])

                    if isinstance(original_batch, (list, tuple)):
                        print("Batch item debug:")
                        for i, item in enumerate(original_batch):
                            if isinstance(item, torch.Tensor):
                                item_cpu = item.detach()
                                print(f"batch[{i}] shape={tuple(item.shape)}, "
                                      f"min={item_cpu.min().item()}, "
                                      f"max={item_cpu.max().item()}")
                            else:
                                print(f"batch[{i}] type={type(item)}")

                    raise ValueError("Could not find valid label tensor in batch.")

                y = found_y

            loss = self.criterion(final_scores.float(), y.long())

        # --- Return output ---
        return ModelOutput(logits=logits, probabilities=probabilities, loss=loss, embeddings=token_scores)


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
