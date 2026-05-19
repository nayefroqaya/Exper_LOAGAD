import torch.nn as nn
import torch
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding
from .time_embed import TimeEmbedding
import torch

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1, is_logkey=True, is_time=False):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=max_len)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.time_embed = TimeEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.is_logkey = is_logkey
        self.is_time = is_time

    #def forward(self, sequence, segment_label=None, time_info=None):
    #    x = self.position(sequence)
    #    # if self.is_logkey:
    #    x = x + self.token(sequence)
    #    if segment_label is not None:
    #        x = x + self.segment(segment_label)
    #    if self.is_time:
    #        x = x + self.time_embed(time_info)
    #    return self.dropout(x)

    def forward(self, sequence, segment_label=None, time_info=None):
        sequence = sequence.long()

        # Fix shape if input is [seq_len, batch] instead of [batch, seq_len]
        if sequence.dim() == 2 and sequence.shape[0] < sequence.shape[1]:
            sequence = sequence.transpose(0, 1).contiguous()

        batch_size, seq_len = sequence.size()

        # Token embedding: event IDs
        token_x = self.token(sequence)

        # Position embedding: positions 0, 1, 2, ..., seq_len - 1
        position_ids = torch.arange(seq_len, dtype=torch.long, device=sequence.device).unsqueeze(0).expand(batch_size,
                                                                                                           seq_len)

        x = self.position(position_ids)
        x = x + token_x

        # Only use segment embedding if labels are real segment IDs: 0 or 1.
        # If segment_label contains event IDs like 146, skip it.
        if segment_label is not None:
            segment_label = segment_label.long()

            if segment_label.shape != sequence.shape:
                if segment_label.dim() == 2 and segment_label.shape[0] < segment_label.shape[1]:
                    segment_label = segment_label.transpose(0, 1).contiguous()

            if segment_label.min().item() >= 0 and segment_label.max().item() <= 1:
                x = x + self.segment(segment_label)

        if self.is_time:
            x = x + self.time_embed(time_info)

        return self.dropout(x)


