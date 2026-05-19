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


    def forward(self, x, segment_info=None, time_info=None):
            x = x.long()

            # Fix shape if input is [seq_len, batch] instead of [batch, seq_len]
            if x.dim() == 2 and x.shape[0] < x.shape[1]:
                x = x.transpose(0, 1).contiguous()

                if segment_info is not None and segment_info.dim() == 2:
                    segment_info = segment_info.transpose(0, 1).contiguous()

                if time_info is not None and time_info.dim() == 2:
                    time_info = time_info.transpose(0, 1).contiguous()

            batch_size, seq_len = x.size()

            # Correct attention mask shape: [batch_size, 1, seq_len, seq_len]
            mask = (x > 0).unsqueeze(1).unsqueeze(2)
            mask = mask.expand(batch_size, 1, seq_len, seq_len)

            x = self.embedding(x, segment_info=segment_info, time_info=time_info)

            for transformer in self.transformer_blocks:
                x = transformer.forward(x, mask)

            return x