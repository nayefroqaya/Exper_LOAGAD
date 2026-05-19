import torch.nn as nn
import torch

from .transformer import TransformerBlock
from .embedding import BERTEmbedding

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, max_len=512, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, is_logkey=True, is_time=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        print(hidden)
        hidden = 300
    
        self.feed_forward_hidden = hidden * 2

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, max_len=max_len, is_logkey=is_logkey, is_time=is_time)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 2, dropout) for _ in range(n_layers)])

    #def forward(self, x, segment_info=None, time_info=None):
    def forward(self, sequence, segment_info=None, time_info=None):
        x = x.long()

        # Fix shape if input is [seq_len, batch] instead of [batch, seq_len]
        if x.dim() == 2 and x.shape[0] < x.shape[1]:
            x = x.transpose(0, 1).contiguous()

            if segment_info is not None and segment_info.dim() == 2:
                segment_info = segment_info.transpose(0, 1).contiguous()

            if time_info is not None and time_info.dim() == 2:
                time_info = time_info.transpose(0, 1).contiguous()

        batch_size, seq_len = x.size()

        # Attention mask shape: [batch_size, 1, seq_len, seq_len]
        mask = (x > 0).unsqueeze(1).unsqueeze(2)
        mask = mask.expand(batch_size, 1, seq_len, seq_len)

        x = self.embedding(x, segment_info=segment_info, time_info=time_info)
        #x = self.embedding(x, segment_label=segment_info, time_info=time_info)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
