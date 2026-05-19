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

        # Fix shape if batch is transposed: [seq_len, batch] -> [batch, seq_len]
        # Your earlier debug showed (121, 512), but expected is (512, 121)
        if sequence.dim() == 2 and sequence.shape[0] < sequence.shape[1]:
            sequence = sequence.transpose(0, 1).contiguous()

        batch_size, seq_len = sequence.size()

        # -------------------------------
        # Helper to get embedding size
        # -------------------------------
        def get_num_embeddings(module):
            if hasattr(module, "embedding"):
                return module.embedding.num_embeddings
            if hasattr(module, "weight"):
                return module.weight.size(0)
            if hasattr(module, "token"):
                return module.token.weight.size(0)
            if hasattr(module, "pe"):
                return module.pe.size(1) if module.pe.dim() >= 2 else module.pe.size(0)
            return None

        # -------------------------------
        # 1. Check token IDs
        # -------------------------------
        token_vocab_size = get_num_embeddings(self.token)
        seq_cpu = sequence.detach().cpu()

        if token_vocab_size is not None:
            seq_min = int(seq_cpu.min().item())
            seq_max = int(seq_cpu.max().item())

            if seq_min < 0 or seq_max >= token_vocab_size:
                print("BAD TOKEN INDEX FOUND")
                print("sequence shape:", tuple(sequence.shape))
                print("sequence min:", seq_min)
                print("sequence max:", seq_max)
                print("token vocab size:", token_vocab_size)
                print("bad values:", seq_cpu[(seq_cpu < 0) | (seq_cpu >= token_vocab_size)][:50])
                raise ValueError("Token index out of range for token embedding.")

        token_x = self.token(sequence)
        torch.cuda.synchronize()

        # -------------------------------
        # 2. Position IDs should be 0..seq_len-1
        # -------------------------------
        position_ids = torch.arange(seq_len, dtype=torch.long, device=sequence.device).unsqueeze(0).expand(batch_size,
                                                                                                           seq_len)

        position_size = get_num_embeddings(self.position)

        if position_size is not None:
            pos_cpu = position_ids.detach().cpu()
            pos_max = int(pos_cpu.max().item())

            if pos_max >= position_size:
                print("BAD POSITION INDEX FOUND")
                print("sequence shape:", tuple(sequence.shape))
                print("seq_len:", seq_len)
                print("max position id:", pos_max)
                print("position embedding size:", position_size)
                raise ValueError("Position index out of range for position embedding.")

        x = self.position(position_ids)
        torch.cuda.synchronize()

        x = x + token_x
        torch.cuda.synchronize()

        # -------------------------------
        # 3. Segment embedding check
        # -------------------------------
        if segment_label is not None:
            segment_label = segment_label.long()

            if segment_label.shape != sequence.shape:
                if segment_label.dim() == 2 and segment_label.shape[0] < segment_label.shape[1]:
                    segment_label = segment_label.transpose(0, 1).contiguous()

            segment_size = get_num_embeddings(self.segment)

            if segment_size is not None:
                seg_cpu = segment_label.detach().cpu()
                seg_min = int(seg_cpu.min().item())
                seg_max = int(seg_cpu.max().item())

                if seg_min < 0 or seg_max >= segment_size:
                    print("BAD SEGMENT INDEX FOUND")
                    print("segment shape:", tuple(segment_label.shape))
                    print("segment min:", seg_min)
                    print("segment max:", seg_max)
                    print("segment embedding size:", segment_size)
                    print("bad values:", seg_cpu[(seg_cpu < 0) | (seg_cpu >= segment_size)][:50])
                    raise ValueError("Segment index out of range for segment embedding.")

            x = x + self.segment(segment_label)
            torch.cuda.synchronize()

        # -------------------------------
        # 4. Time embedding check
        # -------------------------------
        if self.is_time:
            if time_info is None:
                raise ValueError("self.is_time=True but time_info is None")

            time_info = time_info.long()

            if time_info.shape != sequence.shape:
                if time_info.dim() == 2 and time_info.shape[0] < time_info.shape[1]:
                    time_info = time_info.transpose(0, 1).contiguous()

            time_size = get_num_embeddings(self.time_embed)

            if time_size is not None:
                time_cpu = time_info.detach().cpu()
                time_min = int(time_cpu.min().item())
                time_max = int(time_cpu.max().item())

                if time_min < 0 or time_max >= time_size:
                    print("BAD TIME INDEX FOUND")
                    print("time shape:", tuple(time_info.shape))
                    print("time min:", time_min)
                    print("time max:", time_max)
                    print("time embedding size:", time_size)
                    print("bad values:", time_cpu[(time_cpu < 0) | (time_cpu >= time_size)][:50])
                    raise ValueError("Time index out of range for time embedding.")

            x = x + self.time_embed(time_info)
            torch.cuda.synchronize()

        return self.dropout(x)