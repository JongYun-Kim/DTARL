import copy
import torch.nn as nn

from Transformer_modules.residual_connection_layer import ResidualConnectionLayer


class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff, norm, dr_rate=0):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.position_ff = position_ff
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

    def forward(self, src, src_mask):
        # src: shape: (batch_size, src_seq_len, d_embed==d_embed_input)
        # src_mask: shape: (batch_size, 1, src_seq_len, src_seq_len)
        out = src
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residual2(out, self.position_ff)
        # out: shape: (batch_size, src_seq_len, d_embed)
        return out
