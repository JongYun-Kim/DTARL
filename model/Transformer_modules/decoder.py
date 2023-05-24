import copy
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, decoder_block, n_layer, norm):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
        self.norm = norm if norm is not None else nn.Identity()  # may break backward compatibility

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        out = self.norm(out)
        # out = self.norm(out) if self.norm is not None else out
        return out  # shape: (batch_size, tgt_seq_len, d_embed)


# class CustomDecoder(nn.Module):
#
#     def __init__(self, decoder_block, n_layer, norm, include_norm=False):
#         super(CustomDecoder, self).__init__()
#         self.n_layer = n_layer
#         self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
#         self.include_norm = include_norm
#         if self.include_norm:
#             self.norm = norm
#
#     def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
#         out = tgt
#         for layer in self.layers:
#             out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
#         if self.include_norm:
#             out = self.norm(out)
#         return out
