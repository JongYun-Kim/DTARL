import copy
import torch
import torch.nn as nn

from Transformer_modules.residual_connection_layer import ResidualConnectionLayer, IdentityResidualLayer


class DecoderBlock(nn.Module):

    def __init__(self, self_attention, cross_attention, position_ff, norm, dr_rate=0):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.cross_attention = cross_attention
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.position_ff = position_ff
        self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residual2(out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residual3(out, self.position_ff)
        return out


class CustomDecoderBlock(nn.Module):

    def __init__(self, cross_attention, norm, self_attention=None, position_ff=None, dr_rate=0):
        super(CustomDecoderBlock, self).__init__()
        self.self_attention = self_attention
        if self.self_attention is not None: print("self_attention is NOT None, but not used here!!!!!!!!!")
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

        self.cross_attention = cross_attention
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

        self.position_ff = position_ff
        self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        # tgt: (batch_size, tgt_seq_len, d_model)  # TODO: fix d_model; it probably should be d_embed instead
        # encoder_out: (batch_size, src_seq_len, d_model)
        # tgt_mask: (batch_size, 1, tgt_seq_len)
        # src_tgt_mask: (batch_size, tgt_seq_len, src_seq_len)

        # MHA layer with query as the output of the first MHA layer
        # Shape: (batch_size, tgt_seq_len, d_model)
        tgt = self.residual2(tgt, lambda tgt: self.cross_attention(query=tgt, key=encoder_out, value=encoder_out,
                                                                   mask=src_tgt_mask))

        # Position-wise feed-forward network, applied only if include_ffn is True
        # Shape: (batch_size, tgt_seq_len, d_model)
        if self.position_ff is not None:
            tgt = self.residual3(tgt, self.position_ff)

        # Return the output tensor
        # Shape: (batch_size, tgt_seq_len, d_
        return tgt


class ProbablyAlmostUniversalDecoderBlockLol(nn.Module):  # Maybe act as a universal decoder block?
    def __init__(self, cross_attention, norm, self_attention=None, position_ff=None, dr_rate=0):
        super(ProbablyAlmostUniversalDecoderBlockLol, self).__init__()
        self.cross_attention = cross_attention

        if self_attention is not None:
            self.self_attention = self_attention
            self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        else:
            self.self_attention = lambda query, key, value, mask: query
            self.residual1 = IdentityResidualLayer()

        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

        if position_ff is not None:
            self.position_ff = position_ff
            self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        else:
            self.position_ff = lambda x: x
            self.residual3 = IdentityResidualLayer()

    def forward(self, tgt, encoder_out, src_tgt_mask, tgt_mask=None):
        # tgt: (batch_size, tgt_seq_len, d_model)  # d_model used interchangeably with d_embed; TODO: fix this l8er
        # encoder_out: (batch_size, src_seq_len, d_model)
        # tgt_mask: (batch_size, tgt_seq_len, tgt_seq_len); may vary; check your self-attention layer's input shape
        # src_tgt_mask: (batch_size, tgt_seq_len, src_seq_len)

        # 1. Please preprocess your input to the self-attention layer if necessary
        # [Example implementation] Compute the mean of encoder_out along the sequence length dimension
        # avg_enc_out shape: (batch_size, 1, d_model)
        avg_enc_out = torch.mean(encoder_out, dim=1, keepdim=True)
        # Expand avg_enc_out to the same size as tgt and concatenate along the last dimension
        # tgt_concat shape: (batch_size, tgt_seq_len, 2*d_model)
        out_concat = torch.cat((tgt, avg_enc_out.expand_as(tgt)), dim=-1)
        # !!! Make sure that you have the tgt_mask that aligns with the dimension of the preprocessed input

        # 2. Put the preprocessed input into the self-attention layer; Be careful about the dimensions!
        # First MHA layer with query as the concatenation of out and avg_enc_out
        # Shape: (batch_size, tgt_seq_len, d_model)
        out = self.residual1(out_concat, lambda out: self.self_attention(query=out, key=encoder_out, value=encoder_out,
                                                                         mask=tgt_mask))
        # 3. Please process the output of the self-attention layer to the cross-attention layer
        # Your implementation here:

        # 4. Put the (preprocessed) input into the cross-attention layer; Be careful about the dimensions!
        # Second MHA layer with query as the output of the first MHA layer
        # Shape: (batch_size, tgt_seq_len, d_model)
        out = self.residual2(out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out,
                                                                   mask=src_tgt_mask))

        # 5. Put the output of the cross-attention layer into the position-wise feed-forward network
        # Position-wise feed-forward network
        # Shape: (batch_size, tgt_seq_len, d_model)
        out = self.residual3(out, self.position_ff)

        # Return the output tensor
        # Shape: (batch_size, tgt_seq_len, d_model)
        return out


# class CustomDecoderBlock_outdated(nn.Module):
#
#     def __init__(self, cross_attention, position_ff, norm, dr_rate=0, include_ffn=True):
#         super(CustomDecoderBlock_outdated, self).__init__()
#         self.cross_attention = cross_attention
#         self.position_ff = position_ff
#         self.include_ffn = include_ffn  # flag to decide if position_ff is included
#         self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
#         self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
#         if self.include_ffn:
#             self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
#
#     def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
#         # tgt: (batch_size, tgt_seq_len, d_model)
#         # encoder_out: (batch_size, src_seq_len, d_model)
#         # tgt_mask: (batch_size, 1, tgt_seq_len)
#         # src_tgt_mask: (batch_size, tgt_seq_len, src_seq_len)
#         out = tgt
#
#         # Compute the mean of encoder_out along the sequence length dimension
#         # avg_enc_out shape: (batch_size, 1, d_model)
#         avg_enc_out = torch.mean(encoder_out, dim=1, keepdim=True)
#
#         # Expand avg_enc_out to the same size as out and concatenate along the last dimension
#         # out_concat shape: (batch_size, tgt_seq_len, 2*d_model)
#         out_concat = torch.cat((out, avg_enc_out.expand_as(out)), dim=-1)
#
#         # First MHA layer with query as the concatenation of out and avg_enc_out
#         # Shape: (batch_size, tgt_seq_len, d_model)
#         out = self.residual1(out_concat, lambda out: self.self_attention(query=out, key=encoder_out, value=encoder_out,
#                                                                          mask=tgt_mask))
#
#         # Second MHA layer with query as the output of the first MHA layer
#         # Shape: (batch_size, tgt_seq_len, d_model)
#         out = self.residual2(out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out,
#                                                                    mask=src_tgt_mask))
#
#         # Position-wise feed-forward network, applied only if include_ffn is True
#         # Shape: (batch_size, tgt_seq_len, d_model)
#         if self.include_ffn:
#             out = self.residual3(out, self.position_ff)
#
#         # Return the output tensor
#         # Shape: (batch_size, tgt_seq_len, d_model)
#         return out
