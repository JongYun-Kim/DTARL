# Ray and RLlib
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

# Python Modules
import numpy as np
import copy

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# from ray.rllib.utils import try_import_torch
# # Try to import PyTorch
# torch, nn = try_import_torch()

# My Modules
# from model.Transformer_modules
# from model.Transformer_modules.token_embedding import TokenEmbedding
# from model.Transformer_modules.positional_encoding import PositionalEncoding
# from model.Transformer_modules.transformer_embedding import TransformerEmbedding
from model.Transformer_modules.token_embedding import LinearEmbedding
from model.Transformer_modules.multi_head_attention_layer import MultiHeadAttentionLayer
from model.Transformer_modules.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from model.Transformer_modules.encoder_block import EncoderBlock
# from model.Transformer_modules.decoder_block import DecoderBlock
from model.Transformer_modules.decoder_block import CustomDecoderBlock as DecoderBlock
from model.Transformer_modules.encoder import Encoder
from model.Transformer_modules.decoder import Decoder
from model.Transformer_modules.pointer_net import PointerGenerator

# TODO (1): [o] Create model layers individually if necessary (e.g. attention_encoder, ...)
# TODO (2): [x] Check masks
#       TODO (2-1): [x] padding mask in attention (encoder, decoder)
#       TODO (2-2): [x] action mask in attention (decoder ONLY!)
# TODO (3): [x] Determine how to use state of the model
#      TODO (3-1): [x] Determine how do we use task-forward pass
#      TODO (3-2): [x] Determine the shape of it
#      TODO (3-3): [x] Use two learnable placeholders to assemble the context vector in the decoder
#                      allow it only when task-forward count==1
# TODO (4): [x] Get generator to work (consider clipping the vals before masking)
# TODO (5): [x] In the comments get the right dimensions of tensors that are actually used across all the modules
#               (everywhere. particularly d_embed; it is used interchangeably with d_model...!)
# TODO (6): [x] Consider dropping out the dropout layers in the modules of the model as we are dealing with RL.
#               see others' implementations;

# TODO: [o] Add the embedding layer (src embedding)
# TODO: [o] Take copy out of the model (Plz ask him!)


class MyCustomTransformerModel(nn.Module):
    def __init__(self,
                 src_embed,
                 # tgt_embed,
                 encoder,
                 decoder,
                 generator,):
        super(MyCustomTransformerModel, self).__init__()
        self.src_embed = src_embed
        # self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    # def encode_out_dated(self, src, src_mask):
    #     return self.encoder(self.src_embed(src), src_mask)
    #
    # def decode_out_dated(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
    #     return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)

    def forward(self,
                src,  # shape: (batch_size, src_seq_len, d_embed)  # already padded and embedded
                tgt,  # shape: (batch_size, tgt_seq_len, d_embed)
                # src_pad_tokens,  # shape: (batch_size, src_seq_len)
                # tgt_pad_tokens,  # shape: (batch_size, tgt_seq_len)
                ):
        src_mask = self.make_src_mask(src)  # TODO: Get the right pad tokens; from src_pad_tokens!
        tgt_mask = self.make_src_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)  # shape: (batch_size, tgt_seq_len, d_embed)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask

    # def make_tgt_mask_outdated(self, tgt):
    #     pad_mask = self.make_pad_mask(tgt, tgt)
    #     seq_mask = self.make_subsequent_mask(tgt, tgt)
    #     mask = pad_mask & seq_mask
    #     return pad_mask & seq_mask

    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask

    def make_pad_mask(self, query, key, pad_idx=1, dim_check=False):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        # If input_token==pad_idx, then the mask value is 0, else 1
        # In the MHA layer, (no attention) == (mask value is 0) == (input_token==pad_idx)

        # Check if the query and key have the same dimension
        if dim_check:
            assert len(query.shape) == 2, "query must have 2 dimensions: (n_batch, query_seq_len)"
            assert len(key.shape) == 2, "key must have 2 dimensions: (n_batch, key_seq_len)"
            assert query.size(0) == key.size(0), "query and key must have the same batch size"

        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)  # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask  # (n_batch, 1, query_seq_len, key_seq_len)

    # def make_pad_mask(self, query, key, pad_value=0, dim_check=False):
    #     """
    #     This pad masking is used when the query and key are already embedded
    #     """
    #     # query: (n_batch, query_seq_len, d_embed)
    #     # key: (n_batch, key_seq_len, d_embed)
    #     # pad_value: the value that is used to pad the sequence TODO: Check what value you want to use (0, 1, or else)
    #
    #     # Check if the query and key have the dimensionality we want
    #     if dim_check:
    #         assert len(query.shape) == 3, "query tensor must be 3-dimensional"
    #         assert len(key.shape) == 3, "key tensor must be 3-dimensional"
    #         assert query.shape[0] == key.shape[0], "query and key batch sizes must be the same"
    #         assert query.shape[2] == key.shape[2], "query and key embedding sizes must be the same"
    #
    #     # Get the query and key sequence lengths
    #     query_seq_len, key_seq_len = query.size(1), key.size(1)
    #
    #     # create mask where the whole vector equals to pad_value
    #     key_mask = (key != pad_value).any(dim=-1).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
    #     key_mask = key_mask.repeat(1, 1, query_seq_len, 1)  # (n_batch, 1, query_seq_len, key_seq_len)
    #
    #     query_mask = (query != pad_value).any(dim=-1).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
    #     query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)
    #
    #     mask = key_mask & query_mask
    #     mask.requires_grad = False
    #     return mask

    # def make_subsequent_mask(self, query, key):
    #     query_seq_len, key_seq_len = query.size(1), key.size(1)
    #
    #     tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')  # lower triangle without diagonal
    #     mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
    #     return mask


class MyCustomRLlibModel(TorchModelV2, nn.Module):
    # TODO: (1) Get right dimensions for the model from the obs_space
    # TODO: (2) Get the right dimensions for the model from the action_space (generator part)
    # TODO: (4) Check the vals: d_k, d_model, d_embed, h, d_ff, n_layer
    # TODO: (5) Modify the decoder to compute the attention we want
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # tgt_vocab_size = 100
        # device = torch.device("cpu")  # RLlib takes care of device placement
        # max_len = 256

        d_subobs = 2  # dimension of the input tokens!
        d_embed_input = 128
        d_embed_context = 3 * d_embed_input  # 384 (==128*3)
        d_model = 128  # dimension of model (:=d_k * h);
        # d_model_decoder = 128
        # TODO: Would you define d_model_decoder as well? Consider [d_embed_query]==(3*d_embed_input)==(d_embed_context)
        n_layer_encoder = 6
        n_layer_decoder = 1
        h = 8  # number of heads
        d_ff = 512  # dimension of feed forward; usually 2-4 times d_model
        dr_rate = 0  # dropout rate; 0 in our case as we use reinforcement learning... Or may not?
        norm_eps = 1e-5  # epsilon parameter of layer normalization  # TODO: Check if this value works fine

        dpcopy = copy.deepcopy  # TODO: Would you kindly improve the readability?

        # tgt_token_embed = TokenEmbedding(
        #     d_embed=d_embed,
        #     vocab_size=tgt_vocab_size)
        # pos_embed = PositionalEncoding(
        #     d_embed=d_embed,
        #     max_len=max_len,
        #     # device=device
        # )
        # src_embed = TransformerEmbedding(
        #     token_embed=src_token_embed,
        #     pos_embed=copy(pos_embed),
        #     dr_rate=dr_rate)
        # tgt_embed = TransformerEmbedding(
        #     token_embed=tgt_token_embed,
        #     pos_embed=copy(pos_embed),
        #     dr_rate=dr_rate)

        # Module Level: Encoder
        # Need an embedding layer for the input; 2->128 in the case of Kool2019
        input_embed = LinearEmbedding(
            d_env=d_subobs,
            d_embed=d_embed_input,
        )
        attention_encoder = MultiHeadAttentionLayer(
            d_model=d_model,
            h=h,
            q_fc=nn.Linear(d_embed_input, d_model),
            kv_fc=nn.Linear(d_embed_input, d_model),
            out_fc=nn.Linear(d_model, d_embed_input),
            dr_rate=dr_rate)
        position_ff_encoder = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed_input, d_ff),
            fc2=nn.Linear(d_ff, d_embed_input),
            dr_rate=dr_rate)
        norm_encoder = nn.LayerNorm(d_embed_input, eps=norm_eps)
        # Module Level: Decoder
        attention_decoder = MultiHeadAttentionLayer(
            d_model=d_model,
            h=h,
            q_fc=nn.Linear(d_embed_context, d_model),
            kv_fc=nn.Linear(d_embed_input, d_model),
            out_fc=nn.Linear(d_model, d_embed_context),
            dr_rate=dr_rate)
        # position_ff_decoder = PositionWiseFeedForwardLayer(
        #     fc1=nn.Linear(d_embed_context, d_ff),
        #     fc2=nn.Linear(d_ff, d_embed_context),
        #     dr_rate=dr_rate)
        norm_decoder = nn.LayerNorm(d_embed_context, eps=norm_eps)

        # Block Level
        encoder_block = EncoderBlock(
            self_attention=dpcopy(attention_encoder),  # TODO: Can remove dpcopy; the block is deep-copied in Encoder
            position_ff=dpcopy(position_ff_encoder),
            norm=dpcopy(norm_encoder),
            dr_rate=dr_rate)
        decoder_block = DecoderBlock(
            self_attention=None,  # No self-attention in the decoder in this case!
            cross_attention=dpcopy(attention_decoder),
            position_ff=None,  # No position-wise FFN in the decoder in this case!
            norm=dpcopy(norm_decoder),
            dr_rate=dr_rate)

        # Transformer Level (Encoder + Decoder)
        encoder = Encoder(
            encoder_block=encoder_block,
            n_layer=n_layer_encoder,
            norm=dpcopy(norm_encoder))
        decoder = Decoder(
            decoder_block=decoder_block,
            n_layer=n_layer_decoder,
            # norm=dpcopy(norm_decoder),)
            norm=None, )  # TODO: Decide whether to include norm in the decoder!! you can pass None if not needed
        action_size = action_space.n  # it gives n given that action_space is Discrete(n).
        generator = PointerGenerator()  # outputs a probability distribution over the input sequence

        # Initialize state
        # self.state

        # Define the actor
        self.policy_net = MyCustomTransformerModel(
            src_embed=input_embed,
            # tgt_embed=tgt_embed,
            encoder=encoder,
            decoder=decoder,
            generator=generator
        )
        # Define the critic layer
        self.values = None
        self.value_net = nn.Linear(d_model, 1)

    def forward(self, input_dict, state, seq_lens):
        """
        :param input_dict: ["obs", "obs_flat", "prev_action", "prev_reward"]
               input_dict["obs"] is a dictionary, each value of which has shape of (batch_size, {data_shape})
        :param state:
        :return: x, state
        """
        x = input_dict["obs"]
        x, state = self.policy_net(x, state)
        self.values = x
        # Do the rest of the forward pass here, if necessary

        return x, state

    def value_function(self):
        return torch.mean(self.value_net(self.values))

