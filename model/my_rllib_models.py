# Ray: most versions suit for the script; but 1.13 recommended at the moment
import ray
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

# Numpy and math stuffs
import numpy as np

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.Transformer_modules; Check your python path and permission to it
from model.Transformer_modules.multi_head_attention_layer import MultiHeadAttentionLayer
from model.Transformer_modules.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from model.Transformer_modules.encoder_block import EncoderBlock
from model.Transformer_modules.decoder_block import DecoderBlock
from model.Transformer_modules.encoder import Encoder
from model.Transformer_modules.decoder import Decoder
from model.Transformer_modules.pointer_net import PointerGenerator


class MyCustomTransformerModel(nn.Module):
    def __init__(self, encoder, decoder, generator):
        super(MyCustomTransformerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)

    def forward(self,
                src,  # shape: (batch_size, src_seq_len, d_embed)  # already padded
                tgt  # shape: (batch_size, tgt_seq_len, d_embed)
                ):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_src_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask

    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask

    def make_pad_mask(self, query, key, pad_value=0, dim_check=False):
        """
        This pad masking is used when the query and key are already embedded
        """
        # query: (n_batch, query_seq_len, d_embed)
        # key: (n_batch, key_seq_len, d_embed)
        # pad_value: the value that is used to pad the sequence  TODO: Check what value you want to use (0, 1, or else)

        # Check if the query and key have the dimensionality we want
        if dim_check:
            assert len(query.shape) == 3, "query tensor must be 3-dimensional"
            assert len(key.shape) == 3, "key tensor must be 3-dimensional"
            assert query.shape[0] == key.shape[0], "query and key batch sizes must be the same"
            assert query.shape[2] == key.shape[2], "query and key embedding sizes must be the same"

        # Get the query and key sequence lengths
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        # create mask where the whole vector equals to pad_value
        key_mask = (key != pad_value).any(dim=-1).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)  # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = (query != pad_value).any(dim=-1).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask

    # def make_subsequent_mask(self, query, key):
    #     query_seq_len, key_seq_len = query.size(1), key.size(1)
    #
    #     tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')  # lower triangle without diagonal
    #     mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
    #     return mask


class MyCustomRLlibModel(TorchModelV2, nn.Module):
    # TODO: (1) Get right dimensions for the model from the obs_space
    # TODO: (2) Get the right dimensions for the model from the action_space (generator part)
    # TODO: (3) move copy away
    # TODO: (4) Check the vals: d_k, d_model, d_ebmed, h, d_ff, n_layer
    # TODO: (5) Modify the decoder to compute the attention we want
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # d_k = obs_space.original_space[-----]  # get it from the obs and check the dimensionality (d_model)
        d_embed = 32  # embedding dimension
        n_layer = 6
        d_model = 32  # dimension of model (:=d_k * h); in most cases, d_model = d_embed
        h = 8  # number of heads
        d_ff = 128  # dimension of feed forward; usually 2-4 times d_model
        dr_rate = 0  # dropout rate; 0 in our case as we are working on RL
        norm_eps = 1e-5  # epsilon parameter of layer normalization

        import copy
        copy = copy.deepcopy  # TODO (3): take it out of the class.
        
        attention = MultiHeadAttentionLayer(
            d_model=d_model,
            h=h,
            qkv_fc=nn.Linear(d_embed, d_model),
            out_fc=nn.Linear(d_model, d_embed),
            dr_rate=dr_rate)
        position_ff = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed, d_ff),
            fc2=nn.Linear(d_ff, d_embed),
            dr_rate=dr_rate)
        norm = nn.LayerNorm(d_embed, eps=norm_eps)

        encoder_block = EncoderBlock(
            self_attention=copy(attention),
            position_ff=copy(position_ff),
            norm=copy(norm),
            dr_rate=dr_rate)
        decoder_block = DecoderBlock(
            self_attention=copy(attention),
            cross_attention=copy(attention),
            position_ff=copy(position_ff),
            norm=copy(norm),
            dr_rate=dr_rate)

        encoder = Encoder(
            encoder_block=encoder_block,
            n_layer=n_layer,
            norm=copy(norm))
        decoder = Decoder(
            decoder_block=decoder_block,
            n_layer=n_layer,
            norm=copy(norm))
        action_size = action_space.n
        generator = PointerGenerator()  # output is a probability distribution over the input sequence

        # Define the actor
        self.policy_net = MyCustomTransformerModel(
            # src_embed=src_embed,
            # tgt_embed=tgt_embed,
            encoder=encoder,
            decoder=decoder,
            generator=generator
        )
        # Define the critic layer
        self.values = None
        self.value_net = nn.Linear(d_model, 1)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = self.policy_net(x, x)
        self.values = x
        # do the rest of the forward pass here
        # NOT COMPLETED! Check if it is applicable for the pointernet style (particularly, the decoder!)

        return x, state

    def value_function(self):
        return torch.mean(self.value_net(self.values))
