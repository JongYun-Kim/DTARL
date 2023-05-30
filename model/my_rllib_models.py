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
from model.Transformer_modules.pointer_net import PointerProbGenerator

# TODO (1): [o] Create model layers individually if necessary (e.g. attention_encoder, ...)
# TODO (2): [o] Check masks
#       TODO (2-1): [o] padding mask in attention (encoder, decoder)
#       TODO (2-2): [o] action mask in attention (decoder ONLY!)
# TODO (3): [o] Determine how to use state of the model
#      TODO (3-1): [o] Determine how do we use task-forward pass
#      TODO (3-2): [o] Determine the shape of it
#      TODO (3-3): [o] Use two learnable placeholders to assemble the context vector in the decoder
#                      allow it only when task-forward count==1
# TODO (4): [o] Get generator to work (consider clipping the vals before masking)
# TODO (5): [x; waiting] In the comments get the right dimensions of tensors that are actually used across
#               all the modules (everywhere. particularly d_embed; it is used interchangeably with d_model...!)
# TODO (6): [x; PENDING] Consider dropping out the dropout layers in the modules of the model as we are dealing with RL.
#               see others' implementations;
# TODO (7): [x] Get rid of the lint or linting errors

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

        action_context_dim = self.src_embed.d_embed  # embedding dimension size of an encoder input
        # self.action_context_vec1 = nn.Parameter(torch.randn(1, 1, action_context_dim))  # (batch_size, 1, d_embed)
        # self.action_context_vec2 = nn.Parameter(torch.randn(1, 1, action_context_dim))  # (batch_size, 1, d_embed)

        # Learnable parameters: shape (1, 1, data_size); it will be extended to (batch_size, 1, data_size) at maximum
        self.h_first_learnable = nn.Parameter(torch.randn(1, 1, action_context_dim))  # (1, 1, d_embed)
        self.h_last_learnable = nn.Parameter(torch.randn(1, 1, action_context_dim))  # (1, 1, d_embed)

    # def encode_outdated(self, src, src_mask):
    #     return self.encoder(self.src_embed(src), src_mask)
    #
    # def decode_outdated(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
    #     return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    def encode(self, src_dict, src_mask):
        src = src_dict["task_embeddings"]
        src = self.src_embed(src)
        return self.encoder(src, src_mask)
        # return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)
        # return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    def get_context_node(self, obs, prev_action, first_action):
        # obs: shape (batch_size, num_task, data_size)
        # prev_action: shape (batch_size,)
        # first_action: shape (batch_size,)

        # Calculate batch_size, num_task, data_size from obs
        batch_size, num_task, data_size = obs.shape

        # Compute the average observation: shape (batch_size, 1, data_size)
        obs_avg = torch.mean(obs, dim=1, keepdim=True)

        # Initializing h_first and h_last with zeros: shape (batch_size, 1, data_size)
        h_first = torch.zeros(batch_size, 1, data_size)
        h_last = torch.zeros(batch_size, 1, data_size)

        # Create masks for the condition where first_action and prev_action are -1
        # These masks will have shape (batch_size,)
        mask_first_action_minus = first_action == -1
        mask_prev_action_minus = prev_action == -1

        # If the condition is met, fill in h_first and h_last with the corresponding learnable parameters
        # Expand the learnable parameters to match the sum of the conditions in the batch
        h_first[mask_first_action_minus] = self.h_first_learnable.expand(mask_first_action_minus.sum(), 1, data_size)
        h_last[mask_prev_action_minus] = self.h_last_learnable.expand(mask_prev_action_minus.sum(), 1, data_size)

        # Create masks for the condition where first_action and prev_action are not -1
        # These masks will have shape (batch_size,)
        mask_first_action_not_minus = first_action != -1
        mask_prev_action_not_minus = prev_action != -1

        # If the condition is met, fill in h_first and h_last with the corresponding part from obs
        # Unsqueeze is used to match the dimensions: shape (batch_size, 1, data_size)
        h_first[mask_first_action_not_minus] = obs[
                                               mask_first_action_not_minus,
                                               first_action[mask_first_action_not_minus].long(),
                                               :
                                               ].unsqueeze(1)
        h_last[mask_prev_action_not_minus] = obs[
                                             mask_prev_action_not_minus,
                                             prev_action[mask_prev_action_not_minus].long(),
                                             :
                                             ].unsqueeze(1)

        # Concatenate obs_avg, h_first, and h_last along the data_size dimension
        # The resulting tensor, h_c, will have shape (batch_size, 1, 3*data_size)
        h_c = torch.cat([obs_avg, h_first, h_last], dim=-1)

        return h_c

    def forward(self,
                src_dict,  # shape: (batch_size, src_seq_len, d_embed)  # already padded and embedded
                # tgt,  # shape: (batch_size,)  # They are previous actions
                # src_pad_tokens,  # shape: (batch_size, src_seq_len)
                # tgt_pad_tokens,  # shape: (batch_size, tgt_seq_len)
                ):
        # Please keep in mind that here sequence is
        # What is src mask?
        # This is the mask that is used in the encoder to mask the padding tokens in the MultiHeadAttentionLayer
        # It does not mask

        # Prepare tokens for mask generations and context node
        # TODO: Check dims of the following tensors;
        #       See the dims of Box(shape=()), Box(shape=(1,)), Discrete(n)
        pad_tokens = src_dict["pad_tokens"]
        action_tokens = src_dict["completion_tokens"]
        context_token = torch.zeros_like(pad_tokens[:, 0])  # represents the context vector h_c (Never padded, val==0)
        context_token = context_token.unsqueeze(1)  # shape: (batch_size, 1)
        prev_actions = src_dict["prev_action"]  # shape: (batch_size,)
        first_actions = src_dict["first_action"].squeeze(1)  # shape: (batch_size,)

        # Encoder mask
        src_mask_pad = self.make_src_mask(pad_tokens)
        src_mask = src_mask_pad

        # Decoder masks
        # tgt_mask: shape: (batch_size, seq_len_tgt, seq_len_tgt); used for self-attention in the decoder
        # src_tgt_mask: shape: (batch_size, seq_len_tgt, seq_len_src); used for cross-attention in the decoder
        tgt_mask = None
        src_tgt_submask_pad = self.make_src_tgt_mask(pad_tokens, context_token)  # query is the context vector; ...
        src_tgt_submask_action = self.make_src_tgt_mask(action_tokens, context_token)
        src_tgt_mask = src_tgt_submask_pad & src_tgt_submask_action

        # Encoder
        # encoder_out: shape: (batch_size, src_seq_len, d_embed)
        encoder_out = self.encode(src_dict, src_mask.unsqueeze(1))  # a set of task embeddings; permutation invariant
        # get the context vector
        # h_c_N: shape: (batch_size, 1, d_embed_context);  d_embed_context == 3 * d_embed
        h_c_N = self.get_context_node(obs=encoder_out, prev_action=prev_actions, first_action=first_actions)
        # decoder_out: (batch_size, tgt_seq_len, d_embed_context)
        # tgt_seq_len == 1 in our case
        decoder_out = self.decode(h_c_N, encoder_out, tgt_mask, src_tgt_mask.unsqueeze(1))  # h_c^(N+1)
        # Generator: query==decoder_out; key==encoder_out; return==logits
        # out: (batch_size, 1, seq_len_src
        out = self.generator(query=decoder_out, key=encoder_out, mask=src_tgt_mask)
        assert out.shape[1] == 1  # TODO: Remove it later once the model is stable
        # Get the logits
        out = out.squeeze(1)
        # out: (batch_size, src_seq_len) == (batch_size, n_actions) == (batch_size, num_task_max)
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
        # src: key/value; tgt: query
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask

    def make_pad_mask(self, query, key, pad_idx=1, dim_check=False):
    # def make_pad_mask(self, query, key, pad_idx=0, dim_check=True):  # TODO: dim_check=False later!!
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        # If input_token==pad_idx, then the mask value is 0, else 1
        # In the MHA layer, (no attention) == (attention_score: -inf) == (mask value is 0) == (input_token==pad_idx)
        # WARNING: Choose pad_idx carefully, particularly about the data type (e.g. float, int, ...)

        # Check if the query and key have the same dimension
        if dim_check:
            assert len(query.shape) == 2, "query must have 2 dimensions: (n_batch, query_seq_len)"
            assert len(key.shape) == 2, "key must have 2 dimensions: (n_batch, key_seq_len)"
            assert query.size(0) == key.size(0), "query and key must have the same batch size"

        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1)  # (n_batch, 1, key_seq_len)
        # TODO: Maybe we don't need to repeat the masks; The & operation will broadcast the mask
        #       Check it and remove the repeat operation if it is not necessary
        key_mask = key_mask.repeat(1, query_seq_len, 1)  # (n_batch, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(2)  # (n_batch, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, key_seq_len)  # (n_batch, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask  # output shape: (n_batch, query_seq_len, key_seq_len)

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
        clip_in_generator = 10
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
            norm=None,)  # TODO: Decide whether to include norm in the decoder!! you can pass None if not needed
        action_size = action_space.n  # it gives n given that action_space is Discrete(n).
        generator = PointerProbGenerator(
            d_model=d_model,
            q_fc=nn.Linear(d_embed_context, d_model),
            k_fc=nn.Linear(d_embed_input, d_model),
            clip_value=clip_in_generator,
            dr_rate=dr_rate,
        )  # outputs a probability distribution over the input sequence

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
        num_actions = action_space.n
        self.value_net = nn.Linear(num_actions, 1)

    def forward(self, input_dict, state, seq_lens):
        """
        :param input_dict: ["obs", "obs_flat", "prev_action", "prev_reward"]
               input_dict["obs"] is a dictionary, each value of which has shape of (batch_size, {data_shape})
        :param state:
        :return: x, state
        """
        x = input_dict["obs"]

        # Check if the data type of the pad tokens is torch's int32; if not, output a warning and convert it to int32
        # temp_token = x["pad_tokens"][0][0]
        # if temp_token.dtype != torch.int32:
        #     print("Warning: The data type of the pad tokens is not torch's int32. Converting it to int32...")
        #     for _ in range(10): print("The data type of the pad tokens was: ", type(temp_token))
        #     x["pad_tokens"] = x["pad_tokens"].type(torch.int32)

        x, _ = self.policy_net(x)  # x: logits
        self.values = x
        # Do the rest of the forward pass here, if necessary

        return x, state

    def value_function(self):
        # return torch.mean(self.value_net(self.values))
        return self.value_net(self.values).squeeze(-1)

