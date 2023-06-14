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
from model.Transformer_modules.pointer_net import PointerProbGenerator, PointerPlaceholder

# For NN Model
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.torch_utils import FLOAT_MIN


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
# TODO (5): [o?] In the comments get the right dimensions of tensors that are actually used across
#               all the modules (everywhere. particularly d_embed; it is used interchangeably with d_model...!)
# TODO (6): [x; PENDING] Consider dropping out the dropout layers in the modules of the model as we are dealing with RL.
#               see others' implementations;
# TODO (7): [4] Get rid of the lint or linting errors
# TODO (8): [x] See the real data
#      TODO (8-1): [o] Masks with tokens
#      TODO (8-2): [o] logits with masks
#      TODO (8-3): [?] actions; any invalid actions?
#      TODO (8-4): [x]
# TODO (9): [o] Value Branch !!!!!!!!!!!!!!!!! or should we
# TODO (10): [o] FATAL: MUST GET RID OF SOFTMAX IN THE GENERATOR as it allows the invalid actions to be chosen

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.src_embed = src_embed
        # self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

        action_context_dim = self.src_embed.d_embed  # embedding dimension size of an encoder input

        # Learnable parameters: (1, 1, data_size); it will be extended to (batch_size, 1, data_size) at maximum
        self.h_first_learnable = nn.Parameter(torch.randn(1, 1, action_context_dim))  # (1, 1, d_embed)
        self.h_last_learnable = nn.Parameter(torch.randn(1, 1, action_context_dim))  # (1, 1, d_embed)

    def encode(self, src, src_mask):
        # src: (batch_size, num_task==seq_len, d_embed_input)
        # src_mask: (batch_size, 1, seq_len_src, seq_len_src)  # Mask MUST have the heading dim as 1 at dim=1
        # return: (batch_size, seq_len_src, d_embed_input)
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        # tgt: (batch_size, seq_len_tgt==1, d_embed_context)
        # encoder_out: (batch_size, seq_len_src, d_embed_input)
        # tgt_mask: (batch_size, 1, seq_len_tgt, seq_len_tgt)      # Mask MUST have the heading dim as 1 at dim=1
        # src_tgt_mask: (batch_size, 1, seq_len_tgt, seq_len_src)  # Mask MUST have the heading dim as 1 at dim=1
        # return: (batch_size, seq_len_tgt, d_embed_context)
        # If you need tgt_embed, use self.tgt_embed(tgt) instead of tgt in self.decoder()
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)

    def get_context_node(self, obs, prev_action, first_action, pad_tokens, use_obs_mask=True, debug=False):
        # obs: shape (batch_size, num_task_max==seq_len_src, data_size==d_embed_input)
        # prev_action: shape (batch_size,)
        # first_action: shape (batch_size,)
        # pad_tokens: shape (batch_size, num_task_max==seq_len_src)

        # Obtain batch_size, num_task, data_size from obs
        batch_size, num_task_max, data_size = obs.shape

        if use_obs_mask:
            # Expand the dimensions of pad_tokens to match the shape of obs
            mask = pad_tokens.unsqueeze(-1).expand_as(obs)  # (batch_size, num_task_max, data_size)

            # Replace masked values with zero for the average computation
            # obs_masked: (batch_size, num_task_max, data_size)
            obs_masked = torch.where(mask == 1, obs, torch.zeros_like(obs))

            # Compute the sum and count non-zero elements
            obs_sum = torch.sum(obs_masked, dim=1, keepdim=True)  # (batch_size, 1, data_size)
            obs_count = torch.sum((mask == 0), dim=1, keepdim=True).float()  # (batch_size, 1, data_size)

            # Check if there is any sample where all tasks are padded
            if debug:
                if torch.any(obs_count == 0):
                    raise ValueError("All tasks are padded in at least one sample.")

            # Compute the average observation, only for non-masked elements
            obs_avg = obs_sum / obs_count
        else:
            # Compute the average observation: shape (batch_size, 1, data_size)
            obs_avg = torch.mean(obs, dim=1, keepdim=True)  # num_task_max dim is reduced

        # Initializing h_first and h_last with zeros: shape (batch_size, 1, data_size)
        # h_first = torch.zeros(batch_size, 1, data_size).to(self.device)
        # h_last = torch.zeros(batch_size, 1, data_size).to(self.device)
        h_first = torch.zeros_like(obs_avg)  # shape (batch_size, 1, data_size); same device as obs_avg
        h_last = torch.zeros_like(obs_avg)

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
        # The resulting tensor, h_c, will have shape (batch_size, 1, 3*data_size==d_embed_context
        h_c = torch.cat([obs_avg, h_first, h_last], dim=-1)

        return h_c

    def forward(self,
                src_dict,  # shape["task_embeddings"]: (batch_size, src_seq_len, d_embed); already padded & embedded
                # tgt,  # shape: (batch_size,)  # They are previous actions <-- This is not used
                # src_pad_tokens,  # shape: (batch_size, src_seq_len)
                # tgt_pad_tokens,  # shape: (batch_size, tgt_seq_len)
                ):
        # self.device = src_dict["task_embeddings"].device

        # Prepare tokens for mask generations and context node
        # TODO: Check dims of the following tensors;
        #       See the dims of Box(shape=()), Box(shape=(1,)), Discrete(n);
        pad_tokens = src_dict["pad_tokens"]  # shape: (batch_size, seq_len_src==num_task_max)
        action_tokens = src_dict["completion_tokens"]  # shape: (batch_size, num_task_max)
        # context_token: (batch_size, 1); on the same device as pad_tokens
        context_token = torch.zeros_like(pad_tokens[:, 0:1])  # represents the context vector h_c (Never padded, val==0)
        prev_actions = src_dict["prev_action"]  # shape: (batch_size,)
        first_actions = src_dict["first_action"].squeeze(1)  # shape: (batch_size,)

        # Encoder mask
        src_mask_pad = self.make_src_mask(pad_tokens)
        src_mask = src_mask_pad  # shape: (batch_size, seq_len_src, seq_len_src); Be careful: head dimension removed!!

        # Decoder masks
        # tgt_mask: shape: (batch_size, seq_len_tgt, seq_len_tgt); used for self-attention in the decoder
        tgt_mask = None  # tgt_mask is not used in the current implementation
        src_tgt_submask_pad = self.make_src_tgt_mask(pad_tokens, context_token)  # query is the context vector; ...
        src_tgt_submask_action = self.make_src_tgt_mask(action_tokens, context_token)
        # src_tgt_mask: shape: (batch_size, seq_len_tgt, seq_len_src); used for cross-attention in the decoder
        src_tgt_mask = src_tgt_submask_pad & src_tgt_submask_action

        # ENCODER
        # encoder_out: shape: (batch_size, src_seq_len, d_embed)
        # A set of task embeddings encoded; permutation invariant
        # unsqueeze(1) has been applied to src_mask to add head dimension for broadcasting in multi-head attention
        encoder_out = self.encode(src_dict["task_embeddings"], src_mask.unsqueeze(1))

        # DECODER
        # get the context vector
        # h_c_N: shape: (batch_size, 1, d_embed_context);  d_embed_context == 3 * d_embed
        h_c_N = self.get_context_node(obs=encoder_out,
                                      prev_action=prev_actions, first_action=first_actions, pad_tokens=pad_tokens)
        # decoder_out: (batch_size, tgt_seq_len, d_embed_context)
        # tgt_seq_len == 1 in our case
        decoder_out = self.decode(h_c_N, encoder_out, tgt_mask, src_tgt_mask.unsqueeze(1))  # h_c^(N+1)
        # Generator: query==decoder_out; key==encoder_out; return==logits
        # out: (batch_size, 1, seq_len_src==num_task_max)
        out = self.generator(query=decoder_out, key=encoder_out, mask=src_tgt_mask)  # Watch out for the mask dim!!
        # assert out.shape[1] == 1  # TODO: Remove it later once the model runs stable
        # Get the logits
        out = out.squeeze(1)

        # out: (batch_size, src_seq_len) == (batch_size, n_actions) == (batch_size, num_task_max)
        # return out, decoder_out
        return out, h_c_N  # TODO: This is a test solution; remove it very soon

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask  # (batch_size, seq_len_src, seq_len_src)

    # def make_tgt_mask_outdated(self, tgt):
    #     pad_mask = self.make_pad_mask(tgt, tgt)
    #     seq_mask = self.make_subsequent_mask(tgt, tgt)
    #     mask = pad_mask & seq_mask
    #     return pad_mask & seq_mask

    def make_src_tgt_mask(self, src, tgt):
        # src: key/value; tgt: query
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask  # (batch_size, seq_len_tgt, seq_len_src)

    def make_pad_mask(self, query, key, pad_idx=1, dim_check=False):
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

        key_mask = key.ne(pad_idx).unsqueeze(1)  # (n_batch, 1, key_seq_len); on the same device as key
        key_mask = key_mask.repeat(1, query_seq_len, 1)  # (n_batch, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(2)  # (n_batch, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, key_seq_len)  # (n_batch, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask  # output shape: (n_batch, query_seq_len, key_seq_len)  # Keep in mind: 'NO HEADING DIM' here!!

    # def make_subsequent_mask(self, query, key):
    #     query_seq_len, key_seq_len = query.size(1), key.size(1)
    #
    #     tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')  # lower triangle without diagonal
    #     mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
    #     return mask


class MyCustomRLlibModel(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):

        # super(TorchModelV2, self).__init__()
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        # TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = device

        d_subobs = 2  # dimension of the input tokens!
        d_embed_input = 32
        d_embed_context = 3 * d_embed_input  # 384 (==128*3)
        d_model = 32  # dimension of model (:=d_k * h);
        d_model_decoder = 96  # maybe, should be the same as d_embed_context
        # TODO: Would you define d_model_decoder as well? Consider [d_embed_query]==(3*d_embed_input)==(d_embed_context)
        n_layer_encoder = 2
        n_layer_decoder = 1
        h = 8  # number of heads
        d_ff = 64  # dimension of feed forward; usually 2-4 times d_model
        # d_ff_decoder = 512
        clip_in_generator = 10
        dr_rate = 0  # dropout rate; 0 in our case as we use reinforcement learning... Or may not?
        norm_eps = 1e-5  # epsilon parameter of layer normalization  # TODO: Check if this value works fine

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
            d_model=d_model_decoder,
            h=h,
            q_fc=nn.Linear(d_embed_context, d_model_decoder),
            kv_fc=nn.Linear(d_embed_input, d_model_decoder),
            out_fc=nn.Linear(d_model_decoder, d_embed_context),
            dr_rate=dr_rate)
        # position_ff_decoder = PositionWiseFeedForwardLayer(
        #     fc1=nn.Linear(d_embed_context, d_ff_decoder),
        #     fc2=nn.Linear(d_ff_decoder, d_embed_context),
        #     dr_rate=dr_rate)
        # norm_decoder = nn.LayerNorm(d_embed_context, eps=norm_eps)

        # Block Level
        encoder_block = EncoderBlock(
            self_attention=copy.deepcopy(attention_encoder),
            # TODO: Can remove deepcopy; the block is deep-copied in Encoder
            position_ff=copy.deepcopy(position_ff_encoder),
            norm=copy.deepcopy(norm_encoder),
            dr_rate=dr_rate)
        decoder_block = DecoderBlock(
            self_attention=None,  # No self-attention in the decoder in this case!
            cross_attention=copy.deepcopy(attention_decoder),
            position_ff=None,  # No position-wise FFN in the decoder in this case!
            # norm=copy.deepcopy(norm_decoder),
            norm=nn.Identity(),
            dr_rate=dr_rate,
            efficient=True)

        # Transformer Level (Encoder + Decoder)
        encoder = Encoder(
            encoder_block=encoder_block,
            n_layer=n_layer_encoder,
            norm=copy.deepcopy(norm_encoder))
        decoder = Decoder(
            decoder_block=decoder_block,
            n_layer=n_layer_decoder,
            # norm=copy.deepcopy(norm_decoder),)
            norm=nn.Identity(),)
        action_size = action_space.n  # it gives n given that action_space is Discrete(n).
        generator = PointerProbGenerator(
            d_model=d_model,
            q_fc=nn.Linear(d_embed_context, d_model),
            k_fc=nn.Linear(d_embed_input, d_model),
            clip_value=clip_in_generator,
            dr_rate=dr_rate,
        )  # outputs a probability distribution over the input sequence

        # Define the actor
        self.policy_net = MyCustomTransformerModel(
            src_embed=input_embed,
            # tgt_embed=tgt_embed,
            encoder=encoder,
            decoder=decoder,
            generator=generator,)
        # Define the critic layers
        self.values = None
        self.share_layer = True
        if not self.share_layer:
            self.value_net = MyCustomTransformerModel(
                src_embed= copy.deepcopy(input_embed),
                # tgt_embed=tgt_embed,
                encoder=copy.deepcopy(encoder),
                decoder=copy.deepcopy(decoder),
                generator=PointerPlaceholder(),)

        self.value_branch = nn.Sequential(
            nn.Linear(d_embed_context, d_embed_context),
            nn.ReLU(),
            nn.Linear(d_embed_context, 1),
        )

    def forward(self, input_dict, state, seq_lens):
        """
        :param input_dict: ["obs", "obs_flat", "prev_action", "prev_reward"]
               input_dict["obs"] is a dictionary, each value of which has shape of (batch_size, {data_shape})
        :param state:
        :return: x, state
        """
        obs = input_dict["obs"]

        # Check if the data type of the pad tokens is torch's int32; if not, output a warning and convert it to int32
        # temp_token = x["pad_tokens"][0][0]
        # if temp_token.dtype != torch.int32:
        #     print("Warning: The data type of the pad tokens is not torch's int32. Converting it to int32...")
        #     for _ in range(10): print("The data type of the pad tokens was: ", type(temp_token))
        #     x["pad_tokens"] = x["pad_tokens"].type(torch.int32)

        # x: (batch_size, num_task_max)
        # h_c_N1: (batch_size, 1, d_embed_context)
        if self.share_layer:
            x, h_c_N1 = self.policy_net(obs)  # x: logits; RLlib expects raw logits, NOT softmax probabilities
            self.values = h_c_N1.squeeze(1)  # self.values: (batch_size, d_embed_context)
        else:
            x, _ = self.policy_net(obs)
            self.values = self.value_net(obs)[1].squeeze(1)  # self.values: (batch_size, d_embed_context)

        # Check batch dimension size
        # if x.shape[0] != 1:
        #     print(f"batch size = {x.shape[0]} != 1")
        #     print("Stop!")
        # else:
        #     print(f"batch size = {x.shape[0]} == 1")

        return x, state

    def value_function(self):
        out = self.value_branch(self.values).squeeze(-1)  # out: (batch_size,)
        return out


class MyMLPModel(TorchModelV2, nn.Module):
    """
    MLP
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Get configuration for this custom model
        '''
        # Config example:
        model_config = {
            "custom_model_config": {
                "fc_sizes": [128, 64],
                "fc_activation": "relu",
                "value_fc_sizes": [128, 64],
                "value_fc_activation": "relu",
                "is_same_shape": False,
                "share_layers": False,
            }
        }
        '''
        if "is_same_shape" in model_config["custom_model_config"]:
            # TODO: this may cause some confusion...
            self.is_same_shape = model_config["custom_model_config"]["is_same_shape"]
        else:
            self.is_same_shape = False
            print("is_same_shape not received!!")
            print("is_same_shape == False")
        if "fc_sizes" in model_config["custom_model_config"]:
            self.fc_sizes = model_config["custom_model_config"]["fc_sizes"]
        else:
            self.fc_sizes = [256, 256]
            print(f"fc_sizes param in custom_model_config has NOT been received!")
            print(f"It goes with: fc_sizes = {self.fc_sizes}")
        if "fc_activation" in model_config["custom_model_config"]:
            self.fc_activation = model_config["custom_model_config"]["fc_activation"]
        else:
            self.fc_activation = "relu"
        if "value_fc_sizes" in model_config["custom_model_config"]:
            if self.is_same_shape:
                self.value_fc_sizes = self.fc_sizes.copy()
            else:
                self.value_fc_sizes = model_config["custom_model_config"]["value_fc_sizes"]
        else:
            self.value_fc_sizes = [256, 256]
            print(f"value_fc_sizes param in custom_model_config has NOT been received!")
            print(f"It goes with: value_fc_sizes = {self.value_fc_sizes}")
        if "value_fc_activation" in model_config["custom_model_config"]:
            self.value_fc_activation = model_config["custom_model_config"]["value_fc_activation"]
        else:
            self.value_fc_activation = "relu"
        # Define shared_layers flag
        if "share_layers" in model_config["custom_model_config"]:
            self.share_layers = model_config["custom_model_config"]["share_layers"]
        else:
            self.share_layers = False
            print("share_layers not received!!")
            print("share_layers == False")

        # Define observation size
        self.obs_size = get_preprocessor(obs_space)(obs_space).size  # (4 * num_task_max) + 2

        # Initialize logits
        self._logits = None
        # Holds the current "base" output (before logits/value_out layer).
        self._features = None
        self._values = None

        # Build the Module from fcs + 2xfc (action + value outs).
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))  # TODO: why np?????
        # Create layers and get fc_net
        for size in self.fc_sizes[:]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=self.fc_activation,
                )
            )
            prev_layer_size = size
        self.fc_net = nn.Sequential(*layers)

        if not self.share_layers:
            # Get VALUE fc layers
            value_layers = []
            prev_value_layer_size = int(np.product(obs_space.shape))
            # Create layers and get fc_net
            for size in self.value_fc_sizes[:]:
                value_layers.append(
                    SlimFC(
                        in_size=prev_value_layer_size,
                        out_size=size,
                        initializer=normc_initializer(1.0),
                        activation_fn=self.value_fc_activation,
                    )
                )
                prev_value_layer_size = size
            self.value_fc_net = nn.Sequential(*value_layers)
        else:
            self.value_fc_net = self.fc_net

        # Get last layers
        self.last_size = self.fc_sizes[-1]
        self.last_value_size = self.value_fc_sizes[-1]
        # Policy network's last layer
        self.action_branch = nn.Linear(self.last_size, num_outputs)
        # Value network's last layer
        self.value_branch = nn.Linear(self.last_value_size, 1)

    @override(ModelV2)
    def value_function(self):
        assert self._values is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._values), [-1])

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Fetch the observation
        obs = input_dict["obs_flat"]

        # Forward pass through fc_net
        self._features = self.fc_net(obs)

        # If not sharing layers, forward pass through value_fc_net
        # Else, _values are the same as _features
        if not self.share_layers:
            self._values = self.value_fc_net(obs)
        else:
            self._values = self._features

        # Calculate logits
        self._logits = self.action_branch(self._features)

        # Apply action masking
        pad_tasks = input_dict["obs"]["pad_tokens"]  # (batch_size, num_task_max); 0: not padded, 1: padded
        done_tasks = input_dict["obs"]["completion_tokens"]  # (batch_size, num_task_max); 0: not done, 1: done
        # Get action mask
        action_mask = torch.zeros_like(self._logits)
        action_mask[pad_tasks == 1] = FLOAT_MIN
        action_mask[done_tasks == 1] = FLOAT_MIN
        # Apply action mask
        self._logits = self._logits + action_mask

        # JUST FOR DEBUGGING
        # if obs.shape[0] != 1:
        #     print(f"batch size = {obs.shape[0]} != 1")
        #     print("Stop!")
        # else:
        #     print(f"batch size = {obs.shape[0]} == 1")

        # Return logits and state
        return self._logits, state



