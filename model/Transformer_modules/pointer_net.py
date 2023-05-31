import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerGenerator(nn.Module):
    def __init__(self):
        super(PointerGenerator, self).__init__()

    def forward(self, decoder_output, encoder_output):
        # decoder_output : (batch_size, seq_len, d_model)
        # encoder_output : (batch_size, seq_len, d_model)
        attention_scores = torch.bmm(decoder_output, encoder_output.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attention_probs = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        return attention_probs


class PointerProbGenerator(nn.Module):

    def __init__(self, d_model, q_fc, k_fc, clip_value=10, dr_rate=0):
        super(PointerProbGenerator, self).__init__()
        self.d_model = d_model  # The dimension of the internal layers
        self.clip_value = clip_value  # Value used for clipping the attention scores

        # Linear layers for transforming the input query and key to the internal dimension
        self.q_fc = copy.deepcopy(q_fc)  # (d_embed_query, d_model)
        self.k_fc = copy.deepcopy(k_fc)  # (d_embed_key,   d_model)

        self.dropout = nn.Dropout(p=dr_rate)  # Dropout layer

    def calculate_attention(self, query, key, mask):
        # query:  (n_batch, seq_len_query, d_model) - Batch of query vectors
        # key:    (n_batch, seq_len_key,   d_model) - Batch of key vectors
        # mask:   (n_batch, seq_len_query, seq_len_key) - Mask tensor
        batch_size = query.size(0)  # Get the batch size; TODO remove it

        d_k = key.shape[-1]  # Get the last dimension of the key
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Calculate the dot product: (Q x K^T)
        attention_score = self.clip_value * torch.tanh(attention_score)  # Apply clipping to the attention scores
        attention_score = attention_score / math.sqrt(d_k)  # Scale the attention scores
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)  # Apply the mask to the attention scores
        attention_prob = F.softmax(attention_score, dim=-1)  # Apply softmax to get the attention probabilities
        attention_prob = self.dropout(attention_prob)  # Apply dropout
        # if batch_size != 1:
        #     print("batch_size != 1")
        return attention_prob  # (n_batch, seq_len_query, seq_len_key) - The attention probabilities

    def forward(self, *args, query, key, mask=None):
        # query:      (n_batch, seq_len_query, d_embed_query) - Batch of query vectors
        # key:        (n_batch, seq_len_key,   d_embed_key) - Batch of key vectors
        # mask:       (n_batch, seq_len_query, seq_len_key) - Mask tensor

        n_batch = query.size(0)  # Get the batch size

        # Apply the linear transformations to the query and key
        query = self.q_fc(query)  # (n_batch, seq_len_query, d_model)
        key = self.k_fc(key)  # (n_batch, seq_len_key,   d_model)

        attention_score = self.calculate_attention(query, key, mask)  # (n_batch, seq_len_query, seq_len_key)

        return attention_score  # (n_batch, seq_len_query, seq_len_key) - The attention probabilities

