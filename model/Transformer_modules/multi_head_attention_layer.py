import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, q_fc, kv_fc, out_fc, dr_rate=0):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h

        # W^Q, W^K, W^V transform the input query, key, value to d_model dimension
        self.q_fc = copy.deepcopy(q_fc)   # (d_embed_query, d_model)
        self.k_fc = copy.deepcopy(kv_fc)  # (d_embed_key,   d_model)
        self.v_fc = copy.deepcopy(kv_fc)  # (d_embed_value, d_model)
        # TODO: Please remove the copy.deepcopy() if it is not necessary.
        #  (maybe not necessary for q_fc but necessary for k_fc and v_fc)
        #  In my understanding, uses of copy.deepcopy() are fine if it is used in __init__().
        #  But, if I don't need a copy of the (input) object, I shouldn't use copy.deepcopy() for better performance.

        # W^O transforms the
        self.out_fc = out_fc              # (d_model, d_embed_MHA_out)
        self.dropout = nn.Dropout(p=dr_rate)

    def calculate_attention(self, query, key, value, mask):
        # query:      (n_batch, h, seq_len_query, d_k)
        # key, value: (n_batch, h, seq_len_key,   d_k)
        # mask: (n_batch, 1, seq_len_query, seq_len_key)???  # TODO: Dimension (n_batch, seq_len_query, seq_len_key)???
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Q x K^T
        attention_score = attention_score / math.sqrt(d_k)   # (n_batch, h, seq_len_query, seq_len_key)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1)  # (n_batch, h, seq_len_query, seq_len_key)
        attention_prob = self.dropout(attention_prob)
        out = torch.matmul(attention_prob, value)  # TODO: check if this is correct, dimension-wise
        return out  # (n_batch, h, query_seq_len, d_k)

    def forward(self, *args, query, key, value, mask=None):
        # query:      (n_batch, seq_len_query, d_embed_query)
        # key, value: (n_batch, seq_len_key,   d_embed_key)
        # mask: (n_batch, seq_len_query, seq_len_key)
        # return value: (n_batch,
        n_batch = query.size(0)

        def transform(x, x_fc):
            out = x_fc(x)           # (n_batch, seq_len_x, d_embed_x) -> (n_batch, seq_len_x, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h)  # (n_batch, seq_len_x, h, d_k )
            out = out.transpose(1, 2)
            return out  # (n_batch, h, x_seq_len, d_k)

        query = transform(query, self.q_fc)  # (n_batch, h, seq_len_query, d_k)
        key = transform(key, self.k_fc)      # (n_batch, h, seq_len_key,   d_k)
        value = transform(value, self.v_fc)  # (n_batch, h, seq_len_key,   d_k)

        out = self.calculate_attention(query, key, value, mask)  # (n_batch, h, query_seq_len, d_k)
        out = out.transpose(1, 2)                                # (n_batch, query_seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model)   # (n_batch, query_seq_len, d_model)
        out = self.out_fc(out)

        return out  # (n_batch, query_seq_len, d_embed_MHA_out); d_embed_MHA_out == d_embed_query in most cases.

    # def get_attention_score(self, query, key, mask=None):
    #     # query:      (n_batch, seq_len_query, d_embed_query)
    #     # key, value: (n_batch, seq_len_key,   d_embed_key)
    #     # mask: (n_batch, seq_len_query, seq_len_key)
    #     # return value: (n_batch, seq_len_query, seq_len_key)
    #     n_batch = query.size(0)
    #
    #     def transform(x, x_fc):
    #         out = x_fc(x)           # (n_batch, seq_len_x, d_embed_x) -> (n_batch, seq_len_x, d_model)
    #         out = out.view(n_batch, -1, self.h, self.d_model//self.h)
    #         out = out.transpose(1, 2)
    #         return out  # (n_batch, h, x_seq_len, d_k)
    #
    #     query = transform(query, self.q_fc)  # (n_batch, h, seq_len_query, d_k)
    #     key = transform(key, self.k_fc)      # (n_batch, h, seq_len_key,   d_k)
    #
    #     attention_score = torch.matmul(query, key.transpose(-2, -1))  # Q x K^T
    #
    #     if mask is not None:
    #         attention_score = attention_score.masked_fill(mask == 0, -1e9)
    #     attention_prob = F.softmax(attention_score, dim=-1)  # (n_batch, h, seq_len_query, seq_len_key)
    #     attention_prob = self.dropout(attention_prob)
    #     return attention_prob  # (n_batch, h, seq_len_query, seq_len_key)
