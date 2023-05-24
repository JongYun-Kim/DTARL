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
