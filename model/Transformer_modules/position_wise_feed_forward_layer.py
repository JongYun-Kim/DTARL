import torch.nn as nn


class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, fc1, fc2, dr_rate=0):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1  # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dr_rate)
        self.fc2 = fc2  # (d_ff, d_embed)

    def forward(self, x):
        # x: shape: (batch_size, src_seq_len, d_embed==~d_embed_MHA_out)
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out: shape: (batch_size, src_seq_len, d_embed==d_embed_FFN_out)
        return out
