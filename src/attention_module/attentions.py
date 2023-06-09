import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List, Dict, Optional

# class MultiHeadAtt(nn.Module):
#     def __init__(self, config: Dict):
#         super(MultiHeadAtt, self).__init__()        
#         self.d_model = config['attention']['d_model']
#         self.heads = config['attention']['heads']
#         self.d_k = config['attention']['d_key']
#         self.d_v = config['attention']['d_value']
#         self.dropout = config['attention']['dropout']
#         self.matt = nn.MultiheadAttention(embed_dim=self.d_model,num_heads=self.heads,dropout=self.dropout,kdim=self.d_k,vdim=self.d_v)

#     def forward(self, queries, keys, values, attention_mask=None):
#         attention_mask=None
#         out, _ = self.matt(queries, keys, values, attention_mask)
#         return out
    


class self_attention(nn.Module):

    def __init__(self, config: Dict):
        super(self_attention, self).__init__()

        d_model = config['attention']['d_model']
        h = config['attention']['heads']
        d_k = config['attention']['d_key']
        d_v = config['attention']['d_value']

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            att += attention_mask
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out, att
    

class MultiHeadAtt(nn.Module):
    '''
        Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, config: Dict):
        super(MultiHeadAtt, self).__init__()
        
        d_model = config['attention']['d_model']

        self.use_aoa = config['attention']['use_aoa'] # whether to use Attention on Attention (AoA) mechanism or not
        
        if self.use_aoa:    # define additionally AoA layers
            self.informative_attention = nn.Linear(2*d_model, d_model)
            self.gated_attention = nn.Linear(2*d_model, d_model)

        self.attention = self_attention(config)

        self.dropout = nn.Dropout(p=config['attention']['dropout'])
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, queries, keys, values, attention_mask, **kwargs):
        # if self.can_be_stateful and self._is_stateful:
        #     self.running_keys = torch.cat([self.running_keys, keys], 1)
        #     keys = self.running_keys

        #     self.running_values = torch.cat([self.running_values, values], 1)
        #     values = self.running_values

        out, _ = self.attention(queries, keys, values, attention_mask, **kwargs)
        
        # normalization after residual connection
        out = self.dropout(out)
        out = self.layer_norm(queries + out)

        if self.use_aoa:
            aoa_input = torch.cat([queries, out], dim=-1)
            i = self.informative_attention(aoa_input)
            g = torch.sigmoid(self.gated_attention(aoa_input))
            out = i * g
            
        return out