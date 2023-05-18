import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from typing import List, Dict, Optional
from transformers import BertGenerationDecoder
from attention_module.attentions import MultiHeadAtt
from utils.positional_feed_forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.mhatt = MultiHeadAtt(config)
        self.pwff = PositionWiseFeedForward(config)

    def forward(self, queries, keys, values, attention_mask=None, **kwargs):
        att = self.mhatt(queries=queries, keys=keys, values=values, attention_mask=attention_mask)
        ff = self.pwff(att)

        return ff


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.gen = BertGenerationDecoder.from_pretrained(config['decoder']['text_decoder'])
        for param in self.gen.parameters():
            param.requires_grad = False
        self.layer_norm = nn.LayerNorm(config["decoder"]['d_model'])
        self.max_len = config["decoder"]['max_len']
        self.N = config["decoder"]['layers']
        self.criterion = nn.CrossEntropyLoss()
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(self.N)])
        self.linear = nn.Linear(config["model"]["intermediate_dims"],config['decoder']['d_model'])
    def generate_encoder_mask(self, encoder_features: torch.Tensor) -> torch.Tensor:
        mask = torch.sum(encoder_features, dim=2) != 0
        mask = mask.to(torch.float32)
        return mask.unsqueeze(1)
    
    def forward(self, encoder_features: torch.Tensor,encoder_attention_mask: torch.Tensor, answer_ids: torch.Tensor=None):
        # Add padding to labels
        answer_ids = F.pad(answer_ids, (0, self.max_len - answer_ids.size(1)), value=0)                
        #Sử dụng max pooling để giảm kích thước
        encoder_features = F.adaptive_max_pool1d(encoder_features.permute(0, 2, 1), answer_ids.size(1)).permute(0, 2, 1)
        for layer in self.layers:
            encoder_features = layer(queries=encoder_features, keys=encoder_features, values=encoder_features)
        encoder_features = self.linear(encoder_features)
        encoder_attention_mask=self.generate_encoder_mask(encoder_features)
        outputs = self.layer_norm(encoder_features)
        outputs = self.gen(inputs_embeds=outputs, attention_mask=encoder_attention_mask)
        loss = self.criterion(outputs.logits.view(-1,outputs.logits.size(-1)),  answer_ids.view(-1))    
        return outputs.logits, loss