import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from typing import List, Dict, Optional
from transformers import BertGenerationDecoder


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.gen = BertGenerationDecoder.from_pretrained(config['decoder']['text_decoder'])
        self.layer_norm = nn.LayerNorm(config["decoder"]['d_model'])
        self.max_len = config["decoder"]['max_len']
    
    def generate_encoder_mask(self, encoder_features: torch.Tensor) -> torch.Tensor:
      mask = torch.sum(encoder_features, dim=2) != 0
      mask = mask.to(torch.float32)
      return mask.unsqueeze(1)
    
    def forward(self, encoder_features: torch.Tensor, answer_ids: torch.Tensor=None): 
        # Add padding to labels
        # max_length = self.max_len
        # answer_ids = F.pad(answer_ids, (0, max_length - answer_ids.size(1)), value=0)
                
        #Sử dụng max pooling để giảm kích thước
        encoder_features = F.adaptive_max_pool1d(encoder_features.permute(0, 2, 1), answer_ids.size(1)).permute(0, 2, 1)
        encoder_attention_mask=self.generate_encoder_mask(encoder_features)
        outputs = self.layer_norm(encoder_features)
        outputs = self.gen(inputs_embeds=outputs, attention_mask=encoder_attention_mask, labels=answer_ids)    
        return outputs.logits, outputs.loss