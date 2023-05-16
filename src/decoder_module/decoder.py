from accelerate import debug_launcher
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Dict, Optional
from transformers import BertGenerationDecoder

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.gen = BertGenerationDecoder.from_pretrained(config['decoder']['text_decoder'])
        # for param in self.gen.parameters():
        #     param.requires_grad = False

    def forward(self,encoder_features: torch.Tensor, encoder_attention_mask: torch.Tensor,answer_ids: torch.Tensor):
        out = self.gen(inputs_embeds=encoder_features,attention_mask=encoder_attention_mask,labels=answer_ids)
        
        return out.logits, out.loss