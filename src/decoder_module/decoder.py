import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from typing import List, Dict, Optional
from transformers import BertGenerationDecoder,BertGenerationConfig

class Decoder(nn.Module):
    def __init__(self, config, len_vocab):
        super(Decoder, self).__init__()
        
        config_gen = BertGenerationConfig.from_pretrained(config['decoder']['text_decoder'])
        config_gen.hidden_size=config['decoder']['d_model']
        config_gen.num_hidden_layers=config['decoder']['layers']
        config_gen.num_attention_heads=config['decoder']['heads']
        config_gen.hidden_dropout_prob=config['decoder']['dropout']
        config_gen.is_decoder=True
        self.gen = BertGenerationDecoder.from_pretrained(config['decoder']['text_decoder'],config=config_gen)
        self.gen.resize_token_embeddings(len_vocab)
        self.linear=nn.Linear(config['model']['intermediate_dims'],config['decoder']['d_model'])
    def forward(self, encoder_features: torch.Tensor, encoder_attention_mask: torch.Tensor=None, answer_ids: torch.Tensor=None):
        encoder_features = self.linear(encoder_features)
        outputs = self.gen(inputs_embeds=encoder_features, attention_mask=encoder_attention_mask, labels=answer_ids)
        return outputs
