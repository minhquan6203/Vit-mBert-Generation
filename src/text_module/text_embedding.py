import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5Tokenizer,T5EncoderModel
from typing import List, Dict, Optional
from mask.masking import generate_padding_mask

import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5Tokenizer,T5EncoderModel
from typing import List, Dict, Optional
from mask.masking import generate_padding_mask
from data_utils.vocab import create_vocab

def Text_tokenizer(config):
    tokenizer = T5Tokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
    new_tokens,_ = create_vocab(config)
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))
    return tokenizer

class Text_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super(Text_Embedding,self).__init__()
        self.tokenizer = Text_tokenizer(config)
        self.embedding = T5EncoderModel.from_pretrained(config["text_embedding"]["text_encoder"])
        self.embedding=self.embedding.resize_token_embeddings(len(self.tokenizer))
        # freeze all parameters of pretrained model
        if config['text_embedding']['freeze']:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config["text_embedding"]['d_features'], config["text_embedding"]['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["text_embedding"]['dropout'])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.padding = config["tokenizer"]["padding"]
        self.max_length = config["tokenizer"]["max_length"]
        self.truncation = config["tokenizer"]["truncation"]

    def forward(self, questions: List[str]):
        input_ids = self.tokenizer(
                          text=questions,
                          max_length=self.max_length,
                          truncation = self.truncation,
                          return_tensors='pt', padding=self.padding).input_ids.to(self.device)
        padding_mask = generate_padding_mask(input_ids, padding_idx=self.tokenizer.pad_token_id)

        features = self.embedding(input_ids=input_ids).last_hidden_state
        # features = self.proj(features)
        out = self.dropout(self.gelu(features))
        return out, padding_mask