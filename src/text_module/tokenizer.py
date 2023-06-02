import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from typing import List, Dict, Optional

class Text_Tokenizer(nn.Module):
    def __init__(self, config: Dict):
        super(Text_Tokenizer,self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_Tokenizer"]["text_encoder"])
        self.padding = config["tokenizer"]["padding"]
        self.max_length = config["tokenizer"]["max_length"]
        self.truncation = config["tokenizer"]["truncation"]
        self.return_token_type_ids = config["tokenizer"]["return_token_type_ids"]
        self.return_attention_mask = config["tokenizer"]["return_attention_mask"]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, text: List[str]):
        outs = self.tokenizer(
            text = text,
            padding = self.padding,
            max_length = self.max_length,
            truncation = self.truncation,
            return_tensors = 'pt',
            return_token_type_ids = self.return_token_type_ids,
            return_attention_mask = self.return_attention_mask,
        ).to(self.device)
        return outs