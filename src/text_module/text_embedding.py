import torch
from torch import nn
from torch.nn import functional as F
from transformers import  AutoModel
from typing import List, Dict, Optional
from mask.masking import generate_padding_mask
from text_module.tokenizer import Text_Tokenizer

class Text_Embedding(nn.Module):
    def __init__(self, config: Dict):
        super(Text_Embedding,self).__init__()
        self.tokenizer = Text_Tokenizer(config)
        self.embedding = AutoModel.from_pretrained(config["text_embedding"]["text_encoder"])
        # freeze all parameters of pretrained model
        if config['text_embedding']['freeze']:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(config["text_embedding"]['d_features'], config["text_embedding"]['d_model'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config["text_embedding"]['dropout'])

    def forward(self, questions: List[str]):
        inputs = self.tokenizer(questions)
        features = self.embedding(**inputs).last_hidden_state

        padding_mask = generate_padding_mask(inputs['input_ids'], padding_idx=self.tokenizer.pad_token_id)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask




# from transformers.models.bert.modeling_bert import (
#     BertConfig,
#     BertEmbeddings,
#     BertEncoder,
#     BertPreTrainedModel,
# )


# class TextBert(BertPreTrainedModel):
#     def __init__(self, config):
#         super(TextBert,self).__init__(config)

#         self.embeddings = BertEmbeddings(config)
#         self.encoder = BertEncoder(config)
#         self.init_weights()

#     def forward(self, txt_inds, txt_mask):
#         encoder_inputs = self.embeddings(txt_inds)
        
#         attention_mask = txt_mask
#         head_mask = [None] * self.config.num_hidden_layers
#         encoder_outputs = self.encoder(
#             encoder_inputs, attention_mask, head_mask=head_mask
#         )
#         seq_output = encoder_outputs[0]

#         return seq_output

# class Text_Embedding(nn.Module):
#     def __init__(self, config):
#         super(Text_Embedding,self).__init__()

#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#         bert_config = BertConfig(
#             hidden_size=config["text_embedding"]['d_features'],
#             num_hidden_layers=config["text_embedding"]['layers'],
#             num_attention_heads=config["text_embedding"]['heads']
#         )

#         self.tokenizer = BertTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
#         self.embedding = TextBert(bert_config)
#         if config["text_embedding"]['load_pretrained']:
#             self.embedding = self.embedding.from_pretrained(config["text_embedding"]["text_encoder"])
#         if config["text_embedding"]['frezee']:
#             # freeze all parameters of pretrained model
#             for param in self.embedding.parameters():
#                 param.requires_grad = False

#         self.proj = nn.Linear(config["text_embedding"]['d_features'], config["text_embedding"]['d_model'])
#         self.gelu = nn.GELU()
#         self.dropout = nn.Dropout(config["text_embedding"]['dropout'])

#     def forward(self, questions: List[str]):
#         inputs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids.to(self.device)
#         padding_mask = generate_padding_mask(inputs, padding_idx=self.tokenizer.pad_token_id)
#         features = self.embedding(inputs, padding_mask)

#         out = self.proj(features)
#         out = self.dropout(self.gelu(out))

#         return out, padding_mask