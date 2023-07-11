import torch.nn as nn
import torch
from transformers import T5ForConditionalGeneration
import torch.nn.functional as F
from text_module.text_embedding import Text_tokenizer
class Decoder(nn.Module):
    def __init__(self, config, num_labels):
 
        super(Decoder, self).__init__()
        self.gen = T5ForConditionalGeneration.from_pretrained(config['text_embedding']['text_encoder'])
        self.gen.resize_token_embeddings(num_labels)

    def forward(self, 
                    inputs_embeds,
                    attention_mask,
                    decoder_inputs_embeds=None,
                    decoder_attention_mask=None,
                    labels=None):
        outputs = self.gen(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels)
        return outputs

