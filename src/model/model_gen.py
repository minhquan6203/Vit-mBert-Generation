from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.text_embedding import Text_Embedding
from vision_module.vision_embedding import  Vision_Embedding
from attention_module.attentions import MultiHeadAtt
from encoder_module.encoder import CoAttentionEncoder
from decoder_module.decoder import Decoder
from transformers import AutoTokenizer

#lấy ý tưởng từ MCAN
class MultimodalVQAModel(nn.Module):
    def __init__(self,config: Dict):
     
        super(MultimodalVQAModel, self).__init__()
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.num_attention_heads=config["attention"]['heads']
        self.d_text = config["text_embedding"]['d_features']
        self.d_vision = config["vision_embedding"]['d_features']
        self.text_embbeding = Text_Embedding(config)
        self.vision_embbeding = Vision_Embedding(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        self.decoder = Decoder(config)
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.encoder = CoAttentionEncoder(config)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.CrossEntropyLoss()
        self.linear = nn.Linear(self.intermediate_dims,config['decoder']['d_model'])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, questions: List[str], images: List[str]):
        embbed_text, text_mask= self.text_embbeding(questions)
        embbed_vision, vison_mask = self.vision_embbeding(images)
        encoded_text, encoded_image = self.encoder(embbed_text, text_mask, embbed_vision, vison_mask)        
        fused_output = self.fusion(torch.cat([encoded_text, encoded_image], dim=1))
        fused_output = self.linear(fused_output)
        fused_mask = self.fusion(torch.cat([text_mask.squeeze(1).squeeze(1),vison_mask.squeeze(1).squeeze(1)],dim=1))
        return fused_output,fused_mask

def createMultimodalModelForVQA(config: Dict) -> MultimodalVQAModel:
    model = MultimodalVQAModel(config)
    return model