from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embbeding
from vision_module.init_vision_embedding import  build_vision_embedding
from encoder_module.init_encoder import build_encoder
from data_utils.load_data import create_ans_space
from decoder_module.init_decoder import build_decoder

class VQA_Model(nn.Module):
    def __init__(self,config: Dict):
     
        super(VQA_Model, self).__init__()
        self.num_labels = len(create_ans_space(config))
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.num_attention_heads=config["attention"]['heads']
        self.d_text = config["text_embedding"]['d_features']
        self.d_vision = config["vision_embedding"]['d_features']

        self.text_embbeding = build_text_embbeding(config)
        self.vision_embbeding = build_vision_embedding(config)

        self.fusion = nn.Sequential(
            nn.Linear(self.intermediate_dims +self.intermediate_dims, self.intermediate_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config,self.num_labels)
     
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, questions: List[str], images: List[str], labels: Optional[torch.LongTensor] = None):
        embbed_text, text_mask = self.text_embbeding(questions)
        embbed_vision, vison_mask = self.vision_embbeding(images)
        encoded_text, encoded_image = self.encoder(embbed_text, text_mask, embbed_vision, vison_mask)
        
        fused_output = self.fusion(torch.cat([encoded_text, encoded_image], dim=1))
        logits = self.decoder(fused_output)

        if labels is not None:
            logits=logits.view(-1,self.num_labels)
            labels = labels.view(-1)
            loss = self.criterion(logits, labels)
            return logits,loss
        else:
            return logits

def createVQA_Model(config: Dict) -> VQA_Model:
    model = VQA_Model(config)
    return model