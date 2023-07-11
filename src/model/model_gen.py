from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.text_embedding import Text_Embedding
from vision_module.vision_embedding import  Vision_Embedding
from encoder_module.encoder import CoAttentionEncoder
from decoder_module.decoder import Decoder
from text_module.text_embedding import Text_tokenizer

#lấy ý tưởng từ MCAN
class MultimodalVQAModel(nn.Module):
    def __init__(self,config: Dict):
     
        super(MultimodalVQAModel, self).__init__()
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.text_embbeding = Text_Embedding(config)
        self.vision_embbeding = Vision_Embedding(config)

        self.tokenizer = Text_tokenizer(config)
        self.len_vocab=len(self.tokenizer)
        self.padding = config["decoder"]["padding"]
        self.max_length = config["decoder"]["max_length"]
        self.truncation = config["decoder"]["truncation"]
        self.decoder = Decoder(config,self.len_vocab)
        self.encoder = CoAttentionEncoder(config)
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.trans=nn.Linear(217, self.max_length)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, questions: List[str], images: List[str], answers: List[str]):
        embbed_text, text_mask= self.text_embbeding(questions)
        embbed_vision, vison_mask = self.vision_embbeding(images)
        embbed_text, embbed_vision = self.encoder(embbed_text, text_mask, embbed_vision, vison_mask)        
        
        fused_output = self.fusion(torch.cat([embbed_text, embbed_vision], dim=1))
        fused_output = self.trans(fused_output.permute(0,2,1))
        fused_output = fused_output.permute(0,2,1)
        fused_mask = self.fusion(torch.cat([text_mask.squeeze(1).squeeze(1),vison_mask.squeeze(1).squeeze(1)],dim=1))
        fused_mask=None
        
        if answers is not None:
            labels = self.tokenizer(answers,padding = self.padding, max_length = self.max_length,
                                    truncation = self.truncation,return_tensors = 'pt').to(self.device)                   
            out = self.decoder(fused_output,fused_mask,labels['input_ids'])
            return out.logits,out.loss
        else:
            out = self.decoder(fused_output,fused_mask,labels['input_ids'])
            return out.logits
        
def createMultimodalModelForVQA(config: Dict) -> MultimodalVQAModel:
    model = MultimodalVQAModel(config)
    return model