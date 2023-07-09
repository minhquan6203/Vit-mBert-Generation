import torch.nn as nn
import torch
from transformers import T5ForConditionalGeneration


class Decoder(nn.Module):
    def __init__(self, config, num_labels):
 
        super(Decoder, self).__init__()

        model = T5ForConditionalGeneration.from_pretrained(config['text_embedding']['text_encoder'])
        dummy_encoder = list(nn.Sequential(*list(model.encoder.children())[1:]).children())   ## Removing the Embedding layer
        dummy_decoder = list(nn.Sequential(*list(model.decoder.children())[1:]).children())   ## Removing the Embedding Layer

        ## Using the T5 Encoder

        self.list_encoder = nn.Sequential(*list(dummy_encoder[0]))
        self.residue_encoder = nn.Sequential(*list(dummy_encoder[1:]))
        self.list_decoder = nn.Sequential(*list(dummy_decoder[0]))
        self.residue_decoder = nn.Sequential(*list(dummy_decoder[1:]))

        self.seq = config['decoder']['seq_len']

        self.classification_head = nn.Linear(config['decoder']['d_model'], num_labels)

    def forward(self, total_feat, predict_proba = False, predict_class = False):

        ## Extracting the feature

        for layer in self.list_encoder:
          total_feat = layer(total_feat)[0]
        total_feat = self.residue_encoder(total_feat)

        for layer in self.list_decoder:
          total_feat = layer(total_feat)[0]
        total_feat = self.residue_decoder(total_feat)

        if predict_proba:
          return total_feat.softmax(axis = -1)
        
        if predict_class:
          return total_feat.argmax(axis = -1)

        answer_vector = self.classification_head(total_feat)[:, :self.seq, :]

        return answer_vector


