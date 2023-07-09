from decoder_module.decoder import Decoder

def build_decoder(config,num_labels):
    return Decoder(config,num_labels)