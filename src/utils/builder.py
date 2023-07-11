from model.model_gen import createMultimodalModelForVQA,MultimodalVQAModel
def build_model(config):
    if config['model']['type_model']=='gen':
        return createMultimodalModelForVQA(config)

def get_model(config):
    if config['model']['type_model']=='gen':
        return MultimodalVQAModel(config)

