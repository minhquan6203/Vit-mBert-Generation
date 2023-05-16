from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import wordnet
from transformers import  AutoModel, AutoTokenizer
import torch


class WuPalmerScoreCalculator:
    def __init__(self,config: Dict):
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
    def wup_measure(self, a: str, b: str, similarity_threshold: float = 0.925):
        """
        Returns Wu-Palmer similarity score.
        More specifically, it computes:
            max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
            where interp is a 'interpretation field'
        """
        def get_semantic_field(a):
            weight = 1.0
            semantic_field = wordnet.synsets(a,pos=wordnet.NOUN)
            return (semantic_field,weight)


        def get_stem_word(a):
            """
            Sometimes answer has form word\d+:wordid.
            If so we return word and downweight
            """
            weight = 1.0
            return (a,weight)


        global_weight=1.0

        (a,global_weight_a)=get_stem_word(a)
        (b,global_weight_b)=get_stem_word(b)
        global_weight = min(global_weight_a,global_weight_b)

        if a==b:
            # they are the same
            return 1.0*global_weight

        if a==[] or b==[]:
            return 0


        interp_a,weight_a = get_semantic_field(a) 
        interp_b,weight_b = get_semantic_field(b)

        if interp_a == [] or interp_b == []:
            return 0

        # we take the most optimistic interpretation
        global_max=0.0
        for x in interp_a:
            for y in interp_b:
                local_score=x.wup_similarity(y)
                if local_score > global_max:
                    global_max=local_score

        # we need to use the semantic fields and therefore we downweight
        # unless the score is high which indicates both are synonyms
        if global_max < similarity_threshold:
            interp_weight = 0.1
        else:
            interp_weight = 1.0

        final_score=global_max*weight_a*weight_b*interp_weight*global_weight
        return final_score
    
    def batch_wup_measure(self, labels: List[str], preds: List[str]) -> float:
        wup_scores = [self.wup_measure(label, pred) for label, pred in zip(labels, preds)]
        return np.mean(wup_scores)

    def accuracy(self,labels: List[str], preds: List[str]) -> float:
        return accuracy_score(labels,preds)
    
    def f1(self,labels: List[str], preds: List[str]) -> float:
        return f1_score(labels,preds, average='macro')

    def compute_metrics(self, labels: List[str], logits: torch.Tensor) -> Dict[str, float]:
        prediction_probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction_indices = prediction_probabilities.argmax(dim=-1)
        preds = self.tokenizer.batch_decode(prediction_indices)

        return self.batch_wup_measure(labels, preds), self.accuracy(labels, preds), self.f1(labels, preds)
