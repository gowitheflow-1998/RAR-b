from beir.retrieval import models
from typing import Union, Tuple
from sentence_transformers import SentenceTransformer

class PooledSentenceBERT(models.SentenceBERT):
    def __init__(self, model_path: Union[str, Tuple] = None, sep: str = " ", 
                 pooling = "mean", **kwargs):
        super().__init__(**kwargs)
        self.sep = sep
        
        if isinstance(model_path, str):
            self.q_model = SentenceTransformer(model_path)
            # if a model's ST config already is cls, then the default pooling="mean" 
            # defined above wouldn't affect it
            # the following change only happens when the model needs a cls pooling (such as dragon+), 
            # but doesn't have a ST cls config, making its default pooling mode mean.
            if self.q_model[1].pooling_mode_mean_tokens == True:
                if pooling == "cls":
                    self.q_model[1].pooling_mode_mean_tokens = False
                    self.q_model[1].pooling_mode_cls_token = True
            self.doc_model = self.q_model
        
        elif isinstance(model_path, tuple):
            self.q_model = SentenceTransformer(model_path[0])
            if self.q_model[1].pooling_mode_mean_tokens == True:
                if pooling == "cls":
                    self.q_model[1].pooling_mode_mean_tokens = False
                    self.q_model[1].pooling_mode_cls_token = True
            self.doc_model = SentenceTransformer(model_path[1])
            if self.doc_model[1].pooling_mode_mean_tokens == True:
                if pooling == "cls":
                    self.doc_model[1].pooling_mode_mean_tokens = False
                    self.doc_model[1].pooling_mode_cls_token = True
        if "bge" not in model_path:
            self.q_model.max_seq_length = 512
            self.doc_model.max_seq_length = 512
        print("q model max sequence length:", self.q_model.max_seq_length, 
              "doc model max sequence length:",self.doc_model.max_seq_length)
        print(f"initialized a rarb evaluation model with {model_path}")