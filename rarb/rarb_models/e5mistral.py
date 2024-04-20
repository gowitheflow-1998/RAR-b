from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np

class E5Mistral:
    def __init__(self, model_path: str = None, sep: str = " ", max_length = 8196, **kwargs):
        self.sep = sep
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.q_model = AutoModel.from_pretrained(model_path, **kwargs)
        self.doc_model = self.q_model
        self.max_length = max_length
        
    def encode_queries(self, queries, batch_size:int=4, **kwargs):

        encoded_embeds = []
        for start_idx in tqdm(range(0, len(queries), batch_size)):
            batch_texts = queries[start_idx: start_idx + batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=self.max_length - 1, return_attention_mask=False, padding=False, truncation=True)
            batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
            batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt').to("cuda:0")
            with torch.no_grad():
                outputs = self.q_model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1).float()
            encoded_embeds.append(embeddings.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def encode_corpus(self, corpus, batch_size:int=4, **kwargs):
        # our current implementation put instruction into title in the pre-processing step
        sentences = [doc["text"].strip() for doc in corpus]
        encoded_embeds = []
        for start_idx in tqdm(range(0, len(sentences), batch_size)):
            batch_texts = sentences[start_idx: start_idx + batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=self.max_length - 1, return_attention_mask=False, padding=False, truncation=True)
            batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
            batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt').to("cuda:0")
            with torch.no_grad():
                outputs = self.doc_model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1).float()
            encoded_embeds.append(embeddings.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)
    
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]