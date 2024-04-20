import cohere
from tqdm import tqdm
import numpy as np

class CohereEmbedding:
    def __init__(self, model_name = "embed-english-v3.0", truncate = "end", **kwargs):
        self.co = cohere.Client("placeholder")
        self.model_name = model_name
        self.truncate = truncate
 
    def encode_queries(self, queries, **kwargs):
        batch_size = kwargs.get('batch_size')  
        embeddings = []
        for i in tqdm(range(0, len(queries), batch_size)):
            batch_queries = queries[i:i + batch_size]
            batch_embeddings = self.co.embed(
                texts=batch_queries,
                model=self.model_name,
                input_type='search_query',
                truncate=self.truncate
            ).embeddings            
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    def encode_corpus(self, corpus, **kwargs):
        batch_size = kwargs.get('batch_size')
        sentences = [doc["title"].strip() + " " + doc["text"].strip() for doc in corpus]
        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_corpus = sentences[i:i + batch_size]
            batch_embeddings = self.co.embed(
                texts=batch_corpus,
                model=self.model_name,
                input_type='search_document',
                truncate=self.truncate
            ).embeddings   
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    