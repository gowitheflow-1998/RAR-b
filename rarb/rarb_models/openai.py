from openai import OpenAI
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import numpy as np

class OpenAIEmbedding:
    def __init__(self, model_name = "text-embedding-3-large",max_tokens = 8191, **kwargs):
        self.client = OpenAI(api_key = "placeholder")
        self.model_name = model_name
        # an opensource ada tokenizer, checked, length not exactly matching, 
        # but typically a bit shorter than the number the API counts, so okay.
        self.tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/text-embedding-ada-002')
        self.max_tokens = max_tokens
        
    def truncate_input(self, input_text):
        tokens = self.tokenizer(input_text)['input_ids']
        truncated_sentences = [token[:self.max_tokens] for token in tokens]
        truncated_text = [self.tokenizer.decode(truncated_sentence) for truncated_sentence in truncated_sentences]
        return truncated_text

    def encode_queries(self, queries, **kwargs):
        batch_size = kwargs.get('batch_size')  
        embeddings = []
        for i in tqdm(range(0, len(queries), batch_size)):
            batch_queries = queries[i:i + batch_size]
            batch_queries = self.truncate_input(batch_queries)
            batch_embeddings = self.client.embeddings.create(
                input=batch_queries, model=self.model_name
            )
            batch_embeddings = np.array([x.embedding for x in batch_embeddings.data])
            embeddings.extend(batch_embeddings)
            # time.sleep(2.5)

        return np.array(embeddings)

    def encode_corpus(self, corpus, **kwargs):
        batch_size = kwargs.get('batch_size')
        sentences = [doc["title"].strip() + " " + doc["text"].strip() for doc in corpus]
        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_corpus = sentences[i:i + batch_size]
            batch_corpus = self.truncate_input(batch_corpus)
            batch_embeddings = self.client.embeddings.create(
                input=batch_corpus, model=self.model_name
            )
            batch_embeddings = np.array([x.embedding for x in batch_embeddings.data])
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)