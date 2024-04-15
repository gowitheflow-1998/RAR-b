from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import torch
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union, Tuple, cast
import json
from openai import OpenAI
from sentence_transformers.cross_encoder import CrossEncoder
import cohere
import time
from transformers import GPT2TokenizerFast, AutoTokenizer, AutoModel, AutoModelForCausalLM
from torch import Tensor
import torch.nn.functional as F
from functools import partial


def load_reasoning_raw(dataset_name:str, 
              split:str = 'dev'):
    import json
    
    if split == 'dev' or split == 'valid':
        for split in ['dev','valid']:
            if os.path.exists(f'./{dataset_name}/{split}.jsonl'):
                with open(f'./{dataset_name}/{split}.jsonl', 'r', encoding='utf-8') as file:
                    data = [json.loads(n) for n in file]
    elif 'train' in split:
        with open(f'./{dataset_name}/{split}.jsonl', 'r', encoding='utf-8') as file:
            data = [json.loads(n) for n in file]

    if split == 'dev' or split == 'valid':
        for split in ['dev','valid']:
            if os.path.exists(f'./{dataset_name}/{split}-labels.lst'):
                with open(f'./{dataset_name}/{split}-labels.lst', 'r', encoding='utf-8') as file:
                    labels = [label.split()[0] for label in file]
    elif 'train' in split:
        with open(f'./{dataset_name}/{split}-labels.lst', 'r', encoding='utf-8') as file:
            labels = [label.split()[0] for label in file]
    return data, labels

def capitalize_first(string):
  if len(string) == 0:
    string = string
  elif len(string) == 1:
    string = string.upper()
  else:
    string = string[0].upper() + string[1:]
  return string

def make_mcr_frame(instructions, queries, documents, labels):
    import pandas as pd
    mcr = pd.DataFrame()
    mcr['instructions'] = instructions
    mcr['queries'] = queries
    mcr['documents'] = documents
    mcr['labels'] = labels
    return mcr
  
def load_mcr_frame(dataset_name: str,
                   split: str = "dev",
                   level: str = "l2",
                   setting: str = "pure"):
    import pandas as pd
    if dataset_name != "TempReason":
        frame = pd.read_csv(f'./mcr/{dataset_name}-{split}.csv',converters={'documents': eval})
    elif dataset_name == "TempReason":
        frame = pd.read_csv(f'./mcr/TempReason/{split}-{level}-{setting}.csv',converters={'documents': eval})
    instructions = frame['instructions'].values
    queries = frame['queries'].values
    documents = frame['documents'].values
    labels = frame['labels'].values
    return instructions, queries, documents, labels

class AutoRetriever:

    def __init__(self, model_path: Union[str, Tuple] = None,
                 pooling = "mean", device='cuda:0'):
        
        if isinstance(model_path, str):
            self.q_model = SentenceTransformer(model_path,device=device)
            # if the model already has a sbert config it's cls pooling, 
            # the default init arg pooling = "mean" wont affect
            if self.q_model[1].pooling_mode_mean_tokens == True:
                if pooling == "cls":
                    self.q_model[1].pooling_mode_mean_tokens = False
                    self.q_model[1].pooling_mode_cls_token = True
            self.doc_model = self.q_model
        
        elif isinstance(model_path, tuple):
            self.q_model = SentenceTransformer(model_path[0],device=device)
            if self.q_model[1].pooling_mode_mean_tokens == True:
                if pooling == "cls":
                    self.q_model[1].pooling_mode_mean_tokens = False
                    self.q_model[1].pooling_mode_cls_token = True
            self.doc_model = SentenceTransformer(model_path[1],device=device)
            if self.doc_model[1].pooling_mode_mean_tokens == True:
                if pooling == "cls":
                    self.doc_model[1].pooling_mode_mean_tokens = False
                    self.doc_model[1].pooling_mode_cls_token = True

    def evaluate_mcr(self, instructions, queries, documents, labels, sim = 'cos_sim'):

        results = []

        num_example = len(queries)
        num_correct = 0

        for i, q, d, l in tqdm(zip(instructions, queries, documents, labels)):
            
            iq = i + ' ' + q
            e_iq = self.q_model.encode(iq)
            e_d = self.doc_model.encode(d)
            if sim == 'cos_sim':
                result = cosine_similarity([e_iq], e_d)
            elif sim == 'dot':
                result = np.dot([e_iq], e_d.T)
                
            # if results is None:
            #     results = result
            # else:
                # results = np.concatenate((results, result))
            results.append(result)
            if np.argmax(result) == l:
                num_correct+=1
        accuracy = num_correct/num_example
        return accuracy, results
    def evaluate_mcr_without_instructions(self, queries, documents, labels, sim = 'cos_sim'):

        results = []

        num_example = len(queries)
        num_correct = 0

        for q, d, l in tqdm(zip(queries, documents, labels)):
            
            e_q = self.q_model.encode(q)
            e_d = self.doc_model.encode(d)
            
            if sim == 'cos_sim':
                result = cosine_similarity([e_q], e_d)
            elif sim == 'dot':
                result = np.dot([e_q], e_d.T)
                
            # if results is None:
            #     results = result
            # else:
            #     results = np.concatenate((results, result))
            results.append(result)
            
            if np.argmax(result) == l:
                num_correct+=1
        accuracy = num_correct/num_example
        return accuracy, results

class AutoCrossEncoder:
    def __init__(self, model_name:str = None, device = None):
        if device is not None:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)
    
    def evaluate_mcr(self, queries, documents, labels, instructions, batch_size = 16):
        results = []
        num_example = len(queries)
        num_correct = 0
        if instructions == None:
            for q, d, l in tqdm(zip(queries, documents, labels)):
                pairs = [[q, option] for option in d]
                similarity_scores = self.model.predict(pairs, batch_size=batch_size)
                prediction = np.argmax(similarity_scores)
                results.append(similarity_scores)
                if np.argmax(prediction) == l:
                    num_correct+=1
        else:
            for i, q, d, l in tqdm(zip(instructions, queries, documents, labels)):
                iq = i + ' ' + q 
                pairs = [[iq, option] for option in d]
                similarity_scores = self.model.predict(pairs, batch_size=batch_size)
                prediction = np.argmax(similarity_scores)
                results.append(similarity_scores)
                if np.argmax(prediction) == l:
                    num_correct+=1           
        accuracy = num_correct/num_example
        return results, accuracy
    
        
def evaluate_mcr_INSTRUCTOR(instructions, queries, documents, labels):
    results = None
    num_correct = 0
    num_example = len(queries)
    for i, q, d, l in tqdm(zip(instructions, queries, documents, labels)):
        e_iq = model.encode([[i,q]])
        e_d = model.encode(d)
        result = cosine_similarity(e_iq, e_d)
        
        # if results is None:
        #     results = result
        # else:
        #     results = np.concatenate((results, result))
        if np.argmax(result) == l:
            num_correct+=1
    accuracy = num_correct/num_example
    return accuracy, results

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
            # time.sleep(0.5)
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
            # time.sleep(0.5)
        return np.array(embeddings)
    
class PooledSentenceBERT(models.SentenceBERT):
    def __init__(self, model_path: Union[str, Tuple] = None, sep: str = " ", 
                 pooling = "mean", **kwargs):
        super().__init__(**kwargs)
        self.sep = sep
        
        if isinstance(model_path, str):
            self.q_model = SentenceTransformer(model_path)
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
        print(self.q_model.max_seq_length, self.doc_model.max_seq_length)
        
def capitalize_first(string):
  if len(string) == 0:
    string = string
  elif len(string) == 1:
    string = string.upper()
  else:
    string = string[0].upper() + string[1:]
  return string

def save(queries, documents, qrels, instruction, dataset,
         split = 'dev'):
    path = f'full/{dataset}'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'full/{dataset}/queries.jsonl', 'w') as f:
        for item in queries:
            f.write(json.dumps(item) + '\n')
    with open(f'full/{dataset}/corpus.jsonl', 'w') as f:
        for item in documents:
            f.write(json.dumps(item) + '\n')
    path = f'full/{dataset}/qrels'
    if not os.path.exists(path):
        os.makedirs(path)
    qrels.to_csv(f'full/{dataset}/qrels/{split}.tsv', sep='\t', index=False)
    f = open(f"full/{dataset}/instruction.txt", "w")
    f.write(instruction)
    f.close()

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'
    
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
    
class Instructor:
    def __init__(self, model_path: str = None, sep: str = " ", **kwargs):
        self.sep = sep

        self.q_model = INSTRUCTOR(model_path)
        self.doc_model = self.q_model

    def encode_queries(self, queries, batch_size:int = 4, **kwargs):
        return self.q_model.encode(queries, batch_size = batch_size, **kwargs)

    def encode_corpus(self, corpus, batch_size:int = 4, **kwargs):
        # our current implementation put the instruction as title in pre-processing
        sentences = [[doc["title"].strip(),doc["text"].strip()] for doc in corpus]
        return self.doc_model.encode(sentences, batch_size = batch_size, **kwargs)

class SGPT:
    def __init__(self, model_path: str = None, sep: str = " ", **kwargs):
        self.sep = sep

        self.q_model = INSTRUCTOR(model_path)
        self.doc_model = self.q_model

    def encode_queries(self, queries, batch_size:int = 4, **kwargs):
        return self.q_model.encode(queries, batch_size = batch_size, **kwargs)

    def encode_corpus(self, corpus, batch_size:int = 4, **kwargs):
        sentences = [str(doc["title"].strip() + " " + doc["text"].strip()) for doc in corpus]
        return self.doc_model.encode(sentences, batch_size = batch_size, **kwargs)
    
def initialize_retriever(model_name,
                         batch_size = 16,
                         sim = "cos_sim",
                         pooling = "mean"):
    if "text-embedding" in model_name:
        model = DRES(OpenAIEmbedding(model_name),batch_size = batch_size)
        retriever = EvaluateRetrieval(model, score_function= 'cos_sim')
    elif "embed-" in model_name:
        model = DRES(CohereEmbedding(model_name),batch_size = batch_size)
        retriever = EvaluateRetrieval(model, score_function= 'cos_sim')
    elif ('SGPT' in model_name) or ('gpt' in model_name):
        model = DRES(SGPT(model_name), batch_size = batch_size)
        retriever = EvaluateRetrieval(model, score_function= 'cos_sim')        
    elif 'instructor' in model_name:
        model = DRES(Instructor(model_name), batch_size = batch_size)
        retriever = EvaluateRetrieval(model, score_function= 'cos_sim')
    elif 'e5' in model_name:
        model = DRES(E5Mistral(model_name, torch_dtype=torch.float16, device_map = "auto"), batch_size = batch_size)
        retriever = EvaluateRetrieval(model, score_function= 'cos_sim')
    elif "Grit" in model_name:
        model = DRES(GritLM(model_name, torch_dtype=torch.float16, device_map = "auto",mode = "embedding"), batch_size=batch_size)
        retriever = EvaluateRetrieval(model, score_function= 'cos_sim')
    else:
        model = DRES(PooledSentenceBERT(model_name, pooling = pooling), batch_size=batch_size)
        retriever = EvaluateRetrieval(model, score_function= sim)        
    return retriever

def evaluate_full_instructor(retriever, queries, documents, qrels, 
                             instruction = None, doc_instruction = None,
                             evaluate_with_instruction = False):
    if evaluate_with_instruction == False:
        for key in queries.keys():
          queries[key] = ['', queries[key]]
    else:
        # prepend query with instruction in instructOR format
        for key in queries.keys():
          queries[key] = [instruction, queries[key]]
        for key in documents.keys():
          documents[key]['title'] = doc_instruction
    results = retriever.retrieve(documents, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

def evaluate_full(retriever, queries, documents, qrels,
                  instruction = None,
                  evaluate_with_instruction = False,
                  do_rerank = False,
                  rerank_top_k=100,
                  reranker_batch_size = 32,
                  reranker = None
                  ):
    if evaluate_with_instruction == False:
        results = retriever.retrieve(documents, queries)
    else:
        # prepend every query with instruction
        qid = [k for k,v in queries.items()]
        for id in qid:
            queries[id] = instruction + ' ' + queries[id]
        results = retriever.retrieve(documents, queries)

    if do_rerank:
        print("doing reranking")
        def get_top_k_documents(doc_similarity_dict, k=rerank_top_k):
            # Sort the dictionary by similarity score in descending order
            sorted_docs = sorted(doc_similarity_dict.items(), key=lambda x: x[1], reverse=True)
            # Keep the top k items
            top_k_docs = dict(sorted_docs[:k])
            return top_k_docs
        retriever.k_values = [value for value in retriever.k_values if value <= rerank_top_k]
        original_results = [(q_id, doc_scores) for q_id, doc_scores in results.items()]
        reranked_results = {}
        
        if reranker == "cohere":
            co = cohere.Client("placeholder")
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                response = co.rerank(
                    model = 'rerank-english-v2.0',
                    query = q,
                    documents = docs)
                cohere_index = [r.index for r in response]
                reranked_doc_ids = [doc_ids[ind] for ind in cohere_index]
                pseudo_sim_scores = [float(num) for num in np.linspace(1, 0, rerank_top_k)]
                reranked_dict = {}
                for doc_id, similarity_score in zip(reranked_doc_ids, pseudo_sim_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
        else:
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                # print(q)
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                # reranker prediction:
                pairs = [[q, doc] for doc in docs]
                # print(pairs[0])
                similarity_scores = reranker.predict(pairs, batch_size=reranker_batch_size)
                similarity_scores = [float(similarity_score) for similarity_score in similarity_scores]
                reranked_dict = {}
                for doc_id, similarity_score in zip(doc_ids, similarity_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
            
        ndcg, _map, recall, precision = retriever.evaluate(qrels, reranked_results, retriever.k_values)
    else:
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision



def evaluate_full_E5Mistral(retriever, queries, documents, qrels,
                             instruction = None,
                             evaluate_with_instruction = False,
                             do_rerank = False,
                             rerank_top_k=100,
                             reranker_batch_size = 32,
                             reranker = None):
    if evaluate_with_instruction == False:
        for key in queries.keys():
          queries[key] = f'Instruct: \nQuery: {queries[key]}'
    else:
        for key in queries.keys():
          queries[key] = f'Instruct: {instruction}\nQuery: {queries[key]}'
    print(queries[key])

    results = retriever.retrieve(documents, queries)
    if do_rerank:
        print("doing reranking")
        def get_top_k_documents(doc_similarity_dict, k=rerank_top_k):
            # Sort the dictionary by similarity score in descending order
            sorted_docs = sorted(doc_similarity_dict.items(), key=lambda x: x[1], reverse=True)
            # Keep the top k items
            top_k_docs = dict(sorted_docs[:k])
            return top_k_docs
        retriever.k_values = [value for value in retriever.k_values if value <= rerank_top_k]
        original_results = [(q_id, doc_scores) for q_id, doc_scores in results.items()]
        reranked_results = {}
        
        if reranker == "cohere":
            co = cohere.Client("placeholder")
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                response = co.rerank(
                    model = 'rerank-english-v2.0',
                    query = q,
                    documents = docs)
                cohere_index = [r.index for r in response]
                reranked_doc_ids = [doc_ids[ind] for ind in cohere_index]
                pseudo_sim_scores = [float(num) for num in np.linspace(1, 0, rerank_top_k)]
                reranked_dict = {}
                for doc_id, similarity_score in zip(reranked_doc_ids, pseudo_sim_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
        else:
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                print(q)
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                # reranker prediction:
                pairs = [[q, doc] for doc in docs]
                print(pairs[0])
                similarity_scores = reranker.predict(pairs, batch_size=reranker_batch_size)
                similarity_scores = [float(similarity_score) for similarity_score in similarity_scores]
                reranked_dict = {}
                for doc_id, similarity_score in zip(doc_ids, similarity_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
            
        ndcg, _map, recall, precision = retriever.evaluate(qrels, reranked_results, retriever.k_values)
    else:
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def evaluate_full_Grit(retriever, queries, documents, qrels,
                             instruction = None,
                             evaluate_with_instruction = False,
                             do_rerank = False,
                             rerank_top_k=100,
                             reranker_batch_size = 32,
                             reranker = None):
    
    if evaluate_with_instruction == False:
        retriever.retriever.model.encode_queries = partial(retriever.retriever.model.encode_queries,
                                                 instruction=gritlm_instruction(""))
    else:
        retriever.retriever.model.encode_queries = partial(retriever.retriever.model.encode_queries, 
                                                 instruction=gritlm_instruction(instruction))
    retriever.retriever.model.encode_corpus = partial(retriever.retriever.model.encode_corpus, 
                                                instruction=gritlm_instruction(""))    
    results = retriever.retrieve(documents, queries)
    if do_rerank:
        print("doing reranking")
        def get_top_k_documents(doc_similarity_dict, k=rerank_top_k):
            # Sort the dictionary by similarity score in descending order
            sorted_docs = sorted(doc_similarity_dict.items(), key=lambda x: x[1], reverse=True)
            # Keep the top k items
            top_k_docs = dict(sorted_docs[:k])
            return top_k_docs
        retriever.k_values = [value for value in retriever.k_values if value <= rerank_top_k]
        original_results = [(q_id, doc_scores) for q_id, doc_scores in results.items()]
        reranked_results = {}
        
        if reranker == "cohere":
            co = cohere.Client("placeholder")
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                response = co.rerank(
                    model = 'rerank-english-v2.0',
                    query = q,
                    documents = docs)
                cohere_index = [r.index for r in response]
                reranked_doc_ids = [doc_ids[ind] for ind in cohere_index]
                pseudo_sim_scores = [float(num) for num in np.linspace(1, 0, rerank_top_k)]
                reranked_dict = {}
                for doc_id, similarity_score in zip(reranked_doc_ids, pseudo_sim_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
        else:
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                print(q)
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                # reranker prediction:
                pairs = [[q, doc] for doc in docs]
                print(pairs[0])
                similarity_scores = reranker.predict(pairs, batch_size=reranker_batch_size)
                similarity_scores = [float(similarity_score) for similarity_score in similarity_scores]
                reranked_dict = {}
                for doc_id, similarity_score in zip(doc_ids, similarity_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
            
        ndcg, _map, recall, precision = retriever.evaluate(qrels, reranked_results, retriever.k_values)
    else:
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision


class GritLM(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str = None,
        mode: str = 'unified', # One of ['unified', 'embedding', 'generative']        
        pooling_method: str = 'mean', # One of ['cls', 'lasttoken', 'mean', 'weightedmean']
        normalized: bool = True,
        projection: int = None,
        is_inference: bool = True,
        embed_eos: str = "",
        attn: str = 'bbcc',
        **kwargs, # Passed to the model, e.g. `attn_implementation`, `torch_dtype` etc.
    ) -> None:
        super().__init__()
        if mode == 'embedding':
            if any([x in model_name_or_path for x in ['gtr', 't5', 'instructor']]):
                # Somehow AutoModel does not pick the right one by default
                from transformers import T5EncoderModel
                self.model = T5EncoderModel.from_pretrained(model_name_or_path, **kwargs)
            else:
                self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)
            self.embedding_attr = None
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)
            self.generate = self.model.generate

            if hasattr(self.model, 'model'): # LLama2 & Mistral
                self.embedding_attr = 'model'
            elif hasattr(self.model, 'transformer'): # GPT-Neo & GPT-J
                self.embedding_attr = 'transformer'
            else: 
                raise ValueError("Could not find attribute to use for embedding: ", self.model)

        self.projection = torch.nn.Linear(
            in_features=self.model.config.hidden_size, 
            out_features=int(projection),
            dtype=self.model.dtype
        ) if projection is not None else None
        self.normalized = normalized
        self.pooling_method = pooling_method

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_gpus = 1
        self.embed_eos = embed_eos
        self.attn = attn
        if (self.attn is not None) and self.attn not in ['bbcc', 'cccc', 'bb', 'cc']:
            raise ValueError(f"Mixed attention no longer supported: {self.attn}. Only bbcc, cccc, bb, cc are supported")

        print(f"Created GritLM: {self.model.dtype} dtype, {pooling_method} pool, {mode} mode, {attn} attn")

        if is_inference:
            # Padding side right is necessary for `embed_instruction` to index correctly
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='right')
            if not(self.tokenizer.pad_token) and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print('Set pad token to eos token: ' + self.tokenizer.pad_token)        
            if self.embed_eos:
                assert self.embed_eos in self.tokenizer.vocab, f"EOS token {self.embed_eos} not in vocab"
            self.model.eval()
            if not("device_map" in kwargs):
                self.model.to(self.device)
                # Parallelize embedding model
                if mode == 'embedding':
                    self.num_gpus = torch.cuda.device_count()
                    if self.num_gpus > 1:
                        print(f"----------Using {self.num_gpus} data-parallel GPUs----------")
                        self.model = torch.nn.DataParallel(self.model)

    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: Union[List[str], str, List[Dict[str, str]]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        return self.encode(corpus, **kwargs)

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 8192,
        instruction: str = "",
        embed_instruction: bool = False,
        get_cache: bool = False,
        convert_to_tensor: bool = False,
        recast: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> np.ndarray:
        if self.num_gpus > 1:
            batch_size *= self.num_gpus

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_embeddings, all_kv_caches = [], []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=len(sentences)<256):
            sentences_batch = [
                instruction + s + self.embed_eos for s in sentences[start_index:start_index + batch_size]
            ]
            # This will prepend the bos token if the tokenizer has `add_bos_token=True`
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            ).to(self.device)

            if (self.attn is not None) and (self.attn[:2] == 'bb'):
                inputs["is_causal"] = False
            if get_cache:
                inputs['use_cache'] = True
            outputs = (
                getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model
            )(**inputs)
            last_hidden_state = outputs[0]
            if get_cache:
                # Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`
                assert len(all_kv_caches) == 0, "Can only get cache for one batch at a time"
                all_kv_caches = outputs[1]

            if self.projection:
                last_hidden_state = self.projection(last_hidden_state)
            if (instruction) and (embed_instruction is False) and ("mean" in self.pooling_method):
                # Remove instruction tokens from the embeddings by masking them
                instruction_tokens = self.tokenizer(
                    instruction,
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                )["input_ids"]
                inputs['attention_mask'][:, :len(instruction_tokens)] = 0
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'], recast=recast)
            # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
            if self.normalized: 
                in_dtype = embeddings.dtype
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1).to(in_dtype)
            embeddings = cast(torch.Tensor, embeddings)
            if convert_to_tensor:
                all_embeddings.append(embeddings)
            else:
                # NumPy does not support bfloat16
                all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())

        all_embeddings = (
            torch.cat(all_embeddings, dim=0) if convert_to_tensor else np.concatenate(all_embeddings, axis=0)
        )
        if input_was_string:
            all_embeddings = all_embeddings[0]
        if get_cache:
            # all_kv_caches = (
            #     torch.stack(all_kv_caches, dim=0) if convert_to_tensor else np.concatenate(all_kv_caches, axis=0)
            # )
            return all_embeddings, all_kv_caches
        return all_embeddings

    def pooling(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor = None, recast: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: [b, n, d]
            attention_mask: [b, n]
        """
        # In case the model is distributed across multiple devices; hidden_state may end up on diff device
        hidden_state = hidden_state.to(attention_mask.device)
        if self.pooling_method == 'cls':
            embedding = hidden_state[:, 0]
        elif self.pooling_method == 'lasttoken':
            b, n, d = hidden_state.size()
            # Get the last `1` in the attention mask of each item
            # Often it is just `gather_indices = torch.argmin(attention_mask, 1, keepdim=False) - 1`
            # except when 1) There's all 1's 2) There's 0's before the 1's
            reversed_mask = torch.flip(attention_mask, dims=(1,))
            argmax_reverse = torch.argmax(reversed_mask, dim=1, keepdim=False)
            gather_indices = attention_mask.size(1) - argmax_reverse - 1
            # If there are empty sequences, where the index would become -1 it will crash so set them to 0
            gather_indices = torch.clamp(gather_indices, min=0)
            # Turn indices from shape [b] -> [b, 1, d]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, d)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (b, 1, d)
            # Gather along the seq len: [b, n, d] -> [b, d]
            # Actually no need for the attention mask as we gather the last token where attn_mask=1 but
            # as some indices (which shouldn't be attended to) may be 0 due to clamp, use mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand((b, n, d)).float()
            embedding = torch.gather(hidden_state * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        elif self.pooling_method in ['mean', 'weightedmean']:
            if self.pooling_method == 'weightedmean':
                attention_mask *= attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            embedding = s / d
        else: raise NotImplementedError(f"Unknown pooling method: {self.pooling_method}")
        # Recasting performs slightly worse but saves 50% space
        if recast: return embedding.to(hidden_state.dtype)
        return embedding