from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
from .sgpt import SGPT
from .pooled_sentence_transformers import PooledSentenceBERT
from .e5mistral import E5Mistral
from .grit import GritLM
from .instructor import Instructor
from .openai import OpenAIEmbedding
from .cohere import CohereEmbedding
import torch

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
