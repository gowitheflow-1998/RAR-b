import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union, Tuple
from sklearn.metrics import accuracy_score
import cohere
from .tart_utils import EncT5ForSequenceClassification
from .tart_utils import EncT5Tokenizer
from sentence_transformers.cross_encoder import CrossEncoder
from ..rarb_models.grit import GritLM
from openai import OpenAI

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class AutoCrossEncoder:
    def __init__(self, model_name:str = None, device = "cuda:0", api_key = "") -> None:
        self.model_name = model_name
        self.co = None
        self.device = device
        if ("cohere" not in model_name)&("tart" not in model_name)&("Mistral" not in model_name)\
            &("mistral" not in model_name):
            self.model = CrossEncoder(model_name, device=device)
            self.model.max_length = 512

        elif "cohere" in model_name:
            self.co = cohere.Client(api_key)
        
        elif "tart" in model_name:
            self.model = EncT5ForSequenceClassification.from_pretrained("facebook/tart-full-flan-t5-xl")
            self.tokenizer =  EncT5Tokenizer.from_pretrained("facebook/tart-full-flan-t5-xl")
            self.model.eval()
            self.model.to(device)
        
        elif ("Mistral" in model_name) or ("mistral" in model_name):
            print("evaluating LLM-based rerankers")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map = "auto")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.token_false_id = self.tokenizer.get_vocab()["false"]
            self.token_true_id = self.tokenizer.get_vocab()["true"]
            self.template = """<s> [INST] Instruction: {instruction} Decide if the following document is the correct answer to the query (true/false). Answer using only one word, one of those two choices.

            Query: {query}
            Document: {text}
            Relevant (only output one word, either "true" or "false"): [/INST] """

    def evaluate_mcr(self, queries, documents, labels, instructions=None, batch_size = 16):
        results = []
        if self.co is not None:
            
            print("evaluating cohere re-rankers")
            predictions = []
            if instructions is None:
                print("evaluating without instruction")
                for q,d in tqdm(zip(queries, documents)):
                    response = self.co.rerank(
                        model = 'rerank-english-v2.0',
                        query = q,
                        documents = d)
                    predictions.append(response[0].index)  
            else:
                print("evaluating with instruction")
                for i,q,d in tqdm(zip(instructions, queries, documents)):
                    iq = i + " " + q
                    response = self.co.rerank(
                        model = 'rerank-english-v2.0',
                        query = iq,
                        documents = d)
                    predictions.append(response[0].index)

            accuracy = accuracy_score(labels, predictions)

        else:
            print("evaluating open-source rerankers")
            results = []
            num_example = len(queries)
            num_correct = 0
            if "tart" in self.model_name:
                print("evaluating TART")
                if instructions is None:
                    print("evaluating without instruction")
                    for q, d, l in tqdm(zip(queries, documents, labels)):
                        features = self.tokenizer([f'[SEP] {q}']*len(d), d, padding=True, truncation=True, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            scores = self.model(**features).logits
                            normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]
                            prediction = np.argmax(normalized_scores)
                            if prediction == l:
                                num_correct+=1
                else:
                    print("evaluating with instruction")
                    for i, q, d, l in tqdm(zip(instructions, queries, documents, labels)):
                        features = self.tokenizer([f'{i} [SEP] {q}']*len(d), d, padding=True, truncation=True, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            scores = self.model(**features).logits
                            normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]
                            prediction = np.argmax(normalized_scores)
                            results.append(normalized_scores)
                            if prediction == l:
                                num_correct+=1
            elif ("Mistral" in self.model_name) or ("mistral" in self.model_name):
                print(f"evaluating LLM-based rerankers")
                if instructions is None:
                    print("evaluating without instruction")
                    for q, d, l in tqdm(zip(queries, documents, labels)):
                        i = ""
                        prompts = [
                            self.template.format(instruction = i,
                                                 query = q, 
                                                 text = text) for text in d
                        ]
                        tokens = self.tokenizer(
                            prompts,
                            padding = True,
                            truncation = True,
                            return_tensors = "pt",
                            pad_to_multiple_of = None
                        )
                        for key in tokens:
                            tokens[key] = tokens[key].cuda()

                        with torch.no_grad():
                            batch_scores = self.model(**tokens).logits[:,-1,:]
                        true_vector = batch_scores[:,self.token_true_id]
                        false_vector = batch_scores[:,self.token_false_id]
                        batch_scores = torch.stack([false_vector, true_vector], dim =1)
                        batch_scores = torch.nn.functional.log_softmax(batch_scores,dim=1)
                        scores = batch_scores[:,1].exp().tolist()
                        prediction = np.argmax(scores)
                        if prediction == l:
                            num_correct+=1
                else:
                    print("evaluating with instruction")
                    for i, q, d, l in tqdm(zip(instructions, queries, documents, labels)):
                        prompts = [
                            self.template.format(instruction = i,
                                                 query = q, 
                                                 text = text) for text in d
                        ]
                        tokens = self.tokenizer(
                            prompts,
                            padding = True,
                            truncation = True,
                            return_tensors = "pt",
                            pad_to_multiple_of = None
                        )
                        for key in tokens:
                            tokens[key] = tokens[key].cuda()

                        with torch.no_grad():
                            batch_scores = self.model(**tokens).logits[:,-1,:]
                        true_vector = batch_scores[:,self.token_true_id]
                        false_vector = batch_scores[:,self.token_false_id]
                        batch_scores = torch.stack([false_vector, true_vector], dim =1)
                        batch_scores = torch.nn.functional.log_softmax(batch_scores,dim=1)
                        scores = batch_scores[:,1].exp().tolist()
                        prediction = np.argmax(scores)
                        if prediction == l:
                            num_correct+=1
            else:
                if instructions is None:
                    print("evaluating without instruction")
                    for q, d, l in tqdm(zip(queries, documents, labels)):
                        pairs = [[q, option] for option in d]
                        similarity_scores = self.model.predict(pairs, batch_size=batch_size)
                        prediction = np.argmax(similarity_scores)
                        results.append(similarity_scores)
                        if prediction == l:
                            num_correct+=1
                else:
                    print("evaluating with instruction")
                    for i, q, d, l in tqdm(zip(instructions, queries, documents, labels)):
                        iq = i + ' ' + q 
                        pairs = [[iq, option] for option in d]
                        similarity_scores = self.model.predict(pairs, batch_size=batch_size)
                        prediction = np.argmax(similarity_scores)
                        results.append(similarity_scores)
                        if prediction == l:
                            num_correct+=1           
            accuracy = num_correct/num_example
        return results, accuracy

def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

class AutoRetriever:

    def __init__(self, model_path: Union[str, Tuple] = None,
                 pooling = "mean",
                 api_key = ""):
        
        self.model_name = model_path

        if isinstance(model_path, str):
            
            # weighted mean pooling is not updated in the installed ST, so we use instructor to take in sgpt
            if ("instructor" in model_path) or ("SGPT" in model_path) or ("gpt" in model_path):
                self.q_model = INSTRUCTOR(model_path,device = "cuda:0")
                
            elif "e5" in model_path:
                self.max_length = 8192
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.q_model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map = "auto")
            elif "Grit" in model_path:
                self.q_model = GritLM(model_path, torch_dtype=torch.float16, device_map = "auto",mode = "embedding")
            elif "cohere" in model_path:
                self.co = cohere.Client(api_key)
                self.model_name = "embed-english-v3.0" # overwrite the model name for now
            elif "text-embedding" in model_path:
                self.client = OpenAI(api_key = api_key)
            else:
                self.q_model = SentenceTransformer(model_path,device = "cuda:0")
            # if the model already has a sbert config it's cls pooling, 
            # the default init arg pooling = "mean" wont affect
            if ("e5" not in model_path) and ("Grit" not in model_path) \
            and ("cohere" not in model_path) and ("text-embedding" not in model_path):

                if self.q_model[1].pooling_mode_mean_tokens == True:
                    if pooling == "cls":
                        self.q_model[1].pooling_mode_mean_tokens = False
                        self.q_model[1].pooling_mode_cls_token = True
            if ("cohere" not in model_path) and ("text-embedding" not in model_path):
                self.doc_model = self.q_model

        elif isinstance(model_path, tuple):
            if "instructor" in model_path[0]:
                self.q_model = INSTRUCTOR(model_path[0])
            else:
                self.q_model = SentenceTransformer(model_path[0])
            if self.q_model[1].pooling_mode_mean_tokens == True:
                if pooling == "cls":
                    self.q_model[1].pooling_mode_mean_tokens = False
                    self.q_model[1].pooling_mode_cls_token = True
                    
            if "instructor" in model_path[0]:
                self.doc_model = INSTRUCTOR(model_path[1])         
            else:   
                self.doc_model = SentenceTransformer(model_path[1])
            if self.doc_model[1].pooling_mode_mean_tokens == True:
                if pooling == "cls":
                    self.doc_model[1].pooling_mode_mean_tokens = False
                    self.doc_model[1].pooling_mode_cls_token = True

        if ("cohere" not in model_path) and ("text-embedding" not in model_path):
            if "bge" not in model_path:
                self.q_model.max_seq_length = 512
                self.doc_model.max_seq_length = 512
            if "e5" not in model_path:
                print(self.q_model.max_seq_length, self.doc_model.max_seq_length)
        
    def evaluate_mcr(self, instructions, queries, documents, labels, sim = 'cos_sim'):

        results = []

        num_example = len(queries)
        num_correct = 0

        for i, q, d, l in tqdm(zip(instructions, queries, documents, labels)):
            
            iq = (i + ' ' + q).strip()
            e_iq = self.q_model.encode(iq)
            e_d = self.doc_model.encode(d)
            if sim == 'cos_sim':
                result = cosine_similarity([e_iq], e_d)
            elif sim == 'dot':
                result = np.dot([e_iq], e_d.T)

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

            results.append(result)
            
            if np.argmax(result) == l:
                num_correct+=1
        accuracy = num_correct/num_example
        return accuracy, results

    def evaluate_mcr_INSTRUCTOR(self, instructions, queries, documents, labels):

        results = None
        num_correct = 0
        num_example = len(queries)
        for i, q, d, l in tqdm(zip(instructions, queries, documents, labels)):
            e_iq = self.q_model.encode([[i,q]])
            e_d = self.doc_model.encode([["",doc] for doc in d])
            
            result = cosine_similarity(e_iq, e_d)
            if np.argmax(result) == l:
                num_correct+=1
        accuracy = num_correct/num_example
        return accuracy, results

    def evaluate_mcr_Grit(self, instructions, queries, documents, labels, sim = 'cos_sim'):
        results = []
        num_example = len(queries)
        num_correct = 0

        for index, (i, q, d, l) in tqdm(enumerate(zip(instructions, queries, documents, labels))):
            q_embeddings = self.doc_model.encode([q], instruction=gritlm_instruction(i))
            doc_embeddings = self.q_model.encode(d, instruction=gritlm_instruction(""))

            if sim == 'cos_sim':
                result = cosine_similarity(q_embeddings, doc_embeddings)
            elif sim == 'dot':
                result = np.dot(q_embeddings, doc_embeddings.T)

            results.append(result[0])
            if np.argmax(result) == l:
                num_correct+=1
        accuracy = num_correct/num_example
        return accuracy, results
    
    def evaluate_mcr_E5Mistral(self, instructions, queries, documents, labels, sim = 'cos_sim'):

        results = []
        num_example = len(queries)
        num_correct = 0

        for index, (i, q, d, l) in tqdm(enumerate(zip(instructions, queries, documents, labels))):

            iq = f"Instruct: {i} \nQuery: {q}"
            if index <1:
                print(iq)
            q_inputs = self.tokenizer([iq], max_length=self.max_length - 1, return_attention_mask=False, padding=False, truncation=True)
            q_inputs['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in q_inputs['input_ids']]
            q_inputs = self.tokenizer.pad(q_inputs, padding=True, return_attention_mask=True, return_tensors='pt').to("cuda:0")
            with torch.no_grad():
                outputs = self.q_model(**q_inputs)
            q_embeddings = last_token_pool(outputs.last_hidden_state, q_inputs['attention_mask'])
            q_embeddings = F.normalize(q_embeddings, p=2, dim=1).float().cpu()
            
            doc_inputs = self.tokenizer(d, max_length=self.max_length - 1, return_attention_mask=False, padding=False, truncation=True)
            doc_inputs['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in doc_inputs['input_ids']]
            doc_inputs = self.tokenizer.pad(doc_inputs, padding=True, return_attention_mask=True, return_tensors='pt').to("cuda:0")
            with torch.no_grad():
                outputs = self.doc_model(**doc_inputs)
            doc_embeddings = last_token_pool(outputs.last_hidden_state, doc_inputs['attention_mask'])
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1).float().cpu()
            if sim == 'cos_sim':
                result = cosine_similarity(q_embeddings, doc_embeddings)
            elif sim == 'dot':
                result = np.dot(q_embeddings, doc_embeddings.T)

            results.append(result[0])
            if np.argmax(result) == l:
                num_correct+=1
        accuracy = num_correct/num_example
        return accuracy, results

    def evaluate_mcr_OpenAI(self, instructions, queries, documents, labels, sim = "cos_sim"):
        
        results = []

        num_example = len(queries)
        num_correct = 0

        for i, q, d, l in tqdm(zip(instructions, queries, documents, labels)):
            
            iq = (i + ' ' + q).strip()
            e_iq = self.client.embeddings.create(
                    input=iq, model=self.model_name
                ).data
            e_d = self.client.embeddings.create(
                    input=d, model=self.model_name
                ).data
            e_iq = np.array([x.embedding for x in e_iq])
            e_d = np.array([x.embedding for x in e_d])
            
            if sim == 'cos_sim':
                result = cosine_similarity(e_iq, e_d)
            elif sim == 'dot':
                result = np.dot(e_iq, e_d.T)

            results.append(result)
            if np.argmax(result) == l:
                num_correct+=1
        accuracy = num_correct/num_example
        return accuracy, results
    def evaluate_mcr_cohere(self, instructions, queries, documents, labels, sim = 'cos_sim'):

        results = []

        num_example = len(queries)
        num_correct = 0

        for i, q, d, l in tqdm(zip(instructions, queries, documents, labels)):
            
            iq = (i + ' ' + q).strip()
            e_iq = self.co.embed(
                texts=[iq],
                model=self.model_name,
                input_type='search_query',
                truncate="end"
            ).embeddings    
            
            e_d = self.co.embed(
                texts=d,
                model=self.model_name,
                input_type='search_document',
                truncate="end"
            ).embeddings  

            if sim == 'cos_sim':
                result = cosine_similarity(e_iq, e_d)
            elif sim == 'dot':
                result = np.dot(e_iq, e_d.T)

            results.append(result)
            if np.argmax(result) == l:
                num_correct+=1
        accuracy = num_correct/num_example
        return accuracy, results
        
    def evaluate(self, instructions, queries, documents, labels, sim = 'cos_sim',
                 mode = "both",save_accuracy = True):
        
        if save_accuracy == True:
            accuracy_list = []
            
        if mode == "both":
            if "instructor" in self.model_name:
                no_instructions = [""]*len(queries)
                accuracy, results = self.evaluate_mcr_INSTRUCTOR(no_instructions, queries, documents, labels)
                print(f"{self.model_name} w/0 instruction: {accuracy}")
                accuracy_list.append(accuracy)
                
                accuracy, results = self.evaluate_mcr_INSTRUCTOR(instructions, queries, documents, labels)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "e5" in self.model_name:
                no_instructions = [""]*len(queries)
                accuracy, results = self.evaluate_mcr_E5Mistral(no_instructions, queries, documents, labels)
                print(f"{self.model_name} w/0 instruction: {accuracy}")
                accuracy_list.append(accuracy)
                
                accuracy, results = self.evaluate_mcr_E5Mistral(instructions, queries, documents, labels)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "Grit" in self.model_name:
                no_instructions = [""]*len(queries)
                accuracy, results = self.evaluate_mcr_Grit(no_instructions, queries, documents, labels)
                print(f"{self.model_name} w/0 instruction: {accuracy}")
                accuracy_list.append(accuracy)
                
                accuracy, results = self.evaluate_mcr_Grit(instructions, queries, documents, labels)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "embed-english-v3.0" in self.model_name:
                no_instructions = [""]*len(queries)
                accuracy, results = self.evaluate_mcr_cohere(no_instructions, queries, documents, labels)
                print(f"{self.model_name} w/0 instruction: {accuracy}")
                accuracy_list.append(accuracy)
                
                accuracy, results = self.evaluate_mcr_cohere(instructions, queries, documents, labels)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "text-embedding" in self.model_name:
                no_instructions = [""]*len(queries)
                accuracy, results = self.evaluate_mcr_OpenAI(no_instructions, queries, documents, labels)
                print(f"{self.model_name} w/0 instruction: {accuracy}")
                accuracy_list.append(accuracy)
                
                accuracy, results = self.evaluate_mcr_OpenAI(instructions, queries, documents, labels)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)
            else:
                accuracy, results = self.evaluate_mcr_without_instructions(queries, documents, labels, sim=sim)
                print(f"{self.model_name} w/o instruction: {accuracy}")
                accuracy_list.append(accuracy)
                
                accuracy, results = self.evaluate_mcr(instructions, queries, documents, labels, sim=sim)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)
        elif mode == "with":
            if "instructor" in self.model_name:
                accuracy, results = self.evaluate_mcr_INSTRUCTOR(instructions, queries, documents, labels)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "e5" in self.model_name:
                accuracy, results = self.evaluate_mcr_E5Mistral(instructions, queries, documents, labels)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "Grit" in self.model_name:
                accuracy, results = self.evaluate_mcr_Grit(instructions, queries, documents, labels)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "embed-english-v3.0" in self.model_name:
                accuracy, results = self.evaluate_mcr_cohere(instructions, queries, documents, labels)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "text-embedding" in self.model_name:
                accuracy, results = self.evaluate_mcr_OpenAI(instructions, queries, documents, labels)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)
            else:
                accuracy, results = self.evaluate_mcr(instructions, queries, documents, labels, sim=sim)
                print(f"{self.model_name} w/ instruction: {accuracy}")
                accuracy_list.append(accuracy)

                
        elif mode == "without":
            if "instructor" in self.model_name:
                no_instructions = [""]*len(queries)
                accuracy, results = self.evaluate_mcr_INSTRUCTOR(no_instructions, queries, documents, labels)
                print(f"{self.model_name} w/0 instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "e5" in self.model_name:
                no_instructions = [""]*len(queries)
                accuracy, results = self.evaluate_mcr_E5Mistral(no_instructions, queries, documents, labels)
                print(f"{self.model_name} w/0 instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "Grit" in self.model_name:
                no_instructions = [""]*len(queries)
                accuracy, results = self.evaluate_mcr_Grit(no_instructions, queries, documents, labels)
                print(f"{self.model_name} w/0 instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "embed-english-v3.0" in self.model_name:
                no_instructions = [""]*len(queries)
                accuracy, results = self.evaluate_mcr_cohere(no_instructions, queries, documents, labels)
                print(f"{self.model_name} w/0 instruction: {accuracy}")
                accuracy_list.append(accuracy)
            elif "text-embedding" in self.model_name:
                no_instructions = [""]*len(queries)
                accuracy, results = self.evaluate_mcr_OpenAI(no_instructions, queries, documents, labels)
                print(f"{self.model_name} w/0 instruction: {accuracy}")
                accuracy_list.append(accuracy)
            else:
                accuracy, results = self.evaluate_mcr_without_instructions(queries, documents, labels, sim=sim)
                print(f"{self.model_name} w/o instruction: {accuracy}")
                accuracy_list.append(accuracy)
        return accuracy_list