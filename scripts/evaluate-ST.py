from beir.datasets.data_loader import GenericDataLoader
import pickle
from rarb.rarb_models import *
from rarb import *
import random

random.seed(41)


def evaluate_at_once(dataset, split = "dev", batch_size = 128):

    metrics = []

    for model_name in ['facebook/contriever',
                    'sentence-transformers/all-mpnet-base-v2',
                    'sentence-transformers/all-MiniLM-L6-v2',
                    ("facebook/dragon-plus-query-encoder","facebook/dragon-plus-context-encoder")]:
        
        f = open(f"full/{dataset}/instruction.txt", "r")
        instruction = f.read()
        f.close()
        corpus, queries, qrels = GenericDataLoader(data_folder=f'full/{dataset}').load(split=split)
        
        sim = 'dot' if (model_name == 'facebook/contriever') or isinstance(model_name, tuple) else 'cos_sim'# contriever or dragon

        print(f'evaluating {model_name}')
        if isinstance(model_name, tuple): #dragon
            retriever = initialize_retriever(("facebook/dragon-plus-query-encoder","facebook/dragon-plus-context-encoder"),
                                        pooling = "cls", batch_size = batch_size, sim = sim)
        else:
            retriever = initialize_retriever(model_name, batch_size = batch_size, sim = sim)

        ndcg, _map, recall, precision = evaluate_full(retriever, queries, corpus, qrels,
                                                      instruction = instruction, 
                                                      evaluate_with_instruction = False
                                                      )

        metrics.append([model_name, "without", ndcg["NDCG@10"],recall["Recall@10"]])
        print('result without instruction')
        print(ndcg, recall)

        # reloading is not necessary for models other than E5Mistral and Instructor
        # because we evaluate without instruction first without anything changed in the query and doc dicts.
        # but we still leave it here in case users forget to do it when flipping the order
        corpus, queries, qrels = GenericDataLoader(data_folder=f'full/{dataset}').load(split=split) 

        ndcg, _map, recall, precision = evaluate_full(retriever, queries, corpus, qrels,
                                                      instruction = instruction, 
                                                      evaluate_with_instruction = True)

        metrics.append([model_name, "with", ndcg["NDCG@10"],recall["Recall@10"]])
        print('result with instruction')
        print(ndcg, recall)

    return metrics


def evaluate_all_models(dataset, bge_version = 1.5,
                        doc_instruction = 'Represent the following text to end an unfinished story',
                        split ='dev',
                        reranker = None,
                        batch_size = 128):
    results = []
    results.extend(evaluate_at_once(dataset, split = split, batch_size = batch_size)) 
    # results.extend(evaluate_bge(dataset, version = bge_version, split = split))
    # results.extend(evaluate_at_once_instructor(dataset,
                                # doc_instruction = doc_instruction, split = split,
                                #batch_size = 128))
    return results
    
doc_instruction_lookup = {     
    "alphanli": 'Represent the following text that leads to the end of a story',
    "hellaswag": 'Represent the following text to end an unfinished story',
    "winogrande": 'Represent this entity',
    "piqa": 'represent the solution for a goal',
    "siqa": 'represent the sentence to answer a question',
    "ARC-Challenge": 'Represent the answer to a question',
    "quail": 'Represent the answer to a question given the context',
    "TR1": "Represent the following month-year.",
    "TR2-pure": "Represent this answer",
    "TR2-fact": "Represent this answer",
    "TR2-context": "Represent this answer",
    "TR3-pure": "Represent this answer",
    "TR3-fact": "Represent this answer",
    "TR3-context": "Represent this answer",
    "spartqa":"Represent the following answer to a spatial reasoning question",
    "TinyCode": "Represent the answer for a coding problem",
    "code-pooled": "Represent the answer for a coding problem",
    "humanevalpack-mbpp-pooled":"Represent the answer for a coding problem",
    "math-pooled": "Represent the answer to answer a math problem"}




if __name__ == "__main__":

    dataset_list = ["piqa","siqa","alphanli","hellaswag","winogrande","ARC-Challenge","quail",
                    "TempReason-l1", "TempReason-l2-pure", "TempReason-l2-fact", "TempReason-l2-context", 
                    "TempReason-l3-pure", "TempReason-l3-fact", "TempReason-l3-context"
                    "math-pooled","humanevalpack-mbpp-pooled"]

    for dataset in dataset_list:
        split = 'test'
        result_path = f"results/{dataset}-{split}"
        if not os.path.exists(result_path):
            # If it does not exist, create the directory and all required intermediate directories
            os.makedirs(result_path)
            
        doc_instruction = doc_instruction_lookup[dataset]
        print(doc_instruction)
        results = evaluate_all_models(dataset, bge_version = "m3",
                                doc_instruction = doc_instruction, split = split,
                                reranker = None,
                                batch_size = 128)
        
        with open(f"{result_path}/ST-models.pkl", 'wb') as file:
            pickle.dump(results, file)