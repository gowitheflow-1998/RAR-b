from beir.datasets.data_loader import GenericDataLoader
import pickle
from rarb.rarb_models import *
from rarb import *
import random

random.seed(41)

def evaluate_bge(dataset, version = 1, split = "dev"):

    metrics = []
    
    if version == 1:
        for model_name in ['BAAI/bge-small-en',
                        'BAAI/bge-base-en',
                        'BAAI/bge-large-en']:

            if "large" in model_name:
                batch_size = 64
            elif "base" in model_name:
                batch_size = 128
            else:
                batch_size = 256
                
            corpus, queries, qrels = GenericDataLoader(data_folder=f'full/{dataset}').load(split=split)
            
            f = open(f"full/{dataset}/instruction.txt", "r")
            instruction = f.read()
            f.close()

            sim = 'dot'
            print(f'evaluating {model_name}')
            retriever = initialize_retriever(model_name, batch_size = batch_size, sim = sim)
            ndcg, _map, recall, precision = evaluate_full(retriever, queries, corpus, qrels,
                                                        instruction = instruction, evaluate_with_instruction = False)

            metrics.append([model_name, "without", ndcg["NDCG@10"],recall["Recall@10"]])
            print('result without instruction')
            print(ndcg, recall)

            ndcg, _map, recall, precision = evaluate_full(retriever, queries, corpus, qrels,
                                                        instruction = instruction, evaluate_with_instruction = True)

            metrics.append([model_name, "with", ndcg["NDCG@10"],recall["Recall@10"]])
            print('result with instruction')
            print(ndcg, recall)
    
    elif version == 1.5:

        for model_name in ['BAAI/bge-small-en-v1.5',
                          'BAAI/bge-base-en-v1.5',
                          'BAAI/bge-large-en-v1.5']:

            if "large" in model_name:
                batch_size = 64
            elif "base" in model_name:
                batch_size = 128
            else:
                batch_size = 256
            corpus, queries, qrels = GenericDataLoader(data_folder=f'full/{dataset}').load(split=split)

            f = open(f"full/{dataset}/instruction.txt", "r")
            instruction = f.read()

            sim = 'dot'
            print(f'evaluating {model_name}')
            retriever = initialize_retriever(model_name, batch_size = batch_size, sim = sim)
            ndcg, _map, recall, precision = evaluate_full(retriever, queries, corpus, qrels,
                                                        instruction = instruction, evaluate_with_instruction = False)

            metrics.append([model_name, "without", ndcg["NDCG@10"],recall["Recall@10"]])
            print('result without instruction')
            print(ndcg, recall)

            ndcg, _map, recall, precision = evaluate_full(retriever, queries, corpus, qrels,
                                                        instruction = instruction, evaluate_with_instruction = True)

            metrics.append([model_name, "with", ndcg["NDCG@10"],recall["Recall@10"]])
            print('result with instruction')
            print(ndcg, recall)

    elif version == "m3":
        model_name = "BAAI/bge-m3"
        if (dataset == "TR3-context") or (dataset == "TR2-context"):
            batch_size = 2
            print(batch_size)
        elif ("mbpp" in dataset) or ("code" in dataset) or ("humaneval" in dataset):
            batch_size = 1
        elif dataset == "math-pooled":
            batch_size = 16
        else:
            batch_size = 32
        corpus, queries, qrels = GenericDataLoader(data_folder=f'full/{dataset}').load(split=split)

        f = open(f"full/{dataset}/instruction.txt", "r")
        instruction = f.read()

        sim = 'dot'
        print(f'evaluating {model_name}')
        retriever = initialize_retriever(model_name, batch_size = batch_size, sim = sim)
        ndcg, _map, recall, precision = evaluate_full(retriever, queries, corpus, qrels,
                                                    instruction = instruction, evaluate_with_instruction = False)

        metrics.append([model_name, "without", ndcg["NDCG@10"],recall["Recall@10"]])
        print('result without instruction')
        print(ndcg, recall)

        # again technically, reloading the dataset is not necessary here (except for E5Mistral and Instructor),
        # but we still leave it here in case users forget that this is a problem when flipping without and with-instruction settings 
        corpus, queries, qrels = GenericDataLoader(data_folder=f'full/{dataset}').load(split=split)
        
        ndcg, _map, recall, precision = evaluate_full(retriever, queries, corpus, qrels,
                                                    instruction = instruction, evaluate_with_instruction = True)

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
    # results.extend(evaluate_at_once(dataset, split = split, batch_size = batch_size)) 

    # not taking in batch size because of too many model sizes; pre-set in the function
    results.extend(evaluate_bge(dataset, version = 1.5, split = split))
    results.extend(evaluate_bge(dataset, version = bge_version, split = split)) 

    # results.extend(evaluate_at_once_instructor(dataset,
                                # doc_instruction = doc_instruction, split = split,
                                #batch_size = 128))
    return results
    
# only needed for instructor
doc_instruction_lookup = {     
    "alphanli": 'Represent the following text that leads to the end of a story',
    "hellaswag": 'Represent the following text to end an unfinished story',
    "winogrande": 'Represent this entity',
    "piqa": 'represent the solution for a goal',
    "siqa": 'represent the sentence to answer a question',
    "ARC-Challenge": 'Represent the answer to a question',
    "quail": 'Represent the answer to a question given the context',
    "TempReason-l1": "Represent the following month-year.",
    "TempReason-l2-pure": "Represent this answer",
    "TempReason-l2-fact": "Represent this answer",
    "TempReason-l2-context": "Represent this answer",
    "TempReason-l3-pure": "Represent this answer",
    "TempReason-l3-fact": "Represent this answer",
    "TempReason-l3-context": "Represent this answer",
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
            os.makedirs(result_path)
            
        doc_instruction = doc_instruction_lookup[dataset]
        print(doc_instruction)
        results = evaluate_all_models(dataset, bge_version = "m3", # by default, run an 1.5 first, then m3 again by passing in this
                                      doc_instruction = doc_instruction, split = split,
                                      reranker = None,
                                      batch_size = 128)
        
        with open(f"{result_path}/BGE-models.pkl", 'wb') as file:
            pickle.dump(results, file)