from beir.datasets.data_loader import GenericDataLoader
import pickle
from rarb.rarb_models import *
from rarb import *
import random

random.seed(41)


def evaluate_at_once_instructor(dataset,
                                doc_instruction = 'Represent the following text to end an unfinished story',
                                sizes = ["base","large","XL"],split="dev"):
    f = open(f"full/{dataset}/instruction.txt", "r")
    instruction = f.read()
    f.close()
    metrics= []
    for model_size in sizes:
        if model_size == "XL":
            batch_size = 32
        elif model_size == "large":
            batch_size = 64
        else:
            batch_size = 128
        corpus, queries, qrels = GenericDataLoader(data_folder=f'full/{dataset}').load(split=split)
        
        model_name = f'hkunlp/instructor-{model_size}'
        print(f'evaluating {model_name}')
        retriever = initialize_retriever(model_name, batch_size=batch_size)
        print(len(corpus))
        ndcg, _map, recall, precision = evaluate_full_instructor(retriever, queries, corpus, qrels,
                                                                instruction = instruction, doc_instruction = doc_instruction,
                                                                evaluate_with_instruction = False)

        metrics.append([f"instructor-{model_size}", "without", ndcg["NDCG@10"],recall["Recall@10"]])
        print('result without instruction')
        print(ndcg, recall)

        # reloading is necessary here, because evaluate_full_instructor changes the dataset dict.
        corpus, queries, qrels = GenericDataLoader(data_folder=f'full/{dataset}').load(split=split)

        ndcg, _map, recall, precision = evaluate_full_instructor(retriever, queries, corpus, qrels,
                                                                instruction = instruction, doc_instruction = doc_instruction,
                                                                evaluate_with_instruction = True)

        metrics.append([f"instructor-{model_size}", "with", ndcg["NDCG@10"],recall["Recall@10"]])
        print('result with instruction')
        print(ndcg, recall)
        print(instruction, doc_instruction)
    
    return metrics


def evaluate_all_models(dataset, bge_version = 1.5,
                        doc_instruction = 'Represent the following text to end an unfinished story',
                        split ='dev',
                        reranker = None,
                        batch_size = 128):
    results = []
    # results.extend(evaluate_at_once(dataset, split = split, batch_size = batch_size)) 
    # results.extend(evaluate_bge(dataset, version = bge_version, split = split))
    results.extend(evaluate_at_once_instructor(dataset,
                                doc_instruction = doc_instruction, split = split)) # not passing in the batch size as defined in class for different sizes
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
        results = evaluate_all_models(dataset, bge_version = "m3",
                                doc_instruction = doc_instruction, split = split,
                                reranker = None,
                                batch_size = 128)
        
        with open(f"{result_path}/ST-models.pkl", 'wb') as file:
            pickle.dump(results, file)