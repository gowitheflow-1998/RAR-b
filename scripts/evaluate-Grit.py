from beir.datasets.data_loader import GenericDataLoader
import pickle
from rarb.rarb_models import *
from rarb import *
import random

random.seed(41)


def evaluate_at_once_Grit(dataset,split="test",batch_size = 1):
    f = open(f"full/{dataset}/instruction.txt", "r")
    instruction = f.read()
    f.close()
    metrics= []

    if ("context" in dataset) or ("math" in dataset) or ("mbpp" in dataset) or ("code" in dataset) or ("humaneval" in dataset):
        batch_size = 8
    if ("fact" in dataset):
        batch_size = 32

    corpus, queries, qrels = GenericDataLoader(data_folder=f'full/{dataset}').load(split=split)
    model_name = "GritLM/GritLM-7B"
    print(f'evaluating {model_name}')
    retriever = initialize_retriever(model_name, batch_size=batch_size)
    print(len(corpus))
    ndcg, _map, recall, precision = evaluate_full_Grit(retriever, queries, corpus, qrels,
                                                            instruction = instruction,
                                                            evaluate_with_instruction = False)

    metrics.append([model_name, "without", ndcg["NDCG@10"],recall["Recall@10"]])
    print('result without instruction')
    print(ndcg, recall)

    corpus, queries, qrels = GenericDataLoader(data_folder=f'full/{dataset}').load(split=split)

    ndcg, _map, recall, precision = evaluate_full_Grit(retriever, queries, corpus, qrels,
                                                            instruction = instruction,
                                                            evaluate_with_instruction = True)

    metrics.append([model_name, "with", ndcg["NDCG@10"],recall["Recall@10"]])
    print('result with instruction')
    print(ndcg, recall)
    
    return metrics


def evaluate_all_models(dataset, 
                        split ='dev',
                        batch_size = 64):
    results = []
    # results.extend(evaluate_at_once_E5Mistral(dataset,split = split, batch_size = batch_size))
    results.extend(evaluate_at_once_Grit(dataset,split = split, batch_size=batch_size))
    return results

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
            
        results = evaluate_all_models(dataset,
                                split = split,
                                batch_size = 64)
        
        with open(f"{result_path}/E5-Mistral.pkl", 'wb') as file:
            pickle.dump(results, file)