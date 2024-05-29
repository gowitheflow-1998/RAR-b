import pickle
from rarb.mcr_utils import *
from rarb import *
import random

model_name_lookup = {"E5Mistral":"intfloat/e5-mistral-7b-instruct",
                     "mpnet":"sentence-transformers/all-mpnet-base-v2",
                     "minilm":"sentence-transformers/all-MiniLM-L6-v2",
                     "contriever":"facebook/contriever",
                     "dragon":("facebook/dragon-plus-query-encoder","facebook/dragon-plus-context-encoder"),
                     "bge-small":"BAAI/bge-small-en-v1.5",
                     "bge-base":"BAAI/bge-base-en-v1.5",
                     "bge-large":"BAAI/bge-large-en-v1.5",
                     "bge-m3":"BAAI/bge-m3",
                     "instructor-base":"hkunlp/instructor-base",
                     "instructor-large":"hkunlp/instructor-large",
                     "instructor-XL":"hkunlp/instructor-XL",
                     "openai-3-small":"text-embedding-3-small",
                     "openai-3-large":"text-embedding-3-large",
                     "bge-rerank-base":"BAAI/bge-reranker-base",
                     "bge-rerank-large":"BAAI/bge-reranker-large",
                     "colbert":"colbert-ir/colbertv2.0",
                     "cohere":"cohere",
                     "tart":"facebook/tart-full-flan-t5-xl",
                     "Grit":"GritLM/GritLM-7B",
                     "Mistral-instruct-v1":"mistralai/Mistral-7B-Instruct-v0.1",
                     "Mistral-instruct-v2":"mistralai/Mistral-7B-Instruct-v0.2"}
mode = "bi-encoder" # {"bi-encoder","cross-encoder"}
split = "test"

if mode == "bi-encoder":
    for model_abbr in ["bge-small"]:
        model_name = model_name_lookup[model_abbr]
        if model_abbr == "dragon":
            model = AutoRetriever(model_name, pooling = "cls")
        else:
            model = AutoRetriever(model_name)
        sim = 'dot' if (model_name =='facebook/contriever') or ("dragon" in model_name[0]) else 'cos_sim'
        result_list = []

        for dataset in ["ARC-Challenge"]:
        # for dataset in ["alphanli","hellaswag","physicaliqa","socialiqa","winogrande",
        #                      "ARC-Challenge","csts-easy","csts-hard"]:
            instructions, queries, documents, labels = load_mcr_frame(dataset,split)
            accuracy_list = model.evaluate(instructions, queries, documents, labels, sim = sim,
                                        mode = "both")
            save_list = [f"{dataset}"]
            save_list.extend(accuracy_list)
            print(save_list)
            result_list.append(save_list)

            result_path = f"results/mcr/{dataset}-{split}"
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            
            with open(f"{result_path}/{model_abbr}.pkl", 'wb') as file:
                pickle.dump(result_list, file)

elif mode == "cross-encoder":
    for model_abbr in ["bge-rerank-base"]:
        model_name = model_name_lookup[model_abbr]
        model = AutoCrossEncoder(model_name)
        for dataset in ["ARC-Challenge"]:
        # for dataset in ["ARC-Challenge","physicaliqa","socialiqa","quail","csts-easy","csts-hard","winogrande","alphanli","hellaswag"]:
            instructions, queries, documents, labels = load_mcr_frame(dataset,split)
            _, accuracy_without_instructions = model.evaluate_mcr(queries, documents, labels)
            _, accuracy_with_instructions = model.evaluate_mcr(queries, documents, labels, instructions)
            
            result_list = []
            # print(accuracy_list)
            save_list = [f"{dataset}"]
            save_list.extend([accuracy_without_instructions,accuracy_with_instructions])
            print(save_list)
            result_list.append(save_list)
            print(f"{dataset}")
            print(f'accuracy_without_instructions: {accuracy_without_instructions}')            
            print(f'accuracy_with_instructions: {accuracy_with_instructions}')
            print('')

            result_path = f"results/mcr/{dataset}-{split}"
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            
            with open(f"{result_path}/{model_abbr}.pkl", 'wb') as file:
                pickle.dump(result_list, file)