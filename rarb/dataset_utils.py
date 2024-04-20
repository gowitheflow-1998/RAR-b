import pandas as pd
import os
import json

def load_reasoning_raw(dataset_name:str, 
              split:str = 'dev'):
    
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
    
    if dataset_name != "TempReason":
        frame = pd.read_csv(f'./mcr/{dataset_name}-{split}.csv',converters={'documents': eval})
    elif dataset_name == "TempReason":
        frame = pd.read_csv(f'./mcr/TempReason/{split}-{level}-{setting}.csv',converters={'documents': eval})
    instructions = frame['instructions'].values
    queries = frame['queries'].values
    documents = frame['documents'].values
    labels = frame['labels'].values
    return instructions, queries, documents, labels

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