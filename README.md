# RAR-b


Official repo of [*RAR-b: Reasoning as Retrieval Benchmark*](https://arxiv.org/abs/2404.06347)

## Updates
**[April 15, 2024]** Uploaded all rar-b processed datasets and utils. Added a demo for evaluating with Grit.

**[April 9, 2024]** We released the [RAR-b paper](https://arxiv.org/abs/2404.06347). 

## Installation

```
git clone https://github.com/gowitheflow-1998/RAR-b.git
cd RAR-b
pip install -e .
```
## Download Datasets

All of our datasets for the full-dataset retrieval (full) setting are hosted on [Huggingface](https://huggingface.co/RAR-b). And all the datasets for the multiple-choice setting (mcr) are already in the repo along with git clone.

Run the following script under the root folder to set up the datasets you want to evaluate with the format for RAR-b evaluation.

For example, preparing for ARC-Challenge and PIQA:

```python
import os
import subprocess

for dataset_name in ["ARC-Challenge","piqa"]:
    repo_url = f"https://huggingface.co/datasets/RAR-b/{dataset_name}"
    local_path = f"full/{dataset_name}"

    if not os.path.exists(local_path):
        subprocess.run(["git","clone", repo_url, local_path], check=True)
    else:
        print("dataset exists locally")
```
## Evaluation

Check out the `scripts` folder to reproduce evaluation results in RAR-b paper. For example, evaluate BGE models:

Under the `root folder`, run:
```
python scripts/evaluate-BGE.py
```

## Demo

Easily customize the evaluation of models using similar structure.

Below is an example with Grit, evaluated for both without and with instructions:

```python
from beir.datasets.data_loader import GenericDataLoader
from rarb.rarb_models import initialize_retriever
from rarb import evaluate_full_Grit

dataset = "ARC-Challenge"
split = "test"
model_name = "GritLM/GritLM-7B"

f = open(f"full/{dataset}/instruction.txt", "r")
instruction = f.read()

metrics = []

corpus, queries, qrels = GenericDataLoader(data_folder=f'full/{dataset}').load(split=split)
retriever = initialize_retriever(model_name, batch_size=16)

# evaluating without instructions:
ndcg, _map, recall, precision = evaluate_full_Grit(retriever, queries, corpus, qrels,
                                                            instruction = instruction,
                                                            evaluate_with_instruction = False)

metrics.append([model_name, "without", ndcg["NDCG@10"],recall["Recall@10"]])
print('results without instructions:')
print(ndcg, recall)

# evaluating with instructions:
ndcg, _map, recall, precision = evaluate_full_Grit(retriever, queries, corpus, qrels,
                                                        instruction = instruction,
                                                        evaluate_with_instruction = True)

metrics.append([model_name, "with", ndcg["NDCG@10"],recall["Recall@10"]])
print('results with instructions:')
print(ndcg, recall)
```
