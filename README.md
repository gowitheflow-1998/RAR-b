# RAR-b


Official repo of [*RAR-b: Reasoning as Retrieval Benchmark*](https://arxiv.org/abs/2404.06347)

## Updates
**[April 15, 2024]** Uploaded all rar-b processed datasets and utils. Added an example below for evaluating with Grit. Still Updating scripts to run evaluation for all models. Stay tuned!

**[April 9, 2024]** We released the [RAR-b paper](https://arxiv.org/abs/2404.06347). 

## Installation

```
git clone https://github.com/gowitheflow-1998/RAR-b.git
cd RAR-b
pip install e .
```

## Demo

```python
from beir.datasets.data_loader import GenericDataLoader
from rarb.rarb_models import initialize_retriever, evaluate_full_Grit

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
