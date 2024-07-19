# RAR-b


Official repo of [*RAR-b: Reasoning as Retrieval Benchmark*](https://arxiv.org/abs/2404.06347)

## Updates
**[July 2, 2024]**  New dataset/instruction utils; RAR-b has been integrated to [MTEB](https://github.com/embeddings-benchmark/mteb/tree/main) with [leaderboard](https://huggingface.co/spaces/mteb/leaderboard?task=retrieval&language=rar-b). Please submit your evaluation to RAR-b!

**[April 15, 2024]** All RAR-b processed datasets, utils and evaluation scripts are open-sourced.

**[April 9, 2024]**  We released the [RAR-b paper](https://arxiv.org/abs/2404.06347). 

## Installation

```
git clone https://github.com/gowitheflow-1998/RAR-b.git
cd RAR-b
pip install -e .
```

For flexibility of different use cases, we don't put any library requirements in the repo. Based on the specific classes of models you want to evaluate, you might at least want to install some `pytorch`, `sentence-transformers`, `beir`, and if needed, `cohere` and `openai`.

## Download Datasets

All of our datasets for the full-dataset retrieval (full) setting are hosted on [Huggingface](https://huggingface.co/RAR-b). And all the datasets for the multiple-choice setting (mcr) are already in the repo along with git clone (except CSTS, which we provide the code to reproduce in `mcr/create_csts.py`, detailed in `mcr/README.md`).

We provide the adapted `HFDataLoader` to load the Full setting datasets from Huggingface; and getting the task-specific default instruction with `task_to_instruction` mapping - feel free to define the best instruction for your model if the default one is not the optimal!
```python
from rarb import HFDataLoader, task_to_instruction
dataset = "winogrande"
corpus, queries, qrels = HFDataLoader(f"RAR-b/{dataset}").load(split = "test")
instruction = task_to_instruction(dataset)
```
Alternatively, you can git clone the dataset and set them up locally under the `full` folder - We provide the demo for this option at `full/README.md`.


## Evaluation

Check out the `scripts` folder to reproduce evaluation results in RAR-b paper. For example, evaluate BGE models:

Under the `root folder`, run:
```
python scripts/evaluate-BGE.py
```

## Demo

Easily customize the evaluation of models using similar structure, by modifying relevant utils used in the following evaluation pipeline.

Below is an example with Grit, evaluated for both without and with instructions:

```python
from rarb import HFDataLoader, task_to_instruction
from rarb.rarb_models import initialize_retriever
from rarb import evaluate_full_Grit

dataset = "ARC-Challenge"
split = "test"
model_name = "GritLM/GritLM-7B"

instruction = task_to_instruction(dataset)

metrics = []

retriever = initialize_retriever(model_name, batch_size=16)

corpus, queries, qrels = HFDataLoader(f"RAR-b/{dataset}").load(split = "test")
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
