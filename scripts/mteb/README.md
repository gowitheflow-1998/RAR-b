# Evaluate RAR-b with MTEB

RAR-b is now available on the MTEB framework! Here, we provide simple demo regarding the way to flexibly define your `customized models`, and pass in needed arguments (especially complicated settings regarding `instructions`) to evaluate with MTEB.

Under the mteb directory, we define a simple [`InstructionSentenceTransformer`](https://github.com/gowitheflow-1998/RAR-b/blob/main/scripts/mteb/rarb_model/instruct_st.py) which can easily customize your query instructions, doc instructions. If not customized, the model will take in the instructions on MTEB defined by us (the original ones used in the RAR-b paper).

Taking the strong baseline model, the unsupervised Contriever as example. If we don't need instructions at all:

```python
from mteb.models.instructions import task_to_instruction
from rarb_model.instruct_st import InstructionSentenceTransformer

model_name = "facebook/contriever"
model = InstructionSentenceTransformer(model_name)
```

If we do, by setting `do_instruction = True`, query instructions will be prepended according to `prompt_name` mapping we define on MTEB.
```python
model = InstructionSentenceTransformer(model_name, do_instruction = True)
```

If document instructions work the best for your model, by setting `do_instruction = True`, doc instructions will also be prepended according to `prompt_name` mapping we define on MTEB.
```python
model = InstructionSentenceTransformer(model_name, do_instruction = True, do_doc_instruction = True)
```

Feel free to define your own instructions. For example, your own `doc_instruction`.

```python
model = InstructionSentenceTransformer(model_name, do_instruction = True, do_doc_instruction = True,
                                       doc_instruction = "Represent this date to answer a temporal reasoning question.")
```

Finally, use the simple MTEB evaluation pipeline. Don't forget to pass in `score_function = "dot"` as we are using Contriever!
```python
task_names = ["TempReasonL1"]
tasks = mteb.get_tasks(tasks=task_names)

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, score_function = "dot",
                         output_folder=f"results-leaderboard/{model_name}/RAR-b-wo-inst")
```
