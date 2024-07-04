# Evaluate RAR-b with MTEB

RAR-b is now available with the MTEB framework! Here, we provide a simple demo regarding the way to flexibly define your `customized models`, and pass in needed arguments (especially complicated settings regarding `instructions`) to evaluate with MTEB.

Under this page, we define model classes to make RAR-b evaluated models compatible with MTEB framework, such as [`InstructionSentenceTransformer`](https://github.com/gowitheflow-1998/RAR-b/blob/main/scripts/mteb/mteb_rarb_models/instruct_st.py) and [`InstructionInstructor`](https://github.com/gowitheflow-1998/RAR-b/blob/main/scripts/mteb/mteb_rarb_models/instruct_instructor.py) which can easily customize your query instructions, doc instructions. If not customized, the model will take in the instructions on MTEB defined by us (the original ones used in the RAR-b paper).

Below, we provide all demos evaluating RAR-b with MTEB for **SentenceTransformer**, **Instructor**, **E5Mistral**, **GritLM**, **Cohere** and **OpenAI**.

## Sentence Transformers Demo
Taking the strong baseline model, the unsupervised Contriever as example. If we don't need instructions at all:

```python
from mteb.models.instructions import task_to_instruction
from mteb_rarb_models.instruct_st import InstructionSentenceTransformer

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
## Instructor Demo

<details>
  <summary><i>Full Demo</i></summary>
&nbsp;

For Instructor, the without-instruction setting is similar, but for with-instruction setting, pass in `do_doc_instruction = True` to leverage document instructions. 

Similarly, though we have a few defined document instructions, feel free to define your optimal ones like above.

```python
from mteb_rarb_models.instruct_st import InstructionInstructor

model_name = 'hkunlp/instructor-XL'
model = InstructionInstructor(model_name, 
                              do_instruction=True, do_doc_instruction=True)

task_names = ["TempReasonL1"]
tasks = mteb.get_tasks(tasks=task_names)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, batch_size = 16
                        output_folder=f"results-leaderboard/{model_name}/RAR-b-w-inst")
```
                   
</details>

## E5Mistral Demo

<details>
  <summary><i>Full Demo</i></summary>
&nbsp;

</details>

## GritLM Demo

<details>
  <summary><i>Full Demo</i></summary>
&nbsp;

</details>

## Cohere Demo

<details>
  <summary><i>Full Demo</i></summary>
&nbsp;

</details>

## OpenAI Demo

<details>
  <summary><i>Full Demo</i></summary>
&nbsp;

</details>
