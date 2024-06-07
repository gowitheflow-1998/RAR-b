## Full-dataset Retrieval

All of our full-dataset retrieval datasets are hosted on [https://huggingface.co/RAR-b](https://huggingface.co/datasets/RAR-b)

All of our datasets for the full-dataset retrieval (full) setting are hosted on [Huggingface](https://huggingface.co/RAR-b). And all the datasets for the multiple-choice setting (mcr) are already in the repo along with git clone (except C-STS)).

Run the following script under the root folder to set up the datasets you want to evaluate with the format for RAR-b evaluation.

```python
import os
import subprocess

for dataset_name in ["alphanli", "piqa", "siqa", "hellaswag", "quail", "ARC-Challenge", "winogrande", "spartqa", 
                    "math-pooled", "humanevalpack-mbpp-pooled", "TempReason-l1", "TempReason-l2-pure", "TempReason-l2-fact", 
                    "TempReason-l2-context", "TempReason-l3-pure", "TempReason-l3-fact", "TempReason-l3-context"]:
    repo_url = f"https://huggingface.co/datasets/RAR-b/{dataset_name}"
    local_path = f"full/{dataset_name}"

    if not os.path.exists(local_path):
        subprocess.run(["git","clone", repo_url, local_path], check=True)
    else:
        print("dataset exists locally")
```