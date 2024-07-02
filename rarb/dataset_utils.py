import pandas as pd
import os
import json
import logging
from collections import defaultdict
from datasets import Features, Value, load_dataset
from typing import Dict, Tuple
from typing import Optional

logger = logging.getLogger(__name__)

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
    
# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/datasets/data_loader_hf.py#L10
# & https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/AbsTaskRetrieval.py#L21
class HFDataLoader:
    def __init__(
        self,
        hf_repo: Optional[str] = None,
        hf_repo_qrels: Optional[str] = None,
        data_folder: Optional[str] = None,
        prefix: Optional[str] = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
        streaming: bool = False,
        keep_in_memory: bool = False,
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.hf_repo = hf_repo
        if hf_repo:
            self.hf_repo_qrels = hf_repo_qrels if hf_repo_qrels else hf_repo
        else:
            if prefix:
                query_file = prefix + "-" + query_file
                qrels_folder = prefix + "-" + qrels_folder

            self.corpus_file = (
                os.path.join(data_folder, corpus_file) if data_folder else corpus_file
            )
            self.query_file = (
                os.path.join(data_folder, query_file) if data_folder else query_file
            )
            self.qrels_folder = (
                os.path.join(data_folder, qrels_folder) if data_folder else None
            )
            self.qrels_file = qrels_file
        self.streaming = streaming
        self.keep_in_memory = keep_in_memory

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(
                "File {} not present! Please provide accurate file.".format(fIn)
            )

        if not fIn.endswith(ext):
            raise ValueError(
                "File {} must be present with extension {}".format(fIn, ext)
            )

    def load(
        self, split="test"
    ) -> Tuple[Dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
        if not self.hf_repo:
            self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
            self.check(fIn=self.corpus_file, ext="jsonl")
            self.check(fIn=self.query_file, ext="jsonl")
            self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", self.corpus[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        self._load_qrels(split)
        # filter queries with no qrels
        qrels_dict = defaultdict(dict)

        def qrels_dict_init(row):
            qrels_dict[row["query-id"]][row["corpus-id"]] = int(row["score"])

        self.qrels.map(qrels_dict_init)
        self.qrels = qrels_dict
        self.queries = self.queries.filter(lambda x: x["id"] in self.qrels)
        logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
        logger.info("Query Example: %s", self.queries[0])
        self.queries = {query["id"]: query["text"] for query in self.queries}
        self.corpus = {
            doc["id"]: {"title": doc["title"], "text": doc["text"]}
            for doc in self.corpus
        }
        self.qrels = dict(self.qrels)
        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> dict[str, dict[str, str]]:
        if not self.hf_repo:
            self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus))
            logger.info("Doc Example: %s", self.corpus[0])

        return self.corpus

    def _load_corpus(self):
        if self.hf_repo:
            corpus_ds = load_dataset(
                self.hf_repo,
                "corpus",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            corpus_ds = load_dataset(
                "json",
                data_files=self.corpus_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        corpus_ds = next(iter(corpus_ds.values()))  # get first split
        corpus_ds = corpus_ds.cast_column("_id", Value("string"))
        corpus_ds = corpus_ds.rename_column("_id", "id")
        corpus_ds = corpus_ds.remove_columns(
            [
                col
                for col in corpus_ds.column_names
                if col not in ["id", "text", "title"]
            ]
        )
        self.corpus = corpus_ds

    def _load_queries(self):
        if self.hf_repo:
            queries_ds = load_dataset(
                self.hf_repo,
                "queries",
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )
        else:
            queries_ds = load_dataset(
                "json",
                data_files=self.query_file,
                streaming=self.streaming,
                keep_in_memory=self.keep_in_memory,
            )
        queries_ds = next(iter(queries_ds.values()))  # get first split
        queries_ds = queries_ds.cast_column("_id", Value("string"))
        queries_ds = queries_ds.rename_column("_id", "id")
        queries_ds = queries_ds.remove_columns(
            [col for col in queries_ds.column_names if col not in ["id", "text"]]
        )
        self.queries = queries_ds

    def _load_qrels(self, split):
        if self.hf_repo:
            qrels_ds = load_dataset(
                self.hf_repo_qrels,
                keep_in_memory=self.keep_in_memory,
                streaming=self.streaming,
            )[split]
        else:
            qrels_ds = load_dataset(
                "csv",
                data_files=self.qrels_file,
                delimiter="\t",
                keep_in_memory=self.keep_in_memory,
            )
        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            }
        )
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds
        
def task_to_instruction(dataset_name: str = None):
    if dataset_name in ["alphanli", "piqa", "siqa", "hellaswag", "quail", 
                        "ARC-Challenge", "winogrande", "spartqa", 
                        "math-pooled", "humanevalpack-mbpp-pooled", 
                        "TempReason-l1", "TempReason-l2-pure", "TempReason-l2-fact", 
                        "TempReason-l2-context", "TempReason-l3-pure", "TempReason-l3-fact", 
                        "TempReason-l3-context"]:
        task2instruct = {
        "winogrande": "Given the following sentence, retrieve an appropriate answer to fill in the missing underscored part.",
        "ARC-Challenge": "Retrieve the answer to the question.",
        "alphanli": "Given the following start and end of a story, retrieve a possible reason that leads to the end.",
        "hellaswag": "Given the following unfinished context, retrieve the most plausible ending to finish it.",
        "piqa": "Given the following goal, retrieve a possible solution.",
        "quail": "Given the following context and question, retrieve the correct answer.",
        "siqa": "Given the following context and question, retrieve the correct answer.",
        "humanevalpack-mbpp-pooled": "Retrieve the answer for the following coding problem.",
        "math-pooled": "Retrieve the answer for the following math problem.",
        "spartqa": "Given the following spatial reasoning question, retrieve the right answer.",
        "TempReason-l1": "Given the following question about time, retrieve the correct answer.",
        "TempReason-l2-pure": "Given the following question, retrieve the correct answer.",
        "TempReason-l2-fact": "Given the following question and facts, retrieve the correct answer.",
        "TempReason-l2-context": "Given the following question, facts and contexts, retrieve the correct answer.",
        "TempReason-l3-pure": "Given the following question, retrieve the correct answer.",
        "TempReason-l3-fact": "Given the following question and facts, retrieve the correct answer.",
        "TempReason-l3-context": "Given the following question, facts and contexts, retrieve the correct answer.",
        }
        instruction = task2instruct[dataset_name]
    else:
        instruction = ""
        print("task name not defined in RAR-b, please define in def task_to_instruction()")
    return instruction