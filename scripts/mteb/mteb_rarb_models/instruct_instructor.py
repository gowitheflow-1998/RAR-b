from InstructorEmbedding import InstructorTransformer, InstructorPooling, Instructor
from sentence_transformers.util import snapshot_download, import_from_string
from collections import OrderedDict
import os
import json
from mteb.models.instructions import task_to_instruction


class InstructionInstructor(Instructor):

    def __init__(self, model_name_or_path: str, 
                 q_instruction: str = "", 
                 doc_instruction: str = "", 
                 do_instruction = False,
                 do_q_instruction = True,
                 do_doc_instruction = False,
                 **kwargs):
        
        super().__init__(model_name_or_path, **kwargs)
        self.q_instruction = q_instruction # can be self defined, if empty string, mteb defined is used in encode_queries
        self.doc_instruction = doc_instruction # can be self defined, if empty string, mteb defined is used in encode_corpus

        self.do_q_instruction = do_q_instruction
        self.do_doc_instruction = do_doc_instruction
        self.do_instruction = do_instruction
        
        # overwrite if in general False
        if self.do_instruction == False:
            self.do_doc_instruction = False
            self.do_q_instruction = False

    def encode_queries(self, queries, 
                       prompt_name:str= None,
                       **kwargs):
        
        if self.do_q_instruction:
            if self.q_instruction != "":  
                queries = [[instruction,query] for query in queries]
            else:
                instruction = task_to_instruction(
                    prompt_name
                )
                queries = [[instruction,query] for query in queries]
        else:
            queries = [["",query] for query in queries]
        print(self.device)
        return self.encode(queries, device = self.device, **kwargs)
    
    def encode_corpus(self, corpus, 
                      prompt_name:str= None,
                      **kwargs):
        
        sentences = [doc["text"].strip() for doc in corpus]

        if self.do_doc_instruction:
            if self.doc_instruction != "":
                sentences = [[instruction,sentence] for sentence in sentences]
            else: 
                # instruction = task_to_instruction(
                #     prompt_name, is_query = False
                # )
                instruction = doc_instruction_lookup[prompt_name]
                sentences = [[instruction,sentence] for sentence in sentences]    
        
        else:
            sentences = [["",sentence] for sentence in sentences]  
        return self.encode(sentences, device = self.device, **kwargs)
    
    def _load_sbert_model(self, model_path, token=None, cache_folder=None, revision=None, trust_remote_code=False, local_files_only=False, **kwargs):
        """
        Loads a full sentence-transformers model
        """
        # Taken mostly from: https://github.com/UKPLab/sentence-transformers/blob/66e0ee30843dd411c64f37f65447bb38c7bf857a/sentence_transformers/util.py#L544
        download_kwargs = {
            "repo_id": model_path,
            "revision": revision,
            "library_name": "sentence-transformers",
            "token": token,
            "cache_dir": cache_folder,
            # "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "tqdm_class": None,  # disabled_tqdm is not defined, set to None
        }
        model_path = snapshot_download(**download_kwargs)

        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = os.path.join(
            model_path, "config_sentence_transformers.json"
        )
        if os.path.exists(config_sentence_transformers_json_path):
            with open(
                config_sentence_transformers_json_path, encoding="UTF-8"
            ) as config_file:
                self._model_config = json.load(config_file)

        # Check if a readme exists
        model_card_path = os.path.join(model_path, "README.md")
        if os.path.exists(model_card_path):
            try:
                with open(model_card_path, encoding="utf8") as config_file:
                    self._model_card_text = config_file.read()
            except:
                pass

        # Load the modules of sentence transformer
        modules_json_path = os.path.join(model_path, "modules.json")
        with open(modules_json_path, encoding="UTF-8") as config_file:
            modules_config = json.load(config_file)

        modules = OrderedDict()
        for module_config in modules_config:
            if module_config["idx"] == 0:
                module_class = InstructorTransformer
            elif module_config["idx"] == 1:
                module_class = InstructorPooling
            else:
                module_class = import_from_string(module_config["type"])
            module = module_class.load(os.path.join(model_path, module_config["path"]))
            modules[module_config["name"]] = module

        return modules

doc_instruction_lookup = {     
    "AlphaNLI": 'Represent the following text that leads to the end of a story',
    "HellaSwag": 'Represent the following text to end an unfinished story',
    "WinoGrande": 'Represent this entity',
    "PIQA": 'represent the solution for a goal',
    "SIQA": 'represent the sentence to answer a question',
    "ARCChallenge": 'Represent the answer to a question',
    "quail": 'Represent the answer to a question given the context',
    "TempReasonL1": "Represent the following month-year.",
    "TempReasonL2Pure": "Represent this answer",
    "TempReasonL2Fact": "Represent this answer",
    "TempReasonL2Context": "Represent this answer",
    "TempReasonL3Pure": "Represent this answer",
    "TempReasonL3Fact": "Represent this answer",
    "TempReasonL3Context": "Represent this answer",
    "SpartQA":"Represent the following answer to a spatial reasoning question",
    "RARbCode":"Represent the answer for a coding problem",
    "RARbMath": "Represent the answer to answer a math problem"}