from sentence_transformers import SentenceTransformer
from mteb.models.instructions import task_to_instruction

class InstructionSentenceTransformer(SentenceTransformer):

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
                queries = [(self.q_instruction + " " + query).strip() for query in queries]
            else:
                instruction = task_to_instruction(
                    prompt_name
                )
                queries = [(instruction + " " + sentence).strip() for sentence in queries]

        return self.encode(queries, **kwargs)
    
    def encode_corpus(self, corpus, 
                      prompt_name:str= None,
                      **kwargs):
        
        sentences = [doc["text"].strip() for doc in corpus]

        if self.do_doc_instruction:
            if self.doc_instruction != "":
                sentences = [(self.instruction + " " + sentence).strip() for sentence in sentences]
            else: 
                instruction = task_to_instruction(
                    prompt_name, is_query = False
                )
                sentences = [(instruction + " " + sentence).strip() for sentence in sentences]    

        return self.encode(sentences, **kwargs)