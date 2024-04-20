from InstructorEmbedding import INSTRUCTOR

class SGPT:
    def __init__(self, model_path: str = None, sep: str = " ", **kwargs):
        self.sep = sep

        self.q_model = INSTRUCTOR(model_path, **kwargs)
        self.doc_model = self.q_model

    def encode_queries(self, queries, batch_size:int = 4, **kwargs):
        return self.q_model.encode(queries, batch_size = batch_size, **kwargs)

    def encode_corpus(self, corpus, batch_size:int = 4, **kwargs):
        sentences = [str(doc["title"].strip() + " " + doc["text"].strip()) for doc in corpus]
        return self.doc_model.encode(sentences, batch_size = batch_size, **kwargs)