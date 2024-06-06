import cohere
from functools import partial
from tqdm import tqdm
import numpy as np

def evaluate_full(retriever, queries, documents, qrels,
                  instruction = None,
                  evaluate_with_instruction = False,
                  do_rerank = False,
                  rerank_top_k=100,
                  reranker_batch_size = 32,
                  reranker = None,
                  api_key = None
                  ):
    if evaluate_with_instruction == False:
        results = retriever.retrieve(documents, queries)
    else:
        # prepend every query with instruction
        qid = [k for k,v in queries.items()]
        for id in qid:
            queries[id] = instruction + ' ' + queries[id]
        results = retriever.retrieve(documents, queries)

    if do_rerank:
        print("doing reranking")
        def get_top_k_documents(doc_similarity_dict, k=rerank_top_k):
            # Sort the dictionary by similarity score in descending order
            sorted_docs = sorted(doc_similarity_dict.items(), key=lambda x: x[1], reverse=True)
            # Keep the top k items
            top_k_docs = dict(sorted_docs[:k])
            return top_k_docs
        retriever.k_values = [value for value in retriever.k_values if value <= rerank_top_k]
        original_results = [(q_id, doc_scores) for q_id, doc_scores in results.items()]
        reranked_results = {}
        
        if reranker == "cohere":
            co = cohere.Client(api_key)
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                response = co.rerank(
                    model = 'rerank-english-v2.0',
                    query = q,
                    documents = docs)
                cohere_index = [r.index for r in response.results]
                reranked_doc_ids = [doc_ids[ind] for ind in cohere_index]
                pseudo_sim_scores = [float(num) for num in np.linspace(1, 0, rerank_top_k)]
                reranked_dict = {}
                for doc_id, similarity_score in zip(reranked_doc_ids, pseudo_sim_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
        else:
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                # print(q)
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                # reranker prediction:
                pairs = [[q, doc] for doc in docs]
                # print(pairs[0])
                similarity_scores = reranker.predict(pairs, batch_size=reranker_batch_size)
                similarity_scores = [float(similarity_score) for similarity_score in similarity_scores]
                reranked_dict = {}
                for doc_id, similarity_score in zip(doc_ids, similarity_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
            
        ndcg, _map, recall, precision = retriever.evaluate(qrels, reranked_results, retriever.k_values)
    else:
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

def evaluate_full_instructor(retriever, queries, documents, qrels, 
                             instruction = None, doc_instruction = None,
                             evaluate_with_instruction = False,
                             do_rerank = False,
                             rerank_top_k=100,
                             reranker_batch_size = 32,
                             reranker = None,
                             api_key = None):
    if evaluate_with_instruction == False:
        for key in queries.keys():
          queries[key] = ['', queries[key]]
    else:
        # prepend query with instruction in instructOR format
        for key in queries.keys():
          queries[key] = [instruction, queries[key]]
        for key in documents.keys():
          documents[key]['title'] = doc_instruction if doc_instruction is not None else ""
    results = retriever.retrieve(documents, queries)

    if do_rerank:
        print("doing reranking")
        def get_top_k_documents(doc_similarity_dict, k=rerank_top_k):
            # Sort the dictionary by similarity score in descending order
            sorted_docs = sorted(doc_similarity_dict.items(), key=lambda x: x[1], reverse=True)
            # Keep the top k items
            top_k_docs = dict(sorted_docs[:k])
            return top_k_docs
        retriever.k_values = [value for value in retriever.k_values if value <= rerank_top_k]
        original_results = [(q_id, doc_scores) for q_id, doc_scores in results.items()]
        reranked_results = {}
        
        if reranker == "cohere":
            co = cohere.Client(api_key)
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                response = co.rerank(
                    model = 'rerank-english-v2.0',
                    query = q,
                    documents = docs)
                cohere_index = [r.index for r in response.results]
                reranked_doc_ids = [doc_ids[ind] for ind in cohere_index]
                pseudo_sim_scores = [float(num) for num in np.linspace(1, 0, rerank_top_k)]
                reranked_dict = {}
                for doc_id, similarity_score in zip(reranked_doc_ids, pseudo_sim_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
        else:
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                # print(q)
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                # reranker prediction:
                pairs = [[q, doc] for doc in docs]
                # print(pairs[0])
                similarity_scores = reranker.predict(pairs, batch_size=reranker_batch_size)
                similarity_scores = [float(similarity_score) for similarity_score in similarity_scores]
                reranked_dict = {}
                for doc_id, similarity_score in zip(doc_ids, similarity_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
            
        ndcg, _map, recall, precision = retriever.evaluate(qrels, reranked_results, retriever.k_values)
    else:
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

def evaluate_full_E5Mistral(retriever, queries, documents, qrels,
                             instruction = None,
                             evaluate_with_instruction = False,
                             do_rerank = False,
                             rerank_top_k=100,
                             reranker_batch_size = 32,
                             reranker = None,
                             api_key = None):
    if evaluate_with_instruction == False:
        for key in queries.keys():
          queries[key] = f'Instruct: \nQuery: {queries[key]}'
    else:
        for key in queries.keys():
          queries[key] = f'Instruct: {instruction}\nQuery: {queries[key]}'
    print(queries[key])

    results = retriever.retrieve(documents, queries)
    if do_rerank:
        print("doing reranking")
        def get_top_k_documents(doc_similarity_dict, k=rerank_top_k):
            # Sort the dictionary by similarity score in descending order
            sorted_docs = sorted(doc_similarity_dict.items(), key=lambda x: x[1], reverse=True)
            # Keep the top k items
            top_k_docs = dict(sorted_docs[:k])
            return top_k_docs
        retriever.k_values = [value for value in retriever.k_values if value <= rerank_top_k]
        original_results = [(q_id, doc_scores) for q_id, doc_scores in results.items()]
        reranked_results = {}
        
        if reranker == "cohere":
            co = cohere.Client(api_key)
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                response = co.rerank(
                    model = 'rerank-english-v2.0',
                    query = q,
                    documents = docs)
                cohere_index = [r.index for r in response.results]
                reranked_doc_ids = [doc_ids[ind] for ind in cohere_index]
                pseudo_sim_scores = [float(num) for num in np.linspace(1, 0, rerank_top_k)]
                reranked_dict = {}
                for doc_id, similarity_score in zip(reranked_doc_ids, pseudo_sim_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
        else:
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                print(q)
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                # reranker prediction:
                pairs = [[q, doc] for doc in docs]
                print(pairs[0])
                similarity_scores = reranker.predict(pairs, batch_size=reranker_batch_size)
                similarity_scores = [float(similarity_score) for similarity_score in similarity_scores]
                reranked_dict = {}
                for doc_id, similarity_score in zip(doc_ids, similarity_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
            
        ndcg, _map, recall, precision = retriever.evaluate(qrels, reranked_results, retriever.k_values)
    else:
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision


def evaluate_full_Grit(retriever, queries, documents, qrels,
                             instruction = None,
                             evaluate_with_instruction = False,
                             do_rerank = False,
                             rerank_top_k=100,
                             reranker_batch_size = 32,
                             reranker = None,
                             api_key = None):
    
    if evaluate_with_instruction == False:
        retriever.retriever.model.encode_queries = partial(retriever.retriever.model.encode_queries,
                                                 instruction=gritlm_instruction(""))
    else:
        retriever.retriever.model.encode_queries = partial(retriever.retriever.model.encode_queries, 
                                                 instruction=gritlm_instruction(instruction))
    retriever.retriever.model.encode_corpus = partial(retriever.retriever.model.encode_corpus, 
                                                instruction=gritlm_instruction(""))    
    results = retriever.retrieve(documents, queries)
    if do_rerank:
        print("doing reranking")
        def get_top_k_documents(doc_similarity_dict, k=rerank_top_k):
            # Sort the dictionary by similarity score in descending order
            sorted_docs = sorted(doc_similarity_dict.items(), key=lambda x: x[1], reverse=True)
            # Keep the top k items
            top_k_docs = dict(sorted_docs[:k])
            return top_k_docs
        retriever.k_values = [value for value in retriever.k_values if value <= rerank_top_k]
        original_results = [(q_id, doc_scores) for q_id, doc_scores in results.items()]
        reranked_results = {}
         
        if reranker == "cohere":
            co = cohere.Client(api_key)
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                response = co.rerank(
                    model = 'rerank-english-v2.0',
                    query = q,
                    documents = docs)
                cohere_index = [r.index for r in response.results]
                reranked_doc_ids = [doc_ids[ind] for ind in cohere_index]
                pseudo_sim_scores = [float(num) for num in np.linspace(1, 0, rerank_top_k)]
                reranked_dict = {}
                for doc_id, similarity_score in zip(reranked_doc_ids, pseudo_sim_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
        else:
            for result_pair in tqdm(original_results):
                sorted_top_k = get_top_k_documents(result_pair[1])
                doc_ids = [doc_id for doc_id, _ in sorted_top_k.items()]
                q = queries[result_pair[0]]
                print(q)
                docs = [(documents[doc_id]['title'] + " " +documents[doc_id]['text']).strip() for doc_id in doc_ids]
                # reranker prediction:
                pairs = [[q, doc] for doc in docs]
                print(pairs[0])
                similarity_scores = reranker.predict(pairs, batch_size=reranker_batch_size)
                similarity_scores = [float(similarity_score) for similarity_score in similarity_scores]
                reranked_dict = {}
                for doc_id, similarity_score in zip(doc_ids, similarity_scores):
                    reranked_dict[doc_id] = similarity_score
                reranked_results[result_pair[0]] = reranked_dict
            
        ndcg, _map, recall, precision = retriever.evaluate(qrels, reranked_results, retriever.k_values)
    else:
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    return ndcg, _map, recall, precision

def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

