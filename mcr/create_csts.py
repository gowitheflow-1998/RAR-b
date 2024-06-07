import pandas as pd


def make_mcr_frame(instructions, queries, documents, labels):
    mcr = pd.DataFrame()
    mcr['instructions'] = instructions
    mcr['queries'] = queries
    mcr['documents'] = documents
    mcr['labels'] = labels
    return mcr

def create_candidates(sentence1, sentence2, condition, mode = 'sentence2-as-candidate'):
    if mode == 'sentence2-as-candidate':
        candidates = []
        for index, n in enumerate(sentence2):

            s2 = sentence2[index]
            c = condition[index]

            # notably, we do instrution+sentence2 as condidate, while keep s1 the same.
            # because in dual encoders, switching their order is the same.
            instructions = [f'In terms of "{x[:-1]}", retrieve a sentence similar to the following.' for x in c]
            candidate = [i + ' ' + s2 for i in instructions]
            candidates.append(candidate)
        queries = sentence1
        conditioned_document = candidates
    elif mode == 'sentence1-as-candidate':
        candidates = []
        for index, n in enumerate(sentence2):

            s1 = sentence1[index]
            c = condition[index]

            # notably, we do instrution+sentence2 as condidate, while keep s1 the same.
            # because in dual encoders, switching their order is the same.
            instructions = [f'In terms of "{x[:-1]}", retrieve a sentence similar to the following.' for x in c]
            candidate = [i + ' ' + s1 for i in instructions]
            candidates.append(candidate)
        queries = sentence2
        conditioned_document = candidates
            
    return queries, conditioned_document


frame = pd.read_csv('./c-sts/csts_validation.csv')
frame = frame[~frame.duplicated(subset=['sentence1', 'sentence2', 'label'], keep=False)]
grouped = frame.groupby(['sentence1', 'sentence2']).agg({'condition': list, 'label': list}).reset_index()
# the following is also logically necessary but all rows meet the condition 
grouped = grouped[grouped['label'].apply(len) > 1]
sentence1 = grouped['sentence1'].values
sentence2 = grouped['sentence2'].values
condition = grouped['condition'].values
label = grouped['label'].values
# the following is logically necessary but it seems like groupby already makes greater integers in the front 
binary_label = [[1, 0] if label[0] > label[1] else [0, 1] for label in label]
grouped['binary label'] = binary_label


easy_queries, conditioned_documents= create_candidates(sentence1, sentence2, condition, mode = 'sentence2-as-candidate')

queries = []
documents = []

for q,d in zip(easy_queries,conditioned_documents):
    queries.append(q)
    documents.append(d)
    
labels = [0]*len(queries)

instruction = "Retrieve an aspect and a sentence which are similar to the following sentence."
instructions = [instruction]*len(queries)

mcr = make_mcr_frame(instructions, queries, documents, labels)
mcr.to_csv(f'./csts-easy-test.csv',index=False)


queries = []
documents = []

for s1,s2,c in zip(sentence1,sentence2,condition):
    queries.append(f"Sentence1: {s1}; Sentence2: {s2}")
    documents.append(c)
    
labels = [0]*len(queries)

instruction = "Retrieve a condition under which the following two sentences are similar."
instructions = [instruction]*len(queries)

mcr = make_mcr_frame(instructions, queries, documents, labels)
mcr.to_csv(f'./csts-hard-test.csv',index=False)