# Step 1: Load the ACORD dataset

from datasets import load_dataset

# Load corpus, queries, and qrels from the 'train' split
ds = load_dataset("theatticusproject/acord")
corpus = {item['id']: item['text'] for item in ds['corpus']}
queries = {item['id']: item['text'] for item in ds['query']}
qrels = ds['qrels']  # dict: query_id -> {doc_id: relevance}

# Step 2: Implement BM25 baseline

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import numpy as np

# Tokenize corpus
tokenized_corpus = [word_tokenize(corpus[doc_id].lower()) for doc_id in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Retrieve top-k for a query
def bm25_retrieve(query, k=5):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_k = np.argsort(scores)[::-1][:k]
    doc_ids = list(corpus.keys())
    return [(doc_ids[i], scores[i]) for i in top_k]

# Example usage
qid, qtext = list(queries.items())[0]
top_docs = bm25_retrieve(qtext)
print("Top BM25 docs:", top_docs)

# Step 3: Implement SBERT dense retriever

from sentence_transformers import SentenceTransformer, util

# Encode corpus and queries using SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_ids = list(corpus.keys())
corpus_texts = list(corpus.values())
corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True)

def sbert_retrieve(query, k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=k)[0]
    return [(corpus_ids[hit['corpus_id']], hit['score']) for hit in hits]

# Example usage
top_dense = sbert_retrieve(qtext)
print("Top SBERT docs:", top_dense)
