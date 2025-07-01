import os
import re
import csv
import json
import random
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from pprint import pprint
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from sklearn.metrics import ndcg_score, precision_score
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
import mlflow

### Load ACORD dataset
# import pdb;pdb.set_trace()
# dataset = load_dataset("theatticusproject/acord")

# corpus = {item['id']: item['text'] for item in dataset['corpus']}
# queries = {item['id']: item['text'] for item in dataset['query']}
# qrels = defaultdict(dict)
# for row in dataset['qrels']:
#     qrels[row['query-id']][row['corpus-id']] = row['score']  # scores: 0 to 4

# Assuming your base_path is the directory containing corpus.jsonl, queries.jsonl, etc.
base_path = "/Users/rijumone/Nextcloud/WILP/Sem4/Projects/Legal/Datasets/ACORD-dataset"

# Load corpus
corpus_path = os.path.join(base_path, "corpus.jsonl")
with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus = {}
    for line in f:
        item = json.loads(line)
        # import pdb;pdb.set_trace()
        corpus[item['_id']] = item['text']

# Load and shuffle queries
queries_path = os.path.join(base_path, "queries.jsonl")
with open(queries_path, 'r', encoding='utf-8') as f:
    items = [json.loads(line) for line in f]
random.shuffle(items)
# Reconstruct the dictionary with randomized order
queries = {item['_id']: item['text'] for item in items}

# Load qrels (assuming you want to load from 'test.tsv' for example, or you can combine them)
qrels = defaultdict(dict)
qrels_dir = os.path.join(base_path, "qrels")
qrels_files = ["test.tsv", "train.tsv", "valid.tsv"] # Or choose specific ones

for qrels_file in qrels_files:
    current_qrels_path = os.path.join(qrels_dir, qrels_file)
    with open(current_qrels_path, 'r', encoding='utf-8') as f:
        # Assuming TSV format: query-id \t corpus-id \t score
        # Adjust if the header is present or column order is different
        for i, line in enumerate(f):
            if i == 0 and 'tsv' in qrels_file: # Skip header if it exists (common for TSV)
                if line.strip().lower().startswith("query-id"): # Basic check for header
                    continue
            parts = line.strip().split('\t')
            if len(parts) == 3:
                query_id, corpus_id, score = parts
                qrels[query_id][corpus_id] = int(score) # Ensure score is an integer
            else:
                pprint(f"Skipping malformed line in {qrels_file}: {line.strip()}")


def bm25_retrieve(query, k=10):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_k = np.argsort(scores)[::-1][:k]
    return [(corpus_ids[i], scores[i]) for i in top_k]


def sbert_retrieve(query, k=10):
    query_emb = sbert.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, corpus_emb, top_k=k)[0]
    return [(corpus_ids[hit["corpus_id"]], hit["score"]) for hit in hits]



def ollama_rerank(query, candidates, top_k=5, model="llama3:instruct",):
    """
    Args:
        query (str): The legal clause query
        candidates (list): List of (doc_id, text)
    Returns:
        Top-K reranked list of (doc_id, score)
    """

    llm = ChatOllama(
        # model="phi4:latest",
        # model="llama3:instruct",
        model=model,
        base_url="http://192.168.1.11:7869",
    )
    system = SystemMessage(
        content="You are a legal AI trained to evaluate how relevant a clause is to a query."
    )

    clause_list = "\n\n".join([
        f"Clause {i+1} (ID: {cid}):\n{clause.strip()}" 
        for i, (cid, clause) in enumerate(candidates)
    ])

    user_prompt = (
        f"Query:\n{query}\n\n"
        f"Below are {len(candidates)} candidate clauses. "
        "Rate each one from 1 (irrelevant) to 5 (highly relevant) based on how well it answers the query. "
        "Return in the format: Clause <number>: <score>\n\n"
        f"{clause_list}"
    )

    response = llm.invoke([system, HumanMessage(content=user_prompt)])
    content = response.content

    # Parse clause scores from model response
    scores = {}
    for line in content.strip().splitlines():
        match = re.match(r"Clause\s*(\d+)\s*[:\-]?\s*(\d+(\.\d+)?)", line.strip())
        if match:
            idx = int(match.group(1)) - 1
            score = float(match.group(2))
            if 0 <= idx < len(candidates):
                cid, _ = candidates[idx]
                scores[cid] = score

    # Default 0.0 for anything missing
    reranked = [(cid, scores.get(cid, 0.0)) for cid, _ in candidates]
    return sorted(reranked, key=lambda x: -x[1])[:top_k]



def evaluate_with_reranking(
        retrieve_fn, k=5, max_queries=100,reranker=None,
        output_path="results.csv", mlflow_run=None, model="llama3:instruct",
    ):
    '''
    * **NDCG@5 (Normalized Discounted Cumulative Gain)**:
    * Measures **ranking quality**.
    * **P@5 (Precision@5)**:
    * Measures **binary relevance**: how many of the top 5 results are relevant (`score >= 3`).
    '''
    results = []
    ndcg_vals = []
    prec_vals = []
    mrr_vals = []
    hit_vals = []
    recall_vals = []

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for i, (qid, qtext) in enumerate(tqdm(list(queries.items())[:max_queries])):
        true_rels = qrels.get(qid, {})
        if not true_rels:
            continue

        # Step 1: Retrieve
        raw_results = retrieve_fn(qtext, k*2)  # get more for reranker
        candidate_ids = [doc_id for doc_id, _ in raw_results]
        candidate_texts = [(doc_id, corpus[doc_id]) for doc_id in candidate_ids]

        # Step 2: Rerank
        if reranker:
            reranked = reranker(qtext, candidate_texts, top_k=k, model=model)
        else:
            reranked = raw_results[:k]

        # Step 3: Score evaluation
        reranked_ids = [doc_id for doc_id, _ in reranked]
        y_true = [true_rels.get(doc_id, 0) for doc_id in reranked_ids]
        y_score = [1.0 / (rank + 1) for rank in range(len(reranked_ids))]

        y_true_np = np.array([y_true])
        y_score_np = np.array([y_score])
        ndcg_vals.append(ndcg_score(y_true_np, y_score_np, k=k))
        bin_true = [1 if rel >= 3 else 0 for rel in y_true]
        bin_pred = [1] * len(bin_true)
        prec_vals.append(precision_score(bin_true, bin_pred, zero_division=0))

        # MRR@k
        mrr = 0.0
        for rank, rel in enumerate(y_true):
            if rel >= 3:
                mrr = 1.0 / (rank + 1)
                break
        mrr_vals.append(mrr)

        # Hit@k (Success@k)
        hit = 1 if any(rel >= 3 for rel in y_true) else 0
        hit_vals.append(hit)

        # Recall@k
        num_relevant_in_topk = sum(1 for rel in y_true if rel >= 3)
        total_relevant = sum(1 for rel in true_rels.values() if rel >= 3)
        recall = num_relevant_in_topk / total_relevant if total_relevant > 0 else 0.0
        recall_vals.append(recall)

        # Step 4: Save ranking
        for rank, (doc_id, score) in enumerate(reranked):
            results.append({
                "query_id": qid,
                "query_text": qtext,
                "doc_id": doc_id,
                "doc_text": corpus[doc_id],
                "reranker_score": score,
                "true_relevance": true_rels.get(doc_id, 0),
                "rank": rank + 1
            })

    # Write to CSV
    with open(output_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    metrics = {
        f"NDCG@{k}": round(np.mean(ndcg_vals), 3),
        # f"NDCG@{k}_median": round(np.median(ndcg_vals), 3),
        f"P@{k}": round(np.mean(prec_vals), 3),
        # f"P@{k}_median": round(np.median(prec_vals), 3),
        f"MRR@{k}": round(np.mean(mrr_vals), 3),
        # f"MRR@{k}_median": round(np.median(mrr_vals), 3),
        f"Hit@{k}": round(np.mean(hit_vals), 3),
        # f"Hit@{k}_median": round(np.median(hit_vals), 3),
        f"Recall@{k}": round(np.mean(recall_vals), 3),
        # f"Recall@{k}_median": round(np.median(recall_vals), 3),
        "Results File": os.path.abspath(output_path)
    }


    # MLflow logging
    if mlflow_run is not None:
        mlflow.log_metric("NDCG-at-{}".format(k), metrics["NDCG@{}".format(k)])
        # mlflow.log_metric("NDCG-at-{}_median".format(k), metrics["NDCG@{}_median".format(k)])
        mlflow.log_metric("P-at-{}".format(k), metrics["P@{}".format(k)])
        # mlflow.log_metric("P-at-{}_median".format(k), metrics["P@{}_median".format(k)])
        mlflow.log_metric("MRR-at-{}".format(k), metrics["MRR@{}".format(k)])
        # mlflow.log_metric("MRR-at-{}_median".format(k), metrics["MRR@{}_median".format(k)])
        mlflow.log_metric("Hit-at-{}".format(k), metrics["Hit@{}".format(k)])
        # mlflow.log_metric("Hit-at-{}_median".format(k), metrics["Hit@{}_median".format(k)])
        mlflow.log_metric("Recall-at-{}".format(k), metrics["Recall@{}".format(k)])
        # mlflow.log_metric("Recall-at-{}_median".format(k), metrics["Recall@{}_median".format(k)])
        mlflow.log_artifact(output_path)

    return metrics

'''
### Evaluation function
def evaluate(retrieve_fn, k=5, max_queries=100):
    ndcg_vals = []
    prec_vals = []

    for i, (qid, qtext) in enumerate(tqdm(list(queries.items())[:max_queries])):
        true_rels = qrels.get(qid, {})
        if not true_rels:
            continue
        # retrieved = retrieve_fn(qtext, k)
        raw_results = retrieve_fn(qtext, k*2)  # get top-10 before reranking
        retrieved = ollama_rerank(qtext, [(cid, corpus[cid]) for cid, _ in raw_results], top_k=k)

        retrieved_ids = [doc_id for doc_id, _ in retrieved]

        y_true = [true_rels.get(doc_id, 0) for doc_id in retrieved_ids]
        y_score = [1.0 / (rank + 1) for rank in range(len(retrieved_ids))]  # fake scores

        # NDCG assumes relevance grades, precision is binary
        ndcg_vals.append(ndcg_score([y_true], [y_score], k=k))
        bin_true = [1 if rel >= 3 else 0 for rel in y_true]
        bin_pred = [1] * len(bin_true)
        prec_vals.append(precision_score(bin_true, bin_pred, zero_division=0))

    return {
        "NDCG@{}".format(k): np.mean(ndcg_vals),
        "P@{}".format(k): np.mean(prec_vals)
    }
'''

### Run evaluation
if __name__ == "__main__":
    MAX_QUERIES = 20
    # OLLAMA_MODEL = "llama3:instruct"
    # OLLAMA_MODEL = "llama3.2:latest"
    # OLLAMA_MODEL = "gemma3:12b"
    # OLLAMA_MODEL = "phi4-mini:3.8b"
    OLLAMA_MODEL = "qwen2.5:3b"
    # OLLAMA_MODEL = "dolphin-phi:latest"
    
    SBERT_MODEL = "all-MiniLM-L6-v2"
    # SBERT_MODEL = "nlpaueb/legal-bert-base-uncased"
    # SBERT_MODEL = "FacebookAI/roberta-base"

    IS_RERANK = True  # Set to False if you want to run without reranking

    N_K = 5  # Number of top documents to retrieve

    ### BM25 Retriever
    pprint("Building BM25 index...")
    tokenized_corpus = [word_tokenize(corpus[doc_id].lower()) for doc_id in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    corpus_ids = list(corpus.keys())

    

    mlflow.set_experiment("acord_retrieval_eval")

    with mlflow.start_run(run_name=f"BM25_rerank_{IS_RERANK}_{OLLAMA_MODEL}"):
        mlflow.log_param("retriever", "bm25")
        if IS_RERANK:
            mlflow.log_param("reranker", "ollama_rerank")
        mlflow.log_param("k", N_K)
        mlflow.log_param("max_queries", MAX_QUERIES)
        mlflow.log_param("ollama_model", OLLAMA_MODEL)
        mlflow.log_param("sbert_model", SBERT_MODEL)
        pprint("Evaluating BM25...")
        bm25_metrics = evaluate_with_reranking(
            retrieve_fn=bm25_retrieve,
            k=N_K,
            max_queries=MAX_QUERIES,
            reranker=ollama_rerank if IS_RERANK else None,
            output_path="outputs/acord_rerank_results_bm25_retrieve.csv",
            mlflow_run=True,
            model=OLLAMA_MODEL,
        )
        print("BM25:", bm25_metrics)

    with mlflow.start_run(run_name=f"SBERT_rerank_{SBERT_MODEL}_{IS_RERANK}_{OLLAMA_MODEL}"):
        ### SBERT Retriever
        pprint("Encoding corpus with SBERT...")
        sbert = SentenceTransformer(SBERT_MODEL)
        corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
        corpus_emb = sbert.encode(corpus_texts, convert_to_tensor=True)
        
        mlflow.log_param("retriever", "sbert")
        if IS_RERANK:
            mlflow.log_param("reranker", "ollama_rerank")
        mlflow.log_param("k", N_K)
        mlflow.log_param("max_queries", MAX_QUERIES)
        mlflow.log_param("ollama_model", OLLAMA_MODEL)
        mlflow.log_param("sbert_model", SBERT_MODEL)
        pprint("Evaluating SBERT...")
        sbert_metrics = evaluate_with_reranking(
            retrieve_fn=sbert_retrieve,
            k=N_K,
            max_queries=MAX_QUERIES,
            reranker=ollama_rerank if IS_RERANK else None,
            output_path="outputs/acord_rerank_results_sbert_retrieve.csv",
            mlflow_run=True,
            model=OLLAMA_MODEL,
        )
        print("SBERT:", sbert_metrics)
