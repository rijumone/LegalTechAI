"""
Finetuning SBERT on ACORD dataset using GPU
"""
import os
import json
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Paths
base_path = "/Users/rijumone/Nextcloud/WILP/Sem4/Projects/Legal/Datasets/ACORD-dataset"
corpus_path = os.path.join(base_path, "corpus.jsonl")
queries_path = os.path.join(base_path, "queries.jsonl")
qrels_dir = os.path.join(base_path, "qrels")
qrels_files = ["train.tsv", "valid.tsv"]  # Use train/valid for finetuning

# Load corpus
def load_corpus(path):
    corpus = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            corpus[item['_id']] = item['text']
    return corpus

# Load queries
def load_queries(path):
    with open(path, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
    random.shuffle(items)
    return {item['_id']: item['text'] for item in items}

# Load qrels
def load_qrels(qrels_dir, qrels_files):
    from collections import defaultdict
    qrels = defaultdict(dict)
    for qrels_file in qrels_files:
        current_qrels_path = os.path.join(qrels_dir, qrels_file)
        with open(current_qrels_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0 and 'tsv' in qrels_file:
                    if line.strip().lower().startswith("query-id"):
                        continue
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    query_id, corpus_id, score = parts
                    qrels[query_id][corpus_id] = int(score)
    return qrels

corpus = load_corpus(corpus_path)
queries = load_queries(queries_path)
qrels = load_qrels(qrels_dir, qrels_files)

# Prepare training triples (query, relevant_doc, non_relevant_doc)
train_examples = []
for query_id, rels in qrels.items():
    query_text = queries.get(query_id)
    if not query_text:
        continue
    # Get relevant and non-relevant docs
    relevant = [cid for cid, score in rels.items() if score >= 3]
    non_relevant = [cid for cid, score in rels.items() if score < 3]
    for pos_id in relevant:
        pos_text = corpus.get(pos_id)
        if not pos_text:
            continue
        # Sample one negative for each positive
        if non_relevant:
            neg_id = random.choice(non_relevant)
            neg_text = corpus.get(neg_id)
            if not neg_text:
                continue
            train_examples.append(InputExample(texts=[query_text, pos_text, neg_text]))

print(f"Prepared {len(train_examples)} training triples.")

# Finetune SBERT
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device='cuda')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100
)

output_dir = "output/finetuned-sbert"
os.makedirs(output_dir, exist_ok=True)
model.save(output_dir)
print(f"Finetuned model saved to {output_dir}")
