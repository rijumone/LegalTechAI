import os
import json
import click
import numpy as np
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

@click.command()
@click.option('--query', prompt='Enter your query', help='The query to search for.')
@click.option('--retriever', type=click.Choice(['bm25', 'sbert'], case_sensitive=False), default='bm25', help='Retriever type to use.')
@click.option('--bm25-model-path', default=None, help='(Unused, for compatibility) Path to BM25 model (not needed, built on the fly).')
@click.option('--sbert-model', default='nlpaueb/legal-bert-base-uncased', help='SBERT model name or path.')
@click.option('--corpus-path', required=True, help='Path to corpus.jsonl file.')
@click.option('--top-k', default=5, help='Number of top documents to retrieve.')
@click.option('--llm', is_flag=True, default=False, help='Enable LLM reranking (requires Ollama server).')
@click.option('--llm-model', default=None, help='LLM model name for reranking (used only if --llm is set).')
def main(query, retriever, bm25_model_path, sbert_model, corpus_path, top_k, llm, llm_model):
    # Load corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = {}
        for line in f:
            item = json.loads(line)
            corpus[item['_id']] = item['text']
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]

    if retriever.lower() == 'bm25':
        tokenized_corpus = [word_tokenize(text.lower()) for text in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = word_tokenize(query.lower())
        scores = bm25.get_scores(tokenized_query)
        top_k_idx = np.argsort(scores)[::-1][:top_k*2] if llm else np.argsort(scores)[::-1][:top_k]
        results = [(corpus_ids[i], scores[i], corpus[corpus_ids[i]]) for i in top_k_idx]
    elif retriever.lower() == 'sbert':
        sbert = SentenceTransformer(sbert_model)
        corpus_emb = sbert.encode(corpus_texts, convert_to_tensor=True)
        query_emb = sbert.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, corpus_emb, top_k=top_k*2 if llm else top_k)[0]
        results = [(corpus_ids[int(hit['corpus_id'])], hit['score'], corpus[corpus_ids[int(hit['corpus_id'])]]) for hit in hits]
    else:
        click.echo('Unknown retriever type.')
        return

    # LLM reranking (optional)
    if llm:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            click.echo('langchain and dependencies are required for LLM reranking.')
            return
        model = llm_model if llm_model else 'qwen2.5:7b'
        llm_model_instance = ChatOllama(model=model, base_url="http://192.168.1.11:7869")
        system = SystemMessage(content="You are a legal AI trained to evaluate how relevant a clause is to a query.")
        clause_list = "\n\n".join([
            f"Clause {i+1} (ID: {cid}):\n{text.strip()}" for i, (cid, _, text) in enumerate(results)
        ])
        user_prompt = (
            f"Query:\n{query}\n\n"
            f"Below are {len(results)} candidate clauses. "
            "Rate each one from 1 (irrelevant) to 5 (highly relevant) based on how well it answers the query. "
            "Return in the format: Clause <number>: <score>\n\n"
            f"{clause_list}"
        )
        response = llm_model_instance.invoke([system, HumanMessage(content=user_prompt)])
        content = response.content if hasattr(response, 'content') else str(response)
        scores = {}
        import re
        for line in str(content).splitlines():
            match = re.match(r"Clause\s*(\d+)\s*[:\-]?\s*(\d+(\.\d+)?)", line.strip())
            if match:
                idx = int(match.group(1)) - 1
                score = float(match.group(2))
                if 0 <= idx < len(results):
                    cid, _, text = results[idx]
                    scores[cid] = score
        # Default 0.0 for anything missing
        reranked = [(cid, scores.get(cid, 0.0), text) for cid, _, text in results]
        results = sorted(reranked, key=lambda x: -x[1])[:top_k]

    click.echo(f'\nTop {top_k} results:')
    for rank, (doc_id, score, text) in enumerate(results, 1):
        click.echo(f'\nRank {rank}:')
        click.echo(f'ID: {doc_id}')
        click.echo(f'Score: {score:.4f}')
        click.echo(f'Text: {text[:500]}...')

if __name__ == '__main__':
    main()
