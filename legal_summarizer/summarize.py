import click
import torch
import pandas as pd
from extractive.lexrank import lexrank_summary
from abstractive.bart_chunked import bart_chunked_summary
from abstractive.legal_pegasus import pegasus_chunked_summary
from evaluation.evaluate import compute_rouge, compute_bertscore
from run_pipeline import plot_metrics


@click.command()
@click.argument('input_judgement', type=click.Path(exists=True))
@click.argument('input_summary', type=click.Path(exists=True))
@click.option('--model', type=click.Choice(['bart', 'pegasus']), required=False,
              help='Abstractive model to use (required if technique is abstractive)')
@click.option('--technique', type=click.Choice(['extractive', 'abstractive']), required=True,
              help='Summarization technique to use (required)')
@click.option('--num-sentences', default=5, show_default=True,
              help='Number of sentences to extract using LexRank')
@click.option('--chunk-size', default=512, show_default=True,
              help='Max tokens per chunk for abstractive model')
def summarize(input_judgement, input_summary, model, technique, num_sentences, chunk_size):
    """Summarize a legal judgment and evaluate against reference summary."""
    with open(input_judgement, 'r', encoding='utf-8') as f:
        text = f.read()
    with open(input_summary, 'r', encoding='utf-8') as f:
        reference_summary = f.read()

    if technique == 'extractive':
        print(f"🔍 Running LexRank to extract {num_sentences} key sentences...")
        final_summary = lexrank_summary(text, num_sentences=num_sentences)
        """
        lexrank_summary is an extractive summarization technique that uses the LexRank algorithm. 
        This algorithm identifies the most important sentences in a text by constructing a graph 
        where sentences are nodes and the edges are based on cosine similarity. The sentences with 
        the highest "centrality" in this graph are chosen for the summary.
        """
        technique_name = 'extractive'
    elif technique == 'abstractive':
        if not model:
            raise click.UsageError("--model is required when --technique is 'abstractive'.")
        print(f"🧠 Running {model.upper()} abstractive summarization...")
        if model == 'bart':
            final_summary = bart_chunked_summary(text, chunk_size=chunk_size)
            """
            bart_chunked_summary is an abstractive summarization method that uses the BART 
            (Bidirectional and Auto-Regressive Transformers) model. Since BART has a limit 
            on the number of tokens it can process at once, this function breaks the input 
            text into smaller chunks. It summarizes each chunk individually and then combines 
            these partial summaries to create the final summary.
            """
        elif model == 'pegasus':
            final_summary = pegasus_chunked_summary(text, chunk_size=chunk_size)
            """
            pegasus_chunked_summary is another abstractive summarization method, similar to 
            the BART-based one, but it uses the PEGASUS model, which is specifically designed 
            for abstractive summarization. It also processes the text in chunks to handle 
            long documents and then combines the summaries of each chunk.
            """
        technique_name = f'abstractive_{model}'
    else:
        raise ValueError("Invalid technique. Choose 'extractive' or 'abstractive'.")

    print("\n📄 Final Summary:\n")
    print(final_summary)

    # Save final summary to disk
    import os
    input_base = os.path.splitext(os.path.basename(input_judgement))[0]
    summary_filename = f"{input_base}.summary.{technique_name}.txt"
    summary_path = os.path.join(os.path.dirname(input_judgement), summary_filename)
    with open(summary_path, 'w', encoding='utf-8') as sf:
        sf.write(final_summary)
    print(f"📝 Saved summary to {summary_path}")

    # Evaluation
    print("\n📊 Running Evaluation against reference summary...")
    preds = [final_summary]
    refs = [reference_summary]
    rouge_scores = compute_rouge(preds, refs)
    bert_scores = compute_bertscore(preds, refs)

    avg_rouge = rouge_scores[0]
    print("ROUGE Scores:")
    for metric, score in avg_rouge.items():
        print(f"  {metric.upper():<7} | Precision: {score.precision:.3f} | Recall: {score.recall:.3f} | F1: {score.fmeasure:.3f}")
    print(f"\nBERTScore:\n  BERT-F1  | {bert_scores['bert_f1']:.3f}")

    # Plot metrics
    if technique == 'abstractive':
        plot_file_prefix = f"single_eval_abstractive_{model}"
    else:
        plot_file_prefix = f"single_eval_{technique_name}"
    plot_metrics(rouge_scores, bert_scores["bert_f1"], title=f"Evaluation", file_prefix=plot_file_prefix)
    print(f"📈 Saved evaluation plot to plots/{plot_file_prefix}.png")

if __name__ == '__main__':
    summarize()
