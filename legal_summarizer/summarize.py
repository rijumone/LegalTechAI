import click
import torch
from extractive.lexrank import lexrank_summary
from abstractive.bart_chunked import bart_chunked_summary
from abstractive.legal_pegasus import pegasus_chunked_summary

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--model', type=click.Choice(['bart', 'pegasus']), default='bart', show_default=True,
              help='Abstractive model to use')
@click.option('--extractive', is_flag=True, default=True,
              help='Use extractive summarization (LexRank) before abstraction')
@click.option('--num-sentences', default=5, show_default=True,
              help='Number of sentences to extract using LexRank')
@click.option('--chunk-size', default=512, show_default=True,
              help='Max tokens per chunk for abstractive model')
def summarize(input_path, model, extractive, num_sentences, chunk_size):
    """Summarize a legal judgment from a .txt file using chosen model."""
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if extractive:
        print(f"🔍 Running LexRank to extract {num_sentences} key sentences...")
        extractive_summary = lexrank_summary(text, num_sentences=num_sentences)
    else:
        extractive_summary = text  # direct to abstraction

    print(f"🧠 Running {model.upper()} abstractive summarization...")

    if model == 'bart':
        final_summary = bart_chunked_summary(extractive_summary, chunk_size=chunk_size)
    elif model == 'pegasus':
        final_summary = pegasus_chunked_summary(extractive_summary, chunk_size=chunk_size)

    print("\n📄 Final Summary:\n")
    print(final_summary)

if __name__ == '__main__':
    summarize()
