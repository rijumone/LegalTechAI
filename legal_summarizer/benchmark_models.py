import gc
import torch
import time
import pandas as pd
from tqdm import tqdm

from utils.preprocess import load_dataset
from extractive.lexrank import lexrank_summary
from evaluation.evaluate import compute_rouge, compute_bertscore

# Import both but load dynamically
from abstractive.bart_chunked import bart_chunked_summary
from abstractive.legal_pegasus import pegasus_chunked_summary

SAMPLE_SIZE = 30
DATA_PATH = "data/in_abs.csv"

def benchmark(model_name):
    df = load_dataset(DATA_PATH).sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    preds_abs = []
    refs = []

    print(f"\n🚀 Benchmarking {model_name.upper()} on {SAMPLE_SIZE} samples...")
    start = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {model_name}"):
        text = row["text"]
        ext_summary = lexrank_summary(text, num_sentences=5)

        if model_name == "bart":
            summary = bart_chunked_summary(ext_summary)
        elif model_name == "pegasus":
            summary = pegasus_chunked_summary(ext_summary)
        else:
            raise ValueError("Model must be 'bart' or 'pegasus'")

        preds_abs.append(summary)
        refs.append(row["summary"])

    end = time.time()
    runtime = end - start

    rouge = compute_rouge(preds_abs, refs)[0]
    bert = compute_bertscore(preds_abs, refs)

    print(f"\n⏱️  Time taken: {runtime:.2f} seconds")
    print("📊 ROUGE Scores:")
    for metric, score in rouge.items():
        print(f"  {metric.upper():<7} | Precision: {score.precision:.3f} | Recall: {score.recall:.3f} | F1: {score.fmeasure:.3f}")
    print(f"  BERT-F1   | {bert['bert_f1']:.3f}")

def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Cleared GPU memory.\n")

if __name__ == "__main__":
    benchmark("bart")
    clear_gpu()
    benchmark("pegasus")
