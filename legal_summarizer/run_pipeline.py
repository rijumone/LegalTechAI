import os
import click
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.preprocess import load_dataset
from extractive.lexrank import lexrank_summary
from abstractive.bart_chunked import bart_chunked_summary
from abstractive.legal_pegasus import pegasus_chunked_summary
from evaluation.evaluate import compute_rouge, compute_bertscore

# 🔧 Constants
SAMPLE_SIZE = 27

@click.command()
@click.option('--dataset', multiple=True, required=True,
              type=click.Choice(['in_abs', 'in_ext']),
              help='One or more datasets to run the pipeline on (e.g., --dataset in_abs in_ext)')
@click.option('--model', type=click.Choice(['bart', 'pegasus']), default='bart',
              help='Abstractive model to use.')
@click.option('--save-output', type=str, default=None,
              help='Optional path to save results as CSV.')
def run(dataset, model, save_output):
    # import pdb;pdb.set_trace()
    for ds in dataset:
        data_path = f"data/{ds}.csv"
        print(f"\n📂 Loading dataset: {data_path}")
        df = load_dataset(data_path)

        # Sample the dataset
        if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
            df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

        preds_ext = []
        preds_abs = []
        refs = []
        ids = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Summarizing {ds}"):
            text, gold = row["text"], row["summary"]

            ext_summary = lexrank_summary(text, num_sentences=5)

            if model == "bart":
                abs_summary = bart_chunked_summary(ext_summary)
            elif model == "pegasus":
                abs_summary = pegasus_chunked_summary(ext_summary)

            preds_ext.append(ext_summary)
            preds_abs.append(abs_summary)
            refs.append(gold)
            ids.append(i)

        # Evaluation
        rouge_scores = compute_rouge(preds_abs, refs)
        bert_scores = compute_bertscore(preds_abs, refs)

        # Output summary
        print(f"\n📊 Evaluation for {ds.upper()} with {model.upper()}:")
        avg_rouge = rouge_scores[0]
        print("ROUGE Scores:")
        for metric, score in avg_rouge.items():
            print(f"  {metric.upper():<7} | Precision: {score.precision:.3f} | Recall: {score.recall:.3f} | F1: {score.fmeasure:.3f}")
        print(f"\nBERTScore:\n  BERT-F1  | {bert_scores['bert_f1']:.3f}")

        # Save results (optional)
        if save_output:
            output_file = save_output.replace(".csv", f"_{ds}_{model}.csv")
            pd.DataFrame({
                "id": ids,
                "text": df["text"],
                "reference": refs,
                "extractive_summary": preds_ext,
                "abstractive_summary": preds_abs
            }).to_csv(output_file, index=False)
            print(f"\n💾 Saved results to {output_file}")
            plot_metrics(
                rouge_scores, bert_scores["bert_f1"], 
                title=f"Evaluation: {ds.upper()} + {model.upper()}",
                file_prefix=f"{ds}_{model}",
            )


def plot_metrics(rouge_scores, bert_f1, title="Evaluation Metrics", output_dir="plots", file_prefix="eval"):
    scores = rouge_scores[0]

    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    precisions = [scores['rouge1'].precision, scores['rouge2'].precision, scores['rougeL'].precision]
    recalls    = [scores['rouge1'].recall,    scores['rouge2'].recall,    scores['rougeL'].recall]
    f1s        = [scores['rouge1'].fmeasure,  scores['rouge2'].fmeasure,  scores['rougeL'].fmeasure]

    x = range(len(metrics))

    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x], precisions, width=0.2, label='Precision', color='skyblue')
    plt.bar(x, recalls, width=0.2, label='Recall', color='lightgreen')
    plt.bar([i + 0.2 for i in x], f1s, width=0.2, label='F1 Score', color='salmon')

    # BERT-F1 line
    plt.axhline(bert_f1, color='purple', linestyle='--', linewidth=2, label=f'BERT-F1: {bert_f1:.3f}')

    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{file_prefix}.png")
    plt.savefig(filename)
    plt.close()

    print(f"📈 Saved evaluation plot to {filename}")

if __name__ == "__main__":
    run()

'''
📊 Evaluation for IN_ABS with BART:
ROUGE Scores:
  ROUGE1  | Precision: 0.754 | Recall: 0.040 | F1: 0.076
  ROUGE2  | Precision: 0.339 | Recall: 0.018 | F1: 0.034
  ROUGEL  | Precision: 0.579 | Recall: 0.031 | F1: 0.058

BERTScore:
  BERT-F1  | 0.027

📊 Evaluation for IN_EXT with BART:
ROUGE Scores:
  ROUGE1  | Precision: 0.871 | Recall: 0.078 | F1: 0.143
  ROUGE2  | Precision: 0.500 | Recall: 0.044 | F1: 0.081
  ROUGEL  | Precision: 0.484 | Recall: 0.043 | F1: 0.079

BERTScore:
  BERT-F1  | -0.091

---

📊 Evaluation for IN_ABS with PEGASUS:
ROUGE Scores:
  ROUGE1  | Precision: 0.798 | Recall: 0.066 | F1: 0.122
  ROUGE2  | Precision: 0.409 | Recall: 0.034 | F1: 0.062
  ROUGEL  | Precision: 0.528 | Recall: 0.044 | F1: 0.081

BERTScore:
  BERT-F1  | 0.033

📊 Evaluation for IN_EXT with PEGASUS:
ROUGE Scores:
  ROUGE1  | Precision: 0.821 | Recall: 0.083 | F1: 0.151
  ROUGE2  | Precision: 0.486 | Recall: 0.049 | F1: 0.089
  ROUGEL  | Precision: 0.538 | Recall: 0.055 | F1: 0.099

BERTScore:
  BERT-F1  | -0.083
'''