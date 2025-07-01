from utils.preprocess import load_dataset
from extractive.lexrank import lexrank_summary
from abstractive.bart_chunked import bart_chunked_summary
from evaluation.evaluate import compute_rouge, compute_bertscore

def run():
    df = load_dataset("data/in_abs.csv")
    
    preds_ext = []
    preds_abs = []
    refs = []

    for i, row in df.iterrows():
        text, gold = row["text"], row["summary"]
        
        ext_summary = lexrank_summary(text, num_sentences=5)
        abs_summary = bart_chunked_summary(ext_summary)  # Abstractive over extractive
        
        preds_ext.append(ext_summary)
        preds_abs.append(abs_summary)
        refs.append(gold)

        if i % 5 == 0:
            print(f"Processed {i} examples...")

    rouge_scores = compute_rouge(preds_abs, refs)
    bert_scores = compute_bertscore(preds_abs, refs)

    print("ROUGE:", rouge_scores[:1])
    print("BERTScore:", bert_scores)

if __name__ == "__main__":
    run()
