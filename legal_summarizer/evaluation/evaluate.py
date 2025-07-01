from rouge_score import rouge_scorer
# import bert_score
from bert_score import score


def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(r, p) for p, r in zip(preds, refs)]
    return scores

def compute_bertscore(preds, refs):
    P, R, F1 = score(preds, refs, lang="en", verbose=False)
    return {
        "bert_precision": P.mean().item(),
        "bert_recall": R.mean().item(),
        "bert_f1": F1.mean().item()
    }