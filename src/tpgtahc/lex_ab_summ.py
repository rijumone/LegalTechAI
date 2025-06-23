from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from evaluate import load as load_metric
from tqdm import tqdm

# Load LexAbSumm dataset
dataset = load_dataset("MahmoudAly/LexAbSumm", split="train")  # Use "train" or "validation" as needed
print(dataset[0])

# Load LongT5 model (best performer in the paper)
model_ckpt = "google/long-t5-tglobal-base"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

# Set up summarization pipeline
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=0,
)  # set device=-1 if no GPU

# Evaluate with ROUGE
rouge = load_metric("rouge")

# Limit to small sample for demo
num_samples = 10
inputs = dataset.select(range(num_samples))

# Run summarization + collect metrics
predictions = []
references = []

for example in tqdm(inputs):
    input_text = f"Document (Facts Section): {example['facts_source']}"
    reference = example['facts_summary']

    inputs_tok = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=4096
    )
    summary_ids = model.generate(
        inputs_tok.input_ids, attention_mask=inputs_tok.attention_mask, max_length=512
    )
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    predictions.append(summary_text)
    references.append(reference)

    print("\n---")
    print(f"Case: {example['case_name']}")
    print(f"Judgment Date: {example['judgement_date']}")
    print(f"Reference Summary: {reference}")
    print(f"Generated Summary: {summary_text}")


# Compute ROUGE scores
results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
for k, v in results.items():
    # import pdb;pdb.set_trace()
    # print(f"{k}: {v.mid.fmeasure:.4f}")
    print(f"{k}: {v:.4f}")

