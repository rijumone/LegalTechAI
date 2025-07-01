from transformers import BartTokenizer, BartForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

def bart_chunked_summary(text, chunk_size=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = [inputs[i:i+chunk_size] for i in range(0, len(inputs), chunk_size)]
    
    summaries = []
    for chunk in chunks:
        input_chunk = torch.tensor([chunk]).to(device)
        with torch.no_grad():
            summary_ids = model.generate(input_chunk, max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    return " ".join(summaries)
