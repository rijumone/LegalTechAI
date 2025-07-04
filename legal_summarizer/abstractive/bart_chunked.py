from transformers import BartTokenizer, BartForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)

def bart_chunked_summary(text, chunk_size=512):
    # Tokenize text into input IDs (list of integers)
    input_ids = tokenizer.encode(text, truncation=False)
    
    # Create chunks of token IDs
    chunks = [input_ids[i:i+chunk_size] for i in range(0, len(input_ids), chunk_size)]

    summaries = []
    for chunk in chunks:
        input_chunk = torch.tensor(chunk).unsqueeze(0).to(device)  # shape: (1, seq_len)
        with torch.no_grad():
            summary_ids = model.generate(
                input_chunk,
                max_length=150,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)
