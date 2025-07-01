from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You can try `nsi319/legal-pegasus` or use huggingface.co/nsi319/legal-pegasus
MODEL_NAME = "nsi319/legal-pegasus"

tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

def pegasus_chunked_summary(text, chunk_size=512):
    input_ids = tokenizer.encode(text, truncation=False)

    # Create chunks of token IDs
    chunks = [input_ids[i:i+chunk_size] for i in range(0, len(input_ids), chunk_size)]

    summaries = []
    for chunk in chunks:
        input_tensor = torch.tensor(chunk).unsqueeze(0).to(device)
        with torch.no_grad():
            summary_ids = model.generate(
                input_tensor,
                max_length=150,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)
