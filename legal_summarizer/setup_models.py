from transformers import (
    PegasusTokenizer, PegasusForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration
)

def download_bart():
    print("⬇️ Downloading BART...")
    BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def download_pegasus():
    print("⬇️ Downloading Legal-Pegasus...")
    PegasusTokenizer.from_pretrained("nsi319/legal-pegasus")
    PegasusForConditionalGeneration.from_pretrained("nsi319/legal-pegasus")

if __name__ == "__main__":
    download_bart()
    download_pegasus()
    print("✅ All models downloaded and cached.")
