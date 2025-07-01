import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def load_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "summary"])
    return df

def sentence_split(text):
    return sent_tokenize(text)
