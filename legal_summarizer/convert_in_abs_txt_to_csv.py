import os
import pandas as pd

def build_csv_from_txt(judgement_dir, summary_dir, output_path):
    data = []

    for fname in os.listdir(judgement_dir):
        # import pdb;pdb.set_trace()
        judgement_path = os.path.join(judgement_dir, fname)
        summary_path = os.path.join(summary_dir, fname)

        if os.path.exists(judgement_path) and os.path.exists(summary_path):
            with open(judgement_path, 'r', encoding='utf-8') as f1:
                judgement = f1.read().strip()
            with open(summary_path, 'r', encoding='utf-8') as f2:
                summary = f2.read().strip()

            data.append({"text": judgement, "summary": summary})

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} samples to {output_path}")


def build_in_ext_csv(judgement_dir, summary_dir, output_path):
    data = []
    for fname in os.listdir(judgement_dir):
        summary_path = os.path.join(summary_dir, fname)
        judgement_path = os.path.join(judgement_dir, fname)

        if os.path.exists(summary_path) and os.path.exists(judgement_path):
            with open(judgement_path, 'r', encoding='utf-8') as f1:
                text = f1.read().strip()
            with open(summary_path, 'r', encoding='utf-8') as f2:
                summary = f2.read().strip()
            data.append({"text": text, "summary": summary})

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, escapechar='\\',)
    print(f"✅ Saved {len(df)} entries to {output_path}")


# Example usage for IN-Abs/train-data
build_csv_from_txt(
    judgement_dir="dataset/IN-Abs/train-data/judgement",
    summary_dir="dataset/IN-Abs/train-data/summary",
    output_path="data/in_abs.csv"
)

build_in_ext_csv(
    judgement_dir="dataset/IN-Ext/judgement",
    summary_dir="dataset/IN-Ext/summary/full/A1",
    output_path="data/in_ext.csv"
)
