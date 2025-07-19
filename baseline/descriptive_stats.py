import pandas as pd
import os
from transformers import AutoTokenizer

# You can change this to any tokenizer you want (e.g., FinBERT, Longformer, etc.)
DEFAULT_TOKENIZER = 'bert-base-uncased'

def describe_file(path, tokenizer_name=DEFAULT_TOKENIZER):
    print(f'\n===== {os.path.basename(path)} =====')
    df = pd.read_csv(path)
    # 1. Span of data (min/max date)
    if 'parsed_date' in df.columns:
        min_date = df['parsed_date'].min()
        max_date = df['parsed_date'].max()
        print(f"Span of data: {min_date} to {max_date}")
    else:
        print("Span of data: 'parsed_date' column not found.")
    # 2. Number of unique tickers
    n_tickers = df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'
    print(f"Number of unique tickers: {n_tickers}")
    # 3. Number of files (rows)
    print(f"Number of rows/files: {len(df)}")
    # 4. Number of tokens in each transcript
    if 'transcript' in df.columns:
        print("Counting tokens in each transcript (this may take a while)...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        token_counts = df['transcript'].fillna('').apply(lambda x: len(tokenizer.tokenize(x)))
        print(f"Transcript token count: min={token_counts.min()}, max={token_counts.max()}, mean={token_counts.mean():.1f}, median={token_counts.median()}, std={token_counts.std():.1f}")
    else:
        print("Transcript column not found.")

def main():
    for fname in ["merged_data_nasdaq.csv", "merged_data_nyse.csv"]:
        if os.path.exists(fname):
            describe_file(fname)
        else:
            print(f"File not found: {fname}")

if __name__ == "__main__":
    main() 