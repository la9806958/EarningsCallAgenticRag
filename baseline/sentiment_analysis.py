import pandas as pd
import re
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

# --- Configuration ---
LM_DICT_PATH = "baseline/LM/LM_Master.csv"
DATA_PATH = "merged_data_nyse.csv"
OUTPUT_PATH = "baseline/nyse_sentiment_results.csv"

METHOD_NAME = "Loughran-McDonald"

def load_lm_dictionary(path: str) -> tuple[set[str], set[str]]:
    """
    Loads the Loughran-McDonald sentiment word lists.
    The dictionary is expected to have 'Word', 'Positive', and 'Negative' columns.
    A non-zero value in the 'Positive' column indicates a positive word.
    A non-zero value in the 'Negative' column indicates a negative word.
    """
    print(f"Loading Loughran-McDonald dictionary from: {path}")
    try:
        lm_df = pd.read_csv(path)
        
        # Words are converted to uppercase for case-insensitive matching
        positive_words = set(lm_df[lm_df['Positive'] != 0]['Word'].str.upper())
        negative_words = set(lm_df[lm_df['Negative'] != 0]['Word'].str.upper())
        
        print(f"Loaded {len(positive_words)} positive words and {len(negative_words)} negative words.")
        return positive_words, negative_words
    except FileNotFoundError:
        print(f"Error: Loughran-McDonald dictionary not found at {path}")
        return set(), set()

def analyze_sentiment(transcript: str, positive_words: set[str], negative_words: set[str]) -> dict:
    """
    Analyzes the sentiment of a single transcript.
    """
    if not isinstance(transcript, str):
        return {'positive_count': 0, 'negative_count': 0, 'polarity': 0, 'sentiment': 'Neutral'}

    # Tokenize the transcript: convert to uppercase and split by non-alphanumeric characters
    words = re.findall(r'\b\w+\b', transcript.upper())
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    polarity = positive_count - negative_count
    
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
        
    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'polarity': polarity,
        'sentiment': sentiment
    }

def main():
    """
    Main function to load data, perform sentiment analysis, and save results.
    """
    positive_words, negative_words = load_lm_dictionary(LM_DICT_PATH)
    
    if not positive_words and not negative_words:
        print("Could not proceed without sentiment dictionary.")
        return

    print(f"Loading transcripts from: {DATA_PATH}")
    try:
        transcripts_df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Transcripts file not found at {DATA_PATH}")
        return
        
    # Ensure 'transcript' column exists
    if 'transcript' not in transcripts_df.columns:
        print(f"Error: 'transcript' column not found in {DATA_PATH}")
        return

    tqdm.pandas(desc="Analyzing Sentiment")
    
    # Apply the sentiment analysis function to each transcript
    sentiment_results = transcripts_df['transcript'].progress_apply(
        lambda t: analyze_sentiment(t, positive_words, negative_words)
    )
    
    # Combine results into a final DataFrame
    results_df = pd.DataFrame(sentiment_results.tolist())
    final_df = pd.concat([transcripts_df[['ticker', 'q', 'future_3bday_cum_return']], results_df], axis=1)
    
    # Assign ground truth: 1 if future_3bday_cum_return > 0, else 0
    final_df['label'] = (final_df['future_3bday_cum_return'] > 0).astype(int)
    
    # Assign prediction: 1 for Positive, 0 for Negative, ignore Neutral
    final_df['pred'] = final_df['sentiment'].map({'Positive': 1, 'Negative': 0})
    
    # Drop rows with Neutral prediction
    eval_df = final_df.dropna(subset=['pred'])
    
    y_true = eval_df['label']
    y_pred = eval_df['pred']
    
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"\\textbf{{{METHOD_NAME}}} & {bal_acc:.3f} & {macro_precision:.3f} & {macro_recall:.3f} & {macro_f1:.3f} \\")
    
    print(f"Saving sentiment analysis results to: {OUTPUT_PATH}")
    final_df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Sentiment analysis complete.")

if __name__ == "__main__":
    main() 