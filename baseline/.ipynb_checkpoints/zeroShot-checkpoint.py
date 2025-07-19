#!/usr/bin/env python3
"""
baseline_orchestrator.py ─ Minimal LLM baseline with return threshold logic
=======================================================================
• Skips processing if analysis already exists.
• Re-processes entries with empty analysis if actual return ≥ 5%.
• Incrementally updates CSV.
"""

# --- Imports -----------------------------------------------------------
import os
import json
import pandas as pd
import openai
from typing import Dict
from pathlib import Path
from prompts import baseline_prompt

# --- Config ------------------------------------------------------------
credentials_file = "credentials.json"
creds = json.loads(Path(credentials_file).read_text())

DATA_PATH  = "transcripts_nyse.csv"
LOG_PATH   = "baseline_nyse2.csv"
MODEL      = "gpt-4o-mini"
RETURN_THRESHOLD = 0.05  # 5%

openai.api_key = creds["openai_api_key"]

# --- Load full earnings-call metadata ----------------------------------
df = pd.read_csv(DATA_PATH)

# --- Load or create log ------------------------------------------------
if os.path.exists(LOG_PATH):
    processed_df = pd.read_csv(LOG_PATH)
    print(f"📄 Loaded {LOG_PATH} with {len(processed_df)} rows.")
else:
    processed_df = pd.DataFrame(
        columns=["ticker", "q", "analysis", "error", "actual_return"]
    )
    processed_df.to_csv(LOG_PATH, index=False)
    print(f"🆕 Created fresh {LOG_PATH}.")

# --- Helper functions --------------------------------------------------
def already_done(ticker: str, quarter: str) -> bool:
    return ((processed_df["ticker"] == ticker) & 
            (processed_df["q"] == quarter)).any()

def needs_update(ticker: str, quarter: str, actual_return: float) -> bool:
    row = processed_df[(processed_df["ticker"] == ticker) & 
                       (processed_df["q"] == quarter)]
    if row.empty:
        return True
    existing_analysis = row["analysis"].values[0]
    no_analysis = pd.isna(existing_analysis) or existing_analysis.strip() == ""
    return no_analysis and abs(actual_return) >= RETURN_THRESHOLD
    
def append_or_update_log(row_dict: Dict) -> None:
    global processed_df
    mask = (processed_df["ticker"] == row_dict["ticker"]) & (processed_df["q"] == row_dict["q"])

    if processed_df[mask].empty:
        # Append new
        processed_df = pd.concat([processed_df, pd.DataFrame([row_dict])], ignore_index=True)
    else:
        # Update in-place
        for key, value in row_dict.items():
            processed_df.loc[mask, key] = value

    processed_df.to_csv(LOG_PATH, index=False)

def call_gpt(prompt: str) -> str:
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

# --- Main loop ---------------------------------------------------------
ticker_counter: Dict[str, int] = {}

for _, row in df.iterrows():
    ticker = row["ticker"]
    quarter = row["q"]
    actual_return = row.get("future_3bday_cum_return", 0.0)

    ticker_counter[ticker] = ticker_counter.get(ticker, 0) + 1
    appearance = ticker_counter[ticker]

    # Skip if already processed and no need for update
    if not needs_update(ticker, quarter, actual_return):
        print(f"⚡ {ticker}/{quarter} already processed and doesn't meet threshold, skipping.")
        continue

    try:
        print(f"🧠 ({ticker}/{quarter}) appearance {appearance} – calling GPT …")
        prompt = baseline_prompt(row["transcript"])
        analysis = call_gpt(prompt)

        append_or_update_log({
            "ticker": ticker,
            "q": quarter,
            "analysis": analysis,
            "error": "",
            "actual_return": actual_return
        })

        print(f"✅ {ticker}/{quarter} logged (GPT successful)")

    except Exception as e:
        print(f"❌ Error on {ticker}/{quarter}: {e!s}")
        append_or_update_log({
            "ticker": ticker,
            "q": quarter,
            "analysis": "",
            "error": str(e),
            "actual_return": actual_return
        })

print("\n🎯 Baseline processing done!")

