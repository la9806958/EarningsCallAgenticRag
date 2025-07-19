'''prompts.py
Utility functions that return structured prompt templates for various analysis agents
used in an earnings‑call RAG pipeline.
'''
from __future__ import annotations

import json
from typing import Any, Dict, List

__all__ = [
    "comparative_agent_prompt",
    "historical_earnings_agent_prompt",
    "main_agent_prompt",
    "facts_extraction_prompt",
]

def comparative_agent_prompt(fact: Dict[str, Any], related_facts: List[Dict[str, Any]]) -> str:
    """Return the prompt for the *Comparative Peers* analysis agent.

    Parameters
    ----------
    fact
        A single fact extracted from the current firm's earnings call.
    related_facts
        A list of facts from comparable peer firms.
    """
    return f"""
You are analyzing a company’s earnings call transcript alongside statements made by similar firms.

The specific fact about the firm is:
{json.dumps(fact, indent=2)}

Comparable firms discuss the fact in the following way:
{json.dumps(related_facts, indent=2)}

Your task is:
- Describe how the firm's reasoning about their own performance differ from other firms.

- Cite factual evidence from historical calls
""".strip()


def historical_earnings_agent_prompt(
    fact: Dict[str, Any],
    related_facts: List[Dict[str, Any]],
    current_quarter: str
) -> str:
    """
    Return the prompt for the *Historical Earnings* analysis agent.

    Parameters
    ----------
    fact : dict
        The current fact from the firm's latest earnings call.
    related_facts : list of dict
        A list of related facts drawn from the firm's own previous calls.
    current_quarter : str
        The current fiscal quarter (e.g., 'Q2 2025').
    """
    return f"""
You are analyzing a company’s earnings call transcript alongside facts from its own past earnings calls.

The list of current facts are:
{json.dumps(fact, indent=2)}

It is reported in the quarter {current_quarter}

Here is a JSON list of related facts from the firm's previous earnings calls:
{json.dumps(related_facts, indent=2)}

TASK
────
1. **Validate past guidance.**
   ▸ For every forward-looking statement made in previous quarters, state whether the firm met, beat, or missed that guidance in `{current_quarter}`.  
   ▸ Reference concrete numbers (e.g., “Revenue growth was 12 % vs. the 10 % guided in 2024-Q3”).

2. **Provide supporting evidence.**
   ▸ Quote or paraphrase the relevant historical statement, then cite the matching current-quarter metric.  
   ▸ Format each evidence line as  
     `• <metric>: <historical statement> → <current result>`.

3. **Highlight unexpected outcomes.**
   ▸ Identify any areas where management *did not* address an important historical comparison, or where the result diverged sharply from trend/expectations.  
   ▸ Explain why the omission or divergence matters to investors.

""".strip()


def financials_statement_agent_prompt(
    fact: Dict[str, Any],
    quarter:str,
    prior_income: str,
    prior_balance: str,
    prior_cash: str,
) -> str:
    """Prompt template for analysing the current fact in the context of past
    income statements, balance sheets, and cash‑flow statements."""
    return f"""
You are reviewing the company’s {quarter} earnings-call transcript together with its historical financial statements. 

────────────────────────────────────────
Facts extracted from the {quarter} call
{json.dumps(fact, indent=2)}

Historical financial statements
• Income statements (past quarters):
  {prior_income}

• Balance sheets (past quarters):
  {prior_balance}

• Cash-flow statements (past quarters):
  {prior_cash}
────────────────────────────────────────

Your tasks: 

You should compare the content in the earnings call against the historical financial statement. 
──────────
1. **Quarter-over-quarter comparison**  
   • Wherever possible, compare this quarter (derived from the earnings call) with the immediately preceding quarter (from the historical statements).  
   • For each metric, quote or paraphrase the relevant historical figure and cite the current figure on one line, using the format:  
     `• <metric>: <previous-quarter value> → <current value>`

2. **Supported outcomes**  
   • Identify areas where management explicitly addressed historical comparisons and the numbers confirm their comments.

3. **Unexpected outcomes**  
   • Highlight results that management did **not** address or that diverge sharply from historical trends, and explain why this matters to investors.

*Note: Figures may be stated in ten-thousands (万) or hundreds of millions (亿). Make sure to account for these scale differences when comparing values.*

""".strip()


################################################################################################

def memory(all_notes, actual_return):
    return f"""
    You have memory on how your previous prediction on the firm faired. 
    Your previous research note is given as:
    {all_notes},
    The actual return achieved by your previous note was : {actual_return}
    """
    

def main_agent_prompt(notes, all_notes = None) -> str:
    """Prompt for the *Main* decision-making agent, requesting just an 
    Up/Down call plus a confidence score (0-100)."""
    return f"""
You are a portfolio manager and you are reading an earnings call transcript.
decide whether the stock price is likely to **increase (“Up”) or decrease (“Down”)**
one trading day after the earnings call, and assign a **Direction score** from 0 to 10.

---
Financials-vs-History note:
{notes['financials']}

Historical-Calls note:
{notes['past']}

Peer-Comparison note:
{notes['peers']}

Focus more on the bottom line performance of the firm (eg. EBITDA and net income) in order to assess performance. 

Keep your ratings conservative (eg. A small increase in revenue <3% may still yield a neutral response)
---

Instructions:
1. Assign a confidence score (0 = strong conviction of decline, 5 = neutral, 10 = strong conviction of rise).
2. Evaluate all three notes together

Respond in **exactly** this format:

<Couple of sentences of Explanation>
Direction : <0-10>

""".strip()

    
def facts_extraction_prompt(transcript_chunk: str) -> str:
    """
    Build the LLM prompt that asks for four specific data classes
    (Facts, Forward-Looking, Risk Disclosures, and Sentiment)
    from a single earnings-call transcript chunk.
    """
    return f"""
You are a senior equity-research analyst.

### TASK
Extract **only** the following four classes from the transcript below.
Ignore moderator chatter, safe-harbor boiler-plate, and anything that doesn’t match one of these classes.

1. **Fact** – already-achieved financial or operating results  
2. **Forward-Looking** – any explicit future projection, target, plan, or guidance  
3. **Risk Disclosure** – statements highlighting current or expected obstacles  
   (e.g., FX headwinds, supply-chain issues, regulation)  
4. **Sentiment** – management’s overall tone (Positive, Neutral, or Negative);
   cite key wording that informs your judgment.
5. **Macro** - discussion on how the macro-economics landscape is impacting the firm

The transcript is {transcript_chunk}

Output as many item as you can find, ideally 20-70. You MUST output more than 20 facts.
Do not include [ORG] in your response. 
---

### OUTPUT RULES  
* Use the exact markdown block below for **every** extracted item.  
* Increment the item number sequentially (1, 2, 3 …).  
* One metric per block; never combine multiple metrics.  

Example output:
### Fact No. 1  
- **Type:** <Fact | Forward-Looking | Risk Disclosure | Sentiment | Macro>
- **Metric:** FY-2025 Operating EPS  
- **Value:** “at least $14.00”  
- **Reason:** Company reaffirmed full-year earnings guidance.

"""
    
def facts_delegation_prompt(facts: List) -> str:
    """Return the prompt used for extracting individual facts from a transcript chunk.

    Parameters
    ----------
    transcript_chunk
        A chunk of the earnings‑call transcript to be analysed.
    """
    return f""" You are the RAG-orchestration analyst for an earnings-call workflow.

## Objective
For **each fact** listed below, decide **which (if any) of the three tools** will
help you gauge its potential impact on the company’s share price **one trading
day after the call**.

### Available Tools
1. **InspectPastStatements**  
   • Retrieves historical income-statement, balance-sheet, and cash-flow data  
   • **Use when** the fact cites a standard, repeatable line-item
     (e.g., revenue, EBITDA, free cash flow, margins)

2. **QueryPastCalls**  
   • Fetches the same metric or statement from prior earnings calls  
   • **Use when** comparing management’s current commentary with its own
     previous statements adds context

3. **CompareWithPeers**
   • Provides the same metric from peer companies’ calls or filings  
   • **Use when** competitive benchmarking clarifies whether the fact signals
     outperformance, underperformance, or parity

---
The facts are: {facts}

Output your answers in the following form:

InspectPastStatements: Fact No <2, 4, 6>
CompareWithPeers:  Fact No <10>
QueryPastCalls: Fact No <1, 3, 5>

*One fact may appear under multiple tools if multiple comparisons are helpful.*

"""
peer_discovery_ticker_prompt = """
You are a financial analyst. Based on the company with ticker {ticker}, list 5 close peer companies that are in the same or closely related industries.

Only output a Python-style list of tickers, like:
["AAPL", "GOOGL", "AMZN", "MSFT", "ORCL"]
"""

######################################################################################


# Baseline prompts
def baseline_prompt(transcript) -> str:
    return f"""
You are a portfolio manager and you are reading an earnings call transcript.
decide whether the stock price is likely to **increase (“Up”) or decrease (“Down”)**
one trading day after the earnings call, and assign a **Direction score** from 0 to 10.

---
{transcript}

Instructions:
1. Assign a confidence score (0 = strong conviction of decline, 5 = neutral, 10 = strong conviction of rise).

Respond in **exactly** this format:

<Couple of sentences of Explanation>
Direction : <0-10>

""".strip()