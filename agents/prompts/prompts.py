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

def comparative_agent_prompt(facts: List[Dict[str, Any]], related_facts: List[Dict[str, Any]], self_ticker: str = None) -> str:
    """Return the prompt for the *Comparative Peers* analysis agent.

    Parameters
    ----------
    facts
        A list of facts extracted from the current firm's earnings call.
    related_facts
        A list of facts from comparable peer firms.
    self_ticker
        The ticker symbol of the firm being analyzed.
    """
    ticker_section = f"\n\nThe ticker of the firm being analyzed is: {self_ticker}" if self_ticker else ""
    return f"""
You are analyzing a company's earnings call transcript alongside statements made by similar firms.{ticker_section}

The batch of facts about the firm is:
{json.dumps(facts, indent=2)}

Comparable firms discuss the facts in the following way:
{json.dumps(related_facts, indent=2)}

Your task is:
- Describe how the firm's reasoning about their own performance differs from other firms, for each fact if possible.
- Cite factual evidence from historical calls.

Keep your analysis concise. Do not discuss areas not mentioned.

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
You are analyzing a company's earnings call transcript alongside facts from its own past earnings calls.

The list of current facts are:
{json.dumps(fact, indent=2)}

It is reported in the quarter {current_quarter}

Here is a JSON list of related facts from the firm's previous earnings calls:
{json.dumps(related_facts, indent=2)}

TASK
────
1. **Validate past guidanced**
   ▸ For every forward-looking statement made in previous quarters, state whether the firm met, beat, or missed that guidance in `{current_quarter}`.  
   ▸ Reference concrete numbers (e.g., "Revenue growth was 12 % vs. the 10 % guided in 2024-Q3").
   ▸ Omit if you cannot provide a direct comparison

2. **Compare results discussed**
    ▸ Compare the results being discussed.
    ▸ Reference concrete numbers 

3. **Provide supporting evidence.**
   ▸ Quote or paraphrase the relevant historical statement, then cite the matching current-quarter metric.  
   ▸ Format each evidence line as  
     `• <metric>: <historical statement> → <current result>`.

4. **Highlight unexpected outcomes.**
   ▸ Identify any areas where management *did not* address an important historical comparison, or where the result diverged sharply from trend/expectations.  
   ▸ Explain why the omission or divergence matters to investors.

Keep your analysis concise.  Prioritize more recent quarters. Do not discuss areas not mentioned.
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
You are analyzing a company's earnings call transcript alongside facts from its own past earnings calls.

The list of current facts are:
{json.dumps(fact, indent=2)}

It is reported in the quarter {current_quarter}

Here is a JSON list of related facts from the firm's previous earnings calls:
{json.dumps(related_facts, indent=2)}

TASK
────
1. **Validate past guidanced**
   ▸ For every forward-looking statement made in previous quarters, state whether the firm met, beat, or missed that guidance in `{current_quarter}`.  
   ▸ Reference concrete numbers (e.g., "Revenue growth was 12 % vs. the 10 % guided in 2024-Q3").
   ▸ Omit if you cannot provide a direct comparison

2. **Compare results discussed**
    ▸ Compare the results being discussed.
    ▸ Reference concrete numbers 

3. **Provide supporting evidence.**
   ▸ Quote or paraphrase the relevant historical statement, then cite the matching current-quarter metric.  
   ▸ Format each evidence line as  
     `• <metric>: <historical statement> → <current result>`.

Keep your analysis concise. Prioritize more recent quarters. Do not discuss areas not mentioned.
""".strip()


def financials_statement_agent_prompt(
    fact: Dict[str, Any],
    similar_facts: list,
    quarter: str,
) -> str:
    """Prompt template for analysing the current fact in the context of most similar past facts."""
    return f"""
You are reviewing the company's {quarter} earnings-call transcript and comparing a key fact to the most similar historical facts from previous quarters.

────────────────────────────────────────
Current fact (from {quarter}):
{json.dumps(fact, indent=2)}

Most similar past facts (from previous quarters):
{json.dumps(similar_facts, indent=2)}
────────────────────────────────────────

Your tasks:

1. **Direct comparison**
   • Compare the current fact to each of the most similar past facts. For each, note the quarter, the metric, and the value.
   • Highlight similarities, differences, and any notable trends or changes.
   • If the current value is higher/lower/similar to the most recent similar fact, state this explicitly.

2. **Supported outcomes**
   • Identify areas where management explicitly addressed historical comparisons and the numbers confirm their comments.

3. **Unexpected outcomes**
   • Highlight results that management did **not** address or that diverge sharply from historical trends, and explain why this matters to investors.

Focus on improvements on bottom line performance (eg. net income)

*Note: Figures may be stated in ten-thousands (万) or hundreds of millions (亿). Make sure to account for these scale differences when comparing values.*

Keep your analysis concise. Prioritize more recent quarters. Do not discuss areas not mentioned.

""".strip()

def financials_statement_agent_prompt(
    fact: Dict[str, Any],
    similar_facts: list,
    quarter: str,
) -> str:
    """Prompt template for analysing the current fact in the context of most similar past facts."""
    return f"""
You are reviewing the company's {quarter} earnings-call transcript and comparing a key fact to the most similar historical facts from previous quarters.

────────────────────────────────────────
Current fact (from {quarter}):
{json.dumps(fact, indent=2)}

Most similar past facts (from previous quarters):
{json.dumps(similar_facts, indent=2)}
────────────────────────────────────────

Your tasks:

1. **Direct comparison**
   • Compare the current fact to each of the most similar past facts. For each, note the quarter, the metric, and the value.
   • Highlight similarities, differences, and any notable trends or changes.
   • If the current value is higher/lower/similar to the most recent similar fact, state this explicitly.

2. **Supported outcomes**
   • Identify areas where management explicitly addressed historical comparisons and the numbers confirm their comments.

Focus on improvements on bottom line performance (eg. net income)

*Note: Figures may be stated in ten-thousands (万) or hundreds of millions (亿). Make sure to account for these scale differences when comparing values.*

Keep your analysis concise. Prioritize more recent quarters. Do not discuss areas not mentioned.

""".strip()

################################################################################################

def memory(all_notes, actual_return):
    return f"""
    You have memory on how your previous prediction on the firm faired. 
    Your previous research note is given as:
    {all_notes},
    The actual return achieved by your previous note was : {actual_return}
    """
    

def main_agent_prompt(notes, all_notes = None, original_transcript: str = None, memory_txt: str = None, financial_statements_facts: str = None, qoq_section: str = None) -> str:
    """Prompt for the *Main* decision-making agent, requesting just an 
    Up/Down call plus a confidence score (0-100)."""
    transcript_section = f"\nORIGINAL EARNINGS CALL TRANSCRIPT:\n---\n{original_transcript}\n---\n" if original_transcript else ""
    
    financial_statements_section = ""
    if financial_statements_facts:
        financial_statements_section = f"""
---
Financial Statements Facts (YoY):
{financial_statements_facts}
---
"""
    
    qoq_section_str = ""
    if qoq_section:
        qoq_section_str = f"\n---\nQuarter-on-Quarter Changes:\n{qoq_section}\n---\n"
    
    return f"""
You are a portfolio manager and you are reading an earnings call transcript.{transcript_section}
decide whether the stock price is likely to **increase (\"Up\") or decrease (\"Down\")**
one trading day after the earnings call, and assign a **Direction score** from 0 to 10.

The original transcript is:

{original_transcript}

{financial_statements_section}
+{qoq_section_str}
---
Financials-vs-History note:
{notes['financials']}

Historical-Calls note:
{notes['past']}

Peer-Comparison note:
{notes['peers']}


{memory_txt}

---

Instructions:
1. Assign a confidence score (0 = strong conviction of decline, 5 = neutral, 10 = strong conviction of rise).
2. Evaluate all three notes together
3. Consider the financial statements facts when available
4. Pay special attention to the year on year changes section, especially on bottom line figures (eg. net profit)

Respond in **exactly** this format:

<Couple of sentences of Explanation>

**Summary: <Two sentences supporting your verdict with facts and evidence>, Direction : <0-10>**

""".strip()

    
def facts_extraction_prompt(transcript_chunk: str) -> str:
    """
    Build the LLM prompt that asks for five specific data classes
    (Result, Forward-Looking, Risk Disclosure, Sentiment, and Macro)
    from a single earnings-call transcript chunk.
    """
    return f"""
You are a senior equity-research analyst.

### TASK
Extract **only** the following five classes from the transcript below.
Ignore moderator chatter, safe-harbor boiler-plate, and anything that doesn't match one of these classes.

1. **Result** – already-achieved financial or operating results  
2. **Forward-Looking** – any explicit future projection, target, plan, or guidance  
3. **Risk Disclosure** – statements highlighting current or expected obstacles  
   (e.g., FX headwinds, supply-chain issues, regulation)  
4. **Sentiment** – management's overall tone (Positive, Neutral, or Negative);
   cite key wording that informs your judgment.
5. **Macro** – discussion of how the macro-economic landscape affects the firm

The transcript is {transcript_chunk}

Output as many items as you can find, ideally 30-70. You MUST output more than 30 facts.
Do not include [ORG] in your response. 
---

### OUTPUT RULES  
* Use the exact markdown block below for **every** extracted item.  
* Increment the item number sequentially (1, 2, 3 …).  
* One metric per block; never combine multiple metrics.  

Example output:
### Fact No. 1  
- **Type:** <Result | Forward-Looking | Risk Disclosure | Sentiment | Macro>
- **Metric:** Revenue
- **Value:** "3 million dollars"
- **Reason:** Quarter was up on a daily organic basis, driven primarily by core non-pandemic product sales.

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
help you gauge its potential impact on the company's share price **one trading
day after the call**.

### Available Tools
1. **InspectPastStatements**  
   • Retrieves historical income-statement, balance-sheet, and cash-flow data  
   • **Use when** the fact cites a standard, repeatable line-item
     (e.g., revenue, EBITDA, free cash flow, margins)

2. **QueryPastCalls**  
   • Fetches the same metric or statement from prior earnings calls  
   • **Use when** comparing management's current commentary with its own
     previous statements adds context

3. **CompareWithPeers**
   • Provides the same metric from peer companies' calls or filings  
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
decide whether the stock price is likely to **increase ("Up") or decrease ("Down")**
one trading day after the earnings call, and assign a **Direction score** from 0 to 10.

---
{transcript}

Instructions:
1. Assign a confidence score (0 = strong conviction of decline, 5 = neutral, 10 = strong conviction of rise).

Respond in **exactly** this format:

<Couple of sentences of Explanation>
Direction : <0-10>

""".strip()