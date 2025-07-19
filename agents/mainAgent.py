"""main_agent.py – End‑to‑end earnings‑call RAG pipeline
======================================================
• Accepts `credentials_file=` or `credentials_path=` (back‑compat)
• Uses shared prompt templates for extraction, delegation, and verdicts
• **Fix:** reuses the existing OpenAI client instead of re‑instantiating without API key
• Removed duplicate stray code at bottom of file
• **Added:** Token usage tracking for cost monitoring
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil

from openai import OpenAI
from tqdm import tqdm

from collections import defaultdict

# ---- Centralised prompt imports -------------------------------------------
from agents.prompts.prompts import (
    facts_extraction_prompt,  # chunk‑level extraction
    facts_delegation_prompt,  # routing to helper agents
    main_agent_prompt,        # fact‑level verdict
    # Add import for formatting QoQ facts
)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
ITEM_HEADER = re.compile(r"### (?:Item|Fact) No\. (\d+)")
FIELD = re.compile(r"\*\*(.+?):\*\*\s*(.+)")

# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------
class TokenTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.model_used = "gpt-4o-mini"  # default
    
    def add_usage(self, input_tokens: int, output_tokens: int, model: str = None):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        if model:
            self.model_used = model
        
        # Calculate cost based on model
        if "gpt-4o" in model.lower():
            self.total_cost_usd += (input_tokens * 0.000005) + (output_tokens * 0.000015)
        elif "gpt-4" in model.lower():
            self.total_cost_usd += (input_tokens * 0.00003) + (output_tokens * 0.00006)
        elif "gpt-3.5" in model.lower():
            self.total_cost_usd += (input_tokens * 0.0000015) + (output_tokens * 0.000002)
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "model": self.model_used,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cost_usd": self.total_cost_usd
        }

def _parse_items(raw: str) -> List[Dict[str, str]]:
    """Convert markdown blocks into structured dicts."""
    pieces = ITEM_HEADER.split(raw)
    out: List[Dict[str, str]] = []
    for i in range(1, len(pieces), 2):
        num = int(pieces[i])
        body = pieces[i + 1]
        fields = {k.strip(): v.strip() for k, v in FIELD.findall(body)}
        out.append(
            {
                "fact_no": num,
                "type": fields.get("Type", ""),
                "metric": fields.get("Metric", ""),
                "value": fields.get("Value", ""),
                "context": fields.get("Context", fields.get("Reason", "")),
            }
        )
    return out

# ---------------------------------------------------------------------------
# Helper‑agent protocol
# ---------------------------------------------------------------------------

class BaseHelperAgent:
    def run(self, facts: List[Dict[str, Any]], ticker: str, quarter: str, *args) -> str:  # noqa: D401
        """Return a short analysis covering *all* supplied facts."""
        raise NotImplementedError

# ---------------------------------------------------------------------------
# MainAgent
# ---------------------------------------------------------------------------

@dataclass
class MainAgent:
    credentials_file: Union[str, Path] | None = None
    credentials_path: Union[str, Path] | None = None
    model: str = "gpt-4o-mini"

    financials_agent: BaseHelperAgent | None = None
    past_calls_agent: BaseHelperAgent | None = None
    comparative_agent: BaseHelperAgent | None = None

    client: OpenAI = field(init=False)
    token_tracker: TokenTracker = field(default_factory=TokenTracker)

    # ------------ init ----------------------------------------------------
    def __post_init__(self):
        cred_path = Path(self.credentials_file or self.credentials_path or "")
        if not cred_path.exists():
            raise FileNotFoundError("Credentials file not found.")
        api_key = json.loads(cred_path.read_text())["openai_api_key"]
        self.client = OpenAI(api_key=api_key)

    # ------------ internal LLM helper ------------------------------------
    def _chat(self, prompt: str, system: str = "") -> str:
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0, top_p=1)
        
        # Track token usage
        if hasattr(resp, 'usage') and resp.usage:
            self.token_tracker.add_usage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                model=self.model
            )
        
        return resp.choices[0].message.content.strip()

    # ---------------------------------------------------------------------
    # 1) Extraction (single call, no chunking)
    # ---------------------------------------------------------------------
    def extract(self, transcript: str) -> List[Dict[str, str]]:
        raw = self._chat(facts_extraction_prompt(transcript), system="You are a precise extraction bot.")
        return _parse_items(raw)

    # ---------------------------------------------------------------------
    # 2) Delegation (batched)
    # ---------------------------------------------------------------------
    @staticmethod
    def _bucket_by_tool(tool_map: Dict[int, List[str]], items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        id2fact = {i: f for i, f in enumerate(items)}
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for fid, tools in tool_map.items():
            for t in tools:
                if fid in id2fact:
                    buckets[t].append(id2fact[fid])
        return buckets

    def delegate(
        self,
        items: List[Dict[str, Any]],
        ticker: str,
        quarter: str,
        peers: Sequence[str],
        row: Dict[str, Any],
    ) -> None:
        """Route facts, call each helper **in parallel**, store batch-level notes."""

        def _ensure_list_of_dicts(facts):
            if isinstance(facts, str):
                try:
                    facts = json.loads(facts)
                except Exception:
                    print(f"\u274c Could not parse string as JSON: {facts[:200]}")
                    raise TypeError(f"Expected list of dicts, got string: {facts[:200]}")
            if not isinstance(facts, list) or (facts and not isinstance(facts[0], dict)):
                print(f"\u274c Expected list of dicts, got: {type(facts)} with first element {type(facts[0]) if facts else 'empty'}")
                raise TypeError(f"Expected list of dicts, got: {type(facts)} with first element {type(facts[0]) if facts else 'empty'}")
            return facts

        # Step 1: Route facts using LLM
        routing_txt = self._chat(
            facts_delegation_prompt(items).replace("<list of peer tickers>", ", ".join(peers)),
            system="Route each fact.",
        )

        # Step 2: Parse routing map
        tool_map: Dict[int, List[str]] = {}
        for line in routing_txt.splitlines():
            if ":" not in line:
                continue
            tool, nums = line.split(":", 1)
            for n in re.findall(r"\d+", nums):
                idx = int(n) - 1
                tool_map.setdefault(idx, []).append(tool.strip())

        # Step 3: Bucket facts by tool
        buckets = self._bucket_by_tool(tool_map, items)

        # Step 4: Run agents in parallel, with chunking (batch size 10)
        self._batch_notes: Dict[str, str] = {}
        tasks = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            if self.financials_agent and "InspectPastStatements" in buckets:
                def run_financials():
                    facts_for_financials = _ensure_list_of_dicts(buckets["InspectPastStatements"])
                    # Remove chunking: pass all facts at once
                    res = self.financials_agent.run(facts_for_financials, row, quarter, ticker)
                    return ("financials", res)
                tasks.append(executor.submit(run_financials))

            if self.past_calls_agent and "QueryPastCalls" in buckets:
                facts_for_past = _ensure_list_of_dicts(buckets["QueryPastCalls"])
                def run_past_calls():
                    # Remove chunking: pass all facts at once
                    res = self.past_calls_agent.run(facts_for_past, ticker, quarter)
                    return ("past", res)
                tasks.append(executor.submit(run_past_calls))
            
            if self.comparative_agent and "CompareWithPeers" in buckets:
                facts_for_peers = _ensure_list_of_dicts(buckets["CompareWithPeers"])
                sector = row.get("sector") if isinstance(row, dict) else getattr(row, "sector", None)
                def run_comparative():
                    # Remove chunking: pass all facts at once
                    res = self.comparative_agent.run(facts_for_peers, ticker, quarter, peers, sector=sector)
                    return ("peers", res)
                tasks.append(executor.submit(run_comparative))

            for future in as_completed(tasks):
                try:
                    key, result = future.result()
                    self._batch_notes[key] = result
                    if result is None:
                        print(f"[AGENT LOG] Agent '{key}' failed or timed out for ticker={ticker}, quarter={quarter} (returned None)")
                except Exception as e:
                    print(f"[AGENT LOG] Agent '{key}' failed with exception for ticker={ticker}, quarter={quarter}: {e}")
                    print(f"\u26a0\ufe0f Agent failed: {e}")

        # Optional: attach tool usage to each fact
        for i, f in enumerate(items):
            f["tools"] = tool_map.get(i, [])

    # ---------------------------------------------------------------------
    # 3) Summary
    # ---------------------------------------------------------------------

    """
    def _fact_verdict(self, fact: Dict[str, Any]) -> str:
        a = fact.get("agent_analysis", {})
        return self._chat(
            main_agent_prompt(
                metric=fact["metric"],
                value=fact["value"],
                reason=fact["context"],
                financials_summary=a.get("financials_agent", "N/A"),
                past_calls_summary=a.get("past_calls_agent", "N/A"),
                comparative_summary=a.get("comparative_agent", "N/A"),
            )
        
        )
    """
    
    def _flatten_notes(self, note):
        if isinstance(note, list):
            return "\n\n".join([n for n in note if n])
        return note if note is not None else ""

    def summarise(self, items: list[dict[str, Any]],
                  memory_txt: str | None = None,
                  original_transcript: str | None = None,
                  financial_statements_facts: str | None = None) -> str:
        notes = {
            "financials": self._flatten_notes(self._batch_notes.get("financials", None)),
            "past"      : self._flatten_notes(self._batch_notes.get("past", None)),
            "peers"     : self._flatten_notes(self._batch_notes.get("peers", None)),
        }
        # If all three are None, set to dummy string
        if all(v is None or v == "" for v in notes.values()):
            notes = {"financials": "N/A", "past": "N/A", "peers": "N/A"}

        # Format and include QoQChange facts as a dedicated section
        qoq_facts = [f for f in items if f.get('type') == 'YoYChange']
        def format_qoq_facts(qoq_facts):
            if not qoq_facts:
                return "No YoY changes available."
            lines = []
            for f in qoq_facts:
                metric = f.get('metric', '?')
                value = f.get('value', '?')
                quarter = f.get('quarter', '?')
                reason = f.get('reason', '')
                # Format value as a percentage with 2 decimal places
                try:
                    pct_value = float(value) * 100
                    value_str = f"{pct_value:.2f}%"
                except Exception:
                    value_str = str(value)
                lines.append(f"• {metric}: {value_str} ({quarter}) {reason}")
            return '\n'.join(lines)
        qoq_section = format_qoq_facts(qoq_facts)

        core_prompt = main_agent_prompt(notes, original_transcript=original_transcript, memory_txt=memory_txt, financial_statements_facts=financial_statements_facts)

        if core_prompt is None:
            core_prompt = "No summary available."
        final_prompt = f"{memory_txt}\n\n{core_prompt}" if memory_txt else core_prompt

        # TODO: remove
        final_prompt = core_prompt
        # Print the full main agent prompt for debugging
        print("\n==== MAIN AGENT FULL PROMPT ====")
        print(final_prompt)
        print("===============================\n")
        return notes, self._chat(final_prompt,
                                 system="You are a seasoned portfolio manager.")

    # ---------------------------------------------------------------------
    # 4) Orchestrator
    # ---------------------------------------------------------------------
    def run(self, facts: List[Dict[str, Any]], row: Dict[str, Any], mem_txt: str | None = None, original_transcript: str | None = None, financial_statements_facts: str | None = None) -> Dict[str, Any]:
        # Reset token tracker for this run
        self.token_tracker = TokenTracker()
        
        # ------------------------------------------------------------------
        # 1) Peer discovery (best-effort, safe-fallback)
        # ------------------------------------------------------------------
        try:
            peers_resp = self._chat(
                f"Give 3 peer tickers for {row['ticker']} as a JSON list of strings."
            )
            peers = json.loads(peers_resp)
            if not isinstance(peers, list):
                peers = []
        except Exception:
            peers = []
    
        # ------------------------------------------------------------------
        # 2) Delegate facts to helper agents
        # ------------------------------------------------------------------
        self.delegate(facts, row["ticker"], row["q"], peers, row)
    
        # ------------------------------------------------------------------
        # 3) Summarise with memory
        # ------------------------------------------------------------------
        notes, decision = self.summarise(facts, memory_txt=mem_txt, original_transcript=row["transcript"], financial_statements_facts=financial_statements_facts)
    
        # ------------------------------------------------------------------
        # 4) Return everything (memory included for logging/debug)
        # ------------------------------------------------------------------
        return {
            "items": facts,
            "notes": notes,
            "summary": decision,
            "memory": mem_txt or "",
            "token_usage": self.token_tracker.get_summary()
        }
