"""main_agent.py – End‑to‑end earnings‑call RAG pipeline
======================================================
• Accepts `credentials_file=` or `credentials_path=` (back‑compat)
• Uses shared prompt templates for extraction, delegation, and verdicts
• **Fix:** reuses the existing OpenAI client instead of re‑instantiating without API key
• Removed duplicate stray code at bottom of file
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from collections import defaultdict

# ---- Centralised prompt imports -------------------------------------------
from agents.prompts.prompts import (
    facts_extraction_prompt,  # chunk‑level extraction
    facts_delegation_prompt,  # routing to helper agents
    main_agent_prompt,        # fact‑level verdict
)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
ITEM_HEADER = re.compile(r"### (?:Item|Fact) No\. (\d+)")
FIELD = re.compile(r"\*\*(.+?):\*\*\s*(.+)")


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
        resp = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0)
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
        id2fact = {f["fact_no"]: f for f in items}
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for fid, tools in tool_map.items():
            for t in tools:
                buckets[t].append(id2fact[fid])
        return buckets

# ---------------------------------------------------------------------
# 2) Delegation (batched)  ––– REPLACE THE WHOLE METHOD
# ---------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def delegate(
        self,
        items: List[Dict[str, Any]],
        ticker: str,
        quarter: str,
        peers: Sequence[str],
        row: Dict[str, Any],
    ) -> None:
        """Route facts, call each helper **in parallel**, store batch-level notes."""
    
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
                tool_map.setdefault(int(n), []).append(tool.strip())
    
        # Step 3: Bucket facts by tool
        buckets = self._bucket_by_tool(tool_map, items)
    
        # Step 4: Run agents in parallel
        self._batch_notes: Dict[str, str] = {}
        tasks = []
    
        with ThreadPoolExecutor(max_workers=3) as executor:
            if self.financials_agent and "InspectPastStatements" in buckets:
                tasks.append(executor.submit(
                    lambda: ("financials", self.financials_agent.run(buckets["InspectPastStatements"], row,quarter))
                ))

            """
            if self.past_calls_agent and "QueryPastCalls" in buckets:
                tasks.append(executor.submit(
                    lambda: ("past", self.past_calls_agent.run(buckets["QueryPastCalls"], ticker, quarter))
                ))
            """
            
            """
            if self.comparative_agent and "CompareWithPeers" in buckets:
                tasks.append(executor.submit(
                    lambda: ("peers", self.comparative_agent.run(buckets["CompareWithPeers"], ticker, quarter, peers))
                ))

            """
            
            for future in as_completed(tasks):
                try:
                    key, result = future.result()
                    self._batch_notes[key] = result
                except Exception as e:
                    print(f"⚠️ Agent failed: {e}")
    
        # Optional: attach tool usage to each fact
        for f in items:
            f["tools"] = tool_map.get(f["fact_no"], [])

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
    
        # inside class MainAgent -----------------------------------------------
    def summarise(self, items: list[dict[str, Any]],
                  memory_txt: str | None = None) -> str:
        notes = {
            "financials": self._batch_notes.get("financials", "N/A"),
            "past"      : self._batch_notes.get("past", "N/A"),
            "peers"     : self._batch_notes.get("peers", "N/A"),
        }
    
        core_prompt = main_agent_prompt(notes)
        final_prompt = f"{memory_txt}\n\n{core_prompt}" if memory_txt else core_prompt

        # TODO: remove
        final_prompt = core_prompt
        return notes, self._chat(final_prompt,
                                 system="You are a seasoned portfolio manager.")

    # ---------------------------------------------------------------------
    # 4) Orchestrator
    # ---------------------------------------------------------------------
    def run(self, row: Dict[str, Any], mem_txt: str | None = None) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # 1) Fact extraction
        # ------------------------------------------------------------------
        facts = self.extract(row["transcript"])
    
        # ------------------------------------------------------------------
        # 2) Peer discovery (best-effort, safe-fallback)
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
        # 3) Delegate facts to helper agents
        # ------------------------------------------------------------------
        self.delegate(facts, row["ticker"], row["q"], peers, row)
    
        # ------------------------------------------------------------------
        # 4) Summarise with memory
        # ------------------------------------------------------------------
        notes, decision = self.summarise(facts, memory_txt=mem_txt)
    
        # ------------------------------------------------------------------
        # 5) Return everything (memory included for logging/debug)
        # ------------------------------------------------------------------
        return {
            "items": facts,
            "notes": notes,
            "summary": decision,
            "memory": mem_txt or "",
        }
