"""comparative_agent.py – Batch‑aware ComparativeAgent
======================================================
This rewrite lets `run()` accept a **list of fact‑dicts** (rather than one) so
all related facts can be analysed in a single LLM prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase
from openai import OpenAI

from agents.prompts.prompts import comparative_agent_prompt

# -------------------------------------------------------------------------
class ComparativeAgent:
    """Compare a batch of facts against peer data stored in Neo4j."""

    def __init__(self, credentials_file: str = "credentials.json", model: str = "gpt-4o-mini") -> None:
        creds = json.loads(Path(credentials_file).read_text())
        self.client = OpenAI(api_key=creds["openai_api_key"])
        self.model = model
        self.driver = GraphDatabase.driver(
            creds["neo4j_uri"], auth=(creds["neo4j_username"], creds["neo4j_password"])
        )
        self.embedder = OpenAIEmbeddings(openai_api_key=creds["openai_api_key"])

    # ------------------------------------------------------------------
    # Vector search helper
    # ------------------------------------------------------------------
    def _search_similar(self, query: str, top_k: int = 30) -> List[Dict[str, Any]]:
        vec = self.embedder.embed_query(query)
        with self.driver.session() as ses:
            res = ses.run(
                """
                CALL db.index.vector.queryNodes('fact_index', $topK, $vec)
                YIELD node, score
                RETURN node.text AS text, node.metric AS metric, node.value AS value,
                       node.reason AS reason, node.ticker AS ticker,
                       node.quarter AS quarter, score
                ORDER BY score DESC
                """,
                {"topK": top_k, "vec": vec},
            )
            return [dict(r) for r in res]

    # ------------------------------------------------------------------
    def _to_query(self, fact: Dict[str, str]) -> str:
        parts = []
        if fact.get("metric"):
            parts.append(f"Metric: {fact['metric']}")
        if fact.get("value"):
            parts.append(f"Value: {fact['value']}")
        if fact.get("context"):
            parts.append(f"Reason: {fact['context']}")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    def run(
        self,
        facts: List[Dict[str, str]],
        ticker: str,
        quarter: str,
        peers: Sequence[str] | None = None,
        top_k: int = 50,
    ) -> str:
        """Analyse a batch of facts; return one consolidated LLM answer."""
        if not facts:
            return "No facts supplied."

        # --- Build a combined query string (concatenate all metrics/values) ---
        query = " || ".join(self._to_query(f) for f in facts if f)
        related = self._search_similar(query, top_k=top_k)
        print(related)
        # --- Craft prompt --------------------------------------------------
        prompt = comparative_agent_prompt({"batch_of": len(facts)}, related)
        # The template expects a 'fact' field; we pass a stub + list length.
        prompt = (
            "The following is a *batch* of facts for the same company/quarter:\n"
            + json.dumps(facts, indent=2)
            + "\n\n" + prompt
        )
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial forecasting assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            return f"❌ ComparativeAgent error: {exc}"

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.driver.close()
