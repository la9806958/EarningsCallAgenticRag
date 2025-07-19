"""comparative_agent.py – Batch‑aware ComparativeAgent
======================================================
This rewrite lets `run()` accept a **list of fact‑dicts** (rather than one) so
all related facts can be analysed in a single LLM prompt.
• **Added:** Token usage tracking for cost monitoring
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase
from openai import OpenAI

from agents.prompts.prompts import comparative_agent_prompt

# -------------------------------------------------------------------------
# Token tracking
# -------------------------------------------------------------------------
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

class ComparativeAgent:
    """Compare a batch of facts against peer data stored in Neo4j."""

    def __init__(self, credentials_file: str = "credentials.json", model: str = "gpt-4o-mini", sector_map: dict = None) -> None:
        creds = json.loads(Path(credentials_file).read_text())
        self.client = OpenAI(api_key=creds["openai_api_key"])
        self.model = model
        self.driver = GraphDatabase.driver(
            creds["neo4j_uri"], auth=(creds["neo4j_username"], creds["neo4j_password"])
        )
        self.embedder = OpenAIEmbeddings(openai_api_key=creds["openai_api_key"])
        self.token_tracker = TokenTracker()
        self.sector_map = sector_map or {}

    # ------------------------------------------------------------------
    # Vector search helper
    # ------------------------------------------------------------------
    def _search_similar(
        self,
        query: str,
        exclude_ticker: str,
        top_k: int = 10,
        sector: str = None,
        ticker: str = None,
        use_batch_peer_query: bool = False
    ) -> List[Dict[str, Any]]:
        """
        If sector_map is provided, run the query for every ticker in the same sector (excluding exclude_ticker).
        If sector is not provided, infer it from ticker using the sector_map.
        Otherwise, default to original behavior.
        If use_batch_peer_query is True, use a single query with IN $peer_ticker_list.
        """
        tickers_in_sector = None
        if self.sector_map:
            # Infer sector if not provided
            if not sector:
                if ticker:
                    sector = self.sector_map.get(ticker)
                # If still no sector, just return [] instead of raising
                if not sector:
                    return []
            # Get all tickers in the same sector
            tickers_in_sector = [t for t, s in self.sector_map.items() if s == sector and t != exclude_ticker]
        
        try:
            vec = self.embedder.embed_query(query)
            with self.driver.session() as ses:
                all_results = []
                if tickers_in_sector:
                    if use_batch_peer_query:
                        res = ses.run(
                            """
                            CALL db.index.vector.queryNodes('fact_index', $topK, $vec)
                            YIELD node, score
                            WHERE node.ticker IN $peer_ticker_list AND score > $min_score
                            OPTIONAL MATCH (node)-[:HAS_VALUE]->(v:Value)
                            OPTIONAL MATCH (node)-[:EXPLAINED_BY]->(r:Reason)
                            RETURN node.text AS text, node.metric AS metric, v.content AS value,
                                   r.content AS reason, node.ticker AS ticker,
                                   node.quarter AS quarter, score
                            ORDER BY score DESC
                            LIMIT 10
                            """,
                            {"topK": top_k, "vec": vec, "peer_ticker_list": tickers_in_sector, "min_score": 0.3},
                        )
                        all_results.extend([dict(r) for r in res])
                    else:
                        for peer_ticker in tickers_in_sector:
                            res = ses.run(
                                """
                                CALL db.index.vector.queryNodes('fact_index', $topK, $vec)
                                YIELD node, score
                                WHERE node.ticker = $peer_ticker AND score > $min_score
                                OPTIONAL MATCH (node)-[:HAS_VALUE]->(v:Value)
                                OPTIONAL MATCH (node)-[:EXPLAINED_BY]->(r:Reason)
                                RETURN node.text AS text, node.metric AS metric, v.content AS value,
                                       r.content AS reason, node.ticker AS ticker,
                                       node.quarter AS quarter, score
                                ORDER BY score DESC
                                LIMIT 10
                                """,
                                {"topK": top_k, "vec": vec, "peer_ticker": peer_ticker, "min_score": 0.3},
                            )
                            all_results.extend([dict(r) for r in res])
                    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                    return all_results

                
                else:
                    # Default: original behavior
                    res = ses.run(
                        """
                        CALL db.index.vector.queryNodes('fact_index', $topK, $vec)
                        YIELD node, score
                        WHERE node.ticker <> $exclude_ticker AND score > $min_score
                        OPTIONAL MATCH (node)-[:HAS_VALUE]->(v:Value)
                        OPTIONAL MATCH (node)-[:EXPLAINED_BY]->(r:Reason)
                        RETURN node.text AS text, node.metric AS metric, v.content AS value,
                               r.content AS reason, node.ticker AS ticker,
                               node.quarter AS quarter, score
                        ORDER BY score DESC
                        LIMIT 10
                        """,
                        {"topK": top_k, "vec": vec, "exclude_ticker": exclude_ticker, "min_score": 0.3},
                    )
                    return [dict(r) for r in res]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Vector search helper for sector peers
    # ------------------------------------------------------------------
    def _search_similar_sector(self, query: str, sector: str, quarter: str, exclude_ticker: str, top_k: int = 10) -> List[Dict[str, Any]]:
        try:
            vec = self.embedder.embed_query(query)
            with self.driver.session() as ses:
                res = ses.run(
                    """
                    CALL db.index.vector.queryNodes('fact_index', $topK, $vec)
                    YIELD node, score
                    WHERE node.sector = $sector AND node.quarter = $quarter AND node.ticker <> $exclude_ticker
                    OPTIONAL MATCH (node)-[:HAS_VALUE]->(v:Value)
                    OPTIONAL MATCH (node)-[:EXPLAINED_BY]->(r:Reason)
                    RETURN node.text AS text, node.metric AS metric, v.content AS value,
                           r.content AS reason, node.ticker AS ticker,
                           node.quarter AS quarter, node.sector AS sector, score
                    ORDER BY score DESC
                    LIMIT 10
                    """,
                    {"topK": top_k, "vec": vec, "sector": sector, "quarter": quarter, "exclude_ticker": exclude_ticker},
                )
                return [dict(r) for r in res]
        except Exception:
            return []

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
        sector: str | None = None,
        top_k: int = 5,  # Lowered from 50 to 10
    ) -> str:
        """Analyse a batch of facts; return one consolidated LLM answer."""
        if not facts:
            return "No facts supplied."
        # Reset token tracker for this run
        self.token_tracker = TokenTracker()

        # --- Per-fact similarity search and aggregation ---
        all_similar = []
        for fact in facts:
            query = self._to_query(fact)
            if sector:
                similar = self._search_similar(query, ticker, top_k=top_k, sector=sector)
            else:
                similar = self._search_similar(query, ticker, top_k=top_k)
            # Optionally, attach the current metric for context
            for sim in similar:
                sim["current_metric"] = fact.get("metric", "")
            all_similar.extend(similar)

        # Optionally deduplicate similar facts (by metric, value, ticker, quarter)
        seen = set()
        deduped_similar = []
        for sim in all_similar:
            key = (sim.get("metric"), sim.get("value"), sim.get("ticker"), sim.get("quarter"))
            if key not in seen:
                deduped_similar.append(sim)
                seen.add(key)

        if not deduped_similar:
            return None

        # --- Craft prompt --------------------------------------------------
        prompt = comparative_agent_prompt(facts, deduped_similar, self_ticker=ticker)
        prompt = (
            "The following is a *batch* of facts for the same company/quarter:\n"
            + json.dumps(facts, indent=2)
            + "\n\n" + prompt
        )
        
        # Print the full prompt for debugging
        #print(f"\n{'='*80}")
        #print(f"COMPARATIVE AGENT PROMPT for {ticker}/{quarter}")
        #print(f"{'='*80}")
        #print(prompt)
        #print(f"{'='*80}\n")

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial forecasting assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                top_p=1,
            )
            
            # Track token usage
            if hasattr(resp, 'usage') and resp.usage:
                self.token_tracker.add_usage(
                    input_tokens=resp.usage.prompt_tokens,
                    output_tokens=resp.usage.completion_tokens,
                    model=self.model
                )
            
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            return f"❌ ComparativeAgent error: {exc}"

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.driver.close()
