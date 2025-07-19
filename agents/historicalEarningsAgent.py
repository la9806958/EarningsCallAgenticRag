"""historical_earnings_agent.py – Batch‑aware version
====================================================
Compare current facts with the firm's own historical facts.
• **Added:** Token usage tracking for cost monitoring
"""

import json
from openai import OpenAI
from neo4j import GraphDatabase
from typing import List, Dict, Any
from pathlib import Path
import re

from langchain_openai import OpenAIEmbeddings
from agents.prompts.prompts import historical_earnings_agent_prompt

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

class HistoricalEarningsAgent:
    """Compare current facts with the firm's own historical facts."""

    def __init__(self, credentials_file: str = "credentials.json", model: str = "gpt-4o-mini") -> None:
        creds = json.loads(Path(credentials_file).read_text())
        self.client = OpenAI(api_key=creds["openai_api_key"])
        self.model = model
        self.driver = GraphDatabase.driver(
            creds["neo4j_uri"], auth=(creds["neo4j_username"], creds["neo4j_password"])
        )
        self.embedder = OpenAIEmbeddings(openai_api_key=creds["openai_api_key"])
        self.token_tracker = TokenTracker()

    # ------------------------------------------------------------------
    # Neo4j fetch helper (simple filter – same ticker, prior quarters)
    # ------------------------------------------------------------------
    def _fetch_past_facts(self, ticker: str, top_k: int = 10) -> List[Dict[str, Any]]:
        with self.driver.session() as ses:
            result = ses.run(
                """
                MATCH (f:Fact {ticker:$ticker})
                RETURN f.metric AS metric, f.value AS value, f.reason AS reason,
                       f.text   AS text,   f.quarter AS quarter
                ORDER BY f.quarter DESC
                LIMIT $top
                """,
                {"ticker": ticker, "top": top_k},
            )
            return [dict(r) for r in result]


    def get_similar_facts_by_embedding(self, fact: Dict[str, Any], ticker: str, current_quarter: str, top_n: int = 5) -> List[Dict[str, Any]]:
        try:
            embedding = fact.get("embedding")
            if embedding is None:
                text = f"{fact['ticker']} | {fact['metric']} | {fact['type']}"
                embedding = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                ).data[0].embedding
            from neo4j import GraphDatabase
            creds = json.loads(Path(self.driver._uri.split('bolt://')[-1].split(':')[0] + '/credentials.json').read_text()) if hasattr(self.driver, '_uri') else json.loads(Path('credentials.json').read_text())
            driver = self.driver
            with driver.session() as session:
                try:
                    # Get previous year's quarter for YoY comparison
                    prev_year_quarter = self._get_prev_year_quarter(current_quarter)
                    
                    result = session.run(
                        """
                        CALL db.index.vector.queryNodes('fact_index', $top_n, $embedding) YIELD node, score
                        WHERE node.ticker = $ticker AND score > $min_score
                        OPTIONAL MATCH (node)-[:HAS_VALUE]->(v:Value)
                        OPTIONAL MATCH (node)-[:EXPLAINED_BY]->(r:Reason)
                        RETURN node.metric AS metric, v.content AS value, r.content AS reason, node.embedding AS embedding, node.quarter AS quarter, node.type AS type, score
                        ORDER BY score DESC
                        LIMIT 10
                        """,
                        embedding=embedding,
                        top_n=top_n,
                        ticker=ticker,
                        min_score=0.3
                    )
                    all_facts = [r.data() for r in result]
                    
                    # Also search specifically for previous year's quarter
                    prev_year_result = session.run(
                        """
                        MATCH (f:Fact {ticker: $ticker, quarter: $prev_year_quarter})
                        OPTIONAL MATCH (f)-[:HAS_VALUE]->(v:Value)
                        OPTIONAL MATCH (f)-[:EXPLAINED_BY]->(r:Reason)
                        RETURN f.metric AS metric, v.content AS value, r.content AS reason, f.embedding AS embedding, f.quarter AS quarter, f.type AS type, 1.0 AS score
                        """,
                        ticker=ticker,
                        prev_year_quarter=prev_year_quarter
                    )
                    prev_year_facts = [r.data() for r in prev_year_result]
                    
                    # Combine and filter facts
                    combined_facts = all_facts + prev_year_facts
                    
                    # Keep facts from strictly earlier quarters AND previous year's quarter
                    # Explicitly exclude the current quarter
                    filtered_facts = [
                        f for f in combined_facts
                        if f.get("quarter") and (
                            (self._q_sort_key(f.get("quarter")) < self._q_sort_key(current_quarter) or
                            f.get("quarter") == prev_year_quarter) and
                            f.get("quarter") != current_quarter  # Explicitly exclude current quarter
                        )
                    ]
                    
                    return filtered_facts
                except Exception as e:
                    print(f"[ERROR] Neo4j vector query failed in get_similar_facts_by_embedding: {e}")
                    # Fallback: fetch all and compute similarity in Python
                    try:
                        result = session.run(
                            """
                            MATCH (f:Fact {ticker: $ticker})
                            OPTIONAL MATCH (f)-[:HAS_VALUE]->(v:Value)
                            OPTIONAL MATCH (f)-[:EXPLAINED_BY]->(r:Reason)
                            WHERE exists(f.embedding)
                            RETURN f.metric AS metric, v.content AS value, r.content AS reason, f.embedding AS embedding, f.quarter AS quarter, f.type AS type
                            """,
                            ticker=ticker
                        )
                        all_facts = [r.data() for r in result]
                        import numpy as np
                        def cosine_sim(a, b):
                            a = np.array(a)
                            b = np.array(b)
                            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                        for f in all_facts:
                            f["score"] = cosine_sim(embedding, f["embedding"])
                        all_facts.sort(key=lambda x: x["score"], reverse=True)
                        # Only keep facts from strictly earlier quarters
                        # Explicitly exclude the current quarter
                        filtered_facts = [
                            f for f in all_facts
                            if f.get("quarter") and (
                                self._q_sort_key(f.get("quarter")) < self._q_sort_key(current_quarter) and
                                f.get("quarter") != current_quarter  # Explicitly exclude current quarter
                            )
                        ]
                        return filtered_facts[:top_n]
                    except Exception as e2:
                        print(f"[ERROR] Fallback similarity search failed in get_similar_facts_by_embedding: {e2}")
                        return None
        except Exception as e:
            print(f"[ERROR] get_similar_facts_by_embedding failed: {e}")
            return None

    # Helper for quarter comparison
    @staticmethod
    def _q_sort_key(q: str):
        m = re.match(r"(\d{4})-Q(\d)", q)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    
    # Helper to get previous year's same quarter
    @staticmethod
    def _get_prev_year_quarter(quarter: str) -> str:
        m = re.match(r"(\d{4})-Q(\d)", quarter)
        if m:
            year, q = int(m.group(1)), int(m.group(2))
            return f"{year-1}-Q{q}"
        return quarter

    # ------------------------------------------------------------------
    def run(
        self,
        facts: List[Dict[str, str]],
        ticker: str,
        quarter: str,
        top_k: int = 5,  # Lowered from 50 to 10
    ) -> str:
        """Batch: For each fact, find similar historical facts, aggregate, and run the LLM prompt once for the batch."""
        if not facts:
            return "❌ No facts supplied."

        # Reset token tracker for this run
        self.token_tracker = TokenTracker()

        all_similar = []
        for fact in facts:
            similar_facts = self.get_similar_facts_by_embedding(fact, ticker, quarter, top_n=top_k)
            if not similar_facts:
                continue
            for sim in similar_facts:
                sim["current_metric"] = fact.get("metric", "")
                sim.pop("embedding", None)
            all_similar.extend(similar_facts)

        # Optionally deduplicate similar facts
        seen = set()
        deduped_similar = []
        for sim in all_similar:
            key = (sim.get("metric"), sim.get("value"), sim.get("ticker"), sim.get("quarter"))
            if key not in seen:
                deduped_similar.append(sim)
                seen.add(key)

        if not deduped_similar:
            return None

        prompt = historical_earnings_agent_prompt(facts, deduped_similar, quarter)
        
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

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.driver.close()
