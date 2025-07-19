"""historical_performance_agent.py – Batch‑aware version
=======================================================
`run()` now accepts a **list of facts** so the model can compare all of them to
past financial statements in a single prompt.
• **Added:** Token usage tracking for cost monitoring
"""

from __future__ import annotations
from pathlib import Path
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import re

from openai import OpenAI
from agents.prompts.prompts import financials_statement_agent_prompt

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

class HistoricalPerformanceAgent:
    """Compare current-quarter facts with prior financial statements."""

    def __init__(self, credentials_file: str = "credentials.json", model: str = "gpt-4o-mini") -> None:
        creds = json.loads(Path(credentials_file).read_text())
        self.client = OpenAI(api_key=creds["openai_api_key"])
        self.model = model
        self.credentials_file = credentials_file  # Save as instance attribute
        self.token_tracker = TokenTracker()

    # ------------------------------------------------------------------
    @staticmethod
    def _pretty_json(obj: Any) -> str:
        if obj in (None, "", "[]"):
            return "[]"
        try:
            return json.dumps(json.loads(obj) if isinstance(obj, str) else obj, indent=2)
        except Exception:
            return str(obj)

    # ------------------------------------------------------------------
    def get_similar_facts_by_embedding(self, fact: Dict[str, Any], ticker: str, quarter: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Find top-N most similar facts for the same ticker and type='Result' using embedding cosine similarity, fetching value via HAS_VALUE."""
        try:
            if isinstance(fact, str):
                return None
            embedding = fact.get("embedding")
            if embedding is None:
                try:
                    text = f"{fact['ticker']} | {fact['metric']} | {fact['type']}"
                except Exception:
                    return None
                try:
                    embedding = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    ).data[0].embedding
                except Exception as e:
                    print(f"[ERROR] Embedding creation failed in get_similar_facts_by_embedding: {e}")
                    return None
            from neo4j import GraphDatabase
            creds = json.loads(Path(self.credentials_file).read_text())
            driver = GraphDatabase.driver(
                creds["neo4j_uri"], auth=(creds["neo4j_username"], creds["neo4j_password"])
            )
            with driver.session() as session:
                try:
                    # Get previous year's quarter for YoY comparison
                    prev_year_quarter = self._get_prev_year_quarter(quarter)
                    
                    result = session.run(
                        """
                        CALL db.index.vector.queryNodes('fact_index', $top_n, $embedding) YIELD node, score
                        WHERE node.ticker = $ticker AND node.type = 'Result' AND score > $min_score
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
                        MATCH (f:Fact {ticker: $ticker, quarter: $prev_year_quarter, type: 'Result'})
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
                            (self._q_sort_key(f.get("quarter")) < self._q_sort_key(quarter) or
                            f.get("quarter") == prev_year_quarter) and
                            f.get("quarter") != quarter  # Explicitly exclude current quarter
                        )
                    ]
                    
                    return filtered_facts
                except Exception as e:
                    print(f"[ERROR] Neo4j vector query failed in get_similar_facts_by_embedding: {e}")
                    # Fallback: fetch all and compute similarity in Python
                    try:
                        result = session.run(
                            """
                            MATCH (f:Fact {ticker: $ticker, type: 'Result'})
                            OPTIONAL MATCH (f)-[:HAS_VALUE]->(v:Value)
                            OPTIONAL MATCH (f)-[:EXPLAINED_BY]->(r:Reason)
                            WHERE exists(f.embedding)
                            RETURN f.metric AS metric, v.content AS value, r.content AS reason, f.embedding AS embedding, f.quarter AS quarter, f.type AS type
                            """,
                            ticker=ticker
                        )
                        all_facts = [r.data() for r in result]
                        def cosine_sim(a, b):
                            a = np.array(a)
                            b = np.array(b)
                            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                        for f in all_facts:
                            f["score"] = cosine_sim(embedding, f["embedding"])
                        all_facts.sort(key=lambda x: x["score"], reverse=True)
                        return all_facts[:top_n]
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

    def run(self, facts: List[Dict[str, str]], row, quarter, ticker: Optional[str] = None, top_n: int = 5) -> str:  # Lowered from 50 to 10
        """Batch: Compare all facts to all top-N similar past facts by embedding, and run the LLM prompt once for the batch."""
        if not facts:
            return "No facts provided."

        # Reset token tracker for this run
        self.token_tracker = TokenTracker()

        from agents.prompts.prompts import financials_statement_agent_prompt
        all_similar = []
        for fact in facts:
            similar_facts = self.get_similar_facts_by_embedding(fact, ticker or row.get("ticker"), quarter, top_n=top_n)
            # Filter out facts with no value or reason, and only keep those from previous quarters
            # Explicitly exclude the current quarter
            filtered_similar = [
                f for f in similar_facts
                if f.get("quarter") and (
                    self._q_sort_key(f.get("quarter")) < self._q_sort_key(quarter) and
                    f.get("quarter") != quarter  # Explicitly exclude current quarter
                )
            ]
            if not filtered_similar:
                filtered_similar = []
            # Attach the metric of the current fact to each similar fact for context
            for sim in filtered_similar:
                sim["current_metric"] = fact.get("metric", "")
                if "embedding" in sim:
                    del sim["embedding"]
            all_similar.extend(filtered_similar)

        if not all_similar:
            return None

        # Call the prompt ONCE for the batch
        prompt = financials_statement_agent_prompt(
            fact=facts,  # pass the list of facts
            similar_facts=all_similar,  # pass the aggregated similar facts
            quarter=quarter
        )
        
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

    def generate_embeddings_for_facts(self, batch_size: int = 50):
        """Generate and store embeddings for Fact nodes using ticker, metric, and type as input."""
        from openai import OpenAI
        creds = json.loads(Path(self.credentials_file).read_text())
        driver = self.driver
        client = OpenAI(api_key=creds["openai_api_key"])

        with driver.session() as session:
            # Fetch all Fact nodes missing an embedding
            result = session.run("""
                MATCH (f:Fact)
                WHERE f.embedding IS NULL
                RETURN id(f) AS id, f.ticker AS ticker, f.metric AS metric, f.type AS type
            """)
            facts = [r.data() for r in result]

            for i in range(0, len(facts), batch_size):
                batch = facts[i:i+batch_size]
                texts = [
                    f"{f['ticker']} | {f['metric']} | {f['type']}" for f in batch
                ]
                # Generate embeddings in batch
                embeddings = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts
                ).data

                # Write embeddings back to Neo4j
                for fact, emb in zip(batch, embeddings):
                    session.run(
                        """
                        MATCH (f:Fact) WHERE id(f) = $id
                        SET f.embedding = $embedding
                        """,
                        id=fact["id"],
                        embedding=emb.embedding
                    )
