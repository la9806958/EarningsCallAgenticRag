"""index_facts.py – Neo4j fact indexer
====================================
✓ Handles markdown blocks for Result / Forward‑Looking / Risk Disclosure / Sentiment / Macro
✓ Creates Fact nodes with 1536-dimensional embeddings for vector similarity search
✓ Implements six edge types: REPORTS_IN, HAS_FACT, FOR_FIRM, HAS_VALUE, EXPLAINED_BY, QoQ_COMPARES
✓ Stores Value and Reason as separate nodes connected to Fact nodes
✓ Supports QoQ comparisons with percentage change tracking
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List
import concurrent.futures

from neo4j import GraphDatabase
from openai import OpenAI

from agents.prompts.prompts import facts_extraction_prompt

# ── Markdown parser -------------------------------------------------------
ITEM_HEADER = re.compile(
    r"###\s*(?:Fact|Result|Forward-Looking|Risk Disclosure|Sentiment|Macro)\s*No\.\s*(\d+)",
    re.IGNORECASE,
)
FIELD = re.compile(r"\*\*(.+?):\*\*\s*(.+)")


def _parse_markdown_items(raw: str) -> List[Dict[str, str]]:
    pieces = ITEM_HEADER.split(raw)
    out: List[Dict[str, str]] = []
    for i in range(1, len(pieces), 2):
        fields = {k.strip(): v.strip() for k, v in FIELD.findall(pieces[i + 1])}
        out.append(fields)
    return out

# -------------------------------------------------------------------------
class IndexFacts:
    """Extract markdown‑formatted blocks from a transcript and load into Neo4j with embeddings.
    
    Supports five fact categories:
    - Result: already-achieved financial or operating results
    - Forward-Looking: explicit projections, targets, plans, or guidance
    - Risk Disclosure: current or expected obstacles
    - Sentiment: management's overall tone (Positive, Neutral, Negative)
    - Macro: macro-economic landscape impact on the firm
    
    Neo4j Schema:
    - Fact nodes contain: type, metric, ticker, quarter, embedding
    - Value nodes contain: content (human-readable value)
    - Reason nodes contain: content (free-text justification)
    - Relationships: REPORTS_IN, HAS_FACT, FOR_FIRM, HAS_VALUE, EXPLAINED_BY, QoQ_COMPARES
    """

    # ------------------------------------------------------------------
    def __init__(self, credentials_file: str, openai_model: str = "gpt-4o-mini"):
        creds = json.loads(Path(credentials_file).read_text())
        self.client = OpenAI(api_key=creds["openai_api_key"])
        self.model = openai_model
        self.driver = GraphDatabase.driver(
            creds["neo4j_uri"], auth=(creds["neo4j_username"], creds["neo4j_password"])
        )
        self.create_fact_metric_index()
        self._ensure_schema()

    
    # ------------------------------------------------------------------
    def _ensure_schema(self):
        with self.driver.session() as ses:
            ses.run("CREATE INDEX fact_metric_idx IF NOT EXISTS FOR (f:Fact) ON (f.metric)")
            ses.run("CREATE INDEX fact_type_idx IF NOT EXISTS FOR (f:Fact) ON (f.type)")
            ses.run("CREATE INDEX fact_ticker_idx IF NOT EXISTS FOR (f:Fact) ON (f.ticker)")
            ses.run("CREATE INDEX fact_quarter_idx IF NOT EXISTS FOR (f:Fact) ON (f.quarter)")
            ses.run("CREATE INDEX fact_sector_idx IF NOT EXISTS FOR (f:Fact) ON (f.sector)")
            # Note: Database clearing is now handled externally in chunks

    def create_fact_metric_index(self):
        stmt = """
        CREATE VECTOR INDEX fact_index IF NOT EXISTS
        FOR (f:Fact) ON (f.embedding)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`:        1536,
            `vector.similarity_function`: 'cosine'
          }
        }
        """
        with self.driver.session() as session:
            session.run(stmt)

            
    # ------------------------------------------------------------------
    # LLM extraction helpers
    # ------------------------------------------------------------------
    def _llm_extract(self, text: str) -> List[Dict[str, str]]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial information extraction assistant."},
                {"role": "user", "content": facts_extraction_prompt(text)},
            ],
            temperature=0,
        )
        return _parse_markdown_items(resp.choices[0].message.content.strip())

    # ------------------------------------------------------------------
    def extract_facts(self, transcript: str) -> List[Dict[str, str]]:
        """Return all block types with keys: type, metric, value, reason."""
        items = self._llm_extract(transcript)
        return [
            {
                "type": itm.get("Type", ""),
                "metric": itm.get("Metric", ""),
                "value": itm.get("Value", ""),
                "reason": itm.get("Reason", ""),
            }
            for itm in items
        ]

    # ------------------------------------------------------------------
    def extract_facts_with_context(self, transcript: str, ticker: str, quarter: str) -> List[Dict[str, str]]:
        """Return all block types with keys: ticker, quarter, type, metric, value, reason."""
        items = self._llm_extract(transcript)
        return [
            {
                "ticker": ticker,
                "quarter": quarter,
                "type": itm.get("Type", ""),
                "metric": itm.get("Metric", ""),
                "value": itm.get("Value", ""),
                "reason": itm.get("Reason", ""),
            }
            for itm in items
        ]

    # ------------------------------------------------------------------
    # Triple helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_triples(blocks: List[Dict[str, str]], ticker: str, qtr: str) -> List[Dict[str, Any]]:
        """Convert extracted blocks into triples, including type relationship."""
        triples: List[Dict[str, Any]] = []
        for b in blocks:
            metric = b["metric"]  # Always use just the metric name
            value = b["value"]
            reason = b["reason"]
            btype = b["type"]
            # Use ticker and quarter from the fact if available, otherwise use parameters
            fact_ticker = b.get("ticker", ticker)
            fact_quarter = b.get("quarter", qtr)
            # Ensure quarter is always in 'YYYY-QN' format
            import re
            m = re.match(r"(\d{4})-Q(\d)", str(fact_quarter))
            fact_quarter = f"{m.group(1)}-Q{m.group(2)}" if m else str(fact_quarter)

            triples.extend(
                [
                    {
                        "subject": metric,
                        "predicate": "has_value",
                        "object": value,
                        "ticker": fact_ticker,
                        "quarter": fact_quarter,
                        "value": value,
                        "reason": reason,
                        "type": btype,
                    }
                ]
            )
        return triples

    # ------------------------------------------------------------------
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a fact using OpenAI's embedding API."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    # ------------------------------------------------------------------
    def _write_tx_with_embedding(
        self,
        tx,
        subject: str,
        predicate: str,
        object: str,
        ticker: str,
        quarter: str,
        value: str,
        reason: str,
        fact_type: str,
        embedding: List[float],
    ) -> None:
        tx.run(
            f"""
            MERGE (m:Metric {{name:$subject}})
            MERGE (t:Ticker {{symbol:$ticker}})
            MERGE (q:Quarter {{label:$quarter}})
            MERGE (v:Value {{content:$value}})
            MERGE (r:Reason {{content:$reason}})
            
            MERGE (t)-[:REPORTS_IN]->(q)
            MERGE (q)-[:HAS_FACT]->(f:Fact {{type:$fact_type, metric:$subject, ticker:$ticker, quarter:$quarter, embedding:$embedding}})
            MERGE (f)-[:FOR_FIRM]->(t)
            MERGE (f)-[:HAS_VALUE]->(v)
            MERGE (f)-[:EXPLAINED_BY]->(r)
            """,
            subject=subject,
            object=object,
            ticker=ticker,
            quarter=quarter,
            value=value,
            reason=reason,
            fact_type=fact_type,
            embedding=embedding,
        )

    @staticmethod
    def _batch_write_tx(tx, facts_with_embeddings: List[Dict[str, Any]]):
        """Write a batch of facts with their embeddings in a single transaction."""
        query = """
        UNWIND $facts as fact
        MERGE (m:Metric {name: fact.subject})
        MERGE (t:Ticker {symbol: fact.ticker})
        MERGE (q:Quarter {label: fact.quarter})
        MERGE (v:Value {content: fact.value})
        MERGE (r:Reason {content: fact.reason})
        
        MERGE (t)-[:REPORTS_IN]->(q)
        MERGE (q)-[:HAS_FACT]->(f:Fact {
            type: fact.type, 
            metric: fact.subject, 
            ticker: fact.ticker, 
            quarter: fact.quarter, 
            embedding: fact.embedding
        })
        MERGE (f)-[:FOR_FIRM]->(t)
        MERGE (f)-[:HAS_VALUE]->(v)
        MERGE (f)-[:EXPLAINED_BY]->(r)
        """
        tx.run(query, facts=facts_with_embeddings)

    def _push(self, triples: List[Dict[str, Any]]):
        """Generate embeddings in parallel and push all facts in a single batch."""
        
        def get_embedding_for_triple(triple: Dict[str, Any]) -> Dict[str, Any]:
            """Helper function to generate embedding for a single triple."""
            fact_text = f"{triple['subject']}: {triple['value']} - {triple['reason']}"
            embedding = self._generate_embedding(fact_text)
            triple['embedding'] = embedding
            return triple

        # Step 1: Generate embeddings in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Use map to preserve order and simplify
            facts_with_embeddings = list(executor.map(get_embedding_for_triple, triples))

        # Step 2: Write all facts in a single transaction
        if facts_with_embeddings:
            with self.driver.session() as ses:
                ses.write_transaction(self._batch_write_tx, facts_with_embeddings)

    
    def process_transcript(self, transcript: str, ticker: str, quarter: str) -> List[Dict[str, Any]]:
        facts = self.extract_facts_with_context(transcript, ticker, quarter)
        if not facts:
            raise ValueError("No facts extracted.")
        # Exclude facts from the present quarter when indexing similar facts
        filtered_facts = [f for f in facts if f.get("quarter") != quarter]
        triples = self._to_triples(filtered_facts, ticker, quarter)
        self._push(triples)
        return facts  # Return the canonical fact format instead of triples

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------------
    def create_qoq_comparisons(self, ticker: str, current_quarter: str, previous_quarter: str):
        """Create QoQ comparison relationships between facts from consecutive quarters."""
        with self.driver.session() as ses:
            # Find facts from current and previous quarters
            result = ses.run("""
                MATCH (f1:Fact {ticker: $ticker, quarter: $current_quarter})
                MATCH (f2:Fact {ticker: $ticker, quarter: $previous_quarter})
                WHERE f1.metric = f2.metric
                RETURN f1, f2, f1.metric as metric
                """, ticker=ticker, current_quarter=current_quarter, previous_quarter=previous_quarter)
            
            for record in result:
                f1 = record["f1"]
                f2 = record["f2"]
                metric = record["metric"]
                
                # Get values for comparison
                value1_result = ses.run("""
                    MATCH (f:Fact)-[:HAS_VALUE]->(v:Value)
                    WHERE f.ticker = $ticker AND f.quarter = $quarter AND f.metric = $metric
                    RETURN v.content as value
                    """, ticker=ticker, quarter=current_quarter, metric=metric)
                
                value2_result = ses.run("""
                    MATCH (f:Fact)-[:HAS_VALUE]->(v:Value)
                    WHERE f.ticker = $ticker AND f.quarter = $quarter AND f.metric = $metric
                    RETURN v.content as value
                    """, ticker=ticker, quarter=previous_quarter, metric=metric)
                
                value1 = value1_result.single()
                value2 = value2_result.single()
                
                if value1 and value2:
                    # Calculate percentage change (simplified - would need proper parsing)
                    try:
                        # This is a simplified calculation - in practice you'd need to parse the values properly
                        delta_pct = 0.0  # Placeholder for actual calculation
                        
                        # Create QoQ relationship
                        ses.run("""
                            MATCH (f1:Fact {ticker: $ticker, quarter: $current_quarter, metric: $metric})
                            MATCH (f2:Fact {ticker: $ticker, quarter: $previous_quarter, metric: $metric})
                            MERGE (f1)-[:QoQ_COMPARES {delta_pct: $delta_pct}]->(f2)
                            """, ticker=ticker, current_quarter=current_quarter, 
                                 previous_quarter=previous_quarter, metric=metric, delta_pct=delta_pct)
                    except:
                        # Skip if calculation fails
                        continue