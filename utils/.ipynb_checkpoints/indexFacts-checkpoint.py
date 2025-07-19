"""index_facts.py – Neo4j fact indexer
====================================
✓ Handles markdown blocks for Fact / Forward‑Looking / Risk Disclosure / Sentiment
✓ Purges existing Fact nodes and ensures index on `Fact.metric`
✓ Returns the `type` field for every extracted block
✓ Fixed syntax/duplication errors in `_push`
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from neo4j import GraphDatabase
from openai import OpenAI

from agents.prompts.prompts import facts_extraction_prompt

# ── Markdown parser -------------------------------------------------------
ITEM_HEADER = re.compile(
    r"###\s*(?:Fact|Forward-Looking|Risk Disclosure|Sentiment)\s*No\.\s*(\d+)",
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
    """Extract markdown‑formatted blocks from a transcript and load into Neo4j."""

    # ------------------------------------------------------------------
    def __init__(self, credentials_file: str, openai_model: str = "gpt-4o-mini"):
        creds = json.loads(Path(credentials_file).read_text())
        self.client = OpenAI(api_key=creds["openai_api_key"])
        self.model = openai_model
        self.driver = GraphDatabase.driver(
            creds["neo4j_uri"], auth=(creds["neo4j_username"], creds["neo4j_password"])
        )
        # self.create_fact_metric_index()
        # self._ensure_schema()

    
    # ------------------------------------------------------------------
    def _ensure_schema(self):
        with self.driver.session() as ses:
            ses.run("CREATE INDEX fact_metric_idx IF NOT EXISTS FOR (f:Fact) ON (f.metric)")
            ses.run("MATCH (f:Fact) DETACH DELETE f")  # starting fresh each run

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
                "reason": itm.get("Context", itm.get("Reason", "")),
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
            metric = b["metric"]
            value = b["value"]
            reason = b["reason"]
            btype = b["type"]

            triples.extend(
                [
                    {
                        "subject": metric,
                        "predicate": "has_value",
                        "object": value,
                        "ticker": ticker,
                        "quarter": qtr,
                        "value": value,
                        "reason": reason,
                        "type": btype,
                    }
                ]
            )
        return triples

    # ------------------------------------------------------------------
    @staticmethod
    def _write_tx(
        tx,
        subject: str,
        predicate: str,
        object: str,
        ticker: str,
        quarter: str,
        value: str,
        reason: str,
    ) -> None:
        rel = predicate.upper().replace(" ", "_")
        tx.run(
            f"""
            MERGE (m:Metric {{name:$subject}})
            MERGE (t:Ticker {{symbol:$ticker}})
            MERGE (q:Quarter {{label:$quarter}})
            MERGE (t)-[:HAS_QUARTER]->(q)
            CREATE (f:Fact {{metric:$subject, value:$value, reason:$reason, ticker:$ticker, quarter:$quarter}})
            MERGE (q)-[:HAS_FACT]->(f)
            MERGE (o:Entity {{content:$object}})
            MERGE (f)-[:{rel}]->(o)
            """,
            subject=subject,
            object=object,
            ticker=ticker,
            quarter=quarter,
            value=value,
            reason=reason,
        )

    def _push(self, triples: List[Dict[str, Any]]):
        with self.driver.session() as ses:
            for tr in triples:
                ses.write_transaction(
                    self._write_tx,
                    subject=tr["subject"],
                    predicate=tr["predicate"],
                    object=tr["object"],
                    ticker=tr["ticker"],
                    quarter=tr["quarter"],
                    value=tr["value"],
                    reason=tr["reason"],
                )

    
    def process_transcript(self, transcript: str, ticker: str, quarter: str) -> List[Dict[str, Any]]:
        facts = self.extract_facts(transcript)
        if not facts:
            raise ValueError("No facts extracted.")
        triples = self._to_triples(facts, ticker, quarter)
        self._push(triples)
        return triples

    def close(self):
        self.driver.close()