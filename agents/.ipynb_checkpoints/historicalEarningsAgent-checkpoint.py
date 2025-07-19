import json
from openai import OpenAI
from neo4j import GraphDatabase
from typing import List, Dict, Any
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from agents.prompts.prompts import historical_earnings_agent_prompt

# -------------------------------------------------------------------------
class HistoricalEarningsAgent:
    """Compare current facts with the firm’s own historical facts."""

    def __init__(self, credentials_file: str = "credentials.json", model: str = "gpt-4o-mini") -> None:
        creds = json.loads(Path(credentials_file).read_text())
        self.client = OpenAI(api_key=creds["openai_api_key"])
        self.model = model
        self.driver = GraphDatabase.driver(
            creds["neo4j_uri"], auth=(creds["neo4j_username"], creds["neo4j_password"])
        )
        self.embedder = OpenAIEmbeddings(openai_api_key=creds["openai_api_key"])

    # ------------------------------------------------------------------
    # Neo4j fetch helper (simple filter – same ticker, prior quarters)
    # ------------------------------------------------------------------
    def _fetch_past_facts(self, ticker: str, top_k: int = 50) -> List[Dict[str, Any]]:
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

    # ------------------------------------------------------------------
    def run(
        self,
        facts: List[Dict[str, str]],
        ticker: str,
        quarter: str,
        top_k: int = 50,
    ) -> str:
        """Batch‑process a list of facts for one ticker/quarter."""
        if not facts:
            return "❌ No facts supplied."

        past = self._fetch_past_facts(ticker, top_k)

        # Build prompt using the *first* fact for context but include list len
        prompt = (
            "\n\n" + historical_earnings_agent_prompt(facts, past, quarter)
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial forecasting assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.driver.close()
