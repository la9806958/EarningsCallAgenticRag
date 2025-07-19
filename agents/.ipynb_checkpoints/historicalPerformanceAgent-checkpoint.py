"""historical_performance_agent.py – Batch‑aware version
=======================================================
`run()` now accepts a **list of facts** so the model can compare all of them to
past financial statements in a single prompt.
"""

from __future__ import annotations
from pathlib import Path
import json
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from agents.prompts.prompts import financials_statement_agent_prompt
# -------------------------------------------------------------------------
class HistoricalPerformanceAgent:
    """Compare current-quarter facts with prior financial statements."""

    def __init__(self, credentials_file: str = "credentials.json", model: str = "gpt-4o-mini") -> None:
        creds = json.loads(Path(credentials_file).read_text())
        self.client = OpenAI(api_key=creds["openai_api_key"])
        self.model = model

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
    def run(self, facts: List[Dict[str, str]], row, quarter) -> str:
        """Batch‑aware comparison with historical financial statements."""
        if not facts:
            return "No facts provided."

        prior_income = self._pretty_json(row.get("income_statement", "[]"))
        prior_balance = self._pretty_json(row.get("balance_sheet", "[]"))
        prior_cash = self._pretty_json(row.get("cash_flow_statement", "[]"))

        # Use the shared template for *each* fact and aggregate responses
        analyses: List[str] = []
        for fact in facts:
            prompt = financials_statement_agent_prompt(
                fact=fact,
                quarter = quarter,
                prior_income=prior_income,
                prior_balance=prior_balance,
                prior_cash=prior_cash,
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial forecasting assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            analyses.append(resp.choices[0].message.content.strip())

        # Return all mini-analyses joined by newlines so the caller gets a single chat-friendly string
        return "".join(analyses)
