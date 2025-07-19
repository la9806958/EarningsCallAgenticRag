# --------------------------------------------------
# orchestrator_parallel_facts.py
# --------------------------------------------------
import os
import json
import re
import pandas as pd
import concurrent.futures
from typing import Dict, List, Tuple
from utils.indexFacts import IndexFacts
import ast
from functools import partial
import time
from datetime import datetime, timedelta
from pathlib import Path
import math
import threading
import cProfile
import pstats
import io
from neo4j import GraphDatabase
import argparse

# ---------- Constants & paths (defaults, can be overridden by args) ---------------------
DEFAULT_DATA_FILE = "merged_data_nyse.csv"
DEFAULT_SECTOR_MAP = "gics_sector_map_nyse.csv"
TIMEOUT_SEC = 1000  # 8+ minutes for each heavy call
MAX_WORKERS = 10    # Number of parallel workers for facts indexation
CHUNK_SIZE = 300    # Process 300 tickers per chunk

# ---------- Token and Timing Logging ------------
TOKEN_LOG_DIR = "token_logs"
TIMING_LOG_DIR = "timing_logs"
NEO4J_LOG_DIR = "neo4j_logs"

STATEMENT_BASE_DIR = Path("/Volumes/LaCie/美股-财务报表")
STATEMENT_FILES = {
    "cash_flow_statement": "_cash_flow_statement.csv",
    "income_statement": "_income_statement.csv",
    "balance_sheet": "_balance_sheet.csv",
}

KEY_METRICS = [
    # Balance-sheet strength
    "Cash and cash equivalents",
    "Accounts receivable",
    "Inventory",
    "Property, plant and equipment",
    "Short-term debt",
    "Total current liabilities",
    "Total liabilities",
    "Total Shareholders' Equity",
    
    # Profitability & margins
    "Main business income",
    "Operating Costs",
    "Net profit",
    "Gross profit",
    
    # Cash-flow drivers
    "Net Cash Flow from Operating Activities",
    "Net cash flow from investing activities",
    "Net Cash Flow from Financing Activities",
    
    # Per-share measure
    "Diluted earnings per share-Common stock"
]

# ---------- Sector Map (will be loaded in main function) ---------------------
SECTOR_MAP_DF = None
SECTOR_MAP_DICT = None

# ---------- Metric Mapping Dictionary (for standardizing metric names) --------
METRIC_MAPPING = {
    "Main business income": "Revenue",
    "主营收入": "Revenue",  # Chinese version
    "Main Business Income": "Revenue",  # Alternative capitalization
    "MAIN BUSINESS INCOME": "Revenue",  # All caps version
}

# Shared OpenAI semaphore (limit to 4 concurrent calls)
openai_semaphore = threading.Semaphore(4)

# ---------- Neo4j Connection Setup ------------
def get_neo4j_driver():
    """Get Neo4j driver from credentials file."""
    creds = json.loads(Path("credentials.json").read_text())
    return GraphDatabase.driver(
        creds["neo4j_uri"], 
        auth=(creds["neo4j_username"], creds["neo4j_password"])
    )

def clear_neo4j_database(driver, chunk_info: str = None):
    """Clear all data from Neo4j database using transaction-based deletion and log deletion counts."""
    print("🗑️  Clearing Neo4j database...")
    
    # Get counts before deletion
    with driver.session() as session:
        # Count nodes and relationships before deletion
        result = session.run("""
        MATCH (n)
        RETURN count(n) as node_count
        """)
        node_count = result.single()["node_count"]
        
        result = session.run("""
        MATCH ()-[r]->()
        RETURN count(r) as relationship_count
        """)
        relationship_count = result.single()["relationship_count"]
        
        print(f"📊 Found {node_count} nodes and {relationship_count} relationships to delete")
        
        # Use transaction-based deletion for better performance
        session.run("""
        CALL {
          MATCH (n)            // return every node id lazily
          RETURN n
        } IN TRANSACTIONS OF 10000 ROWS   // ← tune batch size
        DETACH DELETE n;
        """)
    
    # Log deletion counts to file
    log_deletion_counts(node_count, relationship_count, chunk_info)
    
    print("✅ Neo4j database cleared successfully")
    print(f"📝 Deletion logged: {node_count} nodes, {relationship_count} relationships")

# ---------- Unit Conversion Utilities ---------------------
UNIT_CONVERSION = {
    ("Hundred million", "Ten thousand"): 10000,
    ("Ten thousand", "Hundred million"): 1/10000,
}

def convert_unit(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert value between 'Hundred million' and 'Ten thousand'.
    If units are the same or conversion is not defined, returns the original value.
    """
    key = (from_unit, to_unit)
    if from_unit == to_unit or key not in UNIT_CONVERSION:
        return value
    return value * UNIT_CONVERSION[key]

def ensure_log_directories():
    """Create log directories if they don't exist."""
    os.makedirs(TOKEN_LOG_DIR, exist_ok=True)
    os.makedirs(TIMING_LOG_DIR, exist_ok=True)
    os.makedirs(NEO4J_LOG_DIR, exist_ok=True)

def get_agent_token_log_path(agent_type: str) -> str:
    """Get the token log file path for a specific agent."""
    return os.path.join(TOKEN_LOG_DIR, f"{agent_type}_token_usage.csv")

def get_agent_timing_log_path(agent_type: str) -> str:
    """Get the timing log file path for a specific agent."""
    return os.path.join(TIMING_LOG_DIR, f"{agent_type}_timing.csv")

def initialize_agent_token_log(agent_type: str):
    """Initialize token usage log file for a specific agent."""
    log_path = get_agent_token_log_path(agent_type)
    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            "timestamp", "ticker", "quarter", "agent_type", "model", 
            "input_tokens", "output_tokens", "total_tokens", "cost_usd"
        ]).to_csv(log_path, index=False)

def initialize_agent_timing_log(agent_type: str):
    """Initialize agent timing log file for a specific agent."""
    log_path = get_agent_timing_log_path(agent_type)
    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            "timestamp", "ticker", "quarter", "agent_type", "start_time", 
            "end_time", "duration_seconds", "status"
        ]).to_csv(log_path, index=False)

def log_token_usage(ticker: str, quarter: str, agent_type: str, model: str, 
                   input_tokens: int, output_tokens: int, cost_usd: float = None):
    """Log token usage to agent-specific CSV file."""
    total_tokens = input_tokens + output_tokens
    
    # Estimate cost if not provided (using OpenAI pricing as default)
    if cost_usd is None:
        if "gpt-4o" in model.lower():
            cost_usd = (input_tokens * 0.000005) + (output_tokens * 0.000015)
        elif "gpt-4" in model.lower():
            cost_usd = (input_tokens * 0.00003) + (output_tokens * 0.00006)
        elif "gpt-3.5" in model.lower():
            cost_usd = (input_tokens * 0.0000015) + (output_tokens * 0.000002)
        else:
            cost_usd = 0.0
    
    # Initialize log file for this agent if it doesn't exist
    initialize_agent_token_log(agent_type)
    
    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "quarter": quarter,
        "agent_type": agent_type,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd
    }])
    
    log_path = get_agent_token_log_path(agent_type)
    log_entry.to_csv(log_path, mode="a", header=False, index=False)
    
    # Also log to a combined file for overall statistics
    combined_log_path = os.path.join(TOKEN_LOG_DIR, "combined_token_usage.csv")
    if not os.path.exists(combined_log_path):
        log_entry.to_csv(combined_log_path, index=False)
    else:
        log_entry.to_csv(combined_log_path, mode="a", header=False, index=False)

def log_agent_timing(ticker: str, quarter: str, agent_type: str, 
                    start_time: float, end_time: float, status: str = "success"):
    """Log agent timing to agent-specific CSV file."""
    duration = end_time - start_time
    
    # Initialize log file for this agent if it doesn't exist
    initialize_agent_timing_log(agent_type)
    
    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "quarter": quarter,
        "agent_type": agent_type,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "duration_seconds": duration,
        "status": status
    }])
    
    log_path = get_agent_timing_log_path(agent_type)
    log_entry.to_csv(log_path, mode="a", header=False, index=False)
    
    # Also log to a combined file for overall statistics
    combined_log_path = os.path.join(TIMING_LOG_DIR, "combined_timing.csv")
    if not os.path.exists(combined_log_path):
        log_entry.to_csv(combined_log_path, index=False)
    else:
        log_entry.to_csv(combined_log_path, mode="a", header=False, index=False)

def ensure_neo4j_log_directory():
    """Create Neo4j log directory if it doesn't exist."""
    os.makedirs(NEO4J_LOG_DIR, exist_ok=True)

def log_deletion_counts(node_count: int, relationship_count: int, chunk_info: str = None):
    """Log Neo4j deletion counts to CSV file."""
    ensure_neo4j_log_directory()
    
    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "node_count": node_count,
        "relationship_count": relationship_count,
        "total_deleted": node_count + relationship_count,
        "chunk_info": chunk_info or "initial_clear"
    }])
    
    log_path = os.path.join(NEO4J_LOG_DIR, "neo4j_deletion_log.csv")
    if not os.path.exists(log_path):
        log_entry.to_csv(log_path, index=False)
    else:
        log_entry.to_csv(log_path, mode="a", header=False, index=False)

# ---------- Helper for quarter sorting and calculation ------------
_Q_RX = re.compile(r"(\d{4})-Q([1-4])")
def _q_sort_key(q: str) -> tuple[int, int]:
    """
    Convert a quarter string (e.g., "2023-Q4") into a sortable tuple (e.g., (2023, 4)).

    Args:
        q: The quarter string to convert.

    Returns:
        A tuple of (year, quarter) integers for sorting.
    """
    m = _Q_RX.fullmatch(q)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

# Quarter calculation functions removed - now using "q" column directly

# ---------- Financial Statement Indexing ------------
def load_latest_statements(ticker: str, as_of_date: pd.Timestamp, n: int = 6) -> Dict[str, List[dict]]:
    """Load the latest n statement columns for each statement type for a ticker, as list of dicts."""
    out = {}
    for key, suffix in STATEMENT_FILES.items():
        fname = STATEMENT_BASE_DIR / f"{ticker.upper()}{suffix}"
        if not fname.exists():
            out[key] = []
            continue
        df = pd.read_csv(fname, index_col=0)
        # Filter columns that are valid dates and < as_of_date
        valid_cols = []
        for col in df.columns:
            try:
                d = pd.to_datetime(col.split(".")[0], format="%Y-%m-%d", errors="raise")
                #print("as of date is ")
                #print(as_of_date)
                #print("dates are")
                #print(d)
                if d < as_of_date:
                    valid_cols.append((d, col))
            except Exception:
                continue
        # Sort by date descending, take n latest
        valid_cols = sorted(valid_cols, reverse=True)[:n]
        # Build output list
        out[key] = [
            {"date": d.strftime("%Y-%m-%d"), "rows": df[c].dropna().to_dict()}
            for d, c in valid_cols
        ]
    return out


def extract_number_with_unit(val):
    """Extract number and unit from the string."""
    val_str = str(val).replace(',', '')
    # Extract the number
    num_match = re.search(r'-?\d+(?:\.\d+)?', val_str)
    if not num_match:
        return None, None
    
    number = float(num_match.group(0))
    
    # Extract the unit (everything after the number)
    unit_start = num_match.end()
    unit = val_str[unit_start:].strip()
    
    return number, unit

def map_metric_name(metric_name: str) -> str:
    """
    Map metric names to standardized versions using the METRIC_MAPPING dictionary.
    
    Args:
        metric_name: The original metric name from the financial statement
        
    Returns:
        The standardized metric name
    """
    return METRIC_MAPPING.get(metric_name, metric_name)

def generate_financial_statement_facts(row: pd.Series, ticker: str, quarter: str) -> List[Dict]:
    """
    Generates a list of financial facts from statement CSVs for a single
    earnings call, without writing them to the database.

    Args:
        row: A pandas Series representing one earnings call transcript.
        ticker: The stock ticker symbol.
        quarter: The quarter string (e.g., "2023-Q4").

    Returns:
        A list of fact dictionaries ready for indexing.
    """
    as_of_date_str = row.get("parsed_date")
    if pd.isna(as_of_date_str):
        as_of_date = pd.Timestamp.now().tz_localize(None)
    else:
        as_of_date = pd.to_datetime(as_of_date_str).tz_localize(None)

    print("generating statement facts") 
    stmts = load_latest_statements(ticker, as_of_date, n=6)

    def parse_report_type_to_quarter(report_type: str) -> str:
        """Parse quarter from 'Financial Report Type' field."""
        try:
            if '/' in report_type and 'Q' in report_type:
                parts = report_type.split('/')
                if len(parts) >= 2:
                    year = parts[0]
                    quarter_part = parts[1].split()[0]
                    return f"{year}-{quarter_part}"
            elif '/' in report_type:
                parts = report_type.split('/')
                if len(parts) >= 2:
                    year = parts[0]
                    report_period = parts[1].split()[0]
                    if 'annual' in report_period.lower():
                        return f"{year}-Q4"
                    elif 'semi' in report_period.lower():
                        return f"{year}-Q2"
            return quarter
        except:
            return quarter

    def process_financial_data(data: List[dict], statement_type: str) -> List[Dict]:
        if not data:
            return []
        facts = []
        current_quarter_key = _q_sort_key(quarter)
        
        # Collect all (quarter, metric, value, report_type) for processing
        metric_quarter_facts = {m: {} for m in KEY_METRICS}  # metric -> {quarter: (value, report_type)}
        report_type_priority = {'annual': 3, 'semi': 2, 'q': 1}  # annual > semi > q
        
        def report_type_rank(report_type):
            rt = report_type.lower()
            if 'annual' in rt:
                return 3
            elif 'semi' in rt:
                return 2
            elif 'q' in rt:
                return 1
            return 0
        
        # Step 1: Collect all data points
        for period_data in data:
            if isinstance(period_data, dict) and 'rows' in period_data:
                rows = period_data.get('rows', {})
                report_type = rows.get('Financial Report Type', '')
                statement_quarter = parse_report_type_to_quarter(report_type) if report_type else quarter
                statement_quarter_key = _q_sort_key(statement_quarter)
                if statement_quarter_key > current_quarter_key:
                    continue
                for metric, value in rows.items():
                    # Apply metric name mapping to standardize metric names
                    mapped_metric = map_metric_name(metric)
                    
                    if mapped_metric in KEY_METRICS and value != '--':
                        # Only keep the highest priority report for each (metric, quarter)
                        prev = metric_quarter_facts[mapped_metric].get(statement_quarter)
                        if prev is None or report_type_rank(report_type) > report_type_rank(prev[1]):
                            metric_quarter_facts[mapped_metric][statement_quarter] = (value, report_type)
        
        # Step 2: Convert cumulative data to quarterly data and add facts
        metric_quarterly_values = {}  # metric -> {quarter: quarterly_value}
        
        for metric, qdict in metric_quarter_facts.items():
            metric_quarterly_values[metric] = {}
            # Sort quarters chronologically
            qv_list_sorted = sorted(qdict.items(), key=lambda x: _q_sort_key(x[0]))
            prev_cum_value, prev_unit, prev_quarter, prev_report_type = None, None, None, None
            for i, (statement_quarter, (cumulative_value, report_type)) in enumerate(qv_list_sorted):
                cumulative_value_num, unit = extract_number_with_unit(cumulative_value)
                if cumulative_value_num is None:
                    continue
                is_cumulative = 'annual' in report_type.lower() or 'semi' in report_type.lower() or 'q' in report_type.lower()
                # For all cumulative reports, take the difference from the previous period (regardless of year)
                if is_cumulative:
                    if prev_cum_value is not None and prev_unit == unit:
                        quarterly_value = cumulative_value_num - prev_cum_value
                    elif prev_cum_value is not None and prev_unit != unit:
                        try:
                            prev_cum_value_converted = convert_unit(prev_cum_value, prev_unit, unit)
                            quarterly_value = cumulative_value_num - prev_cum_value_converted
                        except Exception as e:
                            print(f"[CUMULATIVE UNIT ERROR] {ticker} {metric} {statement_quarter}: {e}")
                            quarterly_value = cumulative_value_num
                    else:
                        quarterly_value = cumulative_value_num
                else:
                    quarterly_value = cumulative_value_num
                metric_quarterly_values[metric][statement_quarter] = (quarterly_value, unit)
                prev_cum_value, prev_unit, prev_quarter, prev_report_type = cumulative_value_num, unit, statement_quarter, report_type
                
                # Add the quarterly fact
                statement_quarter_key = _q_sort_key(statement_quarter)
                is_current_quarter = statement_quarter_key == current_quarter_key
                data_type = "Current" if is_current_quarter else "Historical"
                
                # Format value with unit
                value_str = f"{quarterly_value:.2f}" if isinstance(quarterly_value, float) else str(quarterly_value)
                if unit:
                    value_str = f"{value_str}{unit}"
                
                fact = {
                    "ticker": ticker,
                    "quarter": statement_quarter,
                    "type": "Result",
                    "metric": f"{metric}",
                    "value": value_str,
                    "reason": f"{data_type} {statement_type} quarterly data from {report_type}"
                }
                facts.append(fact)
        
        # Step 3: Calculate QoQ and YoY changes from quarterly values
        def get_prev_yoy_q(curr_q):
            """
            Given a period string like '2023-Q4', '2023-semi', '2023-annual', '2023-H1', '2023-H2',
            return the previous year's same period (e.g., '2022-Q4', '2022-semi', etc.).
            """
            import re
            # Try to match year and the rest (e.g., 2023-Q4, 2023-semi, 2023-annual, 2023-H1, 2023-H2)
            m = re.match(r"(\d{4})([-_][A-Za-z0-9]+)", curr_q)
            if m:
                prev_year = str(int(m.group(1)) - 1)
                rest = m.group(2)
                return prev_year + rest
            # Fallback: try to match year and period with no separator (rare)
            m2 = re.match(r"(\d{4})([A-Za-z0-9]+)", curr_q)
            if m2:
                prev_year = str(int(m2.group(1)) - 1)
                rest = m2.group(2)
                return prev_year + rest
            # If all else fails, just return the original string
            print(f"[YoY WARNING] Could not parse period string: {curr_q}")
            return curr_q

        for metric, quarterly_dict in metric_quarterly_values.items():
            # Sort quarters by date
            qv_list_sorted = sorted(quarterly_dict.items(), key=lambda x: _q_sort_key(x[0]))
            prev_q, prev_v_tuple = None, None
            # Build a lookup for YoY (same quarter last year)
            quarter_to_value = {q: v for q, v in qv_list_sorted}
            for curr_q, curr_v_tuple in qv_list_sorted:
                curr_v, curr_unit = curr_v_tuple
                # QoQ: previous quarter (delta-on-delta logic)
                if prev_q is not None:
                    prev_v, prev_unit = prev_v_tuple
                    curr_year, curr_quarter = _q_sort_key(curr_q)
                    prev_year, prev_quarter = _q_sort_key(prev_q)
                    is_consecutive = (
                        (curr_year == prev_year and curr_quarter == prev_quarter + 1) or
                        (curr_year == prev_year + 1 and curr_quarter == 1 and prev_quarter == 4)
                    )
                    if not is_consecutive or curr_q == prev_q:
                        prev_q, prev_v_tuple = curr_q, curr_v_tuple
                        continue
                    prev_v_converted = prev_v
                    if curr_unit != prev_unit and prev_v is not None and curr_unit and prev_unit:
                        print(f"[QoQ UNIT WARNING] {ticker} {metric} {curr_q}: unit changed from '{prev_unit}' to '{curr_unit}'")
                        try:
                            prev_v_converted = convert_unit(prev_v, prev_unit, curr_unit)
                        except Exception as e:
                            print(f"[QoQ UNIT ERROR] Failed to convert {prev_v} from {prev_unit} to {curr_unit}: {e}")
                            prev_v_converted = prev_v
                    try:
                        if prev_v_converted == 0:
                            pct_change = None
                        else:
                            pct_change = (curr_v - prev_v_converted) / abs(prev_v_converted)
                    except Exception as e:
                        print(f"[QoQ ERROR] {ticker} {metric} {curr_q}: {e}")
                        pct_change = None
                    if pct_change is not None:
                        facts.append({
                            "ticker": ticker,
                            "quarter": curr_q,
                            "type": "QoQChange",
                            "metric": metric,
                            "value": pct_change,
                            "reason": f"Quarter-on-quarter percent change in {statement_type} for {metric} from {prev_q} to {curr_q}"
                        })
                    else:
                        print(f"[QoQ SKIP] {ticker} {metric} {curr_q}: prev_v={prev_v} curr_v={curr_v} (division by zero)")
                # YoY: always simple percentage change, no delta-on-delta
                prev_yoy_q = get_prev_yoy_q(curr_q)
                if prev_yoy_q in quarter_to_value.keys():
                    prev_yoy_v, prev_yoy_unit = quarter_to_value[prev_yoy_q]
                    prev_yoy_v_converted = prev_yoy_v
                    if curr_unit != prev_yoy_unit and prev_yoy_v is not None and curr_unit and prev_yoy_unit:
                        print(f"[YoY UNIT WARNING] {ticker} {metric} {curr_q}: unit changed from '{prev_yoy_unit}' to '{curr_unit}'")
                        try:
                            prev_yoy_v_converted = convert_unit(prev_yoy_v, prev_yoy_unit, curr_unit)
                        except Exception as e:
                            print(f"[YoY UNIT ERROR] Failed to convert {prev_yoy_v} from {prev_yoy_unit} to {curr_unit}: {e}")
                            prev_yoy_v_converted = prev_yoy_v
                    try:
                        if prev_yoy_v_converted == 0:
                            yoy_pct_change = None
                        else:
                            yoy_pct_change = (curr_v - prev_yoy_v_converted) / abs(prev_yoy_v_converted)
                    except Exception as e:
                        print(f"[YoY ERROR] {ticker} {metric} {curr_q}: {e}")
                        yoy_pct_change = None
                    if yoy_pct_change is not None:
                        facts.append({
                            "ticker": ticker,
                            "quarter": curr_q,
                            "type": "YoYChange",
                            "metric": metric,
                            "value": yoy_pct_change,
                            "reason": f"Year-over-year percent change in {statement_type} for {metric} from {prev_yoy_q} to {curr_q}"
                        })
                prev_q, prev_v_tuple = curr_q, curr_v_tuple
        
        return facts

    cash_flow_facts = process_financial_data(stmts["cash_flow_statement"], 'CashFlow')
    income_facts = process_financial_data(stmts["income_statement"], 'Income')
    balance_facts = process_financial_data(stmts["balance_sheet"], 'Balance')
    
    #print(f"\n[DEBUG] Raw financial_facts for {ticker} {quarter}:")
    #for fact in cash_flow_facts + income_facts + balance_facts:
    #    print(fact)
    
    return cash_flow_facts + income_facts + balance_facts

def format_financial_statements_facts(financial_facts: List[Dict]) -> str:
    """Format financial statements facts as a readable string for the main agent prompt, with clear QoQ change annotation."""
    if not financial_facts:
        return "No financial statements facts available."
    
    # Build a mapping for (metric, quarter) -> value for lookup
    value_lookup = {}
    for fact in financial_facts:
        if fact.get('type') == 'Result':
            value_lookup[(fact['metric'], fact['quarter'])] = fact['value']
    
    formatted_facts = []
    for fact in financial_facts:
        if fact.get('type') == 'QoQChange':
            # Try to extract prev quarter and value from the reason string
            import re
            m = re.search(r'from ([^ ]+) to ([^ ]+)', fact.get('reason',''))
            prev_q = m.group(1) if m else None
            curr_q = m.group(2) if m else fact['quarter']
            metric = fact['metric']
            curr_value = value_lookup.get((metric, curr_q), '?')
            prev_value = value_lookup.get((metric, prev_q), '?')
            
            try:
                pct = float(fact['value'])*100
                # Format percentage with better precision
                if abs(pct) >= 1:
                    pct_str = f"{abs(int(round(pct)))}%"
                else:
                    pct_str = f"{abs(pct):.1f}%"
                direction = "increase" if pct > 0 else "decrease" if pct < 0 else "no change"
                arrow = "▲" if pct > 0 else "▼" if pct < 0 else "→"
            except Exception:
                pct_str = str(fact['value'])
                direction = "change"
                arrow = ""
            
            formatted_fact = (
                f"• {metric}: {curr_value} ({curr_q}), {pct_str} {direction} from {prev_value} ({prev_q}) {arrow}"
            )
            formatted_facts.append(formatted_fact)
        elif fact.get('type') == 'YoYChange':
            import re
            m = re.search(r'from ([^ ]+) to ([^ ]+)', fact.get('reason',''))
            prev_q = m.group(1) if m else None
            curr_q = m.group(2) if m else fact['quarter']
            metric = fact['metric']
            curr_value = value_lookup.get((metric, curr_q), '?')
            prev_value = value_lookup.get((metric, prev_q), '?')
            try:
                pct = float(fact['value'])*100
                if abs(pct) >= 1:
                    pct_str = f"{abs(int(round(pct)))}%"
                else:
                    pct_str = f"{abs(pct):.1f}%"
                direction = "increase" if pct > 0 else "decrease" if pct < 0 else "no change"
                arrow = "▲" if pct > 0 else "▼" if pct < 0 else "→"
            except Exception:
                pct_str = str(fact['value'])
                direction = "change"
                arrow = ""
            formatted_fact = (
                f"• {metric}: {curr_value} ({curr_q}), {pct_str} {direction} from {prev_value} ({prev_q}) {arrow}"
            )
            formatted_facts.append(formatted_fact)
        else:
            # Format regular facts with better number formatting
            value_str = str(fact['value'])
            
            # Try to clean up floating point precision issues
            try:
                # If it's a number, format it nicely
                if '.' in value_str and 'Hundred million' in value_str:
                    # Extract number and unit
                    num_match = re.search(r'(-?\d+\.\d+)', value_str)
                    if num_match:
                        num = float(num_match.group(1))
                        # Round to 2 decimal places to avoid floating point issues
                        rounded_num = round(num, 2)
                        # Replace the number in the string
                        value_str = value_str.replace(num_match.group(1), f"{rounded_num}")
            except:
                pass
            
            formatted_fact = f"• {fact['metric']}: {value_str} ({fact['quarter']})"
            formatted_facts.append(formatted_fact)
    
    return "\n".join(formatted_facts)

# ---------- Parallel Facts Indexation Functions ------------
def check_financial_statement_files_exist(ticker: str) -> bool:
    """Check if any financial statement files exist for the given ticker."""
    for suffix in STATEMENT_FILES.values():
        fname = STATEMENT_BASE_DIR / f"{ticker.upper()}{suffix}"
        if fname.exists():
            return True
    return False


# ==================================================
# =============  WORKER FUNCTION ===================
# ==================================================
def process_sector(sector_df: pd.DataFrame, log_path: str = None) -> List[dict]:
    """
    Process all rows that belong to one SECTOR.
    This is the target function for the ProcessPoolExecutor.
    """
    from agents.mainAgent import MainAgent
    from agents.comparativeAgent import ComparativeAgent
    from agents.historicalPerformanceAgent import HistoricalPerformanceAgent
    from agents.historicalEarningsAgent import HistoricalEarningsAgent
    import concurrent.futures

    # --- cProfile setup ---
    pr = cProfile.Profile()
    pr.enable()

    sector_name = sector_df.iloc[0]["sector"]
    print(f"🚀 Worker started for sector: {sector_name} with {len(sector_df)} rows")

    # ---------- 1) Initialize objects ONCE per worker (sector) -------
    indexer = IndexFacts(credentials_file="credentials.json")
    main_agent = MainAgent(
        credentials_file   = "credentials.json",
        comparative_agent  = ComparativeAgent(credentials_file="credentials.json", sector_map=SECTOR_MAP_DICT),
        financials_agent   = HistoricalPerformanceAgent(credentials_file="credentials.json"),
        past_calls_agent   = HistoricalEarningsAgent(credentials_file="credentials.json"),
    )

    # --- I/O caching ---
    if log_path is None:
        log_path = "results.csv"  # fallback
    processed_history = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()
    statement_cache = {}
    def get_statement(ticker):
        if ticker not in statement_cache:
            # Load and cache all statement files for this ticker
            statement_cache[ticker] = load_latest_statements(ticker, pd.Timestamp.now(), n=6)
        return statement_cache[ticker]

    # --- Batch triple accumulation ---
    all_triples = []

    try:
        sector_df = sector_df.sort_values("parsed_date")

        def memories_for(df_processed: pd.DataFrame) -> List[Dict[str, str]]:
            if "quarter" in df_processed.columns:
                rows = df_processed.sort_values("quarter", key=lambda s: s.map(_q_sort_key))
                return rows[["quarter", "research_note", "actual_return"]].to_dict("records")
            else:
                rows = df_processed.sort_values("q", key=lambda s: s.map(_q_sort_key))
                return rows[["q", "research_note", "actual_return"]].to_dict("records")

        for _, row in sector_df.iterrows():
            ticker = row['ticker']
            quarter = row.get("q")
            if pd.isna(quarter):
                print(f"   - ⚠️ Skipping {ticker}: No quarter available")
                continue

            result_dict = {
                "ticker"        : ticker,
                "quarter"       : quarter,
                "parsed_and_analyzed_facts": "[]",
                "research_note" : "",
                "actual_return" : row["future_3bday_cum_return"],
                "error"         : "",
            }

            try:
                if not check_financial_statement_files_exist(ticker):
                    print(f"   - ⏭️ Skipping {ticker}/{quarter}: No financial statement files found")
                    result_dict["error"] = "No financial statement files available"
                    pd.DataFrame([result_dict]).to_csv(log_path, mode="a", header=False, index=False)
                    continue

                mem_block = None
                if not processed_history.empty and (processed_history["ticker"] == ticker).any():
                    quarter_col = "quarter" if "quarter" in processed_history.columns else "q"
                    ticker_history = processed_history[processed_history["ticker"] == ticker]
                    lines = ["Previously, you have made the following analysis on the firm in the past quarter:"]
                    for r in memories_for(ticker_history):
                        research_note = str(r.get('research_note', ''))
                        match = re.search(r"(\*\*Summary:.*?Direction\s*:\s*\d{1,2}\*\*)", research_note, re.DOTALL)
                        if match:
                            summary_text = match.group(1).strip().replace('\n', ' ')
                            quarter_key = 'quarter' if 'quarter' in r else 'q'
                            lines.append(
                                f"- {r[quarter_key]}:  {summary_text} "
                                f"(Following your prediction, the firm realised a 1-day return of = {r['actual_return']:+.2%})"
                            )
                    if len(lines) > 1:
                        mem_block = "\n".join(lines)

                # --- Use cached statement data ---
                statement_data = get_statement(ticker)
                # You may need to adapt generate_financial_statement_facts to accept preloaded data
                financial_facts = generate_financial_statement_facts(row, ticker, quarter)
                if financial_facts:
                    # Parallelize triple conversion and embedding/indexing
                    def triples_for_chunk(chunk):
                        return indexer._to_triples(chunk, ticker, quarter)
                    CHUNK_SIZE = 20
                    triples = []
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future_to_chunk = [pool.submit(triples_for_chunk, financial_facts[i:i+CHUNK_SIZE])
                                           for i in range(0, len(financial_facts), CHUNK_SIZE)]
                        for fut in concurrent.futures.as_completed(future_to_chunk):
                            triples.extend(fut.result())
                    # Index these triples immediately before main agent call
                    if triples:
                        indexer._push(triples)

                # Format and include only YoYChange facts in the financial statements section
                def format_yoy_facts(yoy_facts):
                    if not yoy_facts:
                        return "No year-over-year changes available."
                    lines = []
                    for f in yoy_facts:
                        metric = f.get('metric', '?')
                        value = f.get('value', '?')
                        quarter = f.get('quarter', '?')
                        reason = f.get('reason', '')
                        lines.append(f"• {metric}: {value} ({quarter}) {reason}")
                    return '\n'.join(lines)
                yoy_facts = [f for f in financial_facts if f.get('type') == 'YoYChange']
                financial_statements_facts_str = f"YoY Financial Statements Facts (Last 4 Quarters):\n" + format_yoy_facts(yoy_facts)

                transcript_facts = indexer.process_transcript(row["transcript"], ticker, quarter)

                current_qtr_facts = [f for f in financial_facts if f.get('quarter') == quarter]
                transcript_facts = (transcript_facts or [])
                
                curr_facts = current_qtr_facts + transcript_facts
                # Only feed quarter-on-quarter (QoQ) change facts into the main agent
                #qoq_facts = [f for f in transcript_facts if f.get('type') == 'QoQChange']
                #print(f"Filtered to {len(qoq_facts)} QoQChange facts for main agent.")
                #print("Sample QoQChange facts:", qoq_facts[:3])
                with openai_semaphore:
                    parsed = main_agent.run(transcript_facts, row, mem_block, None, financial_statements_facts_str)


                # Index transcript facts immediately
                if transcript_facts:
                    def triples_for_chunk(chunk):
                        return indexer._to_triples(chunk, ticker, quarter)
                    CHUNK_SIZE = 20
                    transcript_triples = []
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future_to_chunk = [pool.submit(triples_for_chunk, transcript_facts[i:i+CHUNK_SIZE])
                                           for i in range(0, len(transcript_facts), CHUNK_SIZE)]
                        for fut in concurrent.futures.as_completed(future_to_chunk):
                            transcript_triples.extend(fut.result())
                    if transcript_triples:
                        indexer._push(transcript_triples)


                result_dict["parsed_and_analyzed_facts"] = json.dumps(parsed or {})
                if isinstance(parsed, dict):
                    notes = parsed.get("notes", {})
                    parts = [
                        (notes.get("financials") or "").strip(),
                        (notes.get("past") or "").strip(),
                        (notes.get("peers") or "").strip(),
                    ]
                    summary_txt = (parsed.get("summary") or "").strip()
                    result_dict["research_note"] = "\n\n".join([p for p in parts if p] + [summary_txt]).strip()
                result_dict["error"] = ""
            except Exception as e:
                result_dict["error"] = str(e)

            # Incremental log write
            pd.DataFrame([result_dict]).to_csv(log_path, mode="a", header=False, index=False)
            
            # Debug: Print progress every 10 rows
            if sector_df.index.get_loc(_) % 10 == 0:
                print(f"   📝 Written {sector_df.index.get_loc(_) + 1}/{len(sector_df)} rows for {ticker}")

    finally:
        try:
            indexer.close()
        except:
            pass

    # --- cProfile summary ---
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 lines
    print(f"\n[cProfile] Top 30 functions for sector {sector_name}:")
    print(s.getvalue())

    return []

def initialize_log_file(log_path: str) -> None:
    """Create a fresh log file with headers, overwriting any existing file."""
    print(f"Initializing fresh log file at: {log_path}")
    pd.DataFrame(
        columns=[
            "ticker", "quarter", "parsed_and_analyzed_facts",
            "research_note", "actual_return", "error"
        ]
    ).to_csv(log_path, index=False)

def parse_arguments():
    """Parse command line arguments for data file and sector map."""
    parser = argparse.ArgumentParser(
        description="Earnings Call Agentic RAG Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process NYSE data
  python orchestrator_parallel_facts.py --data merged_data_nyse.csv --sector-map gics_sector_map_nyse.csv
  
  # Process NASDAQ data  
  python orchestrator_parallel_facts.py --data merged_data_nasdaq.csv --sector-map gics_sector_map_nasdaq.csv
  
  # Process MAEC data
  python orchestrator_parallel_facts.py --data maec_transcripts.csv --sector-map gics_sector_map_maec.csv
        """
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        default=DEFAULT_DATA_FILE,
        help=f"Path to the earnings call data CSV file (default: {DEFAULT_DATA_FILE})"
    )
    
    parser.add_argument(
        "--sector-map", 
        type=str, 
        default=DEFAULT_SECTOR_MAP,
        help=f"Path to the GICS sector mapping CSV file (default: {DEFAULT_SECTOR_MAP})"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Maximum number of parallel workers (default: {MAX_WORKERS})"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Number of tickers to process per chunk (default: {CHUNK_SIZE})"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_SEC,
        help=f"Timeout in seconds for each agent call (default: {TIMEOUT_SEC})"
    )
    
    return parser.parse_args()

# ==================================================
# =============  PARENT / MAIN  ====================
# ==================================================
THRESHOLD = 0.05          # 5 % in decimal form

def main() -> None:
    """
    Main execution function.
    Loads data, processes it in chunks, and clears Neo4j after each chunk.
    Groups data by sector and dispatches each sector to a separate process for parallel processing.
    It initializes log files and prints a final summary of token usage and timing.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print("🚀 Starting Parallel Facts Indexation with Agents and Memory")
    print(f"📁 Data file: {args.data}")
    print(f"🗺️  Sector map: {args.sector_map}")
    print(f"👥 Max workers: {args.max_workers}")
    print(f"📦 Chunk size: {args.chunk_size}")
    print(f"⏰ Timeout: {args.timeout}s")
    
    start_time = time.time()
    
    # Generate log path based on data file
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    log_path = f"{data_name}_results.csv"
    
    # Load sector map globally
    global SECTOR_MAP_DF, SECTOR_MAP_DICT
    try:
        SECTOR_MAP_DF = pd.read_csv(args.sector_map)
        SECTOR_MAP_DICT = dict(zip(SECTOR_MAP_DF['ticker'], SECTOR_MAP_DF['sector']))
        print(f"✅ Loaded sector map with {len(SECTOR_MAP_DICT)} ticker mappings")
    except Exception as e:
        print(f"❌ Error loading sector map {args.sector_map}: {e}")
        return
    
    # ------ 1) Initialize Log Files for a fresh run ---------------------
    if not os.path.exists(log_path):
        initialize_log_file(log_path)
    ensure_log_directories()

    # ------ 2) load & slice --------------------------------------------
    try:
        df = pd.read_csv(args.data).drop_duplicates()
        print(f"📊 Loaded {len(df)} rows from {args.data}")
    except Exception as e:
        print(f"❌ Error loading data file {args.data}: {e}")
        return
    df["returns"] = df["future_3bday_cum_return"]
    # df = df.iloc[878:]
    # df = df[(df["ticker"] == "REX") | (df["ticker"] == "PUMP") | (df["ticker"] == "PSX")]
    # df = df[(df["ticker"] == "NKE")]
    # df = df[(df["ticker"] == "ACIW") | (df["ticker"] == "CRI")]
    # Filter data
    df = df.dropna(subset=["parsed_date"])
    df['parsed_date'] = pd.to_datetime(df['parsed_date']).dt.tz_localize(None)
    df = df.sort_values("parsed_date").reset_index(drop=True)
    
    # Filter for significant returns

    """
    mask = (df["future_3bday_cum_return"] >= THRESHOLD) | \
           (df["future_3bday_cum_return"] <= -THRESHOLD)

    df = df.loc[mask].reset_index(drop=True)
    """
    
    # Drop entries where future_3bday_cum_return is NaN
    df = df.dropna(subset=["future_3bday_cum_return"]).reset_index(drop=True)
    
    # Filter out rows without financial statement data for better indexing
    # df = df.dropna(subset=["cash_flow_statement", "income_statement", "balance_sheet"], how='all')

    # --- Load sector map and merge ---
    df = df.merge(SECTOR_MAP_DF, on="ticker", how="left")
    # Drop rows with missing sector (optional, or handle as 'Unknown')
    df = df.dropna(subset=["sector"])

    # ------ 3) Process data in chunks ------------------
    unique_tickers = df['ticker'].unique()
    total_tickers = len(unique_tickers)
    total_chunks = math.ceil(total_tickers / args.chunk_size)
    
    print(f"📊 Processing {total_tickers} unique tickers in {total_chunks} chunks of {args.chunk_size}")
    
    # Initialize Neo4j driver
    try:
        neo4j_driver = get_neo4j_driver()
        print("✅ Neo4j driver initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Neo4j driver: {e}")
        print("🔄 Continuing without Neo4j database clearing...")
        neo4j_driver = None
    
    # Clear database at the start to ensure clean state
    if neo4j_driver:
        print("🗑️  Initializing clean Neo4j database...")
        try:
            clear_neo4j_database(neo4j_driver, "initial_clear")
        except Exception as e:
            print(f"⚠️  Warning: Failed to clear Neo4j database: {e}")
    
    for chunk_idx in range(total_chunks):
        start_ticker_idx = chunk_idx * args.chunk_size
        end_ticker_idx = min((chunk_idx + 1) * args.chunk_size, total_tickers)
        chunk_tickers = unique_tickers[start_ticker_idx:end_ticker_idx]
        
        print(f"\n🔄 Processing chunk {chunk_idx + 1}/{total_chunks} (tickers {start_ticker_idx + 1}-{end_ticker_idx})")
        print(f"📋 Tickers in this chunk: {', '.join(chunk_tickers[:5])}{'...' if len(chunk_tickers) > 5 else ''}")
        
        # Filter dataframe for current chunk
        chunk_df = df[df['ticker'].isin(chunk_tickers)].copy()
        print(f"📈 Processing {len(chunk_df)} rows for {len(chunk_tickers)} tickers")
        
        # Group by sector for parallel processing
        sector_groups = [g for _, g in chunk_df.groupby("sector")]
        print(f"🏭 Processing {len(sector_groups)} sectors in this chunk")
        
        # ------ 4) parallel dispatch for current chunk ---------------
        print(f"🔄 Launching {len(sector_groups)} sectors on {args.max_workers} CPU workers...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(process_sector, grp, log_path): grp["sector"].iat[0]
                for grp in sector_groups
            }

            for fut in concurrent.futures.as_completed(futures):
                sector = futures[fut]
                try:
                    fut.result()           # just wait for completion
                except Exception as err:  # capture worker failure
                    print(f"❌ Error in sector {sector}: {err}")
                print(f" ✅ finished sector {sector}")
        
        # ------ 5) Clear Neo4j database after processing chunk ---------------
        if neo4j_driver:
            print(f"🗑️  Clearing Neo4j database after chunk {chunk_idx + 1}")
            try:
                chunk_info = f"chunk_{chunk_idx + 1}_of_{total_chunks}"
                clear_neo4j_database(neo4j_driver, chunk_info)
            except Exception as e:
                print(f"⚠️  Warning: Failed to clear Neo4j database: {e}")
        else:
            print(f"⏭️  Skipping Neo4j database clearing for chunk {chunk_idx + 1} (driver not available)")
        
        # Print chunk summary
        chunk_time = time.time() - start_time
        print(f"✅ Completed chunk {chunk_idx + 1}/{total_chunks} in {chunk_time:.2f} seconds")
        print(f"📊 Average time per ticker in chunk: {chunk_time/len(chunk_tickers):.2f} seconds")
        
        # Print progress
        processed_tickers = (chunk_idx + 1) * args.chunk_size
        if processed_tickers > total_tickers:
            processed_tickers = total_tickers
        progress_pct = (processed_tickers / total_tickers) * 100
        print(f"📈 Overall progress: {processed_tickers}/{total_tickers} tickers ({progress_pct:.1f}%)")
        
        # Check how many rows were written to CSV
        if os.path.exists(log_path):
            try:
                df_check = pd.read_csv(log_path)
                print(f"📊 Total rows in CSV after chunk {chunk_idx + 1}: {len(df_check):,}")
            except Exception as e:
                print(f"⚠️  Could not check CSV file: {e}")
    
    # Close Neo4j driver
    if neo4j_driver:
        neo4j_driver.close()
        print("✅ Neo4j driver closed successfully")
    
    total_time = time.time() - start_time
    
    # Print summary with token and timing statistics
    print(f"\n🎉 Parallel Facts Indexation with Agents Complete!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Average time per row: {total_time/len(df):.2f} seconds")
    
    # Print comprehensive token usage summary
    print(f"\n💰 Token Usage Summary:")
    print("=" * 50)
    
    # Check combined token log first
    combined_token_path = os.path.join(TOKEN_LOG_DIR, "combined_token_usage.csv")
    if os.path.exists(combined_token_path):
        token_df = pd.read_csv(combined_token_path)
        if not token_df.empty:
            total_tokens = token_df['total_tokens'].sum()
            total_cost = token_df['cost_usd'].sum()
            print(f"📈 Total tokens used: {total_tokens:,}")
            print(f"💰 Total cost: ${total_cost:.4f}")
            print(f"📊 Average tokens per row: {total_tokens/len(df):.0f}")
            
            # Breakdown by agent type
            print(f"\n📋 Breakdown by Agent:")
            agent_summary = token_df.groupby('agent_type').agg({
                'total_tokens': 'sum',
                'cost_usd': 'sum',
                'ticker': 'count'
            }).rename(columns={'ticker': 'calls'})
            
            for agent, stats in agent_summary.iterrows():
                print(f"  {agent}: {stats['total_tokens']:,} tokens, ${stats['cost_usd']:.4f} ({stats['calls']} calls)")
    else:
        # Check individual agent files
        agent_types = ["main_agent", "comparative_agent", "financials_agent", "past_calls_agent"]
        total_tokens = 0
        total_cost = 0.0
        
        for agent_type in agent_types:
            log_path = get_agent_token_log_path(agent_type)
            if os.path.exists(log_path):
                df_agent = pd.read_csv(log_path)
                if not df_agent.empty:
                    agent_tokens = df_agent['total_tokens'].sum()
                    agent_cost = df_agent['cost_usd'].sum()
                    total_tokens += agent_tokens
                    total_cost += agent_cost
                    print(f"  {agent_type}: {agent_tokens:,} tokens, ${agent_cost:.4f} ({len(df_agent)} calls)")
        
        if total_tokens > 0:
            print(f"\n📈 Total tokens used: {total_tokens:,}")
            print(f"💰 Total cost: ${total_cost:.4f}")
            print(f"📊 Average tokens per row: {total_tokens/len(df):.0f}")
    
    # Print comprehensive timing summary
    print(f"\n⏱️  Timing Summary:")
    print("=" * 50)
    
    # Check combined timing log first
    combined_timing_path = os.path.join(TIMING_LOG_DIR, "combined_timing.csv")
    if os.path.exists(combined_timing_path):
        timing_df = pd.read_csv(combined_timing_path)
        if not timing_df.empty:
            avg_duration = timing_df['duration_seconds'].mean()
            total_duration = timing_df['duration_seconds'].sum()
            print(f"⏱️  Average agent call time: {avg_duration:.2f}s")
            print(f"⏱️  Total agent time: {total_duration:.2f}s")
            print(f"📊 Average time per row: {total_duration/len(df):.2f}s")
            
            # Breakdown by agent type
            print(f"\n📋 Breakdown by Agent:")
            agent_timing = timing_df.groupby('agent_type').agg({
                'duration_seconds': ['mean', 'sum', 'count']
            }).round(2)
            
            for agent, stats in agent_timing.iterrows():
                print(f"  {agent}: avg {stats[('duration_seconds', 'mean')]}s, total {stats[('duration_seconds', 'sum')]}s ({stats[('duration_seconds', 'count')]} calls)")
    else:
        # Check individual agent files
        agent_types = ["main_agent", "comparative_agent", "financials_agent", "past_calls_agent"]
        total_duration = 0.0
        total_calls = 0
        
        for agent_type in agent_types:
            log_path = get_agent_timing_log_path(agent_type)
            if os.path.exists(log_path):
                df_agent = pd.read_csv(log_path)
                if not df_agent.empty:
                    agent_duration = df_agent['duration_seconds'].sum()
                    agent_calls = len(df_agent)
                    agent_avg = df_agent['duration_seconds'].mean()
                    total_duration += agent_duration
                    total_calls += agent_calls
                    print(f"  {agent_type}: avg {agent_avg:.2f}s, total {agent_duration:.2f}s ({agent_calls} calls)")
        
        if total_duration > 0:
            print(f"\n⏱️  Total agent time: {total_duration:.2f}s")
            print(f"📊 Average time per row: {total_duration/len(df):.2f}s")
    
    print(f"\n📁 Log files created in:")
    print(f"  - {TOKEN_LOG_DIR}/ (token usage logs)")
    print(f"  - {TIMING_LOG_DIR}/ (timing logs)")



if __name__ == "__main__":
    # On macOS & Windows the default is "spawn"; on Linux you may want it too
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)      # safer with big libs & pickling
    main() 