# Earnings Call Agentic RAG

A research implementation of context-enriched agentic Retrieval Augmented Generation (RAG) for analyzing earnings call transcripts and predicting post-earnings price movements.

## Overview

This repository implements a multi-agent system that processes earnings call transcripts to extract financial insights and predict stock price movements following earnings announcements. The system combines:

- **Agentic RAG Architecture**: Multiple specialized agents for different aspects of financial analysis
- **Context-Enriched Data**: Integration of earnings call transcripts with quarterly financial statements
- **Parallel Processing**: Efficient handling of large-scale financial datasets

## Dataset

| File | Description | Size |
|------|-------------|------|
| `maec_transcripts.csv` | MAEC benchmark earnings call transcripts (unfiltered) | 2,725 calls |
| `merged_data_nasdaq.csv` | NASDAQ calls with >5% return threshold | 1,772 calls |
| `merged_data_nyse.csv` | NYSE calls with >5% return threshold | 1,386 calls |
| `financial_statements/` | Quarterly financial data by ticker (income, balance sheet, cash flow) | 8-60 rows each |
| `gics_sector_map_*.csv` | GICS sector mappings for different exchanges | - |

## Architecture

### Core Components

- **`orchestrator_parallel_facts.py`**: Main orchestration script for parallel processing
- **`agents/`**: Specialized analysis agents
  - `mainAgent.py`: Primary coordination agent
  - `comparativeAgent.py`: Comparative analysis between periods
  - `historicalEarningsAgent.py`: Historical earnings pattern analysis
  - `historicalPerformanceAgent.py`: Stock performance analysis
- **`baseline/`**: Baseline sentiment analysis implementations
  - `finbert_classifier.py`: FinBERT-based sentiment classification
  - `sentiment_analysis.py`: Traditional sentiment analysis approaches

### Features

- **Multi-Agent Processing**: Specialized agents for different financial analysis tasks
- **Parallel Execution**: Configurable worker pools for efficient processing
- **Context Integration**: Combines transcript text with structured financial data
- **Token Usage Tracking**: Monitors LLM API costs and usage
- **Neo4j Integration**: Graph database support for relationship analysis

## Setup

### Prerequisites

```bash
pip install openai pandas numpy scikit-learn tqdm neo4j transformers torch accelerate
```

### Configuration

1. **Configure credentials**: Update `credentials.json` with your API keys and database credentials:
   ```json
   {
     "openai_api_key": "your-openai-api-key",
     "neo4j_uri": "your-neo4j-database-uri",
     "neo4j_username": "your-neo4j-username", 
     "neo4j_password": "your-neo4j-password"
   }
   ```

## Usage

### Running the Main Orchestrator

The main agentic RAG system processes earnings calls using multiple specialized agents:

```bash
# Process NYSE data (default)
python orchestrator_parallel_facts.py

# Process NASDAQ data
python orchestrator_parallel_facts.py --data merged_data_nasdaq.csv --sector-map gics_sector_map_nasdaq.csv

# Process MAEC data
python orchestrator_parallel_facts.py --data maec_transcripts.csv --sector-map gics_sector_map_maec.csv

# Custom configuration
python orchestrator_parallel_facts.py \
  --data merged_data_nyse.csv \
  --sector-map gics_sector_map_nyse.csv \
  --max-workers 8 \
  --chunk-size 200 \
  --timeout 1200
```

**Command line options**:
- `--data`: Input CSV file (default: `merged_data_nyse.csv`)
- `--sector-map`: GICS sector mapping file (default: `gics_sector_map_nyse.csv`)
- `--max-workers`: Number of parallel workers (default: 10)
- `--chunk-size`: Process tickers in chunks (default: 300)
- `--timeout`: Timeout per call in seconds (default: 1000)

**Output**: Results are saved to `{dataset_name}_results.csv` with predictions and analysis.

### Running Baseline Methods

#### 1. Loughran-McDonald Sentiment Analysis

Traditional sentiment analysis using financial lexicon:

```bash
python baseline/sentiment_analysis.py
```

**Requirements**: 
- Place Loughran-McDonald dictionary at `baseline/LM/LM_Master.csv`
- Updates `DATA_PATH` and `OUTPUT_PATH` constants as needed

**Output**: Sentiment scores and classifications in `baseline/nyse_sentiment_results.csv`

#### 2. FinBERT Classification

Fine-tuned BERT model for financial sentiment with k-fold cross-validation:

```bash
python baseline/finbert_classifier.py
```

**Configuration**:
- `DATA_PATH`: Input data file (default: `merged_data_nasdaq.csv`)
- `N_FOLDS`: Number of cross-validation folds (default: 5)
- `MODEL_NAME`: Pre-trained model (default: `ProsusAI/finbert`)

**Output**: Trained models saved to `baseline/finbert_returns_classifier/`

#### 3. Descriptive Statistics

Generate dataset statistics and token counts:

```bash
python baseline/descriptive_stats.py
```

**Features**:
- Date span analysis
- Unique ticker counts  
- Token count statistics using configurable tokenizer
- Supports multiple CSV files for comparison

## Research Context

This implementation accompanies the research paper "Context-Enriched Agent RAG: Cooperative LLM Retrieval for Predicting Post-Earnings Price Shocks" and demonstrates practical applications of agentic RAG systems in financial analysis.