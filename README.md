# DRIVE-QA

Natural-language question answering over the DISNET drug-repurposing database.

DRIVE-QA is a Master's Thesis / TFM project that explores how large language
models can be combined with schema retrieval, entity grounding, SQL validation,
and controlled database execution to answer biomedical questions over the DISNET
MySQL database. The system receives a question in natural language, identifies
the most relevant database tables, builds a constrained prompt for an LLM,
generates a read-only SQL query, executes the query, and returns both the
database results and a natural-language answer.

## System Overview

The end-to-end pipeline is implemented in `src/drive_qa` and follows these
stages:

1. **Semantic parsing and entity detection**: the input question is analysed to
   infer intents, requested result types, comparison patterns, metrics, and
   candidate biomedical entities.
2. **Schema retrieval**: candidate tables are scored using lexical signals,
   semantic intent mappings, entity matches, metric-specific rules, and
   relation expansion.
3. **Prompt construction**: the selected schema context, join context, semantic
   parse, and resolved entities are assembled into a controlled LLM prompt.
4. **SQL generation**: a supported LLM produces a SQL query for the DRIVE
   schema.
5. **SQL validation**: generated SQL is checked before execution. Only single
   `SELECT` or `WITH ... SELECT` statements are accepted, destructive keywords
   are rejected, and table references are restricted to the allowed schema
   context.
6. **Execution and answer synthesis**: the validated query is executed on MySQL,
   and the result rows are used to produce a natural-language answer.

## Repository Structure

```text
src/drive_qa/                  Core Python package
scripts/demo_query.py           Command-line demo for a single question
scripts/web_qa.py               Local web interface
scripts/evaluation/             Retriever evaluation scripts
data/evaluation_banks/          Spanish and English evaluation banks
outputs/evaluation_results/     Generated evaluation reports
pyproject.toml                  Package metadata
requirements.txt                Runtime and development dependencies
README.md                       Project documentation
```

## Requirements

- Python 3.10 or later.
- Access to the DRIVE MySQL database.
- Database credentials with read access to the `dr` schema.
- An API key for the selected LLM provider.

The default command-line configuration uses Gemini through the
`GEMINI_API_KEY` environment variable. The web interface accepts the model API
key through its configuration form.

## Installation

From the repository root, create a virtual environment and install the package
in editable mode.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

For `cmd.exe`, activate the environment with:

```cmd
.venv\Scripts\activate.bat
```

## Credentials and Security

Database connection parameters are not hard-coded in the project. If a database
URL is not provided explicitly, the command-line scripts request the host, port,
username, and password interactively. The password is read using a hidden input
prompt.

The expected SQLAlchemy URL format is:

```text
mysql+pymysql://USER:PASSWORD@HOST:PORT/dr
```

For Gemini-based command-line usage, set the API key before running the demo:

### macOS / Linux

```bash
export GEMINI_API_KEY="your_api_key"
```

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="your_api_key"
```

### Windows cmd.exe

```cmd
set GEMINI_API_KEY=your_api_key
```

The local web interface keeps database credentials and model API keys in memory
for the active server session. Request logs are sanitised to avoid exposing raw
connection strings or API keys.

## Command-Line Usage

Run a single natural-language question through the full pipeline:

### macOS / Linux

```bash
python3 scripts/demo_query.py "Which drugs are predicted for Asthma in the GNNS method?"
```

### Windows PowerShell

```powershell
python scripts/demo_query.py "Which drugs are predicted for Asthma in the GNNS method?"
```

You may also pass the database URL explicitly:

```bash
python3 scripts/demo_query.py "Which drugs are predicted for Asthma in the GNNS method?" \
  --db-url "mysql+pymysql://USER:PASSWORD@HOST:PORT/dr"
```

Useful options include:

```bash
python3 scripts/demo_query.py "Which drugs are predicted for Alzheimer disease?" \
  --model gemini-3.1-flash-lite \
  --max-rows 20 \
  --log-level INFO
```

The script prints the retrieved tables, generated SQL, natural-language answer,
and returned rows.

## Local Web Interface

Start the web interface with:

### macOS / Linux

```bash
python3 scripts/web_qa.py --port 8502
```

### Windows PowerShell

```powershell
python scripts/web_qa.py --port 8502
```

Then open:

```text
http://127.0.0.1:8502
```

The web interface provides a local browser-based workflow for configuring the
database connection, selecting a model, submitting natural-language questions,
and inspecting the generated SQL and result rows.

## Supported LLM Backends

The model registry currently includes Gemini models and Azure OpenAI-compatible
deployments. Available model identifiers are defined in
`src/drive_qa/model_registry.py`; at the time of this repository version, they
include:

- `gemini-3.5-flash`
- `gemini-3.1-flash-lite`
- `Kimi-K2.6`
- `gpt-5.4-mini`
- `gpt-5.4`
- `deepseek-v4-flash`

The command-line demo uses the legacy Gemini path by default. The web interface
uses the model registry and sends the selected API key to the appropriate client
for the active session.

## Programmatic Usage

The pipeline can also be used as a Python package:

```python
from drive_qa import create_pipeline

pipeline = create_pipeline(
    db_url="mysql+pymysql://USER:PASSWORD@HOST:PORT/dr",
    gemini_model="gemini-3.1-flash-lite",
)

result = pipeline.answer("Which drugs are predicted for Asthma in the GNNS method?")
print(result["sql"])
print(result.get("answer"))
```

## Retriever Evaluation

The project includes bilingual evaluation banks for the schema retriever. These
banks cover the main DRIVE query families, including GNN scores, network
proximity, threshold values, information paths, disease pathways, pathway
methods, pathway counts, and cross-method questions.

Run all Spanish and English evaluation banks:

### macOS / Linux

```bash
python3 scripts/evaluation/run_eval.py --preset bilingual
```

### Windows PowerShell

```powershell
python scripts/evaluation/run_eval.py --preset bilingual
```

Run selected datasets only:

```bash
python3 scripts/evaluation/run_eval.py \
  --datasets gnns network_proximity \
  --preset english
```

Evaluation reports are written by default to:

```text
outputs/evaluation_results/
```

The output directory and version tag can be customised:

```bash
python3 scripts/evaluation/run_eval.py \
  --output-dir outputs/evaluation_results \
  --version-tag v12
```

The evaluator reports retrieval and semantic-parsing metrics such as top-1
accuracy, required-table coverage, contamination-free rate, intent accuracy,
semantic exact accuracy, precision, recall, F1, and mean reciprocal rank.




## Academic Context

This repository is intended to support a Master's Thesis / TFM on natural
language access to biomedical drug-repurposing data.
