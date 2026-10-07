# Investor RAG System

A Python command-line application that answers investor questions using evidence retrieved from local documents. It uses Chroma for persistent vector storage, Hugging Face embeddings for retrieval, and an LLM served through Groq to generate answers with passage citations.

The included data is **synthetic and for testing only**. Its products and rules do not represent real Indian regulations or financial advice.

## Features

- Separate scripts for document indexing and question answering.
- SHA-256 content hashes and document IDs to detect document changes.
- Adds new documents, skips embedding unchanged documents, and replaces outdated chunks.
- Retrieves the top 5 matching chunks and supplies them as evidence to the LLM.
- Prompts the model to cite supporting passages and acknowledge insufficient evidence.
- Remembers the last few turns and rewrites follow-up questions ("what about Harbor?") into standalone questions before retrieval.
- Removes a document's chunks if its file is emptied.

## Project files

| File | Purpose |
| --- | --- |
| `update_db.py` | Load text files, create chunks, and update Chroma |
| `chat.py` | Retrieve evidence and answer questions in a terminal loop |
| `DATA_FILES/` | Source text documents |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes secrets, virtual environments, and generated database files |

The `chroma_db/` directory is generated locally when indexing documents.

## Setup

Use Python 3.11 and a virtual environment. Run the following from the project folder:

```bash
python -m venv .venv
```

Activate it using the command for your terminal:

| Terminal | Command |
| --- | --- |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Linux/macOS | `source .venv/bin/activate` |

```bash
python -m pip install -r requirements.txt
```

Set `GROQ_API_KEY` in the same terminal before starting the chat. Replace the placeholder with your own key:

```powershell
# Windows PowerShell
$env:GROQ_API_KEY = "your_groq_api_key"
```

For Command Prompt, use `set GROQ_API_KEY=your_groq_api_key`. For Linux/macOS, use `export GROQ_API_KEY="your_groq_api_key"`.

The code reads the environment variable with `os.getenv()`. Creating a `.env` file alone does not load it. Do not commit your API key.

## Index documents

Pass the file path and a stable document ID to `update_db.py`. For the included test documents, use:

| File in `DATA_FILES/` | Suggested `doc_id` |
| --- | --- |
| `01_aster_scheme.txt` | `SYN_ASTER_01` |
| `02_harbor_deposit.txt` | `SYN_HARBOR_01` |
| `03_lumen_fund.txt` | `SYN_LUMEN_01` |
| `04_demo_tax_rules.txt` | `SYN_TAX_01` |

```bash
python update_db.py DATA_FILES/01_aster_scheme.txt SYN_ASTER_01
```

The script processes one file per run. Repeat for each document, using a different ID for each. After editing a document, run the updater with that document's **same ID** to replace its stored version.

## Ask questions

```bash
python chat.py
```

Example questions:

- What is the minimum initial investment in Aster Growth Scheme?
- Compare the minimum initial amounts for Aster Growth Scheme and Harbor Fixed Deposit.
- Does a Lumen Liquid Fund portfolio with 75% debt and 25% cash meet its allocation rules?

Follow-up questions work too, e.g. "And what is its lock-in period?" after asking about Aster Growth Scheme. Add `--debug` to print the rewritten question and the retrieved evidence:

```bash
python chat.py --debug
```

Type `exit` or `quit` to stop. The database is always stored in `chroma_db/` next to the scripts. Index documents before starting the chat.
