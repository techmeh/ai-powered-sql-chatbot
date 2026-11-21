# Text-to-SQL Chatbot

A Python chatbot that turns natural-language questions into SQL using schema
information from an uploaded PDF. It uses a Gradio web interface, semantic
retrieval, and a locally running Ollama model.

> **Scope:** This app generates SQL text only. It does not connect to a
> database, execute queries, or validate SQL against a live database. Review
> generated queries before using them.

## Features

- Upload PDF schema documentation in the chat interface.
- Extract and split the PDF into searchable passages.
- Retrieve relevant schema context using Hugging Face embeddings and Chroma.
- Generate SQL with Ollama's `llama3.2` model.
- Prompt the model to use documented table and column names and to avoid
  aggregation when the question asks for individual records.

## How it works

1. Upload a PDF listing the database tables, columns, and relationships.
2. The app extracts the text, splits it into overlapping chunks, and indexes
   those chunks.
3. Ask a question in plain language.
4. The app retrieves up to four relevant passages and asks the model to produce
   a SQL query using that context.

SQL quality depends on the accuracy of the schema PDF and the relevance of the
retrieved passages.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- The `llama3.2` model pulled into Ollama

## Setup

Clone this repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install gradio langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-ollama pypdf sentence-transformers
```

Download the language model:

```bash
ollama pull llama3.2
```

With Ollama running, start the app:

```bash
python app.py
```

Open the local URL printed by Gradio.

## Example

Upload a schema PDF that documents a table and its columns, for example:

```text
Table: items
Columns: item_id, name, price
```

Then ask:

```text
Show me the top 5 most expensive products.
```

For that schema, an expected query is:

```sql
SELECT *
FROM items
ORDER BY price DESC
LIMIT 5;
```

Use the actual table and column names from your schema. This example uses
`LIMIT`; other SQL dialects may use different syntax.

## Project structure

```text
.
├── app.py             # Gradio app, PDF indexing, retrieval, and SQL prompt
├── pdf_chat.py        # Alternate chat implementation
├── pdf_loader.py      # PDF indexing helper
├── sql_knowledge.pdf  # Example knowledge document
├── README.md
└── .gitignore
```

## Limitations

- The model can misunderstand questions or generate incorrect SQL. Review the
  output before using it.
- The app does not connect to a database, execute SQL, or validate queries
  against a live schema.
- Only schema details in the retrieved PDF passages are available to the model.
- Top-N prompt examples use `LIMIT`, which may not match every SQL dialect.
- The embedding model may be downloaded the first time a PDF is indexed.

## Privacy

Review the schema document and all files for private information before
publishing this repository. Local model inference uses Ollama, but consider
where uploaded documents and embedding models are stored. The included
`.gitignore` excludes `.env` files and virtual environments.

## Tech stack

Python · Gradio · LangChain · Ollama · Llama 3.2 · Hugging Face · Chroma
