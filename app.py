
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama

OLLAMA_MODEL = "llama3.2"

model = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)

# Store the indexed PDF database
state = {"db": None}

# Prompt template for schema-grounded SQL generation
prompt = ChatPromptTemplate.from_template("""
You convert natural-language questions into SQL queries using the schema
information in the context.

Rules:
- Return exactly one SQL query and nothing else: no explanation, Markdown,
  comments, or page references.
- Use only tables and columns explicitly present in the context. Never invent
  schema names.
- For requests about individual records or entities, return matching rows.
  For top-N requests such as "top 5 most expensive products", order the
  relevant column in descending order and apply LIMIT N. Do not aggregate or
  GROUP BY unless the user asks for a grouped summary.
- When the user asks to show or find products, records, or other entities
  without naming specific columns, use SELECT * so the result includes the
  full rows. The column used for sorting (such as COST) is not the only
  column to select unless the user explicitly asks only for that value.
- Use aggregate functions and GROUP BY only for explicit aggregate or
  category-summary requests, such as "maximum cost per company".
- Apply requested filters with WHERE. Use HAVING only to filter aggregated
  groups.
- Do not use nested aggregate functions at the same query level.
- Generate one syntactically valid SQL statement.
- If the context does not provide the schema needed to answer, return this
  exact SQL comment: -- Insufficient schema context to generate the query.

Context:
{context}

Question: {question}
""")

# Build the vector database from the uploaded PDF
def build_index(pdf_path):
    pages = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No readable text found in the uploaded PDF.")

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
    )

    return db


# Retrieve relevant chunks and generate an answer
async def ask(db, question):
    chunks = db.similarity_search(question, k=4)

    if not chunks:
        return "I couldn't find that in the document."

    context = "\n\n".join(
        f"[page {chunk.metadata.get('page', 0) + 1}] "
        f"{chunk.page_content}"
        for chunk in chunks
    )

    chain = prompt | model

    response = await chain.ainvoke({
        "context": context,
        "question": question,
    })

    return response.content


# Handle PDF upload and indexing
def upload(pdf):
    if pdf is None:
        return "Please upload a PDF file first."

    try:
        state["db"] = build_index(pdf.name)
        return "PDF indexed successfully! Ask me anything about it."
    except Exception as exc:
        state["db"] = None
        return f"Failed to index PDF: {exc}"


# Handle chat messages
async def chat(message, history):
    if state["db"] is None:
        return "Please upload a PDF first."

    if not message or not message.strip():
        return "Please enter a question."

    try:
        return await ask(state["db"], message.strip())
    except Exception as exc:
        print(f"Chat error: {exc}")
        if "model" in str(exc).lower() and "not found" in str(exc).lower():
            return (
                f'Ollama model "{OLLAMA_MODEL}" is not installed. '
                f"Run `ollama pull {OLLAMA_MODEL}` and try again."
            )
        return f"Sorry, I couldn't process your question. Error: {exc}"


# Build the Gradio interface
with gr.Blocks(title="Text to SQL") as demo:
    gr.Markdown("# Text to SQL")
    gr.Markdown("Upload schema documentation and ask for SQL in plain language.")

    pdf = gr.File(
        label="Upload a PDF",
        file_types=[".pdf"],
        type="filepath",
    )

    status = gr.Markdown()

    pdf.upload(
        fn=upload,
        inputs=pdf,
        outputs=status,
    )

    gr.ChatInterface(fn=chat)

if __name__ == "__main__":
    demo.launch()
