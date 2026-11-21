from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF assistant. Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I couldn't find that in the document."
After your answer, list the page numbers you used as: Sources: page X, page Y.
STRICT SQL GENERATION RULES:

1. Return exactly ONE SQL query.
2. Return SQL ONLY. No introduction, explanation, comments,
   Markdown formatting, or alternative queries.
3. Use only tables and columns present in the schema.
4. Never invent table names, column names, or relationships.
5. For top-N or bottom-N requests about individual records or entities
   (for example, "top 5 most expensive products"), return matching rows:
   - Do not use GROUP BY or aggregate functions unless the user asks for a
     grouped summary or aggregate (such as totals per company).
   - ORDER BY the requested value, then use LIMIT N.
   - Select all columns for the matching rows if the user does not name
     specific columns.
6. Use GROUP BY and aggregate functions only when the user explicitly asks
   for a summary by category or entity (for example, total cost per company).
7. For bottom-N requests, order the requested value in ASC order and use
   LIMIT N.
8. Use WHERE for filtering individual records.
9. Use HAVING for filtering aggregated groups.
10. NEVER use MAX(SUM(column)) or MIN(SUM(column)).
11. Never use nested aggregate functions at the same query level.
12. Use subqueries or CTEs only when genuinely necessary.
13. Use appropriate JOIN conditions when multiple tables are required.
14. Apply all requested filters before aggregation.
15. Ensure that the generated SQL is syntactically valid.
16. Do not generate multiple SQL statements.
Context:
{context}

Question: {question}
""")

def ask(db, question):
    chunks = db.similarity_search(question, k=4)

    context = "\n\n".join(
        f"[page {c.metadata.get('page', 0) + 1}] {c.page_content}"
        for c in chunks
    )

    chain = prompt | model

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content