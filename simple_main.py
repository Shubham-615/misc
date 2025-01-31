import os
import openai
import psycopg2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname="",
    user="",
    password="",
    host=""
)

# Create table if it doesn't exist
with conn.cursor() as cur:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(1536)
        );
    """)
    conn.commit()

# Document Ingestion API
@app.post("/ingest/")
async def ingest_document(file: UploadFile = File(...)):
    try:
        # Read file content
        content = await file.read()
        text = content.decode("utf-8")  # Assuming plain text for simplicity

        # Generate embedding using OpenAI
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']

        # Store in PostgreSQL
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (title, content, embedding) VALUES (%s, %s, %s)",
                (file.filename, text, embedding)
            )
            conn.commit()

        return JSONResponse(content={"message": "Document ingested successfully!"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Q&A API
@app.post("/ask/")
async def ask_question(question: str):
    try:
        # Generate embedding for the question
        response = openai.Embedding.create(
            input=question,
            model="text-embedding-ada-002"
        )
        question_embedding = response['data'][0]['embedding']

        # Find the most similar document in PostgreSQL
        with conn.cursor() as cur:
            cur.execute("""
                SELECT title, content FROM documents
                ORDER BY embedding <-> %s
                LIMIT 1
            """, (question_embedding,))
            result = cur.fetchone()

        if not result:
            return JSONResponse(content={"answer": "No relevant documents found."})

        title, content = result

        # Generate answer using OpenAI GPT
        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Based on the following document:\n{content}\n\nAnswer this question: {question}"}
            ]
        )
        answer = gpt_response['choices'][0]['message']['content']

        return JSONResponse(content={"answer": answer, "source": title})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Document Selection API
selected_documents = []

@app.post("/select-documents/")
async def select_documents(document_ids: List[int]):
    global selected_documents
    selected_documents = document_ids
    return JSONResponse(content={"message": "Documents selected successfully!"})

# Health Check API
@app.get("/health/")
async def health_check():
    return {"status": "OK"}

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)