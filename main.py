import os
import openai
import asyncpg
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Lifespan event handler for managing database connection pool
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create a database connection pool
    app.state.pool = await asyncpg.create_pool(
        user="",
        password="",
        database="",
        host=""
    )
    yield
    # Shutdown: Close the database connection pool
    await app.state.pool.close()

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Helper function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Pydantic model for document selection
class DocumentSelection(BaseModel):
    document_ids: List[int]

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
        async with app.state.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO documents (title, content, embedding) VALUES ($1, $2, $3)",
                file.filename, text, embedding
            )

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

        # Retrieve all documents and their embeddings
        async with app.state.pool.acquire() as conn:
            rows = await conn.fetch("SELECT id, title, content, embedding FROM documents")

        # Find the most similar document using cosine similarity
        best_match = None
        highest_similarity = -1
        for row in rows:
            doc_id, title, content, embedding = row
            similarity = cosine_similarity(question_embedding, embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = (title, content)

        if not best_match:
            return JSONResponse(content={"answer": "No relevant documents found."})

        title, content = best_match

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
@app.post("/select-documents/")
async def select_documents(selection: DocumentSelection):
    try:
        global selected_documents
        selected_documents = selection.document_ids
        return JSONResponse(content={"message": "Documents selected successfully!"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health Check API
@app.get("/health/")
async def health_check():
    return {"status": "OK"}

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)