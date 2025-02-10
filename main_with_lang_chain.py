import os
import asyncpg
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# Configure OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Lifespan event handler for managing database connection pool
@asynccontextmanager
async def lifespan(app: FastAPI):
    # PostgreSQL connection with pgvector
    app.state.pool = await asyncpg.create_pool(
        user="your_db_user",
        password="your_db_password",
        database="your_db_name",
        host="your_db_host"
    )
    yield
    await app.state.pool.close()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Initialize PGVector as a Vector Store
async def get_vector_store():
    async with app.state.pool.acquire() as conn:
        return PGVector(
            connection=conn,
            table_name="documents",
            embedding_function=embeddings_model
        )

# Pydantic model for document selection
class DocumentSelection(BaseModel):
    document_ids: List[int]

# **🔹 Document Ingestion API**
@app.post("/ingest/")
async def ingest_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")  # Assuming plain text for simplicity
        
        # Create LangChain document
        doc = Document(page_content=text, metadata={"title": file.filename})
        
        # Store document embeddings
        vector_store = await get_vector_store()
        await vector_store.add_documents([doc])
        
        return JSONResponse(content={"message": "Document ingested successfully!"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# **🔹 Q&A API**
@app.post("/ask/")
async def ask_question(question: str):
    try:
        vector_store = await get_vector_store()
        
        # Retrieve similar documents using LangChain's vector search
        docs = await vector_store.similarity_search(question, k=1)  # Retrieve top-1 document

        if not docs:
            return JSONResponse(content={"answer": "No relevant documents found."})

        retrieved_doc = docs[0].page_content

        # Generate answer using OpenAI GPT with LangChain
        chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        response = chat_model.predict(f"Based on this document:\n{retrieved_doc}\nAnswer this question: {question}")

        return JSONResponse(content={"answer": response, "source": docs[0].metadata["title"]})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# **🔹 Document Selection API**
@app.post("/select-documents/")
async def select_documents(selection: DocumentSelection):
    try:
        global selected_documents
        selected_documents = selection.document_ids
        return JSONResponse(content={"message": "Documents selected successfully!"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# **🔹 Health Check API**
@app.get("/health/")
async def health_check():
    return {"status": "OK"}
