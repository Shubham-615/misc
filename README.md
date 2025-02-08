# Document Ingestion and RAG-Driven Q&A Backend
This is a Python-based backend application designed for document ingestion , embedding generation , and retrieval-augmented generation (RAG) -based question answering. It leverages OpenAI's APIs for embeddings and GPT models, PostgreSQL for data storage, and FastAPI for building RESTful APIs.

# Features
Document Ingestion : Upload documents, generate embeddings using OpenAI, and store them in PostgreSQL.
RAG-Based Q&A : Retrieve relevant documents and generate answers using OpenAI's GPT models.
Scalable Architecture : Designed to handle large datasets and high query volumes.
Asynchronous APIs : Built with FastAPI for high performance and scalability.
CI/CD Pipeline : Automated testing, building, and deployment using GitHub Actions.

# Table of Contents
Setup
Usage
API Endpoints
Testing
Deployment
Contributing
License

# Setup
# Prerequisites

Python 3.9+
PostgreSQL
OpenAI API Key
Docker (optional, for containerization)

# Installation
Clone the repository:
  git clone https://github.com/your-repo/document-ingestion-rag.git
  cd document-ingestion-rag

# Install dependencies:
  pip install -r requirements.txt

# Set environment variables:

Create a .env file with the following content:
  OPENAI_API_KEY=your_openai_api_key
  DATABASE_URL=postgresql://user:password@localhost:5432/your_db_name

Create the PostgreSQL table:
  CREATE TABLE IF NOT EXISTS documents (
      id SERIAL PRIMARY KEY,
      title TEXT NOT NULL,
      content TEXT NOT NULL,
      embedding float[]
  );





  


