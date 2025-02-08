# Document Ingestion and RAG-Driven Q&A Backend
  This is a Python-based backend application designed for document ingestion , embedding generation , and retrieval-augmented generation (RAG) -based question answering. It leverages      OpenAI's APIs for embeddings and GPT models, PostgreSQL for data storage, and FastAPI for building RESTful APIs.

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

Run the application:
  uvicorn main:app --reload

# Usage
# Document Ingestion

Upload a document to generate embeddings and store it in the database:
  curl -X POST "http://localhost:8000/ingest/" \
  -H "accept: application/json" \
  -F "file=@example.txt"

# Ask a Question:
    Ask a question to retrieve an answer based on stored documents:
      curl -X POST "http://localhost:8000/ask/" \
      -H "Content-Type: application/json" \
      -d '{"question": "What is this document about?"}'


# Select Documents :

  Specify which documents to consider during Q&A:
    curl -X POST "http://localhost:8000/select-documents/" \
    -H "Content-Type: application/json" \
    -d '{"document_ids": [1, 2]}'



# API Endpoints:

  ENDPOINT  METHOD  DESCRIPTION
  /ingest/  POST  Upload a document and store its embedding.
  /ask/  POST  Ask a question and get an answer.
  /select-documents/  POST  Select specific documents for Q&A.
  /health/  GET  Check if the service is running.


# Testing:

  Unit tests are included to ensure the correctness of the application. To run the tests:
    pip install pytest pytest-cov

  Run the tests:
    pytest test_main.py

  Check test coverage:
    pytest --cov=main test_main.py

# For more details, refer to the Unit Test Documentation .

# Deployment:
Build the Docker image:
  docker build -t rag-backend .

Run the container:
docker run -d -p 8000:8000 rag-backend


Kubernetes
Deploy the application using Kubernetes manifests (deployment.yaml and service.yaml) provided in the repository.

CI/CD
A GitHub Actions pipeline is set up to automate testing, building, and deployment. Push changes to the main branch to trigger the pipeline.






    


