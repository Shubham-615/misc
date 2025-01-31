from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
import os

# Load embedding model (for ChromaDB)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load a proper Question-Answering (QA) model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define a persistent directory for ChromaDB
PERSIST_DIRECTORY = "./chromadb_data"

# Initialize ChromaDB client with persistence
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Check if the collection exists; if not, create it
collection_name = "documents"
if collection_name in [coll.name for coll in client.list_collections()]:
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(name=collection_name)

def ingest_document(file_content: bytes, filename: str):
    """
    Function to ingest a document, generate embeddings, and store in ChromaDB.
    """
    text = file_content.decode("utf-8")  # Assuming text files; handle PDFs differently
    
    # Generate embedding
    embedding = embedding_model.encode(text, truncation=True).tolist()
    
    # Insert into ChromaDB
    collection.add(
        documents=[text],
        metadatas=[{"title": filename}],
        ids=[filename]
    )
    
    return {"message": "Document ingested successfully"}

def ask_question(question: str):
    # Generate embedding for the question
    question_embedding = embedding_model.encode(question, truncation=True).tolist()
    
    # Retrieve relevant documents from ChromaDB
    n_results = min(5, collection.count())
    results = collection.query(query_embeddings=[question_embedding], n_results=n_results)
    
    if not results["documents"]:
        return {"error": "No relevant content found."}
    
    # Extract relevant content
    relevant_content = " ".join(results["documents"][0])
    
    # Truncate the context to 512 tokens
    truncated_context = " ".join(relevant_content.split()[:512])
    
    # Use the QA model to answer the question
    answer = qa_pipeline(question=question, context=truncated_context)
    return {"answer": answer["answer"]}

if __name__ == "__main__":
    # Ingest a document (only needs to be done once)
    # if not os.path.exists("./test2.txt"):
    #     print("Please create a test.txt file with some content.")
    # else:
    #     with open("test2.txt", "rb") as file:
    #         response = ingest_document(file.read(), "test2.txt")
    #         print(response)
    
    # Ask a question
    print(ask_question("moral of test2"))