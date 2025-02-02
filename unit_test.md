1. Setup for Testing
    a. Install Dependencies
        Install pytest and httpx (used by TestClient):
            pip install pytest httpx
    b. Create a Test File
        Create a file named test_main.py in the same directory as your main.py.

2. Unit Test Cases
    Here are the test cases for each functionality:

        import pytest
        from fastapi.testclient import TestClient
        from main import app  # Import your FastAPI app

        # Initialize the test client
        client = TestClient(app)

        # Mock OpenAI Embedding Response
        mock_embedding = [0.1, 0.2, 0.3]  # Example embedding vector

        # Mock OpenAI Chat Completion Response
        mock_chat_completion = {
            "choices": [
                {
                    "message": {
                        "content": "This is the answer based on the document."
                    }
                }
            ]
        }

        # Mock Database Connection
        @pytest.fixture
        def mock_db(mocker):
            mock_pool = mocker.AsyncMock()
            mock_conn = mocker.AsyncMock()
            mock_pool.acquire.return_value = mock_conn
            app.state.pool = mock_pool
            return mock_conn

        # Test Document Ingestion
        def test_ingest_document(mock_db, mocker):
            # Mock OpenAI Embedding API
            mocker.patch("openai.Embedding.create", return_value={"data": [{"embedding": mock_embedding}]})

            # Simulate file upload
            response = client.post(
                "/ingest/",
                files={"file": ("test.txt", b"Hello, this is a test document.")}
            )

            # Assertions
            assert response.status_code == 200
            assert response.json()["message"] == "Document ingested successfully!"
            mock_db.execute.assert_called_once()  # Ensure database insertion was called

        # Test Q&A API
        def test_ask_question(mock_db, mocker):
            # Mock OpenAI Embedding API
            mocker.patch("openai.Embedding.create", return_value={"data": [{"embedding": mock_embedding}]})

            # Mock database query result
            mock_db.fetch.return_value = [
                (1, "Test Document", "This is a test document.", mock_embedding)
            ]

            # Mock OpenAI Chat Completion API
            mocker.patch("openai.ChatCompletion.create", return_value=mock_chat_completion)

            # Simulate asking a question
            response = client.post(
                "/ask/",
                json={"question": "What is this document about?"}
            )

            # Assertions
            assert response.status_code == 200
            assert "answer" in response.json()
            assert response.json()["source"] == "Test Document"

        # Test Document Selection API
        def test_select_documents():
            # Simulate selecting documents
            response = client.post(
                "/select-documents/",
                json={"document_ids": [1, 2, 3]}
            )

            # Assertions
            assert response.status_code == 200
            assert response.json()["message"] == "Documents selected successfully!"

        # Test Health Check API
        def test_health_check():
            # Simulate health check
            response = client.get("/health/")

            # Assertions
            assert response.status_code == 200
            assert response.json()["status"] == "OK"


3. Explanation of Test Cases
    a. Document Ingestion
        Purpose : Tests the /ingest/ endpoint to ensure that:
            A file can be uploaded.
            An embedding is generated using OpenAI.
            The document and its embedding are stored in the database.
        Mocking :
            Mocks the OpenAI Embedding.create method to avoid making real API calls.
            Mocks the database connection to simulate insertion.
    b. Q&A API
        Purpose : Tests the /ask/ endpoint to ensure that:
            A question can be asked.
            Relevant documents are retrieved from the database.
            An answer is generated using OpenAI's GPT model.
        Mocking :
            Mocks the OpenAI Embedding.create and ChatCompletion.create methods.
            Mocks the database query to return a predefined document.
    c. Document Selection
        Purpose : Tests the /select-documents/ endpoint to ensure that:
            Specific document IDs can be selected.
        Assertions :
            Verifies that the response contains the correct message.
    d. Health Check
        Purpose : Tests the /health/ endpoint to ensure that:
            The service is running and returns a "OK" status.


4. Running the Tests
    Run the tests using the pytest command:
        pytest test_main.py


5. Test Coverage
    To measure test coverage, install the pytest-cov plugin:
        pip install pytest-cov
        Run the tests with coverage:
        pytest --cov=main test_main.py


6. Negative Test Cases
    Add negative test cases to cover edge scenarios:

        # Test invalid file upload
        def test_invalid_file_upload():
            response = client.post("/ingest/", files={"file": ("", b"")})  # Empty file
            assert response.status_code == 422  # Unprocessable Entity

        # Test invalid question format
        def test_invalid_question():
            response = client.post("/ask/", json={"question": ""})  # Empty question
            assert response.status_code == 422

        # Test no relevant documents found
        def test_no_relevant_documents(mock_db, mocker):
            mocker.patch("openai.Embedding.create", return_value={"data": [{"embedding": mock_embedding}]})
            mock_db.fetch.return_value = []  # No documents in the database

            response = client.post("/ask/", json={"question": "What is this document about?"})
            assert response.status_code == 200
            assert response.json()["answer"] == "No relevant documents found."


