# GitHub RAG – Backend

This project lets users paste a GitHub repository link and ask questions about the code. It fetches the repo files, finds the most relevant parts, and uses AI to answer questions based on the actual code. This helps developers understand repositories faster without reading every file.

This repo is for backend of this project  
[For frontend purposes click here](https://github.com/my-lord1/Git-Rag-FE)

---

## Tech Stack (Backend)

- FastAPI – API framework
- Python
- LangChain – chain orchestration
- Vector Database – Pinecone
- Embedding Models – Pinecone Llama model
- LLM – gemini-2.5-flash
- Fetching repository – GitHub API
- Chunking – Custom logic

---

## Flow of My Code

The frontend sends a GitHub repository URL along with a user question to the backend. The backend first runs a repository ingestion chain that fetches all relevant files from the GitHub repo and filters out unnecessary files like config and lock files. The fetched code then goes through a normalization and chunking step, where large files are split into smaller readable chunks while keeping the file structure and paths undamaged. These chunks are converted into embeddings using an embedding model and stored in a vector database.

When a user asks a question, a retrieval chain embeds the question and fetches the most relevant code chunks from the vector database. These retrieved chunks are then passed into an LLM response chain, where the code context and user question are combined to generate an answer strictly based on the actual repository code. The backend finally returns this answer to the frontend as a JSON response.


---

## Endpoints

- **POST `/api/ingest`**  
  Takes a GitHub repository URL and starts indexing the repo in the background by fetching files, chunking code, and storing embeddings.

- **POST `/api/chat`**  
  Accepts a repository URL and a user question, retrieves the most relevant code chunks, and returns an AI-generated answer based only on the repo code.

- **POST `/api/stats`**  
  Returns basic information about the indexed repository, like how many files and chunks are stored.

- **GET `/health`**  
  Simple health check endpoint to confirm the backend service is running.
