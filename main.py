import os
from fastapi import FastAPI, BackgroundTasks
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from .schemas import IngestRequest, ChatRequest
from .github_fetch import fetch_repo_files, normalize_repo
from .chunking import chunk_files
from .pinecone_db import embed_and_store, search, pc

load_dotenv()
app = FastAPI()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)

@app.post("/api/ingest")
def ingest_repo(req: IngestRequest, bg: BackgroundTasks):
    repo_id = normalize_repo(req.repo_url)
    bg.add_task(run_ingestion, repo_id)
    return {"status": "indexing started", "repo": repo_id}

def run_ingestion(repo_id: str):
    files = fetch_repo_files(repo_id)
    chunks = chunk_files(files)
    embed_and_store(repo_id, chunks)

@app.post("/api/chat")
def chat(req: ChatRequest):
    repo_id = normalize_repo(req.repo_url)

    query_embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[req.query],
        parameters={"input_type": "query"}
    ).data[0].values

    results = search(repo_id, query_embedding)

    context = ""
    sources = []

    for match in results["matches"]:
        meta = match["metadata"]
        context += f"\n---\nFILE: {meta['file_path']}\n{meta['text']}"
        sources.append(meta["file_path"])

    prompt = (
        "You are an expert developer.\n"
        "Answer ONLY using the context below.\n"
        f"{context}\n\nQuestion: {req.query}"
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": list(set(sources))
    }
