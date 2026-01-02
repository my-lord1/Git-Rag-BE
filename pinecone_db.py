import os
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("git-rag")

NAMESPACE = "repos"

def embed_and_store(repo_id: str, chunks):
    records = []

    for i, chunk in enumerate(chunks):
        records.append({
            "id": f"{repo_id}-{i}",
            "text": chunk["text"],
            "repo_id": repo_id,
            "file_path": chunk["path"]
        })

    for i in range(0, len(records), 50):
        index.upsert_records(
            namespace=NAMESPACE,
            records=records[i:i+50]
        )

def search(repo_id: str, query_embedding):
    return index.query(
        namespace=NAMESPACE,
        vector=query_embedding,
        top_k=5,
        filter={"repo_id": {"$eq": repo_id}},
        include_metadata=True
    )
