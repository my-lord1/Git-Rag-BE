import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("git-rag")
NAMESPACE = "repos"
BATCH_SIZE = 90 

def delete_all_repos() -> int:
    """Delete all repositories from the namespace"""
    dummy_vector = [0.0] * 1024
    results = index.query(
        namespace=NAMESPACE,
        vector=dummy_vector,
        top_k=10000,
        include_metadata=True
    )
    if not results.get("matches"):
        return 0
    
    ids_to_delete = [match["id"] for match in results["matches"]]
    deleted_count = 0
    for i in range(0, len(ids_to_delete), 100):
        batch = ids_to_delete[i:i+100]
        index.delete(
            ids=batch,
            namespace=NAMESPACE
        )
        deleted_count += len(batch)
    return deleted_count



def delete_repo_data(repo_id: str) -> int:
    """Delete all existing data for a repository before re-ingesting"""
    dummy_vector = [0.0] * 1024
    
    results = index.query(
        namespace=NAMESPACE,
        vector=dummy_vector,
        filter={"repo_id": {"$eq": repo_id}},
        top_k=10000,
        include_metadata=True
    )
    
    if not results.get("matches"):
        return 0
    
    ids_to_delete = [match["id"] for match in results["matches"]]
    deleted_count = 0
    for i in range(0, len(ids_to_delete), 100):
        batch = ids_to_delete[i:i+100]
        index.delete(
            ids=batch,
            namespace=NAMESPACE
        )
        deleted_count += len(batch)
    return deleted_count
    


def embed_and_store(repo_id: str, chunks):
    """Embed chunks using Pinecone's inference API and store in index"""
    if not chunks:
        return 0
    
    delete_repo_data(repo_id)
    records = []
    
    for i, chunk in enumerate(chunks):
        record_id = f"{repo_id}-{i}"
        records.append({
            "id": record_id,
            "values": None, 
            "metadata": {
                "text": chunk["text"],
                "repo_id": repo_id,
                "file_path": chunk["path"],
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1)
            }
        })
    
    texts = [chunk["text"] for chunk in chunks]
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(texts))
        batch_texts = texts[batch_start:batch_end]
        
        embeddings = pc.inference.embed( model="llama-text-embed-v2", inputs=batch_texts, parameters={"input_type": "passage"})
        for i, embedding in enumerate(embeddings.data):
            records[batch_start + i]["values"] = embedding.values
            
    stored_count = 0
    
    for i in range(0, len(records), 50):
        batch = records[i:i+50]
        index.upsert(
            vectors=batch,
            namespace=NAMESPACE
        )
        stored_count += len(batch)
        
    return stored_count
    



def search(repo_id: str, query_text: str, top_k: int = 5):
    """Search for similar chunks using query text"""

    query_embedding = pc.inference.embed( model="llama-text-embed-v2", inputs=[query_text], parameters={"input_type": "query"})
    query_vector = query_embedding.data[0].values
    
    results = index.query(
        namespace=NAMESPACE,
        vector=query_vector,
        top_k=top_k,
        filter={"repo_id": {"$eq": repo_id}},
        include_metadata=True
    )

    return results



def get_repo_stats(repo_id: str) -> dict:
    """Get statistics about stored data for a repository"""
    results = index.query(
        namespace=NAMESPACE,
        vector=[0.0] * 1024,  
        filter={"repo_id": {"$eq": repo_id}},
        top_k=10000,
        include_metadata=True
    )
    
    matches = results.get("matches", [])
    
    stats = {
        "repo_id": repo_id,
        "total_chunks": len(matches),
        "unique_files": len(set(m["metadata"]["file_path"] for m in matches)) if matches else 0
    }
    
    return stats
    