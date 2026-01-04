import os
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from fastapi.middleware.cors import CORSMiddleware
from github_fetch import fetch_repo_files, normalize_repo
from chunking import chunk_files
from pinecone_db import embed_and_store, search, get_repo_stats, delete_all_repos
from helper import conversation_store, ConversationState, rewrite_question, update_summary, MAX_TURNS, SUMMARY_UPDATE_EVERY

load_dotenv()

app = FastAPI()
origins = ["http://localhost:5173", "http://127.0.0.1:5173", "https://githubrag.vercel.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           
    allow_credentials=True,
    allow_methods=["*"],             
    allow_headers=["*"],             
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GEMINI_API_KEY"), 
    temperature=0.2
)

current_repo = None

class IngestRequest(BaseModel):
    repo_url: str

class ChatRequest(BaseModel):
    repo_url: str
    query: str

class PineconeRetriever(BaseRetriever):
    repo_id: str
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str):
        try:
            results = search(self.repo_id, query, top_k=5)
            documents = []
            for match in results.get("matches", []):
                metadata = match["metadata"]
                doc = Document(
                    page_content=metadata["text"],
                    metadata={
                        "source": metadata["file_path"],
                        "chunk_index": metadata.get("chunk_index", 0)
                    }
                )
                documents.append(doc)
            return documents
        except:
            return []

@app.post("/api/ingest")
def ingest_repo(req: IngestRequest, bg: BackgroundTasks):
    global current_repo
    repo_id = normalize_repo(req.repo_url)
    bg.add_task(delete_and_ingest, repo_id)
    return {
        "status": "indexing started",
        "repo": repo_id,
        "message": "Repository will be indexed in the background"
    }

def delete_and_ingest(repo_id: str):
    global current_repo
    try:
        delete_all_repos()
        files = fetch_repo_files(repo_id)
        if not files:
            return
        
        chunks = chunk_files(files)
        if not chunks:
            return
        
        stored_count = embed_and_store(repo_id, chunks)
        current_repo = repo_id
    except:
        pass

@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        repo_id = normalize_repo(req.repo_url)
        user_query = req.query
        conversation_id = f"{repo_id}-default"

        if conversation_id not in conversation_store:
            conversation_store[conversation_id] = ConversationState(repo_id)

        convo = conversation_store[conversation_id]

        if convo.repo_id != repo_id:
            convo = ConversationState(repo_id)
            conversation_store[conversation_id] = convo

        standalone_question = rewrite_question(
            llm=llm,
            summary=convo.summary,
            recent_messages=convo.recent_messages,
            user_query=user_query
        )

        results = search(repo_id, standalone_question, top_k=5)

        context = ""
        sources = []
        for match in results.get("matches", []):
            meta = match["metadata"]
            context += f"\n---\nFILE: {meta['file_path']}\n{meta['text']}"
            sources.append(meta["file_path"])

        answer_prompt = f"""
            You are an expert software engineer helping a developer understand a GitHub repository.

            Use ONLY the repository context provided.
            Do NOT use outside knowledge.
            IMPORTANT:
            -If the user asking any random questions but not about this repository you can formally decline and say the reason.
            -If the user formally wishes you, you can reply to him.
            -Think of the user as a student who wants to learn about this repo.
            Repository context:
            {context}

            User question:
            {standalone_question}

            FORMATTING RULES (STRICT):

            1. Use inline code (`) ONLY for:
            - Exact identifiers (variables, model names, function names)
            - Exact file paths
            - Exact endpoint paths
            - Exact package names when referenced ONCE

            2. DO NOT use inline code for:
            - General concepts
            - Repeated words
            - Common technologies after first mention
            - English sentences or fragments

            3. If a term is mentioned more than once:
            - Use inline code ONLY on the FIRST occurrence
            - Use plain text afterwards

            4. Prefer readable sentences over tokenized formatting.
            5. Write like technical documentation, not source code.

            CRITICAL WRITING RULES (NON-NEGOTIABLE):

            - Write in full sentences and paragraphs.
            - Do NOT break sentences across multiple lines.
            - Do NOT place each word or phrase on a new line.
            - Inline code must appear inside a sentence, not on its own line.

            INLINE CODE USAGE:
            - Use inline code ONLY for exact identifiers (file paths, env vars, function names).
            - Do NOT use inline code for technologies, concepts, or general terms.
            - Use inline code at most ONCE per sentence.

            BAD EXAMPLE (FORBIDDEN):
            jwt
            and
            JWT_PASSWORD

            GOOD EXAMPLE:
            The server uses JWT for authentication and reads the secret from the JWT_PASSWORD environment variable.


            STRUCTURE:
            - Start with a brief explanation.
            - Use sections with headings if needed.
            - Only show code blocks when strictly required.

            SOURCES:
            End the response with:
            ---
            ### Sources
            - file/path.ts
            - another/file.ts
            """
        response = llm.invoke(answer_prompt)
        answer = response.content.strip()

        convo.recent_messages.append({"role": "user", "content": user_query})
        convo.recent_messages.append({"role": "assistant", "content": answer})
        convo.turn_count += 1
        convo.last_updated = datetime.utcnow()

        if len(convo.recent_messages) > MAX_TURNS * 2:
            convo.recent_messages = convo.recent_messages[-MAX_TURNS * 2:]

        if convo.turn_count % SUMMARY_UPDATE_EVERY == 0:
            convo.summary = update_summary(
                llm=llm,
                old_summary=convo.summary,
                recent_messages=convo.recent_messages
            )

        return {
            "answer": answer,
            "sources": list(set(sources)),
            "repo": repo_id,
            "standalone_question": standalone_question
        }

    except:
        return {
            "answer": "An error occurred while processing your query"
        }

@app.post("/api/stats")
def get_stats(req: IngestRequest):
    repo_id = normalize_repo(req.repo_url)
    return get_repo_stats(repo_id)

@app.get("/health")
def health():
    return {"status": "healthy"}