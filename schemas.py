from pydantic import BaseModel

class IngestRequest(BaseModel):
    repo_url: str

class ChatRequest(BaseModel):
    repo_url: str
    query: str
