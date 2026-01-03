from typing import Dict, List
from datetime import datetime

class ConversationState:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.summary: str = ""
        self.recent_messages: List[dict] = []
        self.turn_count: int = 0
        self.last_updated = datetime.utcnow()

conversation_store: Dict[str, ConversationState] = {}

MAX_TURNS = 5         
SUMMARY_UPDATE_EVERY = 4


def rewrite_question(llm, summary: str, recent_messages: list, user_query: str) -> str:
    history = ""
    for msg in recent_messages:
        history += f"{msg['role'].upper()}: {msg['content']}\n"

    prompt = f"""
    Task: Rewrite the User Question to be self-contained based on context.
    
    Rules:
    - If the user is just saying "hi" or greeting, return ONLY the word "GREETING".
    - If the user is asking a code question, include necessary context from the history.
    - Return only the rewritten text.

    Summary: {summary}
    History: {history}
    User Question: {user_query}
    """
    response = llm.invoke(prompt)
    return response.content.strip()


def update_summary(llm, old_summary: str, recent_messages: list) -> str:
    dialogue = ""
    for msg in recent_messages:
        dialogue += f"{msg['role'].upper()}: {msg['content']}\n"

    prompt = f"""
    You maintain a short conversation summary (3–4 sentences max).

    Existing summary:
    {old_summary if old_summary else "None"}

    Recent conversation:
    {dialogue}

    Update the summary to reflect the conversation so far.
    Focus on user intent and what is being explored.
    """

    response = llm.invoke(prompt)
    return response.content.strip()
