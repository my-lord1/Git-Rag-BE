from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".h": Language.C,
}

def get_language_from_path(path: str):
    for ext, lang in LANGUAGE_MAP.items():
        if path.endswith(ext):
            return lang
    return None

def get_splitter(path: str):
    language = get_language_from_path(path)
    
    if language:
        try:
            return RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=1000,
                chunk_overlap=100,
            )
        except:
            return RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
            )
    
    if path.endswith((".md", ".rst", ".txt")):
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=1000,
            chunk_overlap=100,
        )
    
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

def chunk_files(files):
    chunks = []
    
    for file in files:
        path = file["path"]
        content = file["content"]
        
        try:
            splitter = get_splitter(path)
            splits = splitter.split_text(content)
            
            for i, text in enumerate(splits):
                chunks.append({
                    "text": text,
                    "path": path,
                    "chunk_index": i,
                    "total_chunks": len(splits)
                })
        except:
            chunks.append({
                "text": content,
                "path": path,
                "chunk_index": 0,
                "total_chunks": 1
            })
            
    return chunks