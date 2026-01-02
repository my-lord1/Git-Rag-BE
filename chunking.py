from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def get_splitter(path: str):
    if path.endswith(".js") or path.endswith(".ts"):
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.JS,
            chunk_size=1000,
            chunk_overlap=100,
        )
    elif path.endswith(".py"):
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=100,
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )


def chunk_files(files):
    chunks = []

    for file in files:
        splitter = get_splitter(file["path"])
        splits = splitter.split_text(file["content"])

        for text in splits:
            chunks.append({
                "text": text,
                "path": file["path"]
            })

    return chunks
