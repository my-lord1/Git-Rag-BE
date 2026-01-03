import os
import requests
from typing import List, Dict

GITHUB_API = "https://api.github.com"

ALLOWED_EXTENSIONS = (
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".rb", ".php", ".cpp", ".c", ".h",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".conf",
    ".md", ".rst", ".txt",
    ".sql", ".graphql",
    ".dockerfile"
)

EXCLUDED_FILES_LIST = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Gemfile.lock",
    "poetry.lock",
    "Cargo.lock",
    "go.sum"
}

EXCLUDED_DIRS = {
    "node_modules", "venv", ".venv", "vendor", "__pycache__",
    ".git", "dist", "build", "target", ".next", "coverage",
    "tmp", "temp", "logs", ".pytest_cache", "env", ".env",
    ".gradle", ".maven", ".cargo", ".bundle"
}

EXCLUDED_FILES = {
    ".lock", ".log", ".pyc", ".o", ".a", ".class", ".jar",
    ".min.js", ".min.css", ".exe", ".dll", ".so",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf",
    ".zip", ".tar", ".gz", ".mp4", ".mp3", ".wav", ".mov"
}

def normalize_repo(repo_url: str) -> str:
    if repo_url.startswith("http"):
        parts = repo_url.rstrip("/").split("/")
        return f"{parts[-2]}/{parts[-1]}"
    return repo_url

def should_include_file(path: str, file_size: int = 0) -> bool:
    if file_size > 2 * 1024 * 1024:
        return False
    
    path_parts = path.split("/")
    for excluded_dir in EXCLUDED_DIRS:
        if excluded_dir in path_parts:
            return False
    
    filename = path.split("/")[-1]
    if filename in EXCLUDED_FILES_LIST:
        return False
    
    for excluded_ext in EXCLUDED_FILES:
        if path.endswith(excluded_ext):
            return False
    
    if not any(path.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return False
    
    if path.endswith("Dockerfile") or path.endswith("docker-compose.yml"):
        return True
    
    return True

def fetch_repo_files(repo_id: str) -> List[Dict[str, str]]:
    owner, repo = repo_id.split("/")
    
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not set")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }
    
    tree_url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    
    try:
        res = requests.get(tree_url, headers=headers)
        res.raise_for_status()
    except:
        raise
    
    tree_data = res.json()
    all_items = tree_data.get("tree", [])
    files = []
    
    for item in all_items:
        if item.get("type") != "blob":
            continue
        
        path = item.get("path", "")
        size = item.get("size", 0)
        
        if not should_include_file(path, size):
            continue
        
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
        
        try:
            content_res = requests.get(raw_url, headers=headers, timeout=10)
            if content_res.status_code == 200:
                files.append({
                    "path": path,
                    "content": content_res.text
                })
        except:
            continue
            
    return files