import os
import requests

GITHUB_API = "https://api.github.com"

def normalize_repo(repo_url: str) -> str:
    if repo_url.startswith("http"):
        parts = repo_url.rstrip("/").split("/")
        return f"{parts[-2]}/{parts[-1]}"
    return repo_url

def fetch_repo_files(repo_id: str):
    owner, repo = repo_id.split("/")

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not set")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    tree_url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    res = requests.get(tree_url, headers=headers)
    res.raise_for_status()

    files = []

    for item in res.json().get("tree", []):
        if item.get("type") != "blob":
            continue

        path = item.get("path", "")
        if not path.endswith((".py", ".js", ".ts", ".md")):
            continue

        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
        content_res = requests.get(raw_url)

        if content_res.status_code != 200:
            continue

        files.append({
            "path": path,
            "content": content_res.text
        })

    return files
