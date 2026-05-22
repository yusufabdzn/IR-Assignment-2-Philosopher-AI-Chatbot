import os
import requests
from ddgs import DDGS

def web_search(query: str, api_key: str = None) -> str:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            title = r.get("title", "")
            body = r.get("body", "")
            combined = f"{title} {body}".lower()
            if any(term in combined for term in ["yoga", "zionism", "nazism"]):
                continue
            results.append(f"- {title}: {body}")
    return "\n".join(results)