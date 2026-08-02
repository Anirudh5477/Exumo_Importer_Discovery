import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
from urllib.parse import urlparse

# Add the parent directory to the Python path so it can find config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddgs import DDGS
from config import DOMAIN_BLACKLIST

def extract_domain(url: str) -> str:
    """
    Extracts and normalizes the domain from a URL (stripping leading www.).
    """
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def is_valid_url(url: str) -> bool:
    """
    Checks if the URL belongs to a blacklisted domain or non-HTML file format.
    """
    if not url:
        return False

    try:
        parsed = urlparse(url)
        path = parsed.path.lower()

        # Ignore non-HTML file extensions
        non_html_extensions = ('.pdf', '.zip', '.doc', '.docx', '.jpg', '.jpeg', '.png', '.gif', '.svg')
        if any(path.endswith(ext) for ext in non_html_extensions):
            return False

        domain = extract_domain(url)
        if not domain:
            return False

        for blacklisted in DOMAIN_BLACKLIST:
            if blacklisted in domain:
                return False

        return True
    except Exception:
        return False

def execute_searches(queries: list[str], max_results_per_query: int = 10) -> list[dict]:
    """
    Executes a list of search queries using DuckDuckGo.
    Collects raw search results, deduplicates by domain (retaining first seen URL per domain),
    and pre-filters out non-HTML files & blacklisted domains.
    Returns a list of dicts: {"title": str, "url": str, "snippet": str, "source_query": str}
    """
    print(f"\n[*] Executing {len(queries)} search queries...")
    
    all_results = []
    seen_domains = set()

    for query in queries:
        print(f"    -> Searching: '{query}'")
        try:
            # Fetch text results from DuckDuckGo using context manager
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results_per_query))
            
            if results:
                for res in results:
                    url = res.get("href", "")
                    if not is_valid_url(url):
                        continue

                    domain = extract_domain(url)
                    if domain in seen_domains:
                        continue

                    seen_domains.add(domain)
                    all_results.append({
                        "title": res.get("title", ""),
                        "url": url,
                        "snippet": res.get("body", ""),
                        "source_query": query
                    })
                    
            # Polite delay between calls as requested
            time.sleep(2.5) 

        except Exception as e:
            print(f"[!] Error during search for '{query}': {e}")
            time.sleep(5.0)

    print(f"[*] Search complete. Found {len(all_results)} unique, valid candidate domains.")
    return all_results

if __name__ == "__main__":
    # Test run standalone
    test_queries = [
        "Ceramic Tiles importer Germany",
        "Fliesen Importeur Deutschland"
    ]
    candidates = execute_searches(test_queries, max_results_per_query=5)
    
    for idx, c in enumerate(candidates, 1):
        safe_title = c['title'].encode('ascii', 'ignore').decode('ascii')
        print(f"{idx}. {safe_title} - {c['url']}")