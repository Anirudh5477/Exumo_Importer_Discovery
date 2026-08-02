import os
import sys
import re
import asyncio
from urllib.parse import urljoin, urlparse

# Add the parent directory to the Python path so it can find config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import trafilatura
from bs4 import BeautifulSoup
from config import SCRAPER_TIMEOUT

# Realistic browser User-Agent to avoid bot blocking
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Keyword patterns for sub-page classification
SUBPAGE_PATTERNS = {
    "about_url": ["about", "uber-uns", "ueber-uns", "unternehmen", "company"],
    "products_url": ["product", "produkte", "grosshandel", "wholesale", "sortiment"],
    "contact_url": ["contact", "kontakt", "impressum"],
}

def discover_subpages(base_url: str, html_content: str) -> dict[str, str]:
    """
    Uses BeautifulSoup to scan internal links on the homepage.
    Finds candidate sub-page URLs for about, products, and contact pages.
    Returns a dict mapping page_type -> full resolved URL.
    """
    soup = BeautifulSoup(html_content, "lxml")
    base_domain = urlparse(base_url).netloc
    found = {}

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        # Resolve relative URLs
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        # Only follow internal links on the same domain
        if parsed.netloc != base_domain:
            continue

        href_lower = full_url.lower()

        for page_type, keywords in SUBPAGE_PATTERNS.items():
            if page_type not in found:
                if any(kw in href_lower for kw in keywords):
                    found[page_type] = full_url

        # Stop early if all three types are discovered
        if len(found) == len(SUBPAGE_PATTERNS):
            break

    return found


async def fetch_page(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    """
    Fetches a single page with the provided AsyncClient.
    Returns (url, html_content) or (url, "") on failure.
    """
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return url, resp.text
    except Exception as e:
        print(f"  [!] Failed to fetch {url}: {e}")
        return url, ""


async def scrape_company_deep(domain_url: str) -> dict:
    """
    Deep-scrapes a company domain:
      1. Fetches the homepage.
      2. Discovers about/products/contact sub-pages.
      3. Asynchronously fetches up to 3 sub-pages in parallel.
      4. Extracts clean text from all pages with trafilatura.
    Returns a structured dict with combined_text, pages_scraped, and raw_html_map.
    """
    pages_scraped = []
    raw_html_map = {}
    all_texts = []

    timeout = httpx.Timeout(SCRAPER_TIMEOUT)
    async with httpx.AsyncClient(headers=BROWSER_HEADERS, timeout=timeout) as client:
        # Step 1: Fetch homepage
        print(f"  [~] Fetching homepage: {domain_url}")
        homepage_url, homepage_html = await fetch_page(client, domain_url)

        if not homepage_html:
            print(f"  [!] Could not load homepage for {domain_url}")
            return {
                "domain": domain_url,
                "homepage_url": homepage_url,
                "combined_text": "",
                "pages_scraped": [],
                "raw_html_map": {},
            }

        pages_scraped.append(homepage_url)
        raw_html_map[homepage_url] = homepage_html

        # Extract clean text from homepage
        homepage_text = trafilatura.extract(homepage_html) or ""
        if homepage_text:
            all_texts.append(homepage_text)

        # Step 2: Discover sub-pages
        subpages = discover_subpages(homepage_url, homepage_html)
        print(f"  [~] Discovered sub-pages: {list(subpages.keys())}")

        # Step 3: Fetch up to 3 sub-pages in parallel
        subpage_urls = list(subpages.values())[:3]
        if subpage_urls:
            tasks = [fetch_page(client, url) for url in subpage_urls]
            results = await asyncio.gather(*tasks)

            for url, html in results:
                if html:
                    raw_html_map[url] = html
                    pages_scraped.append(url)
                    text = trafilatura.extract(html) or ""
                    if text:
                        all_texts.append(text)

    # Step 4: Combine and cap text to ~15,000 characters
    combined_text = "\n\n---\n\n".join(all_texts)
    combined_text = combined_text[:15000]

    return {
        "domain": domain_url,
        "homepage_url": homepage_url,
        "combined_text": combined_text,
        "pages_scraped": pages_scraped,
        "raw_html_map": raw_html_map,
    }


if __name__ == "__main__":
    test_url = "https://www.fliesen-zentrum.de/"
    print(f"\n=== Testing scraper on: {test_url} ===\n")

    result = asyncio.run(scrape_company_deep(test_url))

    print(f"\n[Result]")
    print(f"  Domain:         {result['domain']}")
    print(f"  Pages Scraped:  {result['pages_scraped']}")
    print(f"  Combined Text:  {len(result['combined_text'])} chars")
    print(f"\n--- Text Preview ---")
    print(result["combined_text"][:1000])
