import os
import sys
import json
from pydantic import BaseModel
from groq import Groq

# Add the parent directory to the Python path so it can find config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GROQ_API_KEY

class CandidateVerification(BaseModel):
    url: str
    company_name: str
    is_direct_company: bool  # True ONLY if it's an actual company selling/importing/distributing products
    entity_type: str         # Choice: "direct_importer_wholesaler", "directory_marketplace", "blog_news", "other"
    confidence_score: int    # 0 to 100 confidence
    reasoning: str           # 1-line explanation of why it is or isn't a direct company

def verify_candidates(candidates: list[dict]) -> list[dict]:
    """
    Takes a list of raw search candidate dicts:
      {"title": str, "url": str, "snippet": str, "source_query": str}
    Passes each candidate's title, URL, and snippet to Groq (llama-3.3-70b-versatile).
    Returns only verified direct companies where is_direct_company == True and
    entity_type == "direct_importer_wholesaler", sorted by confidence_score descending.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing from environment variables / .env file.")

    client = Groq(api_key=GROQ_API_KEY)
    verified_results = []

    print(f"\n[*] Verifying {len(candidates)} search candidates with Groq (llama-3.3-70b-versatile)...")

    system_prompt = (
        "You are an expert AI B2B data analyst. Your job is to strictly classify web search result snippets "
        "to determine if they represent a genuine direct company (importer, wholesaler, distributor, dealer, or manufacturer) "
        "versus noise such as a B2B directory/portal (e.g. Europages, YellowPages, WLW, Alibaba, Kompass), "
        "blog/news article, PDF/document, or generic service provider."
    )

    for idx, candidate in enumerate(candidates, 1):
        title = candidate.get("title", "") or "Unknown"
        url = candidate.get("url", "")
        snippet = candidate.get("snippet", "")
        source_query = candidate.get("source_query", "")

        user_prompt = f"""
Analyze the following search candidate:
- URL: {url}
- Page Title: {title}
- Snippet: {snippet}
- Source Search Query: {source_query}

Task:
Determine whether this is an actual direct buyer/importer/wholesaler company website (e.g., tile importer/distributor in Germany), 
OR if it is a directory portal, marketplace, news blog, or other irrelevant site.

Required JSON Structure:
{{
  "url": "{url}",
  "company_name": "<Extracted company name or domain brand name>",
  "is_direct_company": true or false,
  "entity_type": "<Exactly one of: 'direct_importer_wholesaler', 'directory_marketplace', 'blog_news', 'other'>",
  "confidence_score": <Integer from 0 to 100>,
  "reasoning": "<1-line explanation>"
}}

Guidelines:
1. is_direct_company must be true ONLY if it is a real company selling/importing/distributing physical products.
2. If it is a business directory (YellowPages, Europages, Wer liefert was, Kompass, Indiamart, etc.), marketplace, or article/blog listing top importers, entity_type MUST be 'directory_marketplace' or 'blog_news', and is_direct_company MUST be false.
3. Return ONLY a valid JSON object matching the required structure.
"""
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            content = response.choices[0].message.content
            data = json.loads(content)

            # Ensure company_name and reasoning fallback cleanly if None or missing
            extracted_name = str(data.get("company_name") or title)
            extracted_reasoning = str(data.get("reasoning") or "No reasoning provided.")
            extracted_entity = str(data.get("entity_type") or "other")
            confidence = int(data.get("confidence_score") or 0)
            is_direct = bool(data.get("is_direct_company", False))

            # Validate using Pydantic model
            verification = CandidateVerification(
                url=data.get("url") or url,
                company_name=extracted_name,
                is_direct_company=is_direct,
                entity_type=extracted_entity,
                confidence_score=confidence,
                reasoning=extracted_reasoning
            )

            if verification.is_direct_company and verification.entity_type == "direct_importer_wholesaler":
                result_item = verification.model_dump()
                result_item["snippet"] = snippet
                result_item["source_query"] = source_query
                verified_results.append(result_item)
                print(f"  [+] Direct Company ({verification.confidence_score}%): {verification.company_name} | {verification.url}")
            else:
                print(f"  [-] Rejected ({verification.entity_type}): {verification.company_name} - {verification.reasoning}")

        except Exception as e:
            print(f"  [!] Error verifying candidate '{url}': {e}")
            continue

    # Sort verified companies by confidence_score in descending order
    verified_results.sort(key=lambda x: x["confidence_score"], reverse=True)
    print(f"[*] Candidate verification finished. Total verified direct companies: {len(verified_results)}")
    return verified_results

if __name__ == "__main__":
    from modules.query_expansion import generate_search_queries
    from modules.search_engine import execute_searches

    test_product = "Ceramic Tiles"
    test_country = "Germany"

    print(f"=== Starting Phase 1 Integration Test for '{test_product}' in '{test_country}' ===\n")
    
    # Step 1: Generate Queries
    print("[1] Generating search queries...")
    queries = generate_search_queries(test_product, test_country)
    print(f"    Generated {len(queries)} queries.")

    # Step 2: Execute Searches
    print("\n[2] Executing DuckDuckGo searches and pre-filtering...")
    raw_candidates = execute_searches(queries, max_results_per_query=5)
    print(f"    Collected {len(raw_candidates)} candidate URLs.")

    # Step 3: Verify Candidates
    print("\n[3] Verifying candidates via LLM...")
    verified_companies = verify_candidates(raw_candidates)

    # Step 4: Display Top 10 Verified Direct Companies
    print("\n" + "="*80)
    print(f" TOP 10 VERIFIED DIRECT COMPANIES: {test_product.upper()} ({test_country.upper()})")
    print("="*80)

    top_10 = verified_companies[:10]
    if not top_10:
        print("No direct importer/wholesaler companies were verified.")
    else:
        for idx, comp in enumerate(top_10, 1):
            print(f"\n{idx}. Company Name: {comp['company_name']}")
            print(f"   Website URL:  {comp['url']}")
            print(f"   Confidence:   {comp['confidence_score']}%")
            print(f"   Reasoning:    {comp['reasoning']}")
            print(f"   Snippet:      {comp['snippet'][:150]}...")
