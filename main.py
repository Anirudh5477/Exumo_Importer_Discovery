"""
main.py — End-to-End AI Importer Discovery Engine Orchestrator

Usage:
    python main.py --product "Ceramic Tiles" --country "Germany" --top_n 10
"""
import os
import sys
import json
import time
import asyncio
import argparse
from datetime import datetime

# Ensure project root is on the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.query_expansion import generate_search_queries
from modules.search_engine import execute_searches
from modules.candidate_verifier import verify_candidates
from modules.scraper import scrape_company_deep
from modules.contact_enrich import extract_contacts
from modules.scoring import score_candidate


def print_banner(product: str, country: str, top_n: int) -> None:
    print("\n" + "=" * 72)
    print("   AI IMPORTER DISCOVERY ENGINE")
    print("=" * 72)
    print(f"   Product:        {product}")
    print(f"   Target Country: {country}")
    print(f"   Top Results:    {top_n}")
    print(f"   Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72 + "\n")


def print_summary(results: list[dict], product: str, country: str) -> None:
    print("\n" + "=" * 72)
    print(f"  TOP {len(results)} VERIFIED IMPORTERS: {product.upper()} ({country.upper()})")
    print("=" * 72)
    for idx, r in enumerate(results, 1):
        breakdown = r["score_breakdown"]
        print(f"\n  {idx}. {r['company_name']}")
        print(f"     Website:        {r['website']}")
        print(f"     Score:          {r['relevance_score']}/100  "
              f"[PM:{breakdown['product_match']} GF:{breakdown['geo_fit']} "
              f"IS:{breakdown['import_signal']} CR:{breakdown['credibility']} SF:{breakdown['scale_fit']}]")
        print(f"     Email:          {r['contact_email']}")
        print(f"     Phone:          {r['contact_phone']}")
        print(f"     Reason:         {r['match_reason']}")
    print()


async def scrape_all(verified: list[dict]) -> list[dict]:
    """Run scrape_company_deep() on all verified candidates concurrently."""
    tasks = [scrape_company_deep(c["url"]) for c in verified]
    return await asyncio.gather(*tasks)


def run_importer_discovery_pipeline(
    product: str,
    target_country: str,
    top_n: int = 10,
) -> list[dict]:
    """
    Full end-to-end pipeline:
      1. Query Expansion
      2. DuckDuckGo Search + Pre-Filtering
      3. LLM Candidate Verification
      4. Async Deep Scraping (top 15 verified)
      5. Contact Enrichment
      6. LLM Scoring
      7. Output Formatting + JSON Export
    """
    print_banner(product, target_country, top_n)

    # -----------------------------------------------------------------------
    # STEP 1: Query Expansion
    # -----------------------------------------------------------------------
    print("[1/6] Generating search queries...")
    queries = generate_search_queries(product, target_country)
    print(f"      Generated {len(queries)} queries.\n")

    # -----------------------------------------------------------------------
    # STEP 2: DuckDuckGo Search Execution
    # -----------------------------------------------------------------------
    print("[2/6] Executing DuckDuckGo searches...")
    raw_candidates = execute_searches(queries, max_results_per_query=10)
    print(f"      Collected {len(raw_candidates)} unique candidate URLs.\n")

    # -----------------------------------------------------------------------
    # STEP 3: LLM Candidate Verification
    # -----------------------------------------------------------------------
    print("[3/6] Verifying candidates with Groq LLM...")
    verified = verify_candidates(raw_candidates)
    print(f"      {len(verified)} direct importer/wholesaler sites verified.\n")

    if not verified:
        print("[!] No verified candidates found. Exiting.")
        return []

    # Limit to top 15 to manage scrape time and API tokens
    verified = verified[:15]

    # -----------------------------------------------------------------------
    # STEP 4: Async Deep Scraping
    # -----------------------------------------------------------------------
    print(f"[4/6] Deep scraping top {len(verified)} verified companies...")
    scraped_results = asyncio.run(scrape_all(verified))
    print(f"      Scraping complete.\n")

    # -----------------------------------------------------------------------
    # STEP 5 + 6: Contact Enrichment & Scoring
    # -----------------------------------------------------------------------
    print("[5/6] Enriching contacts and scoring candidates...")
    final_results = []

    for idx, (candidate, scraped) in enumerate(zip(verified, scraped_results), 1):
        company_name = candidate.get("company_name", scraped.get("domain", "Unknown"))
        domain = candidate.get("url", scraped.get("homepage_url", ""))
        combined_text = scraped.get("combined_text", "")
        pages_scraped = scraped.get("pages_scraped", [])

        print(f"  [{idx}/{len(verified)}] {company_name}")

        # Contact Enrichment
        contacts = extract_contacts(scraped, target_country=target_country)

        # LLM Scoring
        score_data = score_candidate(
            product=product,
            country=target_country,
            company_name=company_name,
            domain=domain,
            combined_text=combined_text,
            contact_data=contacts,
        )

        print(f"         Score: {score_data['relevance_score']}/100  |  "
              f"Email: {contacts['email']}  |  Phone: {contacts['phone']}")

        final_results.append({
            "company_name": company_name,
            "website": domain,
            "relevance_score": score_data["relevance_score"],
            "score_breakdown": score_data["score_breakdown"],
            "match_reason": score_data["match_reason"],
            "contact_email": contacts["email"],
            "contact_phone": contacts["phone"],
            "contact_linkedin": contacts["linkedin"],
            "sources_used": pages_scraped,
        })

        # Polite inter-call delay to avoid exhausting per-minute token limits
        if idx < len(verified):
            time.sleep(3)

    # -----------------------------------------------------------------------
    # STEP 7: Sort, Trim, Save Output
    # -----------------------------------------------------------------------
    print(f"\n[6/6] Sorting and saving results...")
    final_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_results = final_results[:top_n]

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    safe_product = product.replace(" ", "_").lower()
    safe_country = target_country.replace(" ", "_").lower()
    output_path = os.path.join("outputs", f"{safe_product}_{safe_country}_importers.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(top_results, f, ensure_ascii=False, indent=2)

    print(f"      Results saved to: {output_path}")

    print_summary(top_results, product, target_country)
    return top_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Importer Discovery Engine — Full Pipeline"
    )
    parser.add_argument(
        "--product",
        type=str,
        required=True,
        help='Target product (e.g., "Ceramic Tiles")',
    )
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help='Target country (e.g., "Germany")',
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of top results to return (default: 10)",
    )
    args = parser.parse_args()

    run_importer_discovery_pipeline(
        product=args.product,
        target_country=args.country,
        top_n=args.top_n,
    )
