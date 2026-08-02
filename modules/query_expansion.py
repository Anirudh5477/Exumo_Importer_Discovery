import os
import json
from pydantic import BaseModel, Field
from groq import Groq
from config import GROQ_API_KEY

class SearchQueries(BaseModel):
    queries: list[str] = Field(
        description="List of 8 to 15 targeted search queries including local language, directory specific, and B2B keywords."
    )

def generate_search_queries(product: str, target_country: str) -> list[str]:
    """
    Generates 8-15 high-leverage search queries for finding importers/distributors.
    Includes local-language terms, B2B wholesale keywords, and directory searches.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing from environment variables / .env file.")

    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = (
        "You are an international trade and B2B sourcing expert. "
        "Your goal is to generate search engine queries to discover genuine importers, "
        "distributors, and wholesale buyers in a specific country for a given product."
    )

    user_prompt = f"""
    Target Product: {product}
    Target Country: {target_country}

    Generate 10 to 12 highly targeted search queries to find actual COMPANY websites (not B2B directory portals or blogs) that import and distribute this product.

    Guidelines:
    1. Use local terminology for wholesale and distribution (e.g., for Germany, use terms like "Großhandel", "Fachhandel", "Direktimport").
    2. Include search terms real corporate websites use on their "About Us" or "Wholesale" pages.
    3. Include queries with negative keywords to reduce directory hits (e.g., '{product} wholesale distributor {target_country} -directory -yellowpages').
    4.Also include some english only queries too
    
    Return ONLY a JSON object with a single key "queries" containing the list of string queries.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        queries = data.get("queries", [])
        
        # Fallback if list is empty
        if not queries:
            queries = [f"{product} importer {target_country}", f"{product} distributor {target_country}"]

        return queries

    except Exception as e:
        print(f"[!] Error generating queries with LLM: {e}")
        # Default safety fallback
        return [
            f"{product} importer {target_country}",
            f"{product} wholesale distributor {target_country}",
            f"buy {product} bulk {target_country}"
        ]

if __name__ == "__main__":
    # Test run standalone
    test_product = "Ceramic Tiles"
    test_country = "Germany"
    print(f"Testing Query Expansion for {test_product} -> {test_country}...\n")
    generated = generate_search_queries(test_product, test_country)
    for idx, q in enumerate(generated, 1):
        print(f"{idx}. {q}")