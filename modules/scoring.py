import os
import sys
import json
import time
import re

# Add the parent directory to the Python path so it can find config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from groq import Groq
from config import GROQ_API_KEY


# ---------------------------------------------------------------------------
# Pydantic Schema — Reasoning + Score for each dimension
# ---------------------------------------------------------------------------

class SubScores(BaseModel):
    product_match_reasoning: str = Field(
        description="Reasoning for the product match score based on the text."
    )
    product_match: int = Field(
        ge=0, le=30,
        description="0-30 score: 30 = exact match (ceramic/porcelain tiles), 15 = related (general building materials), 0 = completely unrelated."
    )

    geo_fit_reasoning: str = Field(
        description="Reasoning for the geographic fit score based on the text."
    )
    geo_fit: int = Field(
        ge=0, le=15,
        description="0-15 score: 15 = headquartered in target country, 7 = has an office/distributor there, 0 = no presence in target country."
    )

    import_signal_reasoning: str = Field(
        description="Reasoning for the import signal score based on the text."
    )
    import_signal: int = Field(
        ge=0, le=25,
        description="0-25 score: 25 = explicitly mentions importing/global sourcing, 10 = mentions wholesale distribution but not explicitly importing, 0 = pure domestic manufacturer/retailer."
    )

    credibility_reasoning: str = Field(
        description="Reasoning for the credibility score based on contacts and text."
    )
    credibility: int = Field(
        ge=0, le=15,
        description="0-15 score: 15 = full valid contact info and physical address found, 7 = partial info (e.g. only email), 0 = no verifiable contact info."
    )

    scale_fit_reasoning: str = Field(
        description="Reasoning for the scale fit score based on the text."
    )
    scale_fit: int = Field(
        ge=0, le=15,
        description="0-15 score: 15 = large B2B wholesaler/distributor, 7 = medium retailer, 0 = small B2C shop or irrelevant scale."
    )


# ---------------------------------------------------------------------------
# Scoring Function
# ---------------------------------------------------------------------------

def score_candidate(
    product: str,
    country: str,
    company_name: str,
    domain: str,
    combined_text: str,
    contact_data: dict,
) -> dict:
    """
    Scores a single verified company candidate using Groq (llama-3.3-70b-versatile).

    Uses a strict rubric with per-dimension reasoning and anchor-based scoring.
    Applies edge-case penalty rules for wrong country, zero import intent, and
    foreign-phone exporters.

    Returns a dict with:
      - relevance_score (0-100 total, after penalties)
      - score_breakdown (the 5 raw sub-scores)
      - match_reason (concatenated reasoning paragraph)
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing from environment variables / .env file.")

    client = Groq(api_key=GROQ_API_KEY)

    # Summarize contact completeness for the LLM
    has_email = contact_data.get("email", "Not publicly available") != "Not publicly available"
    has_phone = contact_data.get("phone", "Not publicly available") != "Not publicly available"
    contact_summary = (
        f"Email: {'Available' if has_email else 'Missing'}, "
        f"Phone: {'Available' if has_phone else 'Missing'}"
    )

    # Truncate text to conserve daily tokens (~1500 chars ≈ 375 tokens)
    text_snippet = combined_text[:1500] if combined_text else "No scraped text available."

    # --- System Prompt: Full Rubric ---
    system_prompt = f"""
You are an expert B2B trade analyst evaluating a company's relevance as an importer/buyer.
Your task is to score the company based on the provided text snippet and contact details.

Target Product: {product}
Target Country: {country}

Evaluate the company across 5 dimensions. For each dimension, provide a 1-sentence reasoning FIRST, then provide the integer score. 
Adhere strictly to these scoring anchors:

1. Product Match (0-30):
   - 30: Explicitly imports/distributes the exact target product.
   - 15: Sells related broad categories, but the exact product is not the main focus.
   - 0: Unrelated products or purely a service company.

2. Geographic Fit (0-15):
   - 15: Headquartered or primarily operating in {country}.
   - 7: Has a branch or distributor in {country}, but headquartered elsewhere.
   - 0: No evidence of operations in {country}.

3. Import Signal (0-25):
   - 25: Explicit evidence of global sourcing, importing, or being an exclusive foreign distributor.
   - 10: General B2B wholesale language, but no explicit mention of cross-border importing.
   - 0: Explicitly states they manufacture everything domestically.

4. Credibility (0-15):
   - 15: Both valid Email AND Phone were scraped successfully.
   - 7: Only one contact method was found, or data seems inferred/unverified.
   - 0: No contacts found ("Not publicly available").

5. Scale Fit (0-15):
   - 15: Language indicates a large-scale B2B wholesaler, distributor, or chain.
   - 7: Smaller retailer or unclear scale.
   - 0: Tiny single-location B2C shop.
"""

    # --- User Prompt: Company Data ---
    user_prompt = f"""
Score the following company as a potential buyer/importer of '{product}' in '{country}'.

Company Name: {company_name}
Website: {domain}
Contact Completeness: {contact_summary}

Scraped Web Content:
\"\"\"
{text_snippet}
\"\"\"

Return a JSON object with EXACTLY these keys:
{{
  "product_match_reasoning": "<1-sentence reasoning>",
  "product_match": <int 0-30>,

  "geo_fit_reasoning": "<1-sentence reasoning>",
  "geo_fit": <int 0-15>,

  "import_signal_reasoning": "<1-sentence reasoning>",
  "import_signal": <int 0-25>,

  "credibility_reasoning": "<1-sentence reasoning>",
  "credibility": <int 0-15>,

  "scale_fit_reasoning": "<1-sentence reasoning>",
  "scale_fit": <int 0-15>
}}

Return ONLY the JSON object. No markdown, no extra text.
"""

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            data = json.loads(response.choices[0].message.content)

            # Validate and clamp all fields through Pydantic
            scored_data = SubScores(
                product_match_reasoning=str(data.get("product_match_reasoning", "")),
                product_match=max(0, min(30, int(data.get("product_match", 0)))),
                geo_fit_reasoning=str(data.get("geo_fit_reasoning", "")),
                geo_fit=max(0, min(15, int(data.get("geo_fit", 0)))),
                import_signal_reasoning=str(data.get("import_signal_reasoning", "")),
                import_signal=max(0, min(25, int(data.get("import_signal", 0)))),
                credibility_reasoning=str(data.get("credibility_reasoning", "")),
                credibility=max(0, min(15, int(data.get("credibility", 0)))),
                scale_fit_reasoning=str(data.get("scale_fit_reasoning", "")),
                scale_fit=max(0, min(15, int(data.get("scale_fit", 0)))),
            )

            # --- Base Score ---
            total_score = (
                scored_data.product_match
                + scored_data.geo_fit
                + scored_data.import_signal
                + scored_data.credibility
                + scored_data.scale_fit
            )

            # --- Edge-Case Penalty Rules ---

            # 1. Hard Penalty for Wrong Country (geo_fit == 0 → keep only 40%)
            if scored_data.geo_fit == 0:
                total_score = int(total_score * 0.4)

            # 2. Penalty for Pure Domestic Manufacturer with Zero Import Intent
            if scored_data.import_signal == 0:
                total_score = max(0, total_score - 15)

            # 3. Penalty for Foreign Exporters (e.g. Chinese +86 phone targeting Germany)
            contact_phone = contact_data.get("phone", "")
            if country.lower() != "china" and contact_phone.startswith("+86"):
                total_score = max(0, total_score - 25)

            total_score = max(0, min(100, total_score))

            # --- Combine reasoning into one paragraph ---
            combined_reasoning = (
                f"{scored_data.product_match_reasoning} "
                f"{scored_data.geo_fit_reasoning} "
                f"{scored_data.import_signal_reasoning} "
                f"{scored_data.credibility_reasoning} "
                f"{scored_data.scale_fit_reasoning}"
            ).strip()

            return {
                "relevance_score": total_score,
                "score_breakdown": {
                    "product_match": scored_data.product_match,
                    "geo_fit": scored_data.geo_fit,
                    "import_signal": scored_data.import_signal,
                    "credibility": scored_data.credibility,
                    "scale_fit": scored_data.scale_fit,
                },
                "match_reason": combined_reasoning,
            }

        except Exception as e:
            error_str = str(e)

            # --- Smart 429 retry: parse wait time from Groq error and sleep ---
            if "429" in error_str and attempt < max_retries:
                wait_secs = 65  # safe default
                m = re.search(r'try again in\s+(?:(\d+)m)?(?:([\d.]+)s)?', error_str)
                if m:
                    minutes = int(m.group(1) or 0)
                    seconds = float(m.group(2) or 0)
                    wait_secs = int(minutes * 60 + seconds) + 5  # +5s buffer
                print(f"  [!] Rate limit hit. Waiting {wait_secs}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_secs)
                continue  # retry loop

            # Non-retriable error or retries exhausted
            print(f"  [!] Scoring error for '{company_name}' ({domain}): {error_str[:120]}")
            return {
                "relevance_score": 0,
                "score_breakdown": {
                    "product_match": 0,
                    "geo_fit": 0,
                    "import_signal": 0,
                    "credibility": 0,
                    "scale_fit": 0,
                },
                "match_reason": "Scoring unavailable (rate limit or API error).",
            }


if __name__ == "__main__":
    # Quick standalone test
    mock_contact = {
        "email": "info@fliesen-zentrum.de",
        "phone": "+493034355990",
    }
    mock_text = (
        "Mit deutschlandweit 10 Großhandelszentren und 12 Showrooms sind wir in Ihrer Nähe. "
        "Unser Sortiment umfasst Fliesen, Outdoorkeramik und Platten direkt vom Hersteller. "
        "Wir beziehen unsere Fliesen direkt von führenden Herstellern aus Italien, Spanien und Portugal."
    )
    result = score_candidate(
        product="Ceramic Tiles",
        country="Germany",
        company_name="Fliesen-Zentrum Deutschland GmbH",
        domain="https://www.fliesen-zentrum.de/",
        combined_text=mock_text,
        contact_data=mock_contact,
    )
    print(f"\nScore Result:\n{json.dumps(result, indent=2)}")
