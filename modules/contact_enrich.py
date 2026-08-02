import os
import sys
import re
import asyncio
from urllib.parse import urlparse

# Add the parent directory to the Python path so it can find config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phonenumbers
import pycountry
from bs4 import BeautifulSoup

# --- Email Regex (kept for email extraction; regex is still best-in-class here) ---
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
LINKEDIN_REGEX = re.compile(r'https?://(?:www\.)?linkedin\.com/company/[^\s"\'<>]+')

# Common placeholder / false-positive email fragments to exclude
EMAIL_EXCLUDES = {
    "example.com", "sentry.io", "schema.org",
    "@2x.", ".png", ".jpg", ".gif", ".svg",
}


# ---------------------------------------------------------------------------
# Helper: Country -> ISO 3166-1 alpha-2 region code
# ---------------------------------------------------------------------------

def get_iso_code(country_name: str) -> str:
    """
    Converts a country name to a 2-letter ISO 3166-1 alpha-2 region code
    using pycountry's fuzzy search.

    Examples:
        "Germany"       -> "DE"
        "United States" -> "US"
        "France"        -> "FR"

    Falls back to "US" if the country cannot be resolved.
    """
    try:
        results = pycountry.countries.search_fuzzy(country_name)
        return results[0].alpha_2
    except (LookupError, Exception):
        print(f"  [!] Could not resolve ISO code for '{country_name}', defaulting to 'US'.")
        return "US"


# ---------------------------------------------------------------------------
# Helper: Email filter
# ---------------------------------------------------------------------------

def _is_valid_email(email: str) -> bool:
    """Filter out dummy placeholder emails, image filenames, and no-reply addresses."""
    email_lower = email.lower()
    for exc in EMAIL_EXCLUDES:
        if exc in email_lower:
            return False
    if any(x in email_lower for x in ["noreply", "no-reply", "donotreply"]):
        return False
    return True


# ---------------------------------------------------------------------------
# Phone Extraction (phonenumbers library)
# ---------------------------------------------------------------------------

def extract_phone_numbers(raw_html: str, clean_text: str, target_country: str) -> str:
    """
    Extracts and validates phone numbers using the `phonenumbers` library.

    Strategy:
      1. Check HTML <a href="tel:..."> tags first (highest confidence).
      2. Fall back to scanning clean_text with PhoneNumberMatcher.

    Returns the first valid number in E.164 format (e.g., +4930123456),
    or "Not publicly available" if none is found.
    """
    region_code = get_iso_code(target_country)
    soup = BeautifulSoup(raw_html, "lxml")

    # --- Step 1: tel: anchor tags ---
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if href.lower().startswith("tel:"):
            phone_str = href[4:].strip()
            try:
                parsed = phonenumbers.parse(phone_str, region_code)
                if phonenumbers.is_valid_number(parsed):
                    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            except phonenumbers.NumberParseException:
                continue

    # --- Step 2: Fallback — scan clean text with PhoneNumberMatcher ---
    try:
        for match in phonenumbers.PhoneNumberMatcher(clean_text, region_code):
            if phonenumbers.is_valid_number(match.number):
                return phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        pass

    return "Not publicly available"


# ---------------------------------------------------------------------------
# Main Contact Extraction Function
# ---------------------------------------------------------------------------

def extract_contacts(scraped_company_data: dict, target_country: str = "Germany") -> dict:
    """
    Parses contact information from scraped HTML pages.

    Extracts:
      - Email (from mailto: links and regex)
      - Phone (from tel: links via phonenumbers library, then text scan)
      - LinkedIn company profile URL
      - Source URLs where contacts were found

    Args:
        scraped_company_data: Output dict from scrape_company_deep().
        target_country: The target country string (e.g., "Germany") used
                        to resolve the ISO region code for phone parsing.

    Returns a dict with email, phone, linkedin, confidence levels, and sources.
    """
    raw_html_map: dict[str, str] = scraped_company_data.get("raw_html_map", {})
    domain_url: str = scraped_company_data.get("domain", "")
    combined_text: str = scraped_company_data.get("combined_text", "")

    emails_found = []
    phone_result = "Not publicly available"
    linkedin_found = None
    sources = []

    for page_url, html in raw_html_map.items():
        if not html:
            continue

        soup = BeautifulSoup(html, "lxml")
        page_contributed = False
        text_blob = soup.get_text(separator=" ")

        # --- Email Extraction ---
        # 1. From mailto: anchor tags (highest confidence)
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            if href.startswith("mailto:"):
                email = href[7:].split("?")[0].strip()
                if email and _is_valid_email(email):
                    if email not in emails_found:
                        emails_found.append(email)
                        page_contributed = True

        # 2. From raw text regex scan (fallback)
        for match in EMAIL_REGEX.findall(text_blob):
            if _is_valid_email(match) and match not in emails_found:
                emails_found.append(match)
                page_contributed = True

        # --- Phone Extraction (phonenumbers library) ---
        # Try to extract phone on each page's HTML until we find one
        if phone_result == "Not publicly available":
            extracted = extract_phone_numbers(html, text_blob, target_country)
            if extracted != "Not publicly available":
                phone_result = extracted
                page_contributed = True

        # --- LinkedIn Extraction ---
        if not linkedin_found:
            # From href attributes
            for tag in soup.find_all("a", href=True):
                href = tag["href"]
                if "linkedin.com/company/" in href:
                    linkedin_found = href.split("?")[0].rstrip("/")
                    page_contributed = True
                    break

            # From raw HTML regex as fallback
            if not linkedin_found:
                li_match = LINKEDIN_REGEX.search(html)
                if li_match:
                    linkedin_found = li_match.group(0).split("?")[0].rstrip("/")
                    page_contributed = True

        if page_contributed:
            sources.append(page_url)

    # --- Compose Result ---
    # Email
    if emails_found:
        email = emails_found[0]
        email_confidence = "scraped"
    else:
        email = "Not publicly available"
        email_confidence = "unverified"

    # Phone
    phone_confidence = "scraped" if phone_result != "Not publicly available" else "unverified"

    return {
        "email": email,
        "email_confidence": email_confidence,
        "phone": phone_result,
        "phone_confidence": phone_confidence,
        "linkedin": linkedin_found or "Not listed",
        "sources": sources,
    }


if __name__ == "__main__":
    from modules.scraper import scrape_company_deep

    test_domains = [
        "https://www.fliesen-zentrum.de/",
        "https://casa-moro.de/",
    ]
    target_country = "Germany"

    print("=== Phase 2: Deep Scrape + Contact Enrichment Test ===\n")
    print(f"    ISO Region Code for '{target_country}': {get_iso_code(target_country)}\n")

    for domain_url in test_domains:
        print(f"\n{'='*70}")
        print(f"  Company Domain: {domain_url}")
        print(f"{'='*70}")

        # Deep scrape
        scraped_data = asyncio.run(scrape_company_deep(domain_url))

        print(f"  Pages Scraped:       {scraped_data['pages_scraped']}")
        print(f"  Combined Text Len:   {len(scraped_data['combined_text'])} chars")

        # Extract contacts (passing target_country for phone parsing)
        contacts = extract_contacts(scraped_data, target_country=target_country)

        print(f"\n  --- Contact Details ---")
        print(f"  Email:               {contacts['email']}  [{contacts['email_confidence']}]")
        print(f"  Phone:               {contacts['phone']}  [{contacts['phone_confidence']}]")
        print(f"  LinkedIn:            {contacts['linkedin']}")
        print(f"  Sources:             {contacts['sources']}")

        print(f"\n  --- Text Preview (first 500 chars) ---")
        preview = scraped_data["combined_text"][:500].encode("ascii", "ignore").decode("ascii")
        print(f"  {preview}...")
