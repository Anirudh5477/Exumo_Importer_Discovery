import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Rate Limiting & Delays (to prevent getting blocked)
DDG_DELAY_SECONDS = 1.5      # Pause between DuckDuckGo search requests
SCRAPER_TIMEOUT = 10.0        # Seconds before giving up on a slow website
MAX_CONCURRENT_SCRAPES = 5   # Max parallel web requests

# Domain Blacklist (Ignore general portals/directories that aren't specific buyers)
DOMAIN_BLACKLIST = {
    "amazon.com", "alibaba.com", "indiamart.com", "ebay.com", 
    "youtube.com", "facebook.com", "instagram.com", "twitter.com", 
    "linkedin.com", "wikipedia.org", "yellowpages.com",
    "trademo.com", "go4worldbusiness.com", "exportbusinessmart.com"
}