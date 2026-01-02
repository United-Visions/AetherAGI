"""
Path: mind/ingestion/web_crawler.py
Role: Production extraction of educational data using FireCrawl.
"""

from firecrawl import FirecrawlApp
from loguru import logger

class AetherCrawler:
    def __init__(self, api_key: str):
        self.app = FirecrawlApp(api_key=api_key)

    def scrape_educational_content(self, url: str) -> str:
        """
        Crawls a URL and returns sanitized Markdown content.
        """
        logger.info(f"Initiating FireCrawl for URL: {url}")
        
        try:
            # scrape_url returns a dictionary with the markdown content
            result = self.app.scrape_url(url, params={
                'formats': ['markdown'],
                'onlyMainContent': True # Removes headers/footers/ads
            })
            
            markdown = result.get('markdown', '')
            if not markdown:
                logger.warning(f"No content extracted from {url}")
                return ""
                
            logger.success(f"Successfully extracted {len(markdown)} characters of Markdown.")
            return markdown
            
        except Exception as e:
            logger.error(f"FireCrawl failed for {url}: {str(e)}")
            return ""