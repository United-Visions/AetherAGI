"""
Path: mind/ingestion/k12_ingestor.py
Role: Main execution script for populating the AetherMind Knowledge Base.
"""

import os
import asyncio
from loguru import logger
from .web_crawler import AetherCrawler
from .processor import IngestionProcessor
from ..vector_store import AetherVectorStore

class K12Ingestor:
    def __init__(self):
        # Configuration from Environment Variables
        self.crawler = AetherCrawler(api_key=os.getenv("FIRECRAWL_API_KEY"))
        self.processor = IngestionProcessor(chunk_size=1200, overlap=200)
        self.store = AetherVectorStore(api_key=os.getenv("PINECONE_API_KEY"))
        self.target_namespace = "core_k12"

    async def ingest_url(self, url: str):
        """
        The full pipeline: Scrape -> Chunk -> Upsert.
        """
        # 1. Scrape
        raw_markdown = self.crawler.scrape_educational_content(url)
        if not raw_markdown:
            return

        # 2. Chunk
        knowledge_chunks = self.processor.chunk_markdown(raw_markdown)
        logger.info(f"Split content into {len(knowledge_chunks)} semantic chunks.")

        # 3. Upsert into Pinecone
        count = 0
        for chunk in knowledge_chunks:
            metadata = {
                "source_url": url,
                "data_type": "verified_curriculum",
                "phase": "linguistic_genesis"
            }
            try:
                self.store.upsert_knowledge(chunk, self.target_namespace, metadata)
                count += 1
            except Exception as e:
                logger.error(f"Failed to upsert chunk {count}: {str(e)}")

        logger.success(f"Ingestion Complete. {count} chunks added to {self.target_namespace}.")

# Run Example
if __name__ == "__main__":
    # Example target: A high-quality educational resource
    target_urls = [
        "https://openstax.org/books/physics/pages/1-introduction",
        "https://en.wikipedia.org/wiki/Laws_of_motion"
    ]
    
    ingestor = K12Ingestor()
    
    for url in target_urls:
        asyncio.run(ingestor.ingest_url(url))