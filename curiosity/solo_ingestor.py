"""
Path: curiosity/solo_ingestor.py
Role: Autonomous worker that crawls, reads, and embeds new knowledge.
"""
import asyncio
from loguru import logger
from bs4 import BeautifulSoup

from curiosity.research_scheduler import ResearchScheduler
from perception.mcp_client import MCPClient
from mind.vector_store import VectorStore  # Assuming a unified interface
from mind.ingestion.processor import IngestionProcessor

class SoloIngestor:
    def __init__(self, scheduler: ResearchScheduler, mcp_client: MCPClient, vector_store: VectorStore):
        """
        Initializes the SoloIngestor with necessary components.
        """
        self.scheduler = scheduler
        self.mcp = mcp_client
        self.store = vector_store
        self.processor = IngestionProcessor(chunk_size=1000, overlap=200)
        self._stop_event = asyncio.Event()

    def _html_to_markdown(self, html: str) -> str:
        """A simple converter from HTML to Markdown-like text."""
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator='\n', strip=True)

    async def research_worker(self):
        """
        The main worker loop that picks up and executes research jobs.
        """
        logger.info("Research worker started. Waiting for jobs...")
        while not self._stop_event.is_set():
            job = await self.scheduler.pop()
            if job:
                logger.info(f"Processing job: {job['query']}")
                md_chunks = []
                
                for tool in job.get("tools", []):
                    if tool == "browser":
                        # For simplicity, we'll assume the query is a search term for now
                        # A real implementation would generate a URL or use a search API
                        raw_html = await self.mcp.call("browse", {"url": f"https://www.google.com/search?q={job['query']}"})
                        if not raw_html.startswith("Error:"):
                             md_chunks.append(self._html_to_markdown(raw_html))
                    # Add other tools like "arxiv", "youtube" here
                
                if md_chunks:
                    full_text = "\n".join(md_chunks)
                    chunks = self.processor.chunk_text(full_text) # Use a generic text chunker
                    
                    logger.info(f"Upserting {len(chunks)} new chunks of knowledge...")
                    # This is a placeholder for the actual upsert logic
                    # self.store.upsert_knowledge(chunks, namespace="autonomous_research", metadata={"source": "curiosity", "query": job["query"]})
                    logger.success(f"Autonomous research for '{job['query']}' completed.")
                
            else:
                # Sleep for a bit if no job is found
                await asyncio.sleep(5)

    async def start(self):
        """Starts the worker as an asyncio task."""
        self._stop_event.clear()
        asyncio.create_task(self.research_worker())

    async def stop(self):
        """Stops the worker gracefully."""
        self._stop_event.set()
        logger.info("Research worker stopping...")
        await self.mcp.close()

