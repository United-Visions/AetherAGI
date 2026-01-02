"""
Path: mind/ingestion/processor.py
Role: Recursive text chunking and density optimization.
"""

from loguru import logger
import re

class IngestionProcessor:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_markdown(self, text: str) -> list:
        """
        Splits text into overlapping chunks to maintain semantic continuity.
        """
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            # Move the start pointer back by the 'overlap' to keep context
            start += (self.chunk_size - self.overlap)
            
        return chunks