"""
Path: mind/vector_store.py
"""
import hashlib
from pinecone import Pinecone
from loguru import logger

class AetherVectorStore:
    def __init__(self, api_key: str, index_name: str = "aethermind-genesis"):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        # Your chosen OpenAI model
        self.model = "llama-text-embed-v2" 

    def upsert_knowledge(self, text: str, namespace: str, metadata: dict = None):
        """
        Explicitly generates a vector using OpenAI via Pinecone Inference
        and then upserts into the index.
        """
        try:
            # 1. Get the embedding from OpenAI (Truncated to 1024 as you set in Dashboard)
            res = self.pc.inference.embed(
                model=self.model,
                inputs=[text],
                parameters={"input_type": "passage", "dimension": 1024}
            )
            vector_values = res.data[0].values

            # 2. Generate a unique ID
            v_id = hashlib.md5(text.encode()).hexdigest()

            # 3. Upsert with the actual values
            metadata_dict = metadata if isinstance(metadata, dict) else {"info": str(metadata)}
            self.index.upsert(
                vectors=[{
                    "id": v_id,
                    "values": vector_values,
                    "metadata": {**metadata_dict, "text": text}
                }],
                namespace=namespace
            )
        except Exception as e:
            logger.error(f"Failed to upsert to Pinecone: {e}")

    def query_context(self, query_text: str, namespace: str, top_k: int = 5):
        """
        Queries the mind using OpenAI embeddings.
        """
        try:
            # 1. Embed the query
            res = self.pc.inference.embed(
                model=self.model,
                inputs=[query_text],
                parameters={"input_type": "query", "dimension": 1024}
            )
            query_vector = res.data[0].values

            # 2. Search
            results = self.index.query(
                namespace=namespace,
                top_k=top_k,
                vector=query_vector,
                include_metadata=True
            )
            
            contexts = [m['metadata']['text'] for m in results['matches']]
            state_vector = results['matches'][0]['values'] if results['matches'] else [0]*1024
            
            return contexts, state_vector
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return [], [0]*1024