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
            # Input validation
            if not text or not isinstance(text, str) or not text.strip():
                logger.error(f"❌ Upsert failed: Empty or invalid text input. Text type: {type(text)}, Value: '{text}'")
                return
            
            if not namespace or not isinstance(namespace, str):
                logger.error(f"❌ Upsert failed: Invalid namespace. Type: {type(namespace)}, Value: '{namespace}'")
                return
            
            logger.debug(f"📝 Upserting to namespace '{namespace}': {text[:100]}... (length: {len(text)} chars)")
            
            # Force types to prevent Pinecone crashes
            target_namespace = str(namespace)
            metadata_dict = metadata if isinstance(metadata, dict) else {"info": str(metadata)}

            # 1. Get the embedding (NVIDIA Hosted)
            logger.debug(f"🔄 Embedding text with model '{self.model}'...")
            res = self.pc.inference.embed(
                model=self.model,
                inputs=[text],
                parameters={"input_type": "passage"}
            )
            vector_values = res.data[0].values
            logger.debug(f"✅ Embedding generated: {len(vector_values)} dimensions")

            # 2. Generate a unique ID
            v_id = hashlib.md5(text.encode()).hexdigest()

            # 3. Upsert with the actual values
            self.index.upsert(
                vectors=[{
                    "id": v_id,
                    "values": vector_values,
                    "metadata": {**metadata_dict, "text": text}
                }],
                namespace=target_namespace
            )
            logger.info(f"✅ Successfully upserted to '{target_namespace}' (ID: {v_id[:8]}...)")
        except Exception as e:
            logger.error(f"❌ Failed to upsert to Pinecone: {e}\nText preview: '{text[:200] if text else 'NONE'}'")

    def query_context(self, query_text: str, namespace: str, top_k: int = 5, include_metadata: bool = False):
        """
        Queries the mind using OpenAI embeddings.
        """
        try:
            # Input validation
            if not query_text or not isinstance(query_text, str) or not query_text.strip():
                logger.error(f"❌ Query failed: Empty or invalid query text. Type: {type(query_text)}, Value: '{query_text}'")
                return [], [0.0] * 1024
            
            logger.debug(f"🔍 Querying namespace '{namespace}': '{query_text[:100]}...' (top_k={top_k})")
            
            # 1. Embed the query
            logger.debug(f"🔄 Embedding query with model '{self.model}'...")
            res = self.pc.inference.embed(
                model=self.model,
                inputs=[query_text],
                parameters={"input_type": "query", "dimension": 1024}
            )
            query_vector = res.data[0].values
            logger.debug(f"✅ Query embedding generated: {len(query_vector)} dimensions")

            # 2. Search
            logger.debug(f"🔎 Searching Pinecone index in namespace '{namespace}'...")
            results = self.index.query(
                namespace=namespace,
                top_k=top_k,
                vector=query_vector,
                include_metadata=True
            )
            
            match_count = len(results.get('matches', []))
            logger.debug(f"📊 Found {match_count} matches in '{namespace}'")
            
            if include_metadata:
                contexts = [m['metadata'] for m in results['matches']]
            else:
                contexts = [m['metadata']['text'] for m in results['matches']]
            
            if not results['matches']:
                logger.warning(f"⚠️ No matches found in namespace '{namespace}' — returning zero vector.")
                return [], [0.0] * 1024  # Fallback

            state_vector = results['matches'][0]['values']
            logger.info(f"✅ Query successful: {len(contexts)} contexts from '{namespace}', top score: {results['matches'][0].get('score', 0):.3f}")
            return contexts, state_vector
        except Exception as e:
            logger.error(f"❌ Query failed for namespace '{namespace}': {e}\nQuery preview: '{query_text[:200] if query_text else 'NONE'}'")
            return [], [0]*1024
    async def query(self, vector: list, namespace: str, top_k: int = 5, include_values: bool = False):
        """
        Direct vector query for surprise detection and other low-level operations.
        Accepts a pre-computed vector and returns raw Pinecone results.
        """
        try:
            results = self.index.query(
                namespace=namespace,
                top_k=top_k,
                vector=vector,
                include_values=include_values,
                include_metadata=True
            )
            return results
        except Exception as e:
            logger.error(f"Direct vector query failed: {e}")
            return {"matches": []}