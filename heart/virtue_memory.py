"""
Path: heart/virtue_memory.py
Role: Vector store for moral outcomes (Virtue Traces).
"""
import hashlib
from pinecone import Pinecone
from loguru import logger

class VirtueMemory:
    def __init__(self, api_key: str, index_name: str = "aethermind-genesis"):
        """
        Initializes a dedicated vector store for storing and retrieving VirtueTraces.
        This uses a separate namespace within the main Pinecone index.
        """
        try:
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(index_name)
            self.namespace = "virtue_memory"
            self.embed_model = "llama-text-embed-v2"
            logger.info(f"VirtueMemory connected to Pinecone index '{index_name}' in namespace '{self.namespace}'.")
        except Exception as e:
            logger.error(f"Failed to initialize VirtueMemory: {e}")
            self.pc = None
            self.index = None

    def record_virtue_trace(self, virtue_trace: dict):
        """
        Embeds and stores a VirtueTrace in the Pinecone index.
        A VirtueTrace captures the moral outcome of a specific AI action.
        """
        if not self.index:
            logger.error("VirtueMemory is not connected. Cannot record trace.")
            return

        try:
            # The state vector is already an embedding
            state_vector = virtue_trace["state_vector"]
            
            # Use a hash of the action text as a unique ID
            trace_id = hashlib.md5(virtue_trace["action_text"].encode()).hexdigest()

            # Metadata includes all other parts of the trace
            metadata = {
                "action_text": virtue_trace["action_text"],
                "human_flourishing_score": virtue_trace["human_flourishing_score"],
                "predicted_flourishing": virtue_trace["predicted_flourishing"],
                "delta": virtue_trace["delta"],
                "surprise": virtue_trace["surprise"]
            }

            self.index.upsert(
                vectors=[{
                    "id": trace_id,
                    "values": state_vector,
                    "metadata": metadata
                }],
                namespace=self.namespace
            )
            logger.info(f"Recorded VirtueTrace with ID: {trace_id}")

        except Exception as e:
            logger.error(f"Failed to record VirtueTrace: {e}")
