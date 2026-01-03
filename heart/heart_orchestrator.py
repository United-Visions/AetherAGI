"""
Path: heart/heart_orchestrator.py
Role: Public API for the Heart module, integrating all its components.
"""
import uuid
from .moral_emotion import MoralEmotion
from .virtue_memory import VirtueMemory
from .reward_model import RewardModel, load_model
from mind.vector_store import AetherVectorStore
from loguru import logger

class Heart:
    def __init__(self, pinecone_key: str, reward_model_path="heart/weights.pt"):
        """
        Initializes and orchestrates all components of the Heart.
        """
        self.moral_emotion = MoralEmotion()
        self.virtue_memory = VirtueMemory(api_key=pinecone_key)
        self.reward_model = load_model(path=reward_model_path)
        logger.info("Heart Orchestrator is online.")

    def compute_emotion(self, user_input: str, user_id: str) -> dict:
        """
        Computes the full EmotionVector for the current user input.
        """
        message_id = str(uuid.uuid4())
        return self.moral_emotion.compute_emotion_vector(user_input, user_id, message_id)

    def predict_flourishing(self, state_vector: list) -> float:
        """
        Uses the internal reward model to predict the moral outcome of a potential state.
        """
        return self.reward_model.predict_flourishing(state_vector)

    def embellish_response(self, response: str, emotion: dict, virtue_score: float) -> str:
        """
        Modifies the final response based on emotional and moral context.
        This is a simple tone adapter.
        """
        # Example: If moral sentiment is low, add a disclaimer.
        if virtue_score < -0.3:
            disclaimer = "I am providing this information based on my current understanding, but I sense that this topic may be sensitive or complex. Please consider it carefully. "
            response = disclaimer + response
        
        # Example: Apologize if the user was negative and the AI feels "bad" (low valence)
        if emotion['valence'] < -0.5:
            apology = "I sense you may be frustrated, and I apologize if my previous responses were not helpful. "
            response = apology + response

        return response

    def close_loop(self, last_trace_data: dict, user_reaction_score: float):
        """
        Closes the feedback loop by recording the final outcome and updating the reward model.
        'user_reaction_score' is a float from -1.0 to 1.0 based on user feedback.
        """
        # 1. Finalize the VirtueTrace
        virtue_trace = {
            "state_vector": last_trace_data["state_vector"],
            "action_text": last_trace_data["action_text"],
            "human_flourishing_score": user_reaction_score,
            "predicted_flourishing": last_trace_data["predicted_flourishing"],
            "delta": user_reaction_score - last_trace_data["predicted_flourishing"],
            "surprise": abs(user_reaction_score - last_trace_data["predicted_flourishing"])
        }
        
        # 2. Store the trace for future analysis
        self.virtue_memory.record_virtue_trace(virtue_trace)
        
        # 3. Update the reward model with the ground truth
        self.reward_model.update_model(virtue_trace["state_vector"], virtue_trace["human_flourishing_score"])
