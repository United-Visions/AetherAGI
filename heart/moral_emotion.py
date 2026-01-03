"""
Path: heart/moral_emotion.py
Role: Emotion vectoriser & valence regressor for the Heart.
"""
from loguru import logger

class MoralEmotion:
    def __init__(self):
        """
        Initializes the Moral Emotion engine. This is an evolution of the EmpathyEngine,
        now responsible for creating a more detailed EmotionVector.
        """
        self.sentiment_keywords = {
            "positive": ["great", "awesome", "thank you", "love", "happy", "excellent", "wow"],
            "negative": ["hate", "terrible", "bad", "error", "frustrated", "problem", "help", "fix"],
            "curious": ["why", "how", "what", "explain", "who", "investigate"],
        }
        logger.info("MoralEmotion Engine initialized.")

    def compute_emotion_vector(self, text: str, user_id: str, message_id: str) -> dict:
        """
        Analyzes input text to generate a comprehensive EmotionVector.
        """
        text_lower = text.lower()
        valence = 0.0
        arousal = 0.0
        dominance = 0.0 # Placeholder for now

        # Simple keyword-based sentiment analysis
        if any(k in text_lower for k in self.sentiment_keywords["positive"]):
            valence = 0.6
            arousal = 0.3
        elif any(k in text_lower for k in self.sentiment_keywords["negative"]):
            valence = -0.7
            arousal = 0.6
        elif any(k in text_lower for k in self.sentiment_keywords["curious"]):
            valence = 0.1
            arousal = 0.4

        # In a real implementation, 'moral_sentiment' would come from the reward model.
        # For now, we'll derive it from valence as a placeholder.
        moral_sentiment = valence * 0.5

        emotion_vector = {
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "dominance": round(dominance, 4),
            "moral_sentiment": round(moral_sentiment, 4),
            "timestamp": logger.now().isoformat(),
            "user_id": user_id,
            "message_id": message_id
        }
        
        logger.info(f"Generated EmotionVector: {emotion_vector}")
        return emotion_vector
