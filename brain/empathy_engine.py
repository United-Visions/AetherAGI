"""
Path: brain/empathy_engine.py
Role: The "Heart" of AetherMind. Provides emotional context and affective analysis.
"""
from loguru import logger

class EmpathyEngine:
    def __init__(self):
        """
        Initializes the Empathy Engine. In a future version, this could load
        a sophisticated sentiment analysis model. For now, it uses a keyword-based approach.
        """
        self.sentiment_keywords = {
            "positive": ["great", "awesome", "thank you", "love", "happy", "excellent", "wow"],
            "negative": ["hate", "terrible", "bad", "error", "frustrated", "problem", "help", "fix"],
            "curious": ["why", "how", "what", "explain", "who", "investigate"],
        }
        logger.info("Empathy Engine (Heart) initialized.")

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes the input text to determine the user's likely emotional state.
        Returns a dictionary representing the emotional context.
        """
        text_lower = text.lower()
        sentiment = "neutral" # Default sentiment
        urgency = "low"

        for sent, keywords in self.sentiment_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                sentiment = sent
                break
        
        if sentiment == "negative":
            urgency = "high"
        
        emotional_context = {"sentiment": sentiment, "urgency": urgency}
        logger.info(f"Emotional context analyzed: {emotional_context}")
        
        return emotional_context
