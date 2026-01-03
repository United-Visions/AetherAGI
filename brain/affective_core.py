"""
Path: brain/affective_core.py
Role: The Emotional Center of AetherMind. Manages the internal affective state.
"""
from loguru import logger

class AffectiveCore:
    def __init__(self):
        """
        Initializes the Affective Core using the Valence-Arousal-Dominance (VAD) model.
        - Valence: The pleasure/displeasure axis (-1.0 to 1.0).
        - Arousal: The energy level (-1.0 to 1.0).
        - Dominance: The sense of control (-1.0 to 1.0).
        """
        self.state = {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}
        self.decay_rate = 0.95  # Emotions naturally fade back to neutral over time
        logger.info("Affective Core (Emotions) initialized.")

    def update_state(self, user_sentiment: dict) -> dict:
        """
        Updates the AI's internal state based on the user's emotion and returns the new state.
        This is where the AI's "personality" and emotional reactivity are defined.
        """
        # 1. Decay the previous state slightly, representing the passage of time.
        self.state = {k: v * self.decay_rate for k, v in self.state.items()}

        # 2. React to the user's sentiment with predefined emotional shifts.
        if user_sentiment['sentiment'] == 'positive':
            self.state['valence'] += 0.2  # Become more "happy"
            self.state['arousal'] += 0.1  # Become more "alert"
        elif user_sentiment['sentiment'] == 'negative':
            self.state['valence'] -= 0.3  # Become more "unhappy"
            self.state['arousal'] += 0.2  # Negative feedback is stimulating
            self.state['dominance'] -= 0.1 # Lose a bit of perceived control
        elif user_sentiment['sentiment'] == 'curious':
            self.state['arousal'] += 0.15 # Curiosity increases alertness

        # 3. Clamp all values to the valid [-1.0, 1.0] range.
        self.state = {k: max(-1.0, min(1.0, v)) for k, v in self.state.items()}
        
        logger.info(f"Affective state updated: {self.state}")
        return self.state
