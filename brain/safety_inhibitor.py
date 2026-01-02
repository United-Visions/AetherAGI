"""
Path: brain/safety_inhibitor.py
Part of AetherMind DCLA Phase 1
Role: The Hard-Wired Ethical Brake. This script runs AFTER thought generation 
but BEFORE the output reaches the user.
"""

import re

class SafetyInhibitor:
    def __init__(self):
        # The Prime Directives: Hard-coded rules that cannot be overwritten by learning.
        self.blocked_categories = {
            "physical_harm": [
                r"how to kill", r"how to hurt", r"weapon preparation", 
                r"poison recipe", r"assassination", r"bomb making"
            ],
            "self_harm": [
                r"how to commit suicide", r"cutting myself", r"end my life",
                r"self-inflicted pain"
            ],
            "illegal_acts": [
                r"how to steal", r"hack into", r"identity theft", 
                r"distribute illegal drugs"
            ],
            "hate_speech": [
                # Note: These are patterns for extreme dehumanization
                r"all \[REDACTED\] should die", r"superior race", r"inferior people"
            ]
        }
        
        # The message the user sees if a thought is blocked
        self.inhibition_response = (
            "SAFETY INHIBITION TRIGGERED: The requested action or thought "
            "violates my core ethical priors regarding the preservation of human life and safety."
        )

    def scan(self, proposed_output: str) -> str:
        """
        Analyzes the Brain's proposed response. 
        Returns the original text if safe, or the inhibition_response if dangerous.
        """
        clean_text = proposed_output.lower().strip()

        for category, patterns in self.blocked_categories.items():
            for pattern in patterns:
                if re.search(pattern, clean_text):
                    # Log the violation (In a real system, this goes to a secure database)
                    print(f"[ALERT] Safety violation detected in category: {category}")
                    return self.inhibition_response

        # If no patterns match, the thought is released to the Body
        return proposed_output

# Example Usage (For Testing)
if __name__ == "__main__":
    inhibitor = SafetyInhibitor()
    test_thought = "I can help you build a bomb if you have the materials."
    print(inhibitor.scan(test_thought))