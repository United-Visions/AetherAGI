"""
Path: first_contact.py
Role: The official 'First Contact' script for AetherMind Phase 1.
"""

import os
import httpx
import asyncio
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Configuration
API_URL = "https://aetheragi.onrender.com/v1/chat/completions"
# Make sure you added your AM_LIVE_KEY to the .env file
API_KEY = os.getenv("AM_LIVE_KEY") 
logging.info(f"API_KEY loaded: {API_KEY}")

async def main():
    logging.info("--- AetherMind Genesis Console ---")
    
    user_input = "Identify yourself and explain your core objective based on your internal world model."

    headers = { 
        "X-Aether-Key": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "model": "aethermind-v1",
        "user": "deion_owner",
        "messages": [{"role": "user", "content": user_input}]
    }

    logging.info(f"Sending request to AetherMind...")

    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(API_URL, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                answer = data['choices'][0]['message']['content']
                logging.info("\n--- AETHERMIND RESPONSE ---\n" + answer + "\n---------------------------\n")
            else:
                logging.error(f"Error: {response.status_code} - {response.text}")
                
        except httpx.RequestError as e:
            logging.error(f"Connection Failed: An error occurred while requesting {e.request.url!r}. - {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())