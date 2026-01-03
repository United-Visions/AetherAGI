# perception_service/runpod_handler.py

import runpod
from uvicorn import run
from threading import Thread
import os

# --- RunPod Handler ---
# This is the entry point that RunPod calls for each serverless request.
def run_uvicorn():
    """Function to run Uvicorn server in a separate thread."""
    run(app="main:app", host="0.0.0.0", port=8000)

# Start the Uvicorn server in a background thread when the worker initializes.
# This ensures the API is ready to accept requests.
thread = Thread(target=run_uvicorn)
thread.start()

def handler(event):
    """
    This is the main handler function that RunPod will execute.
    It's a simple pass-through; the actual logic is in our FastAPI app.
    RunPod's serverless framework routes the HTTP request to our Uvicorn server.
    """
    # This handler doesn't need to do anything itself because the Uvicorn
    # server is already running and handling the request that triggered this handler.
    # The return value is not sent to the client, but can be useful for logging.
    return "Handled by FastAPI"

# --- Start the server for local testing (optional) ---
if __name__ == "__main__":
    # This block allows you to run the FastAPI app directly for testing.
    # It will not be used by RunPod in production.
    print("Starting FastAPI server for local testing...")
    run(app="main:app", host="0.0.0.0", port=8001)
