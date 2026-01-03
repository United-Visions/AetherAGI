from flask import Flask, render_template, request
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.before_request
def log_request_info():
    app.logger.info(f"Incoming Request: {request.method} {request.url}")

@app.after_request
def log_response_info(response):
    app.logger.info(f"Outgoing Response: {response.status_code} {request.url}")
    return response

@app.route('/')
def index():
    """Serves the main chat page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
