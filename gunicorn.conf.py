import os
import multiprocessing

# Render sets PORT variable
port = os.getenv("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Worker configuration
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
