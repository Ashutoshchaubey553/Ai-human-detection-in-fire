# gunicorn_config.py
# Render production WSGI config for Fire Detection Flask app

import os

# Render provides the port via the PORT environment variable
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# 1 worker to keep RAM usage low with YOLO weights + TensorFlow
workers = 1

# Threads to handle multiple concurrent requests without extra process memory
threads = 4

# Increase timeouts – video processing and model inference can take time on CPU
timeout = 300
graceful_timeout = 60
keepalive = 5

# Logging to stdout/stderr so Render logs capture everything
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Ensure models are cleanly initialized
preload_app = False
