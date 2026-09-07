# gunicorn_config.py
# Render.com production WSGI config for the Fire Detection Flask app

import multiprocessing

# Bind to the PORT env var that Render provides
bind = "0.0.0.0:10000"

# Workers: 1 worker to avoid duplicating the huge YOLO weights in RAM
workers = 1

# Threads per worker (handle concurrent requests without extra processes)
threads = 4

# Increase timeouts – model inference can be slow on CPU
timeout = 300         # 5 minutes per request
graceful_timeout = 60
keepalive = 5

# Logging
accesslog = "-"
errorlog  = "-"
loglevel  = "info"

# Prevent zombie workers
preload_app = True
