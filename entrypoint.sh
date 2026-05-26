#!/bin/bash
set -e

echo "Waiting for MySQL..."
ATTEMPTS=0
until python -c "
import MySQLdb, os
MySQLdb.connect(
    host=os.environ['MYSQL_HOST'],
    user=os.environ['MYSQL_USER'],
    passwd=os.environ['MYSQL_PASSWORD'],
    db=os.environ['MYSQL_DATABASE'],
)
" 2>/dev/null; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ "$ATTEMPTS" -gt 30 ]; then
        echo "MySQL did not come up in 60 seconds, giving up."
        exit 1
    fi
    echo "MySQL not ready, waiting 2 seconds... (attempt $ATTEMPTS/30)"
    sleep 2
done
echo "MySQL ready."

echo "Waiting for Ollama at ${OLLAMA_HOST}:${OLLAMA_PORT}..."
ATTEMPTS=0
until python -c "
import urllib.request, sys, os
try:
    urllib.request.urlopen(
        f\"http://{os.environ['OLLAMA_HOST']}:{os.environ['OLLAMA_PORT']}/api/tags\",
        timeout=3,
    ).read()
except Exception:
    sys.exit(1)
" 2>/dev/null; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ "$ATTEMPTS" -gt 20 ]; then
        echo "Ollama not reachable after 60 seconds. Continuing anyway — startup will fail at first inference."
        break
    fi
    echo "Ollama not ready, waiting 3 seconds... ($ATTEMPTS/20)"
    sleep 3
done
echo "Ollama check complete."

echo "Running migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Gunicorn..."
exec gunicorn jiggasai.asgi:application \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 2 \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
