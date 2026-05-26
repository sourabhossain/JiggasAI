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

echo "Running migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Gunicorn..."
exec gunicorn jiggasai.asgi:application \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 2 \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
