FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN useradd --system --create-home --user-group --home-dir /home/app relay

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

COPY web_login.py .
COPY start_auth.bat .

RUN chown -R relay:relay /app /home/app
RUN chown -R relay:relay /app /home/app && chmod -R u+rwX,go+rX /app

USER relay

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["sh", "-c", "uvicorn src.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
