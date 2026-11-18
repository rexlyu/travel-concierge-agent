FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PYTHONPATH=/app

WORKDIR /app

# Create non-root runtime user for least-privilege container execution
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --shell /bin/bash --create-home appuser

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY travel_concierge ./travel_concierge
COPY evals ./evals

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8080

# Launch the ADK FastAPI web/API server bound to Cloud Run's injected $PORT
CMD ["sh", "-c", "adk api_server --host 0.0.0.0 --port ${PORT:-8080} ."]
