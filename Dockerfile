FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV ENABLE_WEB_INTERFACE=true

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY src/ src/
COPY server/ server/
COPY openenv.yaml .
COPY inference.py .
COPY README.md .

RUN uv pip install --system -e .

EXPOSE 7860

CMD ["python", "-c", "import uvicorn; from server.app import app; uvicorn.run(app, host='0.0.0.0', port=7860)"]
