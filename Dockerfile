FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY src/ src/
COPY server/ server/
COPY openenv.yaml .
COPY inference.py .

RUN uv pip install --system -e .

EXPOSE 8000

CMD ["python", "server/app.py"]
