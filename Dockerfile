FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn openenv-core openai pydantic

CMD ["python", "-m", "server.app"]
