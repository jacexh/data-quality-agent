FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install uv && uv pip install --system .
COPY agent/ ./agent/
COPY models/ ./models/
CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
