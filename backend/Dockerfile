# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install uv for fast package management inside the container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency specifications
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application source code
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]