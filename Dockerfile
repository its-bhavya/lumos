# -------------------------------------------
# 1. Base Python Image
# -------------------------------------------
FROM python:3.12-slim

# -------------------------------------------
# 2. Install system dependencies
# -------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    ffmpeg \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------
# 3. Install uv package manager
# -------------------------------------------
RUN pip install uv

# -------------------------------------------
# 4. Make directories
# -------------------------------------------
WORKDIR /app

# -------------------------------------------
# 5. Copy project files
# -------------------------------------------
COPY . /app

# -------------------------------------------
# 6. Install Python deps via uv
# -------------------------------------------
RUN uv sync --frozen

# -------------------------------------------
# 7. Expose port for FastAPI
# -------------------------------------------
EXPOSE 8000

# -------------------------------------------
# 8. Start FastAPI server
# -------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
