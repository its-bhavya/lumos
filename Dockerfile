# -------------------------------------------
# 1. Base Python Image
# -------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# -------------------------------------------
# 2. Install system dependencies
# -------------------------------------------


# -------------------------------------------
# 4. Make directories
# -------------------------------------------
WORKDIR /app

RUN apt-get update && apt-get install -y graphviz

# -------------------------------------------
# 3. Install uv package manager
# -------------------------------------------
RUN pip install uv

# -------------------------------------------
# 6. Install Python deps via uv
# -------------------------------------------
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen


COPY . .
# -------------------------------------------
# 7. Expose port for FastAPI
# -------------------------------------------
EXPOSE 8000

# -------------------------------------------
# 8. Start FastAPI server
# -------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
