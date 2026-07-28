# The API only: it reads outputs/store.db, which src/pipeline.py builds. No video or model code goes in,
# so the image needs neither PyTorch nor OpenCV.
FROM python:3.11-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/__init__.py src/metrics.py src/api.py src/

RUN useradd --create-home api
USER api

ENV STORE_DB=/app/outputs/store.db
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
