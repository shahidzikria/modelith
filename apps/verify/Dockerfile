# Sandboxed verifier: run with --network none --read-only --memory 4g --cpus 2 --tmpfs /tmp
FROM python:3.12-slim
RUN useradd -m verify
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0 \
 && pip install --no-cache-dir -r requirements.txt
COPY main.py .
USER verify
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
