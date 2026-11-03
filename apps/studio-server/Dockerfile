FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
# albumentations pulls opencv-python-headless; keep the contrib build (same cv2 module) installed last.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall --no-deps opencv-contrib-python-headless==4.12.0.88
COPY main.py .
RUN useradd -m studio
USER studio
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
