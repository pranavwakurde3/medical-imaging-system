FROM python:3.14-slim
WORKDIR /app
RUN pip install --no-cache-dir --default-timeout=200 --retries 10 torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout=200 --retries 10 fastapi uvicorn requests pydicom pillow numpy matplotlib
RUN python -c "import torch, torchvision, pydicom, fastapi, matplotlib; print('deps ok')"
COPY src src
EXPOSE 8001
CMD ["uvicorn", "src.viewer_api:app", "--host", "0.0.0.0", "--port", "8001"]
