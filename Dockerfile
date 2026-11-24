FROM python:3.11-slim

# FFmpeg là bắt buộc cho mọi tính năng xử lý video/âm thanh của Creative Motion Lab.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài thư viện trước để tận dụng cache của Docker (chỉ cài lại khi requirements đổi).
COPY requirements.txt requirements-ai.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-ai.txt

COPY backend ./backend
COPY frontend ./frontend

# Nơi lưu job/video tạm trong lúc máy chủ chạy (không commit lên git, xem .dockerignore).
RUN mkdir -p /app/data

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# $PORT do nền tảng cloud (Render, Railway...) cấp; mặc định 8000 nếu chạy Docker tay.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
