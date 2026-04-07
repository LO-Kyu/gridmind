# Stage 1: Build the Go environment server
FROM golang:1.21 AS builder

WORKDIR /app
COPY go.mod go.sum* ./
RUN go mod download || true

COPY main.go ./
COPY env/ ./env/
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -a -installsuffix cgo -ldflags="-s -w" -o gridmind-server main.go

# Stage 2: Final image with Python runtime and Dashboard
FROM python:3.11-slim

WORKDIR /app

# Install supervisor to run both servers
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY python/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir openai httpx pydantic requests fastapi uvicorn python-dotenv

# Copy Go binary
COPY --from=builder /app/gridmind-server /usr/local/bin/gridmind-server

# Copy Python layer and Dashboard
COPY python/ ./python/
COPY inference.py ./inference.py
COPY dashboard/ ./dashboard/
COPY server/ ./server/
COPY data/ ./data/
COPY openenv.yaml ./

# Configure Supervisor to use /tmp for socket and pid files (writable by any user)
RUN echo "[unix_http_server]" > /etc/supervisor/conf.d/supervisord.conf && \
    echo "file=/tmp/supervisor.sock" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[supervisord]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "pidfile=/tmp/supervisord.pid" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "nodaemon=true" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:go-env]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=/usr/local/bin/gridmind-server" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "environment=PORT=7860" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile=/dev/stdout" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile_maxbytes=0" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile=/dev/stderr" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile_maxbytes=0" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:dashboard]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=python -m uvicorn dashboard.server:app --host 0.0.0.0 --port 7861" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile=/dev/stdout" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile_maxbytes=0" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile=/dev/stderr" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile_maxbytes=0" >> /etc/supervisor/conf.d/supervisord.conf

# EXPOSE 7860 only - this is the main OpenEnv API endpoint (reverse proxy + /dashboard)
# Port 7861 (dashboard) runs internally only and is accessed via /dashboard proxy
EXPOSE 7860

# Add a non-root user (good practice and required for some HF Spaces configs)
RUN useradd -m -u 1000 user && chown -R user:user /app

# Run supervisord to manage both Go server and Python dashboard
# Using /tmp for socket and pid files (writable by any user, including uid 1000)
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
