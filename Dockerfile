# Stage 1: Build the Go environment server
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum* ./
RUN go mod download || true

COPY main.go ./
COPY env/ ./env/
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o gridmind-server main.go

# Stage 2: Final image with Python runtime and Dashboard
FROM python:3.11-slim

WORKDIR /app

# Install supervisor to run both servers
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY python/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir openai httpx pydantic "pydantic>=2.0.0" requests

# Copy Go binary
COPY --from=builder /app/gridmind-server /usr/local/bin/gridmind-server

# Copy Python layer and Dashboard
COPY python/ ./python/
COPY inference.py ./inference.py
COPY dashboard/ ./dashboard/
COPY data/ ./data/
COPY openenv.yaml ./

# Configure Supervisor
RUN echo "[supervisord]" > /etc/supervisor/conf.d/supervisord.conf && \
    echo "nodaemon=true" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:go-env]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=/usr/local/bin/gridmind-server" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "environment=PORT=7860" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile=/dev/stdout" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile_maxbytes=0" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile=/dev/stderr" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile_maxbytes=0" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "[program:dashboard]" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "command=python -m uvicorn dashboard.server:app --host 0.0.0.0 --port 7861" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile=/dev/stdout" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stdout_logfile_maxbytes=0" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile=/dev/stderr" >> /etc/supervisor/conf.d/supervisord.conf && \
    echo "stderr_logfile_maxbytes=0" >> /etc/supervisor/conf.d/supervisord.conf

# Create run directory for supervisor
RUN mkdir -p /var/run/supervisor /var/log/supervisor && \
    chmod 755 /var/run/supervisor /var/log/supervisor

# Add a non-root user (good practice and required for some HF Spaces configs)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app && \
    chown -R user:user /var/run/supervisor /var/log/supervisor

# 7860 = Env Server (main OpenEnv endpoint), 7861 = Dashboard
EXPOSE 7860 7861

# Run supervisor as root to manage both services (required for multi-process supervision)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf", "-n"]
