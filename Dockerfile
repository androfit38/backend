# Dockerfile

# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1
ARG UID=10001

# Create non-privileged user
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Install build deps
RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

USER appuser
RUN mkdir -p /home/appuser/.cache && chown -R appuser /home/appuser/.cache

# Ensure user scripts are on PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /home/appuser

COPY requirements.txt .
RUN python -m pip install --user --no-cache-dir -r requirements.txt

COPY . .

# Download any required models at build-time
RUN python main.py download-files || true

# Expose health-check port
EXPOSE 8081

# Start the app
CMD ["python", "main.py", "start"]
