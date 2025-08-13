# This is a Dockerfile for running LiveKit Agents
# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-slim

# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Install gcc and other build dependencies
RUN apt-get update && \
    apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

USER appuser

RUN mkdir -p /home/appuser/.cache
RUN chown -R appuser /home/appuser/.cache

# ADD THIS LINE: Update PATH to include local bin
ENV PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /home/appuser

COPY requirements.txt .
RUN python -m pip install --user --no-cache-dir -r requirements.txt

COPY . .

# Ensure that any dependent models are downloaded at build-time
RUN python main.py download-files || true

# Expose healthcheck port
EXPOSE 8081

# Run the application
CMD ["python", "main.py", "start"]
