# ── Stage 1: Build ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY watchdog/requirements.txt .
RUN pip install --no-cache-dir --target=/deps -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.11-slim

# Security: non-root user
RUN groupadd -r watchdog && useradd -r -g watchdog -d /app -s /sbin/nologin watchdog

WORKDIR /app

# Install snapshot-cli for Tier 2 GCS fallback
# Download the latest release binary from GitHub (linux/amd64)
ARG SNAPSHOT_CLI_VERSION=latest
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL -o /usr/local/bin/snapshot-cli \
       "https://github.com/castai/snapshot-cli/releases/${SNAPSHOT_CLI_VERSION}/download/snapshot-cli-linux-amd64" \
    && chmod +x /usr/local/bin/snapshot-cli \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy deps from builder
COPY --from=builder /deps /usr/local/lib/python3.11/site-packages/

# Copy application
COPY watchdog/ /app/watchdog/

# State file on persistent volume; /tmp writable for any temp needs
ENV WATCHDOG_STATE_FILE=/data/watchdog_state.json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Own the app dir; /data will be mounted
RUN chown -R watchdog:watchdog /app

USER watchdog

ENTRYPOINT ["python", "-m", "watchdog"]
