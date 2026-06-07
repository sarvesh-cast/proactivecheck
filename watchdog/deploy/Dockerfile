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
