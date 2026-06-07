FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY watchdog/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy watchdog package
COPY watchdog/ /app/watchdog/

# State file lives on a persistent volume
ENV WATCHDOG_STATE_FILE=/data/watchdog_state.json
ENV PYTHONUNBUFFERED=1

# Run the watchdog continuous loop
ENTRYPOINT ["python", "-m", "watchdog"]
