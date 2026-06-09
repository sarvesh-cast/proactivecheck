# Watchdog VM Deployment (systemd)

Target: GCE VM `cs-proactive-monitor` in `success-team-dev`, user `sarveshanand`.


## 1. systemd unit

```bash
sudo tee /etc/systemd/system/proactive-monitor.service > /dev/null << 'EOF'
[Unit]
Description=CAST AI Watchdog Monitor (Grip Security)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=sarveshanand
Group=sarveshanand
WorkingDirectory=/home/sarveshanand/proactivecheck
EnvironmentFile=/home/sarveshanand/proactivecheck/watchdog/.env
Environment=GOOGLE_APPLICATION_CREDENTIALS=/home/sarveshanand/.config/gcloud/application_default_credentials.json
ExecStart=/home/sarveshanand/proactivecheck/.venv/bin/python -m watchdog --force-tier=hybrid
Restart=on-failure
RestartSec=30
StartLimitIntervalSec=300
StartLimitBurst=5

# Logging to journald
StandardOutput=journal
StandardError=journal
SyslogIdentifier=watchdog

[Install]
WantedBy=multi-user.target
EOF
```

## 2. Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable proactive-monitor
sudo systemctl restart proactive-monitor
```

## 3. Verify

```bash
# Status
sudo systemctl status proactive-monitor

# Live logs
journalctl -u proactive-monitor -f

# Last 50 lines
journalctl -u proactive-monitor -n 50 --no-pager
```

## 4. Common operations

```bash
# Restart after code update
sudo systemctl restart proactive-monitor

# Stop
sudo systemctl stop proactive-monitor

# One-off test run (outside systemd)
cd ~/proactivecheck
source .venv/bin/activate
export $(grep -v '^#' /etc/watchdog/env | xargs)
python -m watchdog --once --force-tier=hybrid --tier-report

# Dry run
WATCHDOG_DRY_RUN=true python -m watchdog --once --force-tier=hybrid
```

## Prerequisites already done

- `snapshot-cli` installed in PATH
- ADC configured via `gcloud auth application-default login`
- Long-term fix: grant `roles/storage.objectViewer` to VM SA `1012279037880-compute@developer.gserviceaccount.com` on the snapshot bucket so it doesn't depend on personal ADC
