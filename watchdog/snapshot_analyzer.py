"""Async wrapper around the snapshot-cli binary for direct GCS snapshot access.

Vendored and adapted from sales-engineering-snapshot-analyzer/backend/snapshot_client.py.
Used as Tier 2 fallback when the CAST AI MCP server is unavailable.

Requirements:
  - snapshot-cli binary in PATH (or SNAPSHOT_CLI_PATH env var)
  - GCS auth: workload identity (EKS) or GOOGLE_APPLICATION_CREDENTIALS

The snapshot-cli reads .btrsp files from GCS and returns JSON.  Each call is
a subprocess invocation wrapped in asyncio.to_thread so the event loop stays
non-blocking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from typing import Any

logger = logging.getLogger("watchdog.snapshot_analyzer")


# ── Exceptions ──────────────────────────────────────────────────────────

class SnapshotCLIError(Exception):
    """Base exception for snapshot-cli errors."""


class CLINotFoundError(SnapshotCLIError):
    """snapshot-cli binary not found in PATH."""


class SnapshotNotFoundError(SnapshotCLIError):
    """Snapshot file not found in GCS."""


class ResourceNotFoundError(SnapshotCLIError):
    """Resource not found in snapshot section."""


# ── Client ──────────────────────────────────────────────────────────────

class SnapshotAnalyzer:
    """Async client for reading CAST AI cluster snapshots from GCS.

    Uses the snapshot-cli binary directly — no MCP server dependency.

    Args:
        cluster_id: Cluster UUID
        gcs_bucket: GCS bucket name (e.g. 'prod-master-console-cluster-snapshots-snapshotstore')
        cli_path: Optional path to snapshot-cli binary (searches PATH if not given)
        snapshot_name: Optional specific snapshot (uses latest if omitted)
    """

    def __init__(
        self,
        cluster_id: str,
        gcs_bucket: str,
        cli_path: str | None = None,
        snapshot_name: str | None = None,
    ) -> None:
        self.cluster_id = cluster_id
        self.gcs_bucket = gcs_bucket
        self.snapshot_name = snapshot_name
        self._cli_path = cli_path or self._find_cli()

    @staticmethod
    def _find_cli() -> str:
        cli = shutil.which("snapshot-cli")
        if not cli:
            raise CLINotFoundError(
                "snapshot-cli binary not found in PATH. "
                "Set SNAPSHOT_CLI_PATH or install from https://github.com/castai/snapshot-cli"
            )
        return cli

    def _run_command_sync(self, args: list[str]) -> Any:
        """Run snapshot-cli and return parsed JSON. Synchronous — call via to_thread."""
        cmd = [
            self._cli_path, *args,
            f"--cluster-id={self.cluster_id}",
            f"--gcs-bucket={self.gcs_bucket}",
        ]
        if self.snapshot_name:
            cmd.append(f"--snapshot-name={self.snapshot_name}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=60,
            )
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired as e:
            raise SnapshotCLIError(f"snapshot-cli timed out after 60s: {args}") from e
        except subprocess.CalledProcessError as e:
            self._handle_error(e)
            raise  # unreachable, _handle_error always raises
        except json.JSONDecodeError as e:
            raise SnapshotCLIError(f"Failed to parse snapshot-cli JSON output: {e}") from e

    @staticmethod
    def _handle_error(error: subprocess.CalledProcessError) -> None:
        stderr = (error.stderr or "").lower()
        # Extract last meaningful line before "Usage:" help text
        lines = (error.stderr or "").split("\n")
        msg = ""
        for line in lines:
            line = line.strip()
            if line.startswith("Usage:"):
                break
            if line:
                msg = line
        if msg.startswith("Error: "):
            msg = msg[7:]
        if not msg:
            msg = error.stderr.strip() if error.stderr else f"exit code {error.returncode}"

        if "snapshot not found" in stderr or "no such file" in stderr:
            raise SnapshotNotFoundError(msg)
        elif "resource not found" in stderr or "not found in section" in stderr:
            raise ResourceNotFoundError(msg)
        else:
            raise SnapshotCLIError(msg)

    # ── Async wrappers ─────────────────────────────────────────────────

    async def get_sections(self, sections: list[str]) -> dict[str, Any]:
        """Fetch multiple snapshot sections in one CLI call.

        Returns dict like {"podList": [...], "nodeList": [...], ...}
        """
        args = ["get"]
        for s in sections:
            args.append(f"--sections={s}")
        return await asyncio.to_thread(self._run_command_sync, args)

    async def get_pods(self) -> list[dict]:
        data = await self.get_sections(["pods"])
        return data.get("podList", [])

    async def get_nodes(self) -> list[dict]:
        data = await self.get_sections(["nodes"])
        return data.get("nodeList", [])

    async def get_deployments(self) -> list[dict]:
        data = await self.get_sections(["deployments"])
        return data.get("deploymentList", [])

    async def get_recommendations(self) -> list[dict]:
        data = await self.get_sections(["recommendations"])
        return data.get("recommendationList", [])

    async def get_pod_metrics(self) -> list[dict]:
        data = await self.get_sections(["podmetrics"])
        return data.get("podmetricsList", [])

    async def get_events(self) -> list[dict]:
        data = await self.get_sections(["events"])
        return data.get("eventList", [])

    async def get_all_watchdog_sections(self) -> dict[str, list]:
        """Fetch all 6 sections needed for watchdog analysis in one CLI call.

        Returns dict with keys: podList, nodeList, deploymentList,
        recommendationList, podmetricsList, eventList.
        """
        data = await self.get_sections([
            "pods", "nodes", "deployments",
            "recommendations", "podmetrics", "events",
        ])
        return {
            "pods": data.get("podList", []),
            "nodes": data.get("nodeList", []),
            "deployments": data.get("deploymentList", []),
            "recommendations": data.get("recommendationList", []),
            "pod_metrics": data.get("podmetricsList", []),
            "events": data.get("eventList", []),
        }

    async def health_check(self) -> bool:
        """Verify snapshot-cli can reach GCS and read a snapshot."""
        try:
            # List sections is a lightweight probe
            await asyncio.to_thread(
                self._run_command_sync, ["get", "--sections=namespaces"]
            )
            return True
        except SnapshotCLIError:
            return False
