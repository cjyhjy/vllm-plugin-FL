# SPDX-License-Identifier: Apache-2.0
"""Record and compare the available provenance of operator profiles."""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlsplit

PROVENANCE_FILE = "run_provenance.json"
REQUIRED_FIELDS = ("model", "vllm_plugins", "device", "tensor_parallel_size")
LAUNCH_FIELDS = {
    "--revision": "model_revision",
    "--tensor-parallel-size": "tensor_parallel_size",
    "-tp": "tensor_parallel_size",
}


def model_identity(model: str) -> str | None:
    if not model:
        return None
    path = Path(model)
    if path.exists():
        return str(path.resolve())
    if "://" in model:
        try:
            url = urlsplit(model)
        except ValueError:
            return None
        # A signed or authenticated URL may contain credentials in its arguments.
        if url.username is not None or url.query or url.fragment:
            return None
    return model


def launch_provenance(model: str, args: list[str]) -> dict[str, Any]:
    """Collect the positional model and explicit launch settings from serve.sh."""
    provenance: dict[str, Any] = {"model": model_identity(model)}
    plugins = os.environ.get("VLLM_PLUGINS")
    if plugins is not None:
        # vLLM matches entry point names against the comma-separated allowlist.
        provenance["vllm_plugins"] = sorted(set(plugins.split(",")) - {""})
    # Explicit options take precedence over vLLM's config file and defaults.
    for index, arg in enumerate(args):
        if arg == "--":
            break
        flag, equals, value = arg.partition("=")
        field = LAUNCH_FIELDS.get(flag.replace("_", "-"))
        if field is not None:
            if not equals and index + 1 < len(args):
                value = args[index + 1]
            if field == "tensor_parallel_size":
                try:
                    provenance[field] = int(value)
                except ValueError:
                    provenance[field] = None
            else:
                provenance[field] = value or None
    return provenance


def write_launch_provenance(directory: Path, provenance: dict[str, Any]) -> None:
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            json.dump(provenance, output, indent=2)
            output.write("\n")
            output.close()
            temporary.replace(directory / PROVENANCE_FILE)
        finally:
            temporary.unlink(missing_ok=True)


def trace_device(path: Path) -> str | None:
    """Read Kineto's header without decoding the potentially large event array."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        header, separator, _ = source.read(1024 * 1024).partition('"traceEvents"')
    if not separator:
        return None
    try:
        metadata = json.loads(header.rstrip().removesuffix(",") + "}")
    except json.JSONDecodeError:
        return None
    devices = metadata.get("deviceProperties")
    if not isinstance(devices, list) or not devices:
        return None
    names = [
        device.get("name") if isinstance(device, dict) else None for device in devices
    ]
    # A heterogeneous visible device list does not identify the profiled device.
    first = names[0]
    if (
        isinstance(first, str)
        and first.strip()
        and all(name == first for name in names)
    ):
        return first
    return None


def read_run_provenance(trace_path: Path, traces: list[Path]) -> dict[str, Any]:
    directory = trace_path.parent if trace_path.is_file() else trace_path
    sidecar = directory / PROVENANCE_FILE
    provenance = (
        json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    )
    if not isinstance(provenance, dict):
        raise ValueError(f"run provenance must be a JSON object: {sidecar}")
    devices = {trace_device(trace) for trace in traces}
    # The trace, rather than the machine doing offline extraction, supplies the device.
    provenance["device"] = next(iter(devices)) if len(devices) == 1 else None
    return provenance


def provenance_value(provenance: dict[str, Any], field: str) -> Any:
    value = provenance.get(field)
    if field == "tensor_parallel_size":
        return value if type(value) is int and value > 0 else None
    if field == "vllm_plugins":
        if isinstance(value, list) and all(
            isinstance(name, str) and name for name in value
        ):
            return sorted(set(value))
        return None
    return value if isinstance(value, str) and value.strip() else None


def validate_run_provenance(
    current: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    missing = []
    conflicts = []
    for field in (*REQUIRED_FIELDS, "model_revision"):
        observed = provenance_value(current, field)
        expected = provenance_value(reference, field)
        if field == "model_revision" and observed is None and expected is None:
            continue
        if observed is None or expected is None:
            missing.append(field)
        elif observed != expected:
            conflicts.append(f"{field} (current={observed!r}, eager={expected!r})")
    if conflicts:
        raise ValueError("eager reference provenance mismatch: " + ", ".join(conflicts))
    return {
        "environment_validation": "unverified" if missing else "verified",
        "unverified_fields": missing,
    }


if __name__ == "__main__":
    directory, model, *args = sys.argv[1:]
    write_launch_provenance(Path(directory), launch_provenance(model, args))
