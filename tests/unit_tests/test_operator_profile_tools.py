# SPDX-License-Identifier: Apache-2.0

import csv
import gzip
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_PATH = REPO_ROOT / "tools/operator_profile/extract_operator_shapes.py"
PROFILE_SCRIPT = REPO_ROOT / "tools/operator_profile/profile.sh"
SERVE_SCRIPT = REPO_ROOT / "tools/operator_profile/serve.sh"
sys.path.insert(0, str(EXTRACTOR_PATH.parent))


def load_extractor():
    spec = importlib.util.spec_from_file_location(
        "operator_profile_extractor", EXTRACTOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_iter_events_accepts_common_trace_formats(tmp_path, compact, compressed):
    extractor = load_extractor()
    events = [
        {"name": "aten::add", "cat": "cpu_op", "args": {}},
        {"name": "kernel", "cat": "kernel", "dur": 1.5, "args": {}},
    ]
    payload = {"metadata": "x" * 5000, "traceEvents": events}
    suffix = ".json.gz" if compressed else ".json"
    path = tmp_path / f"trace{suffix}"
    text = json.dumps(
        payload,
        separators=(",", ":") if compact else None,
        indent=None if compact else 4,
    )
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8") as target:
            target.write(text)
    else:
        path.write_text(text, encoding="utf-8")

    assert list(extractor.iter_events(path)) == events


def test_iter_events_rejects_missing_trace_events(tmp_path):
    extractor = load_extractor()
    path = tmp_path / "trace.json"
    path.write_text('{"metadata": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="traceEvents not found"):
        list(extractor.iter_events(path))


def test_profile_script_rejects_nested_stale_trace(tmp_path):
    run_dir = tmp_path / "run"
    nested = run_dir / "profile/rank0"
    nested.mkdir(parents=True)
    (nested / "stale.pt.trace.json.gz").touch()

    commands = [
        ["bash", str(PROFILE_SCRIPT), "model", str(run_dir)],
        ["bash", str(SERVE_SCRIPT), "/model", "model", str(run_dir)],
    ]
    for command in commands:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert "profile directory already contains a trace" in result.stderr


def make_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("plugins", "expected"),
    [("fl", ["fl"]), ("", []), (None, None)],
)
def test_serve_script_cleans_configured_flaggems_oplist(tmp_path, plugins, expected):
    run_dir = tmp_path / "run"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(fake_bin / "vllm", "#!/bin/sh\nexit 0\n")
    oplist = tmp_path / "evidence/custom-oplist.txt"
    oplist.parent.mkdir()
    oplist.write_text("stale", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(oplist),
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    if plugins is None:
        env.pop("VLLM_PLUGINS", None)
    else:
        env["VLLM_PLUGINS"] = plugins

    result = subprocess.run(
        [
            "bash",
            str(SERVE_SCRIPT),
            "example/model",
            "common-alias",
            str(run_dir),
            "--revision=example-revision",
            "-tp",
            "2",
            "--api-key",
            "test-key",
            "--enforce-eager",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert not oplist.exists()
    provenance = json.loads((run_dir / "profile/run_provenance.json").read_text())
    assert provenance.pop("vllm_plugins", None) == expected
    assert provenance == {
        "model": "example/model",
        "model_revision": "example-revision",
        "tensor_parallel_size": 2,
    }


def test_profile_script_extracts_provenance_and_moves_configured_flaggems_oplist(
    tmp_path,
):
    run_dir = tmp_path / "run"
    trace_dir = run_dir / "profile"
    trace_dir.mkdir(parents=True)
    (trace_dir / "run_provenance.json").write_text('{"model": "example/model"}')
    trace = tmp_path / "rank0.pt.trace.json"
    trace.write_text('{"traceEvents": [{"cat": "kernel", "name": "kernel", "dur": 1}]}')
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    make_executable(
        fake_bin / "curl",
        '#!/bin/sh\ncase "$*" in\n  */stop_profile) cp "$TEST_TRACE" "$TEST_TRACE_DIR/" ;;\nesac\n',
    )
    make_executable(
        fake_bin / "python3",
        '#!/bin/sh\ncase "$1" in\n  */extract_operator_shapes.py) exec "$TEST_PYTHON" "$@" ;;\nesac\n',
    )
    oplist = tmp_path / "evidence/custom-oplist.txt"
    oplist.parent.mkdir()
    oplist.write_text("current", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "FLAGGEMS_ENABLE_OPLIST_PATH": str(oplist),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "TEST_PYTHON": sys.executable,
            "TEST_TRACE": str(trace),
            "TEST_TRACE_DIR": str(trace_dir),
        }
    )

    result = subprocess.run(
        ["bash", str(PROFILE_SCRIPT), "model", str(run_dir)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    copied = run_dir / "results/flaggems_enable_oplist.txt"
    assert result.returncode == 0
    assert copied.read_text(encoding="utf-8") == "current"
    assert not oplist.exists()
    summary = json.loads((run_dir / "results/summary.json").read_text())
    assert summary["run_provenance"] == {"model": "example/model", "device": None}


@pytest.mark.parametrize(
    "differences",
    [
        {},
        {"model": "example/other-model"},
        {"device": "AMD Instinct MI300X"},
        {"vllm_plugins": [], "tensor_parallel_size": 4, "model_revision": "other"},
        {"tensor_parallel_size": None},
        {"device": None},
        {"model": "example/other-model", "device": None},
    ],
)
def test_extractor_compares_eager_run_provenance(tmp_path, differences):
    provenance = {
        "model": "example/model",
        "model_revision": "example-revision",
        "vllm_plugins": ["fl"],
        "tensor_parallel_size": 2,
        "device": "NVIDIA H100",
    }

    def extract(name, settings, events, reference=None):
        trace_dir = tmp_path / name
        trace_dir.mkdir()
        settings = settings.copy()
        device = settings.pop("device")
        (trace_dir / "run_provenance.json").write_text(json.dumps(settings))
        with gzip.open(
            trace_dir / "rank0.pt.trace.json.gz", "wt", encoding="utf-8"
        ) as target:
            json.dump(
                {"deviceProperties": [{"name": device}], "traceEvents": events}, target
            )
        output = tmp_path / f"{name}-results"
        command = [
            sys.executable,
            str(EXTRACTOR_PATH),
            "--trace-dir",
            str(trace_dir),
            "--output-dir",
            str(output),
        ]
        if reference is not None:
            command += ["--eager-reference", str(reference)]
        return output, subprocess.run(
            command, capture_output=True, text=True, check=False
        )

    eager, result = extract(
        "eager",
        provenance,
        [
            {"cat": "cpu_op", "name": "aten::mul", "args": {"External id": 1}},
            {
                "cat": "kernel",
                "name": "shared_kernel",
                "dur": 99,
                "args": {"External id": 1},
            },
        ],
    )
    assert result.returncode == 0, result.stderr
    assert (
        json.loads((eager / "summary.json").read_text())["run_provenance"] == provenance
    )
    graph, result = extract(
        "graph",
        {**provenance, **differences},
        [
            {"cat": "kernel", "name": "shared_kernel", "dur": 1},
            {"cat": "kernel", "name": "shared_kernel", "dur": 2},
        ],
        eager,
    )
    conflicts = {
        field: value for field, value in differences.items() if value is not None
    }
    if conflicts:
        assert result.returncode != 0
        assert not graph.exists()
        assert "eager reference provenance mismatch" in result.stderr
        for field, value in conflicts.items():
            assert (
                f"{field} (current={value!r}, eager={provenance[field]!r})"
                in result.stderr
            )
        return

    assert result.returncode == 0, result.stderr
    summary = json.loads((graph / "summary.json").read_text())
    reference = summary["eager_reference"]
    assert reference["environment_validation"] == (
        "unverified" if differences else "verified"
    )
    assert reference["unverified_fields"] == list(differences)
    assert ("warning: eager reference provenance unverified" in result.stderr) == bool(
        differences
    )
    assert summary["kernel_event_count"] == 2
    assert summary["kernel_time_total_us"] == 3
    with (graph / "kernel_time.csv").open() as source:
        assert list(csv.DictReader(source)) == [
            {
                "operator_name": "aten::mul",
                "refer": "eager-refer",
                "kernel_name": "shared_kernel",
                "kernel_call_count": "2",
                "kernel_time_us": "3.0",
                "percent": "100.000%",
            }
        ]
