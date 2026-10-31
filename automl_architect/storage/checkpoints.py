"""Atomic, authenticated local checkpoints. Never deserialize uploaded artifacts."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import secrets
import sys
import threading
import time
from dataclasses import fields
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import joblib
import sklearn

from ..core.schemas import RunStatus
from ..core.state import RunState

FORMAT_VERSION = 1
_KEY_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def runtime_identity():
    """Reject pickles from incompatible code or scientific-library versions."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return {
        "code_sha256": digest.hexdigest(),
        "libraries": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")
        },
    }


def state_payload(state):
    from ..ingestion.base import redact_source

    payload = {
        f.name: getattr(state, f.name)
        for f in fields(state)
        if f.name not in {"settings", "bus", "_monotonic_start"}
    }
    payload["config"] = state.config.model_copy(
        update={"source": redact_source(state.config.source)}
    )
    return payload


def restore_payload(payload, settings, bus, elapsed=0.0):
    state = RunState(settings=settings, bus=bus, **payload)
    state._monotonic_start = time.monotonic() - elapsed
    return state


def _key(settings):
    with _KEY_LOCK:
        return _read_or_create_key(settings)


def _read_or_create_key(settings):
    path = settings.workspace / ".checkpoint-key"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        key = path.read_bytes()
        # Another process may have exclusively created the file and still be
        # completing its first fsync. Never replace that process's key.
        for _ in range(20):
            if len(key) == 32:
                break
            time.sleep(0.05)
            key = path.read_bytes()
        if len(key) != 32:
            raise ValueError("Invalid checkpoint authentication key.") from None
        return key
    key = secrets.token_bytes(32)
    with os.fdopen(fd, "wb") as stream:
        stream.write(key)
        stream.flush()
        os.fsync(stream.fileno())
    return key


def _folder(settings, run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError("Invalid run identifier.")
    root = settings.runs_dir.resolve()
    result = (root / run_id / "checkpoints").resolve()
    if not result.is_relative_to(root):
        raise ValueError("Checkpoint is outside the workspace.")
    return result


def save_checkpoint(state):
    folder = _folder(state.settings, state.run_id)
    folder.mkdir(parents=True, exist_ok=True)
    name = "state-" + secrets.token_hex(8) + ".joblib"
    data = io.BytesIO()
    joblib.dump(state_payload(state), data, compress=3)
    blob = data.getvalue()
    metadata = {
        "version": FORMAT_VERSION,
        "run_id": state.run_id,
        "file": name,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "elapsed": state.elapsed_seconds,
        "python": list(sys.version_info[:2]),
        "sklearn": sklearn.__version__,
        "runtime": runtime_identity(),
        "sequence": state.bus.last_sequence,
        "saved_at": time.time(),
    }
    signed = json.dumps(metadata, sort_keys=True).encode()
    metadata["signature"] = hmac.new(
        _key(state.settings), signed + blob, hashlib.sha256
    ).hexdigest()
    with (folder / name).open("wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    temporary = folder / (name + ".json")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, folder / "latest.json")
    # A completed atomic manifest points only at a fully written artifact.
    # Retain the previous checkpoint for diagnosis; bounded disk use.
    old = sorted(folder.glob("state-*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    for item in old[2:]:
        item.unlink()
    state.extras["checkpoint"] = name
    return name


def load_checkpoint(settings, run_id, bus):
    folder = _folder(settings, run_id)
    metadata = json.loads((folder / "latest.json").read_text(encoding="utf-8"))
    signature = metadata.pop("signature")
    filename = metadata.get("file", "")
    if not re.fullmatch(r"state-[a-f0-9]{16}\.joblib", filename):
        raise ValueError("Invalid checkpoint artifact name.")
    blob = (folder / filename).read_bytes()
    expected = hmac.new(
        _key(settings), json.dumps(metadata, sort_keys=True).encode() + blob, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("Checkpoint authentication failed; refusing to deserialize it.")
    if metadata["version"] != FORMAT_VERSION or metadata["run_id"] != run_id:
        raise ValueError("Checkpoint identity or format does not match.")
    if (
        metadata["python"] != list(sys.version_info[:2])
        or metadata["sklearn"] != sklearn.__version__
    ):
        raise ValueError("Checkpoint runtime differs; start a new run with this version.")
    if metadata.get("runtime") != runtime_identity():
        raise ValueError(
            "Checkpoint code or dependencies differ; start a new run with this version."
        )
    payload = joblib.load(io.BytesIO(blob))
    bus.advance_sequence(metadata["sequence"])
    state = restore_payload(payload, settings, bus, metadata["elapsed"])
    state.status = RunStatus.RUNNING
    state.finished_at = None
    state.error = None
    state.extras["checkpoint"] = filename
    return state
