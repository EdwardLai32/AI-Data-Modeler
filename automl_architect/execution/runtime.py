"""Killable computation workers. Only registered deterministic tasks can run."""

from __future__ import annotations

import io
import multiprocessing
import os
import time

import joblib

from ..core.errors import BudgetExceeded, ExecutionError, RunCancelled


def _worker(connection, payload, settings_data, elapsed, task, argument):
    from ..config import Settings
    from ..core.events import EventBus
    from ..storage.checkpoints import restore_payload, state_payload

    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        os.environ.pop(name, None)
    try:
        settings = Settings(_env_file=None, **settings_data)
        bus = EventBus(payload["config"].run_id)
        state = restore_payload(payload, settings, bus, elapsed)
        state.extras["computation_worker"] = True
        if task == "train":
            from .trainer import run_experiments

            result = run_experiments(state)
        elif task == "tune":
            from .tuner import run_tuning

            result = run_tuning(state, argument)
        else:
            raise ValueError("Unregistered computation task.")
        data = io.BytesIO()
        joblib.dump((state_payload(state), result, bus.since(0)), data)
        connection.send_bytes(b"O" + data.getvalue())
    except BaseException as exc:
        connection.send_bytes(
            b"E" + f"{type(exc).__name__}: {exc}".encode("utf-8", errors="replace")
        )
    finally:
        connection.close()


def run_computation(state, task, argument=None):
    from ..api.service import active_control
    from ..storage.checkpoints import state_payload

    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    settings = state.settings.model_dump()
    settings.update(offline=True, isolate_training=False, enable_checkpoints=False)
    # Child estimators use threads; terminating the worker must not leave nested
    # process pools running after cancellation.
    settings["n_jobs"] = 1
    worker = context.Process(
        target=_worker,
        args=(writer, state_payload(state), settings, state.elapsed_seconds, task, argument),
        daemon=True,
    )
    started = time.monotonic()
    worker.start()
    writer.close()
    control = active_control()
    try:
        while True:
            if control is not None:
                control.raise_if_cancelled()
            cancel = state.extras.get("cancel_requested", False)
            if cancel:
                raise RunCancelled("Run cancelled during computation.")
            if state.config.strict_deadline and state.time_remaining <= 0:
                raise BudgetExceeded("Run deadline reached; computation worker stopped.")
            if task == "tune" and time.monotonic() - started > argument.timeout_seconds:
                raise BudgetExceeded("Tuning deadline reached; computation worker stopped.")
            if reader.poll(0.1):
                message = reader.recv_bytes()
                if message[:1] != b"O":
                    raise ExecutionError(message[1:].decode("utf-8"))
                payload, result, events = joblib.load(io.BytesIO(message[1:]))
                # Preserve identity, timer, credentials, budgets and parent event
                # ordering. Only computation output is adopted from the worker.
                for name in (
                    "experiments",
                    "tuning",
                    "preprocessor",
                    "best_model",
                    "best_pipeline",
                    "label_encoder",
                    "feature_names",
                    "applied_cleaning",
                    "applied_features",
                ):
                    setattr(state, name, payload[name])
                state.warnings = payload["warnings"]
                for key, value in payload["extras"].items():
                    if key not in {"computation_worker", "checkpoint"}:
                        state.extras[key] = value
                for event in events:
                    state.bus.emit(
                        event.kind,
                        event.message,
                        agent=event.agent,
                        step_id=event.step_id,
                        payload=event.payload,
                        duration_seconds=event.duration_seconds,
                    )
                return result
            if not worker.is_alive():
                raise ExecutionError(f"Computation worker exited unexpectedly ({worker.exitcode}).")
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=5)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=5)
        reader.close()
