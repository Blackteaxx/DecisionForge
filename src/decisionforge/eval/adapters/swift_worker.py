"""Spawned worker for ms-swift local inference; the parent stays SDK-free."""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace


def worker_main(connection, options):
    runtime = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from decisionforge.training.backends.swift_runtime import Runtime
            from decisionforge.training.config import SwiftConfig

            config = SimpleNamespace(
                model=SimpleNamespace(
                    name=options["base_model"],
                    revision=options["base_revision"],
                    max_seq_length=options["max_seq_length"],
                ),
                swift=SwiftConfig(
                    model_type=options.get("model_type", "omnijev"),
                    usage=options.get("usage"),
                    allow_none=options.get("allow_none", False),
                ),
            )
            runtime = Runtime(
                config,
                options["path"],
                device=options["device"],
                chat=options["adapter"] == "swift_chat_local",
            )
        connection.send({"kind": "ready", "metadata": runtime.metadata})
        while True:
            command = connection.recv()
            if command["kind"] == "close":
                break
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    response = runtime.predict(command["request"], calibrated=options["calibrated"])
                connection.send({"kind": "prediction", "response": response})
            except (ValueError, TypeError, KeyError) as exc:
                connection.send(
                    {
                        "kind": "error",
                        "error": {"kind": type(exc).__name__, "message": str(exc)},
                        "fatal": False,
                    }
                )
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send(
                {
                    "kind": "error",
                    "error": {"kind": type(exc).__name__, "message": str(exc)},
                    "fatal": True,
                }
            )
        except (OSError, EOFError):
            pass
    finally:
        if runtime is not None:
            runtime.close()
        connection.close()
