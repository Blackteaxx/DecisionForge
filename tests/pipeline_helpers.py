"""Offline fixtures using the current core's checkpoint and audit layout."""

from decisionforge.pipeline import Pipeline
from decisionforge.util import write_json


def fixture_pipeline(config, store, provider):
    if getattr(provider, "multi", False):
        config.task.modes = ["choice", "score"]
    # CLI execute saves this before starting Pipeline; the test avoids a paid client.
    write_json(store.directory / "config.resolved.json", config.model_dump())

    class RecordedProvider:
        async def call(self, role, payload, schema, record, stage):
            result = await provider.call(role, payload, schema, record, stage)
            store.save_step(record, stage, {"key": "fixture", "output": result.model_dump()})
            store.event(record, stage, {"role": role, "raw_output": result.model_dump_json()})
            store.event(record, stage, {"schema_valid": True})
            return result

    return Pipeline(config, store, RecordedProvider())
