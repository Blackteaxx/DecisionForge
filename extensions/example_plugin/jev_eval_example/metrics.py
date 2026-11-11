"""Group agreement metric owned by the example benchmark."""

from decisionforge.eval import Metric


def unanimous_groups(rows, params):
    groups = {}
    for row in rows:
        group = row["sample"].group_id
        if group and not row["excluded"]:
            groups.setdefault(group, []).append(row)
    eligible = [g for g in groups.values() if len(g) >= 2]
    usable = [g for g in eligible if all(r["scoring_valid"] for r in g)]
    return {
        "value": sum(len({r["label"] for r in g}) == 1 for g in usable) / len(usable)
        if usable
        else None,
        "eligible": len(eligible),
        "used": len(usable),
        "failed": len(eligible) - len(usable),
        "unit_of_aggregation": "group",
    }


def group_metric():
    return Metric("unanimous_groups", "1", "higher", "ratio", unanimous_groups)
