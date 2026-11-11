from copy import deepcopy


def sample(kind="choice", target="hard", identity="one"):
    question = {"type": kind, "instructions": {"question": "Decide."}}
    if kind == "choice":
        question["criteria"] = {"A": "one", "B": "two"}
        hard, soft = "B", {"A": 0.2, "B": 0.8}
    elif kind == "noul":
        hard, soft = True, {"false": 0.2, "true": 0.8}
    else:
        question["criteria"] = ["low", "middle", "high"]
        hard, soft = 2, {"0": 0.1, "1": 0.2, "2": 0.7}
    return {
        "schema_version": "decisionforge.eval.v1",
        "id": identity,
        "group_id": None,
        "split": "test",
        "subset": "main",
        "request": {
            "state": {"text": "Content", "condition": "limit"},
            "questions": {"decision": question},
        },
        "target": (
            {"kind": "hard", "value": hard}
            if target == "hard"
            else {"kind": "soft", "probabilities": soft}
            if target == "soft"
            else {"kind": "scalar", "value": 1.7}
            if target == "scalar"
            else None
        ),
        "metadata": {"target_source": "authored_test_fixture"},
    }


def prediction(identity="one", kind="choice", probabilities=None, valid=True):
    answer = {"type": kind}
    if kind == "choice":
        answer.update(
            choice="B", probabilities=probabilities or {"A": 0.3, "B": 0.7}, confidence=0.4
        )
    elif kind == "noul":
        answer["noul"] = 0.8
    else:
        answer.update(
            score=1.6,
            probabilities=probabilities or {"0": 0.1, "1": 0.2, "2": 0.7},
            confidence=0.4,
            legend={"0": "low", "1": "middle", "2": "high"},
        )
    return {
        "id": identity,
        "model": "fixture",
        "probability_source": "native",
        "transport_ok": valid,
        "protocol_ok": valid,
        "answer": deepcopy(answer) if valid else None,
        "error": None,
    }
