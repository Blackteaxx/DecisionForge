import pytest

from decisionforge.storage import Store
from decisionforge.util import BudgetExhausted, CallLimit


def test_budget_counts_persist_and_do_not_overrun(tmp_path):
    store = Store(tmp_path)
    charge = store.reserve("r", "stage", 0.6, 1.0, 3)
    with pytest.raises(BudgetExhausted):
        store.reserve("other", "stage", 0.6, 1.0, 3)
    store.settle(charge, 0.1)
    store.close()
    store = Store(tmp_path)
    assert store.spent() == 0.1
    store.reserve("r", "next", 0.2, 1.0, 2)
    with pytest.raises(CallLimit):
        store.reserve("r", "third", 0, None, 2)
    store.close()


def test_group_split_conflict(tmp_path):
    store = Store(tmp_path)
    store.register_group("document", "train")
    with pytest.raises(ValueError, match="crosses splits"):
        store.register_group("document", "test")
    store.close()
