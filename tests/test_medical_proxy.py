"""External Jev routing must remain confined to this task's proxy process."""

import copy

import pytest


def test_proxy_config_selects_only_the_requested_node_without_mutating_source():
    from examples.domain.medical_triage.scripts.v1.run_jev import task_proxy_config

    original = {
        "mixed-port": 7890,
        "rules": ["MATCH,GLOBAL"],
        "proxies": [
            {"name": "node", "type": "vless", "uuid": "private-test-value"},
            {"name": "other", "type": "http"},
        ],
        "secret": "controller-secret",
    }
    before = copy.deepcopy(original)
    result = task_proxy_config(original, 12345, "node")
    assert original == before
    assert result["mixed-port"] == 12345
    assert result["proxies"] == [original["proxies"][0]]
    assert result["rules"] == ["DOMAIN-SUFFIX,openrouter.ai,node", "MATCH,DIRECT"]
    assert result["allow-lan"] is False
    assert result["bind-address"] == "127.0.0.1"
    assert "secret" not in result and "external-controller" not in result


def test_proxy_refuses_unknown_or_dependent_outbound_node():
    from examples.domain.medical_triage.scripts.v1.run_jev import task_proxy_config

    with pytest.raises(ValueError):
        task_proxy_config({"proxies": []}, 12345, "missing")
    with pytest.raises(ValueError):
        task_proxy_config({"proxies": [{"name": "node", "dialer-proxy": "GLOBAL"}]}, 12345, "node")


def test_proxy_copies_a_fixed_dialer_chain_without_other_routes():
    from examples.domain.medical_triage.scripts.v1.run_jev import task_proxy_config

    original = {
        "proxies": [
            {"name": "CLUSTER", "type": "http"},
            {"name": "node", "type": "vless", "dialer-proxy": "CLUSTER"},
            {"name": "other", "type": "http"},
        ]
    }
    result = task_proxy_config(original, 12345, "node")
    assert [p["name"] for p in result["proxies"]] == ["CLUSTER", "node"]
