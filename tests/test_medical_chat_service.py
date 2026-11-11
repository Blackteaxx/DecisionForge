"""The same-base service must enforce its advertised deterministic generation contract."""

import pytest
from pydantic import ValidationError


def test_chat_request_preserves_original_prompt_and_disables_thinking():
    from examples.domain.medical_triage.scripts.v1.serve_chat import ChatRequest

    messages = [
        {"role": "user", "content": "Exact patient observations.\nReturn the requested JSON."}
    ]
    value = ChatRequest(model="qwen35-9b", messages=messages, temperature=0, max_tokens=1024)
    assert value.messages == messages
    assert value.chat_template_kwargs == {"enable_thinking": False}
    assert value.n == 1 and value.stream is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.7),
        ("n", 2),
        ("stream", True),
        ("chat_template_kwargs", {"enable_thinking": True}),
        ("response_format", {"type": "json_schema"}),
    ],
)
def test_unsupported_generation_contract_is_rejected(field, value):
    from examples.domain.medical_triage.scripts.v1.serve_chat import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {
                "model": "qwen35-9b",
                "messages": [{"role": "user", "content": "A patient message"}],
                field: value,
            }
        )
