"""Local deterministic Chat Completions frontend for the frozen ordinary HF backbone."""

from __future__ import annotations

import argparse
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[dict[str, str]] = Field(min_length=1)
    temperature: Literal[0, 0.0] = 0
    max_tokens: int = Field(default=2048, gt=0, le=2048)
    n: Literal[1] = 1
    stream: Literal[False] = False
    chat_template_kwargs: dict = Field(default_factory=lambda: {"enable_thinking": False})
    response_format: dict | None = None

    @field_validator("messages")
    @classmethod
    def text_messages(cls, messages):
        if any(
            set(m) != {"role", "content"}
            or m["role"] not in {"system", "user", "assistant"}
            or not m["content"]
            for m in messages
        ):
            raise ValueError("Only nonempty text system/user/assistant messages are supported")
        return messages

    @field_validator("chat_template_kwargs")
    @classmethod
    def no_thinking(cls, value):
        if value != {"enable_thinking": False}:
            raise ValueError("This fixed baseline only supports enable_thinking=false")
        return value

    @field_validator("response_format")
    @classmethod
    def unconstrained_text(cls, value):
        if value is not None and value != {"type": "text"}:
            raise ValueError(
                "Use response_format=text; this service does not constrain or repair JSON"
            )
        return value


class ChatRuntime:
    def __init__(self, model_path, revision, served_name, max_input_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.name, self.limit = torch, served_name, max_input_tokens
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, revision=revision, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=revision, dtype=torch.bfloat16, local_files_only=True
        ).to("cuda:0")
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(
            f"Loaded ordinary HF backbone {model_path}@{revision}; bf16; thinking=false; served={served_name}",
            flush=True,
        )

    def generate(self, request):
        encoded = self.tokenizer.apply_chat_template(
            request.messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_tokens = encoded["input_ids"].shape[-1]
        if prompt_tokens > self.limit:
            raise ValueError(
                f"Input has {prompt_tokens} tokens, exceeds {self.limit}; no truncation"
            )
        with self.lock, self.torch.inference_mode():
            result = self.model.generate(
                **encoded.to(self.device), max_new_tokens=request.max_tokens, do_sample=False
            )
            generated = result[0, prompt_tokens:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            # Preserve raw generated content, including malformed/truncated JSON.
            completion_tokens = generated.shape[-1]
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            reason = "stop" if int(generated[-1]) in eos else "length"
        return {
            "id": "chatcmpl-" + uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def create_app(model_path, revision, served_name="qwen35-9b", max_input_tokens=8192):
    from fastapi import FastAPI, HTTPException

    runtime = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal runtime
        runtime = ChatRuntime(model_path, revision, served_name, max_input_tokens)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backbone_path": str(model_path),
            "revision": revision,
            "precision": "bfloat16",
            "enable_thinking": False,
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": served_name, "object": "model", "owned_by": "task-local-hf"}],
        }

    @app.post("/v1/chat/completions")
    def completion(request: ChatRequest):
        if request.model != served_name:
            raise HTTPException(status_code=404, detail="Unknown served model")
        try:
            return runtime.generate(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--served-model-name", default="qwen35-9b")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    args = parser.parse_args()
    import uvicorn

    app = create_app(args.model_path, args.revision, args.served_model_name, args.max_input_tokens)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
