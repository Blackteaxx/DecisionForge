"""Bounded request-local reuse of the pinned HF backend's exact tokenization."""

from __future__ import annotations

from contextlib import contextmanager


class RequestTokenizer:
    """Reuse only the upstream backend's known encode and batch-pad signatures."""

    def __init__(self, tokenizer, *, max_entries=1024, max_tokens=2_000_000):
        self.original = tokenizer
        self.max_entries, self.max_tokens = max_entries, max_tokens
        self._cache, self._tokens = {}, 0
        self.supported = (
            getattr(tokenizer, "is_fast", False)
            and set(getattr(tokenizer, "model_input_names", ())) == {"input_ids", "attention_mask"}
            and callable(getattr(tokenizer, "pad", None))
        )

    def __getattr__(self, name):
        return getattr(self.original, name)

    def encode(self, text, *args, **kwargs):
        known = (
            self.supported
            and isinstance(text, str)
            and not args
            and kwargs == {"add_special_tokens": False}
        )
        if known and text in self._cache:
            return list(self._cache[text])
        result = self.original.encode(text, *args, **kwargs)
        if (
            known
            and isinstance(result, list)
            and all(type(value) is int for value in result)
            and len(self._cache) < self.max_entries
            and self._tokens + len(result) <= self.max_tokens
        ):
            self._cache[text] = tuple(result)
            self._tokens += len(result)
        return result

    def __call__(self, texts, *args, **kwargs):
        if (
            self.supported
            and not args
            and kwargs == {"return_tensors": "pt", "padding": True, "add_special_tokens": False}
            and isinstance(texts, (list, tuple))
            and texts
            and all(isinstance(text, str) and text in self._cache for text in texts)
        ):
            return self.original.pad(
                {"input_ids": [list(self._cache[text]) for text in texts]},
                padding=True,
                return_tensors="pt",
            )
        return self.original(texts, *args, **kwargs)

    def clear(self):
        self._cache.clear()
        self._tokens = 0


@contextmanager
def cached_tokenization(backend, bounded_backend, *, enabled):
    """Temporarily share one tokenizer proxy, restoring owners even on failure."""
    original = backend.tokenizer
    owners = [backend] if bounded_backend is backend else [backend, bounded_backend]
    cached = RequestTokenizer(original) if enabled else None
    if (
        cached is None
        or not cached.supported
        or any(owner.tokenizer is not original for owner in owners)
    ):
        yield None
        return
    try:
        for owner in owners:
            owner.tokenizer = cached
        yield cached
    finally:
        for owner in owners:
            owner.tokenizer = original
        cached.clear()
