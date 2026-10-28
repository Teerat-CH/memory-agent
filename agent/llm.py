"""Thin LLM wrapper: one `chat()` call, with token and cost accounting.

Uses litellm so you can switch providers with a model string:
  gemini/gemini-2.5-flash, vertex_ai/gemini-2.5-flash, gpt-4o-mini,
  anthropic/claude-sonnet-4-5, ollama/qwen2.5:7b, ...
Set the provider's API key in the environment (GEMINI_API_KEY, OPENAI_API_KEY, ...).
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0  # includes thinking tokens
    reasoning_tokens: int = 0   # thinking tokens alone, if the provider reports them
    truncated: int = 0          # replies cut off by max_tokens (thinking counts toward it)
    cost_usd: float = 0.0
    seconds: float = 0.0        # wall-clock time spent in calls (the useful cost for local models)

    def snapshot(self) -> dict:
        return dict(calls=self.calls, prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens, reasoning_tokens=self.reasoning_tokens,
                    truncated=self.truncated, cost_usd=round(self.cost_usd, 6), seconds=round(self.seconds, 1))

    def diff(self, before: dict) -> dict:
        now = self.snapshot()
        return {k: round(now[k] - before[k], 6) if isinstance(now[k], float) else now[k] - before[k]
                for k in now}


class LLM:
    """Real model via litellm."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 4096,
                 num_retries: int = 5, **extra_kwargs):
        import litellm  # imported lazily so tests without litellm still work

        litellm.suppress_debug_info = True
        self._litellm = litellm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.extra_kwargs = extra_kwargs
        self.usage = Usage()
        self.last_reasoning = ""  # native thinking of the last reply, when returned apart from the text

    def chat(self, messages: list[dict], **kwargs) -> str:
        params = dict(model=self.model, messages=messages, temperature=self.temperature,
                      max_tokens=self.max_tokens, num_retries=self.num_retries)
        params.update(self.extra_kwargs)
        params.update(kwargs)
        start = time.monotonic()
        resp = self._litellm.completion(**params)
        self.usage.seconds += time.monotonic() - start
        self.last_reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
        u = getattr(resp, "usage", None)
        self.usage.calls += 1
        if u is not None:
            self.usage.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
            self.usage.completion_tokens += getattr(u, "completion_tokens", 0) or 0
            details = getattr(u, "completion_tokens_details", None)
            self.usage.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0
        if resp.choices[0].finish_reason == "length":
            self.usage.truncated += 1
        try:
            self.usage.cost_usd += self._litellm.completion_cost(completion_response=resp) or 0.0
        except Exception:
            pass  # unknown pricing for this model; tokens are still counted
        return resp.choices[0].message.content or ""

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embedding vectors for `texts` from an embedding model (e.g. gemini/gemini-embedding-001)."""
        resp = self._litellm.embedding(model=model, input=texts, num_retries=self.num_retries)
        return [d["embedding"] if isinstance(d, dict) else d.embedding for d in resp.data]


class FakeLLM:
    """Offline stand-in for smoke tests. Picks a random valid command from the prompt.

    It never solves tasks on purpose; it only proves the loop, logging and
    memory plumbing work without spending money.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.model = "fake"
        self.usage = Usage()
        self.last_reasoning = ""

    def chat(self, messages: list[dict], **kwargs) -> str:
        text = messages[-1]["content"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        self.usage.calls += 1
        self.usage.prompt_tokens += len(text) // 4
        if "expert judge" in system:  # a memory method's judge
            out = '{"success": false, "rationale": "fake judge", "evidence_step": -1}'
        elif "Memory Item" in system:  # a memory method's extractor
            out = ("# Memory Item 1\n## Title Fake insight\n## Description A placeholder.\n"
                   "## Content Open containers before searching inside them.")
        else:  # the agent: pick a random admissible action (quoted in the SkillOS prompt)
            m = re.search(r"admissible actions of the current situation are: (.*?)\n\s*Now it's your turn", text, re.S)
            options = re.findall(r"'([^']*)'", m.group(1)) if m else []
            action = self.rng.choice(options or ["look"])
            out = f"<think>I will try something.</think>\n<action>{action}</action>"
        self.usage.completion_tokens += len(out) // 4
        return out

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        # Deterministic pseudo-embeddings from word hashes, so similar texts share dimensions.
        vecs = []
        for t in texts:
            v = [0.0] * 64
            for w in re.findall(r"\w+", t.lower()):
                v[hash(w) % 64] += 1.0
            vecs.append(v)
        return vecs


def make_llm(model: str, **kwargs):
    return FakeLLM() if model == "fake" else LLM(model, **kwargs)
