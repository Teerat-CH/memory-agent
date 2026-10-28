"""Core data types.

When a task finishes, everything about it is packed into one `Episode` and
handed to `memory.update(episode)`. Each memory method reads only what it needs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Task:
    """One item in the task stream."""

    id: str                      # unique within the benchmark
    instruction: str             # natural-language goal shown to the agent
    meta: dict[str, Any] = field(default_factory=dict)  # task type, file paths, etc.


@dataclass
class Step:
    """One agent decision and what the environment returned."""

    thought: str                 # the agent's reasoning ("Think:" part)
    action: str                  # the action sent to the environment
    observation: str             # what the environment returned
    raw_output: str = ""         # full LLM output, for debugging
    reasoning: str = ""          # the model's native thinking, if the provider returns it separately


@dataclass
class Episode:
    """Everything known about one task attempt."""

    task: Task
    steps: list[Step] = field(default_factory=list)
    success: bool = False        # benchmark ground truth (ALFWorld `won`)
    memory_context: str = ""     # what memory.retrieve() returned for this task
    initial_observation: str = ""  # the environment's opening text, before any action
    judge: dict[str, Any] | None = None  # a memory method's own success verdict, if it makes one
    usage: dict[str, Any] = field(default_factory=dict)  # tokens / cost for this episode
    error: str | None = None     # set if the episode crashed
    memory_error: str | None = None  # set if memory.retrieve() or memory.update() crashed

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["num_steps"] = self.num_steps
        return d

    def transcript(self, max_obs_chars: int = 300) -> str:
        """Compact action/observation log, handy for memory methods."""
        lines = [f"Task: {self.task.instruction}"]
        for i, s in enumerate(self.steps, 1):
            obs = s.observation if len(s.observation) <= max_obs_chars else s.observation[:max_obs_chars] + "..."
            if s.thought:
                lines.append(f"[{i}] Think: {s.thought}")
            lines.append(f"[{i}] Action: {s.action}")
            lines.append(f"[{i}] Observation: {obs}")
        lines.append(f"Outcome: {'SUCCESS' if self.success else 'FAILURE'}")
        return "\n".join(lines)
