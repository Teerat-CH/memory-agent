"""ExpRecent: show the agent its k most recent past episodes as worked examples.

A simple, non-LLM baseline (also used in Evo-Memory). It is here mainly as a
template showing how a memory method reads from the `Episode` it is given.
"""

from __future__ import annotations

from episode import Episode, Task
from .base import Memory


class ExpRecent(Memory):
    def __init__(self, k: int = 2, successes_only: bool = True, same_type_first: bool = True):
        self.k = k
        self.successes_only = successes_only
        self.same_type_first = same_type_first
        self.bank: list[dict] = []

    def retrieve(self, task: Task) -> str:
        if not self.bank:
            return ""
        pool = list(reversed(self.bank))  # most recent first
        if self.same_type_first and "task_type" in task.meta:
            same = [e for e in pool if e["task_type"] == task.meta["task_type"]]
            other = [e for e in pool if e["task_type"] != task.meta["task_type"]]
            pool = same + other
        chosen = pool[: self.k]
        return "\n\n".join(f"### Past example {i + 1}\n{e['text']}" for i, e in enumerate(chosen))

    def update(self, episode: Episode) -> None:
        # Reads only what it needs: the result and the trajectory.
        if self.successes_only and not episode.success:
            return
        self.bank.append({
            "task_id": episode.task.id,
            "task_type": episode.task.meta.get("task_type"),
            "text": episode.transcript(max_obs_chars=200),
        })
