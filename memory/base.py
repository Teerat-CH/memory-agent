"""Memory interface. Every method (no memory, ExpRecent, ReasoningBank, yours) implements this.

Per task in the stream:
    ctx = memory.retrieve(task)   # once the task's goal is known: text pasted into the agent's prompt
    memory.update(episode)        # after the episode: learn from it

`update` receives the whole `Episode` (see episode.py). Read whatever you need:
    episode.success                      -> result only
    episode.steps / episode.transcript() -> full trajectory

`self.llm` is set by run.py to the agent's LLM, for methods that call a model
(e.g. a judge or an extractor). Its tokens count toward the run's usage.
"""

from __future__ import annotations

from episode import Episode, Task


class Memory:
    llm = None  # set by run.py

    def retrieve(self, task: Task) -> str:
        """Return text to show the agent before it starts `task`. Empty string = nothing."""
        return ""

    def update(self, episode: Episode) -> None:
        """Learn from a finished episode."""
        return None


class NoMemory(Memory):
    """Baseline: same agent, empty memory."""
