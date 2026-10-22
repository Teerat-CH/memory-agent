"""The task-stream loop: retrieve -> act -> update, one task at a time.

Outputs in the run directory:
    config.json      the run's arguments and task order
    episodes.jsonl   one full Episode per line, appended as tasks finish
    summary.json     success rate overall / per task type, tokens, cost
"""

from __future__ import annotations

import json
import random
import traceback
from collections import defaultdict
from pathlib import Path

from episode import Episode


def summarize(episodes: list[Episode], usage: dict) -> dict:
    n = len(episodes)
    by_type = defaultdict(list)
    for e in episodes:
        by_type[e.task.meta.get("task_type", "all")].append(e.success)
    summary = {
        "num_tasks": n,
        "success_rate": round(sum(e.success for e in episodes) / n, 4) if n else 0.0,
        "avg_steps": round(sum(e.num_steps for e in episodes) / n, 2) if n else 0.0,
        "errors": sum(e.error is not None for e in episodes),
        "by_task_type": {t: {"n": len(v), "success_rate": round(sum(v) / len(v), 4)}
                         for t, v in sorted(by_type.items())},
        "memory_errors": sum(e.memory_error is not None for e in episodes),
        "usage": usage,
    }
    judged = [e for e in episodes if e.judge]
    if judged:  # how often the memory's own success verdict matched the game's
        summary["judge_accuracy"] = round(sum(e.judge["success"] == e.success for e in judged) / len(judged), 4)
    return summary


def run_stream(env, agent, memory, out_dir: str | Path, config: dict,
               seed: int = 0, limit: int | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Shuffle with a fixed seed so every memory method sees the same task order.
    tasks = env.tasks()
    random.Random(seed).shuffle(tasks)
    tasks = tasks[:limit] if limit else tasks

    config = dict(config, task_order=[t.id for t in tasks])
    (out / "config.json").write_text(json.dumps(config, indent=2))
    ep_path = out / "episodes.jsonl"
    ep_path.write_text("")

    usage = agent.llm.usage
    episodes = []
    for task in tasks:
        before = usage.snapshot()
        try:
            ep = env.run_episode(task, agent, memory)  # calls memory.retrieve once the goal is known
        except Exception as exc:  # keep the stream going; record the failure
            ep = Episode(task=task, error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        try:
            memory.update(ep)
        except Exception as exc:  # a failed memory update shouldn't end the run
            ep.memory_error = f"update: {type(exc).__name__}: {exc}"
        ep.usage = usage.diff(before)  # acting + memory calls for this task

        episodes.append(ep)
        with ep_path.open("a") as f:
            f.write(json.dumps(ep.to_dict()) + "\n")

        sr = sum(e.success for e in episodes) / len(episodes)
        status = "ERROR" if ep.error else ("ok " if ep.success else "-- ")
        print(f"[{len(episodes):>4}/{len(tasks)}] {status} {ep.num_steps:>2} steps "
              f"{task.meta.get('task_type', ''):<8} running SR={sr:.3f}  {task.id}")
        if ep.error:
            print("       " + ep.error.strip().splitlines()[0][:200])
        if ep.memory_error:
            print("       memory " + ep.memory_error[:200])

    summary = summarize(episodes, usage.snapshot())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
