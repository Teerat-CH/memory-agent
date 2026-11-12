"""ReasoningBank (Ouyang et al., ICLR 2026; arXiv 2509.25140), on ALFWorld.

Sources, cited as tags in the comments below:
  [RB-paper]  arXiv 2509.25140, Section 3.2 and Appendix A
  [RB-code]   github.com/google-research/reasoning-bank, WebArena/induce_memory.py,
              WebArena/memory_management.py, WebArena/prompts/memory_instruction.py,
              WebArena/agents/legacy/agent.py
  [SkillOS]   arXiv 2605.06614, Figure 13 (ALFWorld judge prompt) and Appendix B
              (all methods retrieve top-5 items with BM25)
ReasoningBank itself was only evaluated on web and SWE tasks, so a few details have no
ALFWorld source. They are marked ADAPTED.

Three variants (--memory-args '{"variant": ...}'):
  "skillos"       BM25 over memory items, top 5 [SkillOS]. Items shown as title + content.
  "skillos_task"  BM25 over past task instructions; the best-matching tasks' items, up to 5 [SkillOS
                  top-5 BM25; RB-paper retrieval by task]. Items shown SKILL.md-style, with description.
  "paper"         embed each task with gemini-embedding-001, cosine similarity, take all items of the
                  top-1 most similar past task, plus ReasoningBank's memory instruction [RB-paper A.2, RB-code].

Per task:
  retrieve  as above
  update    1. judge: the agent's LLM labels the trajectory success/failure at temperature 0
               [RB-paper A.2], with SkillOS's ALFWorld judge prompt [SkillOS Fig. 13].
               judge="gt" uses ALFWorld's won flag instead, an oracle [RB-code --criteria gt].
            2. extract: at most 3 items {title, description, content} with the success or failure
               prompt at temperature 1.0, the judge's rationale appended [RB-code induce_memory.py].
               The trajectory shown is observations and actions, as SkillOS defines it [SkillOS 3.1]
               (ADAPTED: RB-code shows thoughts and actions).
            3. consolidate: append, no pruning [RB-paper A.2].
"""

from __future__ import annotations

import json
import math
import re

from episode import Episode, Task

from .base import Memory

# [RB-code] prompts/memory_instruction.py SUCCESSFUL_SI / FAILED_SI.
# ADAPTED: "web navigation" -> "household tasks in a text-based simulator", and
# "product names" -> "object names". Everything else is verbatim.
SUCCESSFUL_SI = """
You are an expert in household tasks in a text-based simulator. You will be given a user query, the corresponding trajectory that represents **how an agent successfully accomplished the task**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable procedures over abstract principles. Do not embed specific object names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing similar tasks in the future>
```
"""

FAILED_SI = """
You are an expert in household tasks in a text-based simulator. You will be given a user query, the corresponding trajectory that represents **how an agent attempted to resolve the task but failed**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable recovery procedures over abstract principles. Do not embed specific object names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to avoid such failures and successfully accomplishing similar tasks in the future>
```
"""

# [SkillOS] Figure 13, verbatim.
JUDGE_SI = """You are an expert judge evaluating whether an embodied agent successfully completed a household task in a text-based simulator. Output a single JSON object and nothing else.

# Task
You will be given (1) the task description the agent was asked to complete, and (2) the full interaction trace between the agent and the simulator. Determine whether the agent fully completed the task.

## What "success" means
- The agent's actions must have produced the world state the task description specifies. Every condition stated in the task must hold at the end of the trace.
- If the task implies a transformation must occur before a final placement or interaction, the transformation must be evidenced in the trace before the final step.
- Credit only effects that the simulator's observations confirm. Do not credit effects that the agent merely declared, planned, or assumed.
- Ignore the agent's own claims of completion; rely solely on the simulator's observation strings.
- A trace that ends with the agent stuck in a loop, exhausting its step budget, or repeatedly emitting invalid actions is a failure regardless of partial progress.

## Strictness
- If the trace is ambiguous about whether every required condition is satisfied at the end, output success=false.
- Partial completion is failure. Either every condition holds or the trace is a failure.

# Output
Output exactly one JSON object with these fields, and nothing else: { "success": <true|false>, "rationale": "<one or two sentences citing the specific observations that prove success or failure>", "evidence_step": <integer step index where success was confirmed, or -1 if failure> }"""

JUDGE_INPUT = """# Inputs
## Task description
{task_description}
## Trajectory
The trajectory alternates between simulator OBSERVATION and agent ACTION.
{trajectory}"""

# [RB-code] agents/legacy/agent.py: put before the retrieved items ("paper" variant only;
# SkillOS's executor prompt has its own "with the help of past relevant skills" wording).
RB_INSTRUCTION = ("Below are some memory items that I accumulated from past interaction from the environment "
                  "that may be helpful to solve the task. You can use it when you feel it's relevant. In each "
                  "step, please first explicitly discuss if you want to use each memory item or not, and then "
                  "take action.")

# [RB-code] memory_management.py screening(). ADAPTED: "web navigation" -> "household task".
EMBED_INSTRUCTION = ("Given the prior household task queries, your task is to analyze a current query's intent "
                     "and select relevant prior queries that could help resolve it.")

VARIANTS = {
    "skillos": {"retrieval": "bm25", "k": 5},            # k = memory items
    "skillos_task": {"retrieval": "bm25_task", "k": 5},  # k = memory items, taken task by task
    "paper": {"retrieval": "embedding", "k": 1},         # k = past tasks, all of whose items are returned
}

ITEM_RE = re.compile(r"##\s*Title\s*(.*?)\s*##\s*Description\s*(.*?)\s*##\s*Content\s*(.*?)\s*(?=#\s*Memory Item|```|\Z)",
                     re.S | re.I)


class ReasoningBank(Memory):
    def __init__(self, variant: str = "skillos", k: int | None = None, judge: str = "llm",
                 embed_model: str = "gemini/gemini-embedding-001"):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        if judge not in ("llm", "gt"):
            raise ValueError("judge must be 'llm' or 'gt'")
        self.variant = variant
        self.retrieval = VARIANTS[variant]["retrieval"]
        self.k = k if k is not None else VARIANTS[variant]["k"]
        self.judge = judge
        self.embed_model = embed_model
        # One entry per finished task [RB-paper A.2 "ReasoningBank Storage"]:
        # {task_id, query, status, rationale, items: [{title, description, content}], embedding}
        self.entries: list[dict] = []
        self._retrieved_from: list[str] = []  # task_ids of the items shown for the current task

    # ------------------------------------------------------------------ retrieve
    def retrieve(self, task: Task) -> str:
        if self.retrieval == "bm25":
            pairs = self._bm25(task.instruction)
        elif self.retrieval == "bm25_task":
            pairs = self._bm25_task(task.instruction)
        else:
            pairs = self._by_embedding(task.instruction)
        self._retrieved_from = [e["task_id"] for e, _ in pairs]
        if not pairs:
            return ""
        if self.variant == "skillos_task":
            # ADAPTED: SkillOS gives every method "the same memory interface" [SkillOS B.3.2], and its
            # skills are SKILL.md files: a name + description frontmatter, then the body [SkillOS 3.1, Fig. 7].
            text = "\n\n".join(f"---\nname: {it['title']}\ndescription: {it['description']}\n---\n{it['content']}"
                               for _, it in pairs)
        else:
            # [RB-paper A.2] "each item represented by its title and content".
            text = "\n\n".join(f"### {it['title']}\n{it['content']}" for _, it in pairs)
        return f"{RB_INSTRUCTION}\n\n{text}" if self.variant == "paper" else text

    def _bm25(self, query: str) -> list[tuple[dict, dict]]:
        # [SkillOS] top-k BM25, as rank_bm25's get_top_n (no score cutoff).
        # ADAPTED: documents are each item's title + description + content; the query is the task instruction.
        from rank_bm25 import BM25Okapi

        pairs = [(e, it) for e in self.entries for it in e["items"]]
        if not pairs:
            return []
        bm25 = BM25Okapi([_tok(f"{it['title']} {it['description']} {it['content']}") for _, it in pairs])
        return bm25.get_top_n(_tok(query), pairs, n=self.k)

    def _bm25_task(self, query: str) -> list[tuple[dict, dict]]:
        # [SkillOS] top-k BM25 with k = 5 items; [RB-paper A.2, RB-code] retrieval by past-task similarity.
        # ADAPTED: documents are past task instructions; items are taken from the best-matching tasks,
        # all of a task's items at a time, until k are collected. Ties keep the earlier task first.
        from rank_bm25 import BM25Okapi

        pool = [e for e in self.entries if e["items"]]
        if not pool:
            return []
        scores = BM25Okapi([_tok(e["query"]) for e in pool]).get_scores(_tok(query))
        ranked = [e for _, e in sorted(zip(scores, pool), key=lambda p: -p[0])]
        pairs = []
        for e in ranked:
            pairs.extend((e, it) for it in e["items"])
            if len(pairs) >= self.k:
                break
        return pairs[: self.k]

    def _by_embedding(self, query: str) -> list[tuple[dict, dict]]:
        # [RB-code] memory_management.py: past queries are embedded as documents; the current
        # query is embedded with an instruction prefix; rank by cosine; top-k tasks' items.
        pool = [e for e in self.entries if e["items"] and e.get("embedding")]
        if not pool:
            return []
        q = self.llm.embed([f"Instruct: {EMBED_INSTRUCTION}\nQuery: {query}"], model=self.embed_model)[0]
        ranked = sorted(pool, key=lambda e: _cosine(q, e["embedding"]), reverse=True)
        return [(e, it) for e in ranked[: self.k] for it in e["items"]]

    # -------------------------------------------------------------------- update
    def update(self, episode: Episode) -> None:
        query = episode.task.instruction

        # 1. Correctness signal.
        if self.judge == "gt":
            success, rationale = episode.success, ""
        else:
            success, rationale = self._judge(episode)
        episode.judge = {"success": success, "rationale": rationale, "source": self.judge,
                         "retrieved_from": self._retrieved_from}

        # 2. Extraction. ADAPTED: RB-code (induce_memory.py) passes think/action pairs without observations,
        # because WebArena's are huge. We use SkillOS's trajectory xi = {o1, a1, ..., on, an} [SkillOS 3.1]:
        # ALFWorld's observations are short and are the only record of what each action did.
        parts = [f"<observation>\n{episode.initial_observation}\n</observation>"]
        for s in episode.steps:
            if s.action:
                parts.append(f"<action>\n{s.action}\n</action>\n<observation>\n{s.observation}\n</observation>")
        trajectory = "\n\n".join(parts)
        prompt = f"**Query:** {query}\n\n**Trajectory:**\n{trajectory}"
        if rationale:
            prompt += f"\n\nThe task {'succeeded' if success else 'failed'} because: {rationale}"
        raw = self.llm.chat([{"role": "system", "content": SUCCESSFUL_SI if success else FAILED_SI},
                             {"role": "user", "content": prompt}], temperature=1.0)
        items = [{"title": t.strip(), "description": d.strip(), "content": c.strip()}
                 for t, d, c in ITEM_RE.findall(raw)][:3]
        episode.judge.update(items=items, extraction_raw=raw)  # saved to episodes.jsonl for auditing

        # 3. Consolidation: append [RB-paper A.2].
        entry = {"task_id": episode.task.id, "query": query, "status": "success" if success else "fail",
                 "rationale": rationale, "items": items, "raw": raw}
        if self.retrieval == "embedding":
            entry["embedding"] = self.llm.embed([query], model=self.embed_model)[0]
        self.entries.append(entry)

    def _judge(self, episode: Episode) -> tuple[bool, str]:
        # ADAPTED: SkillOS doesn't show how the trace is written out; this numbers each step.
        lines = [f"[Step 0] OBSERVATION: {episode.initial_observation}"]
        for i, s in enumerate(episode.steps, 1):
            lines.append(f"[Step {i}] ACTION: {s.action}")
            lines.append(f"[Step {i}] OBSERVATION: {s.observation}")
        raw = self.llm.chat([{"role": "system", "content": JUDGE_SI},
                             {"role": "user", "content": JUDGE_INPUT.format(
                                 task_description=episode.task.instruction, trajectory="\n".join(lines))}],
                            temperature=0.0)
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            verdict = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            verdict = {}
        if "success" not in verdict:  # unparseable: fall back to the field alone, else failure (strictness)
            f = re.search(r'"success"\s*:\s*(true|false)', raw, re.I)
            verdict = {"success": bool(f and f.group(1).lower() == "true"), "rationale": raw.strip()[:500]}
        return bool(verdict["success"]), str(verdict.get("rationale", ""))


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
