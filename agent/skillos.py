"""The ALFWorld executor from SkillOS (arXiv 2605.06614).

SkillOS takes its ALFWorld prompt and environment from GiGPO's code (verl-agent);
these parts are copied from there:
    prompt     agent_system/environments/prompts/alfworld.py (one user message, no system prompt)
    history    agent_system/memory/memory.py (last 3 steps, "[Observation N: '...', Action N: '...']")
    parsing    agent_system/environments/env_package/alfworld/projection.py
SkillOS settings (its Table 4): max 30 turns, history length 3, split valid_seen.
"""

from __future__ import annotations

import re

from episode import Episode

TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

# SkillOS Figure 9: the executor prompt when retrieved skills are present. It only shows the
# with-history form; at step 1 we fill it with 0 steps and an empty history (not in the source).
TEMPLATE_SKILLS = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
## Past Relevant Skills
{retrieved_skills}
## Current Progress
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: {admissible_actions}
Now it's your turn to take an action.
You should first reason step-by-step about the current situation with the help of past relevant skills. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and MUST present it within <action> </action> tags.
"""

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


class SkillOSAgent:
    def __init__(self, llm, history_length: int = 3):
        self.llm = llm
        self.history_length = history_length

    def build_messages(self, ep: Episode, obs: str, admissible: list[str]) -> list[dict]:
        actions = "\n ".join(f"'{a}'" for a in admissible if a != "help")
        # Each action is paired with the observation it was taken from.
        seen = [ep.initial_observation] + [s.observation for s in ep.steps[:-1]]
        start = max(0, len(ep.steps) - self.history_length)
        lines = [f"[Observation {i + 1}: '{seen[i]}', Action {i + 1}: '{ep.steps[i].action}']"
                 for i in range(start, len(ep.steps))]
        current = obs if ep.steps else ep.initial_observation
        fields = dict(task_description=ep.task.instruction, step_count=len(ep.steps),
                      history_length=len(lines), action_history="\n".join(lines),
                      current_step=len(ep.steps) + 1, current_observation=current,
                      admissible_actions=actions)
        if ep.memory_context.strip():
            prompt = TEMPLATE_SKILLS.format(retrieved_skills=ep.memory_context.strip(), **fields)
        elif not ep.steps:
            prompt = TEMPLATE_NO_HIS.format(current_observation=current, admissible_actions=actions)
        else:
            prompt = TEMPLATE.format(**fields)
        return [{"role": "user", "content": prompt}]

    def act(self, ep: Episode, obs: str, admissible: list[str]) -> tuple[str, str, str]:
        """Choose the next action. Returns (thought, action, raw_output)."""
        raw = self.llm.chat(self.build_messages(ep, obs, admissible))
        low = raw.lower()
        m = ACTION_RE.search(low)
        # Like verl-agent: without <action> tags, the last 30 characters are sent as the action.
        action = m.group(1).strip() if m else low[-30:].strip()
        t = THINK_RE.search(raw)
        thought = t.group(1).strip() if t else ""
        return thought, action, raw
