"""ALFWorld (text-only TextWorld version) adapter.

Setup (Python 3.9-3.11; textworld's jericho dependency does not build on 3.12+):
    pip install alfworld
    export ALFWORLD_DATA=~/alfworld_data && alfworld-download

Note: alfworld>=0.4 uses "move X to Y" for placing objects; older versions used
"put X in/on Y". The agent sees the admissible commands, so it picks up the grammar from them.

Splits under $ALFWORLD_DATA/json_2.1.1/ ("seen" = rooms that also appear in train):
    valid_seen    140 tasks  <- SkillOS's test set (our default)
    valid_unseen  134 tasks  <- ReAct, Reflexion, Evo-Memory
    train         ~3.5k tasks
"""

from __future__ import annotations

import glob
import json
import os

from episode import Episode, Step, Task

# Directory-name prefix -> task type (six ALFWorld task families)
TASK_TYPES = {
    "pick_and_place_simple": "put",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "look_at_obj_in_light": "examine",
    "pick_two_obj_and_place": "puttwo",
}

WELCOME = "-= Welcome to TextWorld, ALFRED! =-"


class ALFWorldEnv:
    def __init__(self, split: str = "valid_seen", max_steps: int = 30,
                 data_dir: str | None = None):
        self.split = split
        self.max_steps = max_steps
        self.data_dir = data_dir or os.environ.get("ALFWORLD_DATA") or os.path.expanduser("~/.cache/alfworld")

    # ------------------------------------------------------------------ tasks
    def tasks(self) -> list[Task]:
        root = os.path.join(self.data_dir, "json_2.1.1", self.split)
        game_files = sorted(glob.glob(os.path.join(root, "*", "*", "game.tw-pddl")))
        if not game_files:
            raise FileNotFoundError(f"No ALFWorld games under {root}. Run `alfworld-download` "
                                    f"and set ALFWORLD_DATA.")
        tasks = []
        for gf in game_files:
            trial_dir = os.path.dirname(gf)
            task_dir = os.path.basename(os.path.dirname(trial_dir))
            if "movable" in task_dir or "Sliced" in task_dir:
                continue  # same filters as ALFWorld's own loader
            with open(gf) as f:
                if not json.load(f).get("solvable", False):
                    continue
            ttype = next((v for k, v in TASK_TYPES.items() if task_dir.startswith(k)), "unknown")
            task_id = f"{task_dir}/{os.path.basename(trial_dir)}"
            # The goal text is only known after reset(); fill a placeholder for now.
            tasks.append(Task(id=task_id, instruction="",
                              meta={"task_type": ttype, "game_file": gf, "split": self.split}))
        return tasks

    # ---------------------------------------------------------------- episode
    def _make_env(self, game_file: str):
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [game_file], infos, batch_size=1, asynchronous=False,
            max_episode_steps=self.max_steps + 1,  # we enforce max_steps ourselves
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        return textworld.gym.make(env_id)

    def run_episode(self, task: Task, agent, memory) -> Episode:
        env = self._make_env(task.meta["game_file"])
        try:
            obs, info = env.reset()
            obs = obs[0].replace(WELCOME, "").strip()
            ep = Episode(task=task, initial_observation=obs)
            # Split the opening text into scene description and goal.
            if "Your task is to:" in obs:
                scene, goal = obs.split("Your task is to:", 1)
                task.instruction = goal.strip()
                obs = scene.strip()
            admissible = info["admissible_commands"][0]

            # Retrieve only now: the goal text is generated when the game starts.
            try:
                ep.memory_context = memory.retrieve(task)
            except Exception as exc:  # run the task without memory rather than lose it
                ep.memory_error = f"retrieve: {type(exc).__name__}: {exc}"

            for _ in range(self.max_steps):
                thought, action, raw = agent.act(ep, obs, admissible)
                if not action:
                    new_obs, done, won = "Invalid response: no action found.", False, False
                else:
                    o, _, d, info = env.step([action])
                    new_obs, done, won = o[0].strip(), bool(d[0]), bool(info["won"][0])
                    admissible = info["admissible_commands"][0]
                ep.steps.append(Step(thought=thought, action=action, observation=new_obs, raw_output=raw))
                obs = new_obs
                if won:
                    ep.success = True
                    break
                if done:
                    break
            return ep
        finally:
            env.close()
