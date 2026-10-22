"""Run one memory method over the ALFWorld task stream, with the SkillOS agent.

Defaults match SkillOS's "No Memory" setup: 140 valid_seen tasks, 30 steps max.

Examples:
  # free smoke test (random agent, no API key needed)
  python run.py --model fake --limit 5 --out runs/smoke

  # no-memory baseline
  python run.py --model gemini/gemini-3.8-flash --out runs/g38_none

  # same stream with a memory method
  python run.py --model gemini/gemini-3.8-flash --memory exp_recent --out runs/g38_exprecent
"""

import argparse
import json

from agent import SkillOSAgent, make_llm
from benchmarks import ALFWorldEnv
from memory import ExpRecent, NoMemory, ReasoningBank
from runner import run_stream

# Add your memory method here so `--memory <name>` finds it.
MEMORIES = {
    "none": NoMemory,
    "exp_recent": ExpRecent,
    "reasoningbank": ReasoningBank,  # --memory-args '{"variant": "skillos"|"paper", "judge": "llm"|"gt"}'
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="litellm model string, e.g. gemini/gemini-3.8-flash, or 'fake' for an offline smoke test")
    p.add_argument("--memory", default="none", choices=sorted(MEMORIES))
    p.add_argument("--memory-args", default="{}", help='JSON kwargs, e.g. \'{"k": 3}\'')
    p.add_argument("--split", default="valid_seen", choices=["valid_seen", "valid_unseen", "train"])
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0, help="task order; keep fixed across methods")
    p.add_argument("--limit", type=int, default=None, help="only run the first N tasks")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    llm = make_llm(args.model)
    agent = SkillOSAgent(llm)
    env = ALFWorldEnv(split=args.split, max_steps=args.max_steps)
    memory = MEMORIES[args.memory](**json.loads(args.memory_args))
    memory.llm = llm  # memory methods that call a model use the agent's (as in ReasoningBank and SkillOS)

    summary = run_stream(env, agent, memory, args.out, config=vars(args),
                         seed=args.seed, limit=args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
