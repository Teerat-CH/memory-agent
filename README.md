# memagent

A small harness for agents that learn from a **stream of tasks**. For each task it does three things: retrieve memory, act, then update memory. The agent is fixed (SkillOS's) and the memory method is pluggable, so baselines and your own methods run under identical conditions. The model is a flag.

Currently implemented: **ALFWorld** (text-only), the **SkillOS** agent setup, and the memories **`none`**, **`exp_recent`** and **`reasoningbank`**. The defaults reproduce the "No Memory" setup from [SkillOS](https://arxiv.org/abs/2605.06614).

## The core idea

The runner calls the memory in a loop (`runner.py`):

```
for task in stream:
    episode = env.run_episode(task, agent, memory)  # starts the game, calls memory.retrieve(task),
                                                    # then the agent plays the task
    memory.update(episode)                          # learn from the attempt
```

`retrieve()` is called right after the game starts, because ALFWorld only reveals the goal text then. Inside `run_episode`, the agent is called once per step and returns one action, which runs in the game. The memory text goes into the prompt under `## Past Relevant Skills`, using SkillOS's executor prompt for retrieved skills (its Figure 9). Nothing else about the agent changes between memory methods, so any difference in results comes from the memory.

## Setup

```bash
# Python 3.11 recommended. textworld's `jericho` dependency does NOT build on 3.12+.
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

alfworld-download          # game data, to ~/.cache/alfworld (or set ALFWORLD_DATA)

export GEMINI_API_KEY=...  # or the key for whichever provider --model uses
```

## Run

```bash
# Free smoke test: random agent, no API key, checks everything is wired up
python run.py --model fake --limit 5 --out runs/smoke

# No-memory baseline (SkillOS setup: 140 valid_seen tasks, 30 steps max)
python run.py --model gemini/gemini-3.8-flash --out runs/g38_none

# Same task stream with a memory method
python run.py --model gemini/gemini-3.8-flash --memory exp_recent --memory-args '{"k": 2}' --out runs/g38_exprecent
```

`--model` is any [litellm](https://docs.litellm.ai/docs/providers) model string, e.g. `gemini/...`, `anthropic/...`, `openrouter/...` or `ollama_chat/...` for a local model (see [Local models](#local-models-qwen3-via-ollama)). Other flags:

- `--limit N`: run only the first N tasks.
- `--seed`: sets the task order. Keep it fixed across methods.
- `--max-steps`: the step limit per task (default 30, as in SkillOS).
- `--temperature`: the agent's sampling temperature (default 0). Google [recommends 1.0](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/gemini-3-prompting-guide) for Gemini 3 models and warns that lower values can cause looping or worse reasoning. Memory methods set their own temperatures (ReasoningBank: judge 0, extractor 1.0).
- `--thinking`: `minimal`, `low`, `medium` or `high`, sent as litellm's `reasoning_effort`. Omit it to use the model's default, which for Gemini-3.1-Flash-Lite is close to no thinking. For Ollama models it is on/off: `low`/`medium`/`high` turn thinking on, `minimal` turns it off.
- `--max-tokens`: reply length limit per call (default 4096). Thinking counts toward it, so raise it with `--thinking` (e.g. 16384); `summary.json` counts cut-off replies as `truncated`.
- `--llm-args`: extra litellm arguments as JSON, e.g. `'{"top_p": 0.95, "top_k": 20}'`.
- `--split`: `valid_seen` (default, 140 tasks, as in SkillOS) or `valid_unseen` (134 tasks, as in ReAct and Evo-Memory). "Seen" means the rooms also appear in the training split; with no training, both are just test sets.
- `--memory`: pick by name from `MEMORIES` in `run.py`.

Each run writes these files to `runs/<name>/`:

| File | Contents |
|---|---|
| `config.json` | All arguments plus the exact task order |
| `episodes.jsonl` | One `Episode` per task: every step's thought, action, observation, raw model reply and native thinking (`reasoning`, for models that return it separately, such as Qwen3 on Ollama), plus success, tokens and the memory text shown |
| `summary.json` | Success rate overall and per task type, average steps, total tokens, cost and seconds spent in model calls (acting and memory calls together), and `judge_accuracy` if the memory judges its own episodes |

A run starts over if it's interrupted; there is no resume. The program prints progress but writes no log file, so redirect it: `python run.py ... > runs/<name>.log 2>&1`.

## Local models (Qwen3 via Ollama)

SkillOS also reports Qwen3-8B and Qwen3-32B executors (its Table 1). To run Qwen3-8B on your own machine with [Ollama](https://ollama.com):

```bash
brew install ollama
OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_FLASH_ATTENTION=1 ollama serve   # in its own terminal
ollama pull qwen3:8b-q8_0                                          # ~9 GB; qwen3:8b-fp16 is ~16 GB

nohup caffeinate -i python -u run.py --model ollama_chat/qwen3:8b-q8_0 \
  --thinking high --temperature 0.6 --max-tokens 8192 \
  --llm-args '{"top_p": 0.95, "top_k": 20}' \
  --out runs/qwen3_8b_q8_think > runs/qwen3_8b_q8_think.log 2>&1 &
```

- Use the `ollama_chat/` prefix. It returns Qwen's thinking separately from the reply, so the `<action>` parser only sees the reply, and the thinking is saved per step as `reasoning`.
- The sampling values are [Qwen3's recommended settings](https://huggingface.co/Qwen/Qwen3-8B) for thinking mode. With thinking off (`--thinking minimal`), Qwen recommends temperature 0.7, top_p 0.8, top_k 20.
- Ollama's default context is shorter than thinking needs and it silently cuts the prompt, hence `OLLAMA_CONTEXT_LENGTH`.
- Ollama has no bf16 Qwen3; `fp16` is the closest to the paper's likely bf16 weights, and `q8_0` is about twice as fast with little loss.
- Speed on an M4 Max (36 GB): fp16 measured about 10.6 tokens/s, or 45–75 s per step with thinking, so a full 140-task run takes about 2 days; q8_0 should take about half that. `nohup` keeps the run alive when the terminal closes and `caffeinate -i` stops the Mac from sleeping.
- Qwen3-32B only fits on a 36 GB Mac at 4-bit; use a hosted endpoint (e.g. `openrouter/qwen/qwen3-32b`) to stay closer to the paper.

## Files

| File | Role |
|---|---|
| `run.py` | Command-line entry point and the `MEMORIES` name list |
| `runner.py` | The stream loop and the summary |
| `episode.py` | `Task`, `Step`, `Episode` dataclasses |
| `agent/skillos.py` | The agent: SkillOS's prompt, 3-step history and `<think>`/`<action>` format |
| `agent/llm.py` | litellm wrapper with token and cost tracking, plus `FakeLLM` for offline tests |
| `memory/` | `Memory` base class, `NoMemory`, `ExpRecent`, `ReasoningBank` |
| `benchmarks/alfworld.py` | The environment: ALFWorld's task list and episode loop |

## Writing a memory method

Subclass `Memory`, implement `retrieve()` and `update()`, and add the class to `MEMORIES` in `run.py`. `memory/exp_recent.py` is a 40-line example.

`update()` receives the whole `Episode`, so use as much or as little as you want:

- `episode.success` and `episode.task` give the result only.
- `episode.steps` or `episode.transcript()` give the full trajectory.
- `episode.task.meta["task_type"]` gives the ALFWorld task family.

```python
from memory import Memory

class MyMemory(Memory):
    def __init__(self):
        self.items = []

    def retrieve(self, task):
        return "\n".join(self.items[-5:])

    def update(self, episode):
        if episode.success:
            self.items.append(f"Solved: {episode.task.instruction}")
```

If your memory needs a model (a judge, an extractor), use `self.llm`: `run.py` sets it to the agent's LLM, so its tokens count toward the run's usage.

**Success signal.** `episode.success` is ALFWorld's ground-truth `won` flag. SkillOS and ReasoningBank judge success with an LLM instead, so their memory never sees ground truth. For a faithful comparison, have your memory call a judge in `update()` and store its verdict in `episode.judge`; the summary then reports `judge_accuracy`. Report a ground-truth variant separately as an oracle.

## ReasoningBank

`memory/reasoningbank.py` implements [ReasoningBank](https://arxiv.org/abs/2509.25140) (ICLR 2026). After each task, the agent's model judges the trajectory, extracts up to 3 memory items (title, description, content) with a success or failure prompt, and appends them. The file header lists the source of every piece: the paper, the [official code](https://github.com/google-research/reasoning-bank), and SkillOS's ALFWorld judge prompt. ReasoningBank was only evaluated on web and SWE tasks, so the few ALFWorld-specific choices are marked `ADAPTED` in the code.

```bash
# SkillOS's version: BM25 top-5 items. Compare with SkillOS's ReasoningBank row (66.0% on Gemini-3.1-Flash-Lite)
python run.py --model gemini/gemini-3.1-flash-lite --memory reasoningbank --out runs/g31lite_rb

# The original paper's version: gemini-embedding-001, items of the top-1 most similar past task
python run.py --model gemini/gemini-3.1-flash-lite --memory reasoningbank --memory-args '{"variant": "paper"}' --out runs/g31lite_rb_paper

# Oracle: use ALFWorld's true result instead of the LLM judge
python run.py --model gemini/gemini-3.1-flash-lite --memory reasoningbank --memory-args '{"judge": "gt"}' --out runs/g31lite_rb_gt
```

The `paper` variant calls the Gemini embedding API, so it needs `GEMINI_API_KEY` even when `--model` is another provider.

## Reference results (SkillOS, 140 valid_seen tasks, 30 steps)

| Executor | No Memory | ReasoningBank | Avg steps (No Memory) |
|---|---|---|---|
| Qwen3-8B | 47.9% | 55.7% | 21.1 |
| Qwen3-32B | 54.5% | 61.4% | 20.3 |
| Gemini-3.1-Flash-Lite | 61.2% | 66.0% | 18.5 |
| Gemini-2.5-Pro | 66.4% | 71.4% | 17.7 |

From SkillOS Table 1 and Table 6, averaged over 3 runs. In Table 1 (Qwen and Gemini-2.5-Pro rows), ReasoningBank's memories were extracted by Qwen3-8B (the "curator"), not by the executor; Table 6 (Flash-Lite) doesn't say. Our implementation extracts with the agent's own model, as the ReasoningBank paper does, so this may differ from SkillOS's runs. Average steps include failed tasks, which use all 30 steps.

SkillOS does not report thinking or sampling settings for its executors. Our Gemini-3.1-Flash-Lite No Memory runs:

| Setting | Success | Avg steps | Cost |
|---|---|---|---|
| Default thinking (close to none), temperature 0 | 52.1% | 20.5 | $1.02 |
| Default thinking, temperature 1.0 | 53.6% | 20.5 | $1.13 |
| `--thinking high --temperature 1.0 --max-tokens 16384` | 60.0% | 19.1 | $3.73 |

With thinking high the overall rate matches the paper's 61.2%, but per task type Clean is lower (44.4% vs 67.9%) and Cool higher (60.0% vs 38.7%), from a single run. An independent reproduction of SkillOS ([belt-sh/skillos](https://github.com/belt-sh/skillos), DIVERGENCES.md #13) got 39–41% for the Qwen3-8B No Memory baseline (paper 47.9%) and 53.6% for Qwen3-32B (paper 54.5%), and found that thinking on or off made no difference for Qwen.
