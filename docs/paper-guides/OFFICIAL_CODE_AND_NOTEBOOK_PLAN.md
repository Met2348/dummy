# ERIC-3080Ti Official Code and Notebook Plan

> Branch: `ERIC-3080Ti/paper-guides`
>
> Machine target: `ERIC-3080Ti`
>
> Date: 2026-06-08
>
> Scope: 46 anchor papers already covered by `paper/guide_*.md`.

## Goal

The paper guides now explain the papers deeply. The next learning layer is to connect each paper to code:

- pull the closest official code, benchmark, model, or dataset artifact into the matching module when it exists;
- configure the local machine so the official artifact can at least be inspected and smoke-tested;
- build one teaching-oriented Jupyter notebook per paper, focused on what a beginner should reproduce, not on blindly rerunning paper-scale experiments.

This document is the execution contract before downloading and wiring 46 external artifacts.

## Repository Layout Standard

Each module should use the same shape:

```text
learning/<module>/
  paper/                         # source paper PDF + deep guide, already done
  src/                           # repo-owned minimal implementations and tests
  official/
    README.md                    # official source status, license note, local setup
    repos/<upstream-name>/        # preferred: git submodule or shallow clone
    data/                         # optional local cache, not committed if large
  notebooks/
    01_<paper_slug>_walkthrough.ipynb
```

Rules:

- Prefer a git submodule for official GitHub code, pinned to a commit. This keeps the branch lightweight while still giving local source code.
- Do not commit large model weights, datasets, checkpoints, generated logs, or downloaded benchmark corpora.
- If the official artifact is a Hugging Face dataset or gated model, store only the source pointer and a streaming/small-sample loader.
- If no verified official code exists, create `official/README.md` that says so explicitly and teaches from `src/`.
- If the paper is safety-sensitive, official code can be inspected, but notebooks must use benign synthetic examples and must not run harmful prompt datasets.

## Reproduction Levels

`L0 - Source map only`
: No verified official code, or the paper is a survey/concept paper. Notebook reads the paper and maps concepts to local toy code.

`L1 - Mechanism reproduction`
: CPU-runnable implementation of the core formula, tensor shape, scheduler rule, evaluator, or algorithm. Target runtime: under 10 minutes.

`L2 - Official smoke test`
: Import or run a tiny official-code path with synthetic/small data. Target runtime: 10-60 minutes.

`L3 - ERIC-3080Ti GPU lab`
: A meaningful one-GPU experiment or microbenchmark on the local RTX 3080 Ti class machine. Target runtime: 30-180 minutes. Windows-native support is not assumed; many systems projects will likely need WSL2/Linux CUDA.

`L4 - Paper-scale not reproduced`
: Full paper reproduction is out of scope because it needs clusters, proprietary data, gated weights, huge data, paid APIs, or multi-day training. Notebook should explain why and reproduce the smallest honest slice.

## Notebook Teaching Standard

Use tutorial notebooks, not research dump notebooks.

Every `01_<paper_slug>_walkthrough.ipynb` should have:

1. **Learning objective**
   - One sentence: what mechanism the learner should be able to explain from memory.

2. **Paper-to-code map**
   - Point to the exact paper section/figure/table and the matching official source file or local `src/` file.

3. **Environment check**
   - Print Python, CUDA availability when relevant, package versions, and whether official repo exists locally.

4. **Minimal mechanism cell**
   - Small, runnable code that implements the core algorithm without hiding it behind a framework.

5. **Tensor/resource diagram cell**
   - Show shapes, memory sizes, request flow, label schema, or evaluation aggregation.

6. **Official-code smoke cell**
   - Run only when the official repo and dependencies are present. It must fail gracefully with a clear message.

7. **Ablation or knob**
   - One small change the learner can make: rank, beta, gamma, block size, context scale, expert capacity, judge bias, cache size, batch size, etc.

8. **Closed-book questions**
   - Three short prompts for AI-agent-assisted recall: explain, derive, and debug.

Outputs should be cleared before commit unless a tiny printed table is essential for teaching.

## Per-Paper Official Source and Notebook Design

| # | Module | Paper | Official source judgment | Teaching target | Notebook design |
|---:|---|---|---|---|---|
| 01 | `transformer-deep` | Attention Is All You Need | Historical first-party implementation: `https://github.com/tensorflow/tensor2tensor.git`. It is old TensorFlow-era code, so it should be read more than executed. | L1 + L4 | `01_attention_walkthrough.ipynb`: derive scaled dot-product attention, compare local `mha.py`/`gpt_mini.py` to Tensor2Tensor attention layout, run a tiny forward pass. |
| 02 | `dpo-family` | Direct Preference Optimization | Author code: `https://github.com/eric-mitchell/direct-preference-optimization.git`. | L2 | `01_dpo_walkthrough.ipynb`: inspect preference batch shape, implement DPO loss by hand, compare against local `dpo_minimal.py`, optional official config smoke. |
| 03 | `lora-family` | LoRA | Microsoft official code: `https://github.com/microsoft/LoRA.git`. | L2 + optional L3 | `01_lora_walkthrough.ipynb`: inject low-rank adapters into a linear layer, count trainable params, run a toy fit, compare with official `loralib`. |
| 04 | `rlhf-classic` | InstructGPT | OpenAI official repository exists for the paper family: `https://github.com/openai/following-instructions-human-feedback.git`; full InstructGPT training is not reproducible locally. | L1 + L4 | `01_instructgpt_walkthrough.ipynb`: build SFT/RM/PPO toy pipeline, inspect official data/training structure if cloned, explain what cannot be reproduced. |
| 05 | `rl-foundations` | PPO | OpenAI Baselines contains official PPO implementations: `https://github.com/openai/baselines.git`. | L2 | `01_ppo_walkthrough.ipynb`: compute clipped objective on toy advantages, run local CartPole-style PPO logic, optionally compare Baselines config naming. |
| 06 | `reasoning-r1` | DeepSeek-R1 | DeepSeek official model/report repo: `https://github.com/deepseek-ai/DeepSeek-R1.git`; no full RL training code/cluster recipe. | L1 + L4 | `01_deepseek_r1_walkthrough.ipynb`: reproduce GRPO-style advantage normalization and rule rewards, map to local `grpo_minimal.py`, discuss distillation vs full RL. |
| 07 | `kernel-engineering` | FlashAttention | Official implementation: `https://github.com/Dao-AILab/flash-attention.git`. Windows native may be difficult; WSL2/Linux CUDA likely needed. | L1 + optional L3 | `01_flashattention_walkthrough.ipynb`: prove online softmax numerically, compare tiled attention with local implementation, optional GPU benchmark if official package builds. |
| 08 | `inference-engine-core` | vLLM / PagedAttention | Official system repo: `https://github.com/vllm-project/vllm.git`. | L1 + optional L3 | `01_pagedattention_walkthrough.ipynb`: simulate block table and KV fragmentation, compare local `paged_kv.py`, optional vLLM import/server smoke under Linux CUDA. |
| 09 | `long-context` | YaRN | Author-associated code: `https://github.com/jquesnelle/yarn.git`. | L2 | `01_yarn_walkthrough.ipynb`: plot RoPE frequency scaling, inspect YaRN config knobs, run local `rope_yarn.py` passkey-style toy. |
| 10 | `moe-architecture` | Switch Transformer | Closest first-party Google stack: `https://github.com/google-research/t5x.git`; paper-scale Switch training is not local. | L1 + L4 | `01_switch_transformer_walkthrough.ipynb`: route tokens to top-1 experts, visualize capacity overflow and load-balancing loss with local `switch_router.py`. |
| 11 | `scaling-infra` | Scaling Laws for Neural Language Models | No verified OpenAI official code release for Kaplan scaling-law experiments. | L1 | `01_scaling_laws_walkthrough.ipynb`: fit a power law to synthetic curves, derive compute-optimal allocation, compare local estimator. |
| 12 | `data-curation` | The Pile | EleutherAI official data/code repo: `https://github.com/EleutherAI/the-pile.git`; full 800GB data is not committed or downloaded by default. | L2 + L4 | `01_the_pile_walkthrough.ipynb`: inspect component mixture, run small dedup/filter samples, use streaming/tiny local data only. |
| 13 | `pretraining-recipe` | The Llama 3 Herd of Models | Meta official inference/model repo: `https://github.com/meta-llama/llama3.git`; weights may be gated and training recipe is not fully open. | L1 + L4 | `01_llama3_recipe_walkthrough.ipynb`: reconstruct data/compute/tokenizer/config decisions, run local data-mixture and schedule toy code. |
| 14 | `eval-foundations` | HELM | Stanford CRFM official code: `https://github.com/stanford-crfm/helm.git`. | L2 | `01_helm_walkthrough.ipynb`: construct scenario/adaptation/metric triples, run local micro-HELM, optionally inspect HELM schema. |
| 15 | `reasoning-eval` | Training Verifiers to Solve Math Word Problems | OpenAI official GSM8K dataset repo: `https://github.com/openai/grade-school-math.git`; verifier training code is not fully released. | L2 | `01_gsm8k_verifier_walkthrough.ipynb`: parse GSM8K item shape, generate candidate answers, score with a toy verifier, compare local runner. |
| 16 | `process-reward` | Let's Verify Step by Step | OpenAI official PRM800K repo: `https://github.com/openai/prm800k.git`. | L2 | `01_prm800k_walkthrough.ipynb`: inspect step-label schema, implement PRM aggregation, compare best-of-N selection in local code. |
| 17 | `rl-sota-2026` | DAPO | Official DAPO repo verified: `https://github.com/BytedTsinghua-SIA/DAPO.git`; full RL at scale is out of local scope. | L1 + L2 + L4 | `01_dapo_walkthrough.ipynb`: implement Clip-Higher, dynamic sampling, token-level loss on tensors, inspect official configs. |
| 18 | `llm-judge-arena` | MT-Bench / Chatbot Arena | LMSYS official FastChat repo: `https://github.com/lm-sys/FastChat.git`. | L2 | `01_mt_bench_arena_walkthrough.ipynb`: build pairwise judge records, simulate position/verbosity bias, inspect FastChat eval layout. |
| 19 | `eval-graduation` | MT-Bench / Chatbot Arena | Same official FastChat source as #18; this module uses it as a graduation evaluation capstone. | L1 + L2 | `01_eval_graduation_walkthrough.ipynb`: combine judge, arena, safety, and scorecard views; reuse or cross-reference FastChat official clone. |
| 20 | `agent-code-eval` | Evaluating LLMs Trained on Code | OpenAI official HumanEval repo: `https://github.com/openai/human-eval.git`. | L2 | `01_humaneval_walkthrough.ipynb`: compute pass@k, run safe tiny candidate functions in sandboxed local tests, connect to local runners. |
| 21 | `agent-foundations` | ReAct | Author code: `https://github.com/ysymyth/ReAct.git`. | L2 | `01_react_walkthrough.ipynb`: replay thought-action-observation traces with mock tools, inspect official prompts/environments, compare local ReAct loop. |
| 22 | `tool-use-mcp` | Toolformer | No verified Meta official code release. Use local toy code; optional third-party code must be labeled non-official. | L1 | `01_toolformer_walkthrough.ipynb`: sample API-call insertion, execute toy tools, filter by future-token loss improvement. |
| 23 | `agent-memory-context` | MemGPT | Official successor project: `https://github.com/letta-ai/letta.git` for Letta/MemGPT lineage. | L2 | `01_memgpt_walkthrough.ipynb`: simulate main/archival/recall memory pressure, inspect Letta concepts, compare local virtual-context code. |
| 24 | `agent-framework-stack` | Survey on LLM-based Autonomous Agents | Survey material/list repo verified: `https://github.com/Paitesanshi/LLM-Agent-Survey.git`; not an executable paper implementation. | L0 + L1 | `01_agent_survey_walkthrough.ipynb`: turn survey taxonomy into a decision tree, map frameworks to local mocks. |
| 25 | `agent-graduation` | AI Agents That Matter | No verified official code found in this pass. Treat as evaluation-method paper, not implementation paper. | L1 | `01_agents_that_matter_walkthrough.ipynb`: reproduce accuracy-cost Pareto scoring, retry/escalation baselines, and holdout reasoning with local code. |
| 26 | `multi-agent-orchestration` | CAMEL | Official CAMEL-AI repo: `https://github.com/camel-ai/camel.git`. | L2 | `01_camel_walkthrough.ipynb`: run role-play protocol with local mock agents, inspect CAMEL role/task abstractions, add cost/termination guards. |
| 27 | `rag-essential` | Retrieval-Augmented Generation | First-party/near-first-party pieces: DPR `https://github.com/facebookresearch/DPR.git`, Hugging Face Transformers `https://github.com/huggingface/transformers.git`. | L2 | `01_rag_walkthrough.ipynb`: marginalize over retrieved docs on a toy corpus, run local BM25/dense retrieval, optionally inspect official RAG/DPR code. |
| 28 | `prompt-tuning-family` | Prefix-Tuning | Author code: `https://github.com/XiangLi1999/PrefixTuning.git`. | L2 | `01_prefix_tuning_walkthrough.ipynb`: create prefix KV tensors, compare prefix vs prompt tuning, inspect official training entrypoints. |
| 29 | `adapter-tuning-family` | Parameter-Efficient Transfer Learning for NLP | Google Research official code: `https://github.com/google-research/adapter-bert.git`. | L2 | `01_adapters_walkthrough.ipynb`: insert Houlsby adapters, count params, test near-identity behavior, compare local adapter variants. |
| 30 | `quantization-deploy` | GPTQ | Official IST-DASLab repo: `https://github.com/IST-DASLab/gptq.git`. | L2 + optional L3 | `01_gptq_walkthrough.ipynb`: solve layer-wise quantization on a tiny matrix, show Hessian compensation, optional official small model smoke. |
| 31 | `speculative-decoding` | Speculative Decoding | No verified official paper code in this pass; Google Research monorepo exists but specific paper directory still needs confirmation. | L1 | `01_speculative_decoding_walkthrough.ipynb`: implement exact accept/reject sampling and prove distribution match with a toy language model. |
| 32 | `sglang-radixattention` | SGLang | Official SGLang repo: `https://github.com/sgl-project/sglang.git`. | L2 + optional L3 | `01_sglang_walkthrough.ipynb`: build radix cache toy, grammar FSM, inspect official APIs, optional server smoke if dependencies fit. |
| 33 | `distributed-inference` | DistServe | Official repo verified: `https://github.com/LLMServe/DistServe.git`. | L1 + L2 | `01_distserve_walkthrough.ipynb`: simulate prefill/decode queues, TTFT/TPOT/goodput, inspect official placement/config code. |
| 34 | `serving-graduation` | DistServe | Same official DistServe source as #33, used here for serving scorecard/capstone. | L1 + L2 | `01_serving_graduation_walkthrough.ipynb`: connect DistServe goodput to model routing, cost, SLO, and graduation scorecard. |
| 35 | `production-serving` | Clipper | Official RISELab repo: `https://github.com/ucbrise/clipper.git`. | L2 | `01_clipper_walkthrough.ipynb`: simulate adaptive batching, cache, model selection, inspect Clipper architecture. |
| 36 | `cluster-networking` | Demystifying NCCL | Paper-specific code not verified; official implementation/source: `https://github.com/NVIDIA/nccl.git`, tests: `https://github.com/NVIDIA/nccl-tests.git`. | L1 + optional L3 | `01_nccl_walkthrough.ipynb`: simulate ring/tree collectives and protocol chunks, optional local NCCL tests only under compatible CUDA/Linux. |
| 37 | `gpu-architecture` | Roofline | No original-paper official code verified. Useful one-source tools include NERSC-style roofline examples such as `https://github.com/cyanguwa/nersc-roofline.git`, but they are not the 2009 paper's official repo. | L1 | `01_roofline_walkthrough.ipynb`: measure or synthesize FLOP/byte points, draw roofline, classify kernels from local examples. |
| 38 | `cuda-essentials` | CUDA C++ Programming Guide | NVIDIA official samples: `https://github.com/NVIDIA/cuda-samples.git`. | L2 + optional L3 | `01_cuda_guide_walkthrough.ipynb`: map thread/block/grid, memory coalescing, shared-memory bank conflicts, optional compile/run CUDA sample. |
| 39 | `training-orchestration` | Ray | Official Ray repo: `https://github.com/ray-project/ray.git`. | L2 | `01_ray_walkthrough.ipynb`: run task/actor/object-store toy, inspect scheduler concepts, compare local simulator. |
| 40 | `infra-graduation` | MLPerf Training Benchmark | MLCommons official training repo: `https://github.com/mlcommons/training.git`. | L2 + L4 | `01_mlperf_training_walkthrough.ipynb`: reproduce timing/quality-threshold aggregation rules with tiny logs, inspect official benchmark layout. |
| 41 | `storage-dataops` | Accelerating Data Loading | No verified author code found in this pass. | L1 | `01_data_loading_walkthrough.ipynb`: simulate storage bandwidth plateau, caching, and locality-aware batch reassignment with local code. |
| 42 | `ssm-hybrid` | Mamba | Official implementation: `https://github.com/state-spaces/mamba.git`. | L2 + optional L3 | `01_mamba_walkthrough.ipynb`: implement selective scan on tiny tensors, compare local `mamba_original_minimal.py`, optional official import smoke. |
| 43 | `red-team-jailbreak` | Universal Transferable Attacks | Official repo: `https://github.com/llm-attacks/llm-attacks.git`. Safety-limited: inspect code, do not run harmful target prompts. | L1 safety-limited | `01_gcg_safe_walkthrough.ipynb`: use benign target strings to explain coordinate-gradient mechanics and defense evaluation boundaries. |
| 44 | `safety-defense` | Llama Guard | Meta official PurpleLlama repo: `https://github.com/meta-llama/PurpleLlama.git`. | L2 | `01_llama_guard_walkthrough.ipynb`: inspect taxonomy/output format, run local classifier mock, optional official prompt/data smoke. |
| 45 | `multimodal-agent` | Vision-R1 | Official-looking repository verified: `https://github.com/Osilly/Vision-R1.git`; confirm author identity again during clone phase. Full training is not local. | L1 + L2 + L4 | `01_vision_r1_walkthrough.ipynb`: simulate GRPO/HFRRF/PTST tensors, inspect official configs/data scripts, avoid full MLLM training. |
| 46 | `small-model-graduation` | TinyStories | No author GitHub repo verified. Official data resources are Hugging Face datasets: `https://huggingface.co/datasets/roneneldan/TinyStories` and `https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct`. | L2 data + L1 code | `01_tinystories_walkthrough.ipynb`: stream a tiny sample, generate controlled synthetic stories, train/evaluate a tiny local model if feasible. |

## Execution Order for the Next Phase

Do not clone all 46 blindly. Work in four passes:

1. **Manifest pass**
   - Create `official/README.md` for all 46 modules.
   - Record URL, official-status judgment, target level, expected environment, and whether weights/data are gated or large.

2. **Low-risk official code pass**
   - Clone/submodule small or moderate repos first: DPO, LoRA, GSM8K, PRM800K, HumanEval, ReAct, PrefixTuning, adapter-bert, GPTQ, TinyStories HF pointers.
   - Build notebooks that run on CPU.

3. **Systems/GPU pass**
   - FlashAttention, vLLM, SGLang, DistServe, NCCL, CUDA samples, Mamba, Ray, MLPerf.
   - Prefer WSL2/Linux CUDA for packages that do not support native Windows cleanly.
   - Every notebook must have a CPU fallback cell.

4. **Heavy/gated/no-official pass**
   - Llama 3, DeepSeek-R1, InstructGPT, Switch/T5X, Pile, DAPO, Vision-R1, HELM.
   - Use official code for inspection and tiny smoke only; never pretend full reproduction happened.

## AI-Agent Learning Workflow

For each notebook, the learner should use an AI agent in a constrained way:

- Before running code: ask the agent to predict tensor shapes and failure points.
- During debugging: ask only for one hypothesis at a time, then inspect the exact traceback.
- After running: close the notebook and explain the core mechanism from memory.
- Final check: ask the agent to quiz you with three questions, then answer without looking.

The agent is used to accelerate feedback loops, not to outsource memory formation.

## Immediate Next Checklist

- [ ] Create `official/README.md` manifest files for 46 modules.
- [ ] Decide submodule vs shallow clone mechanics after checking `.gitmodules` policy.
- [ ] Scaffold 46 tutorial notebooks using the `jupyter-notebook` skill template.
- [ ] Fill and execute notebooks in the same order as the four-pass execution plan.
- [ ] Run smoke tests and document failures per module.
- [ ] Commit and push the official-code/notebook phase on the ERIC-3080Ti branch.
