# Claim-Evidence Matrix

This matrix records what each central paper establishes and which new claim it
blocks or enables. It is a novelty audit, not a generic summary.

| Paper | Established evidence | Consequence for our idea search |
|---|---|---|
| [AI Agents That Matter](https://arxiv.org/abs/2407.01502) | agent evaluation should include cost, holdouts, standardization, and overfitting controls | broad “agents overfit benchmarks” is not new; an adaptive-harness paper needs a new sequential protocol and estimator |
| [HAL](https://arxiv.org/abs/2510.11977) | standardized large-scale model/scaffold/benchmark evaluation with 21,730 rollouts | supplies evidence and traces; blocks claims that no system-level infrastructure exists |
| [Stop Comparing LLM Agents](https://arxiv.org/abs/2605.23950) | controlled model-harness variance decomposition and ranking reversals | kills the simple factorial attribution paper; explicitly leaves causal mechanisms and perturbation tests open |
| [Harness-Bench](https://arxiv.org/abs/2605.27922) | complete-configuration diagnostics over 5,194 trajectories | blocks another native-harness comparison; explicitly says results are not causal decompositions |
| [General Agent Evaluation](https://arxiv.org/abs/2602.22953) | full factorial across 5 architectures, 5 backbones, and 6 benchmarks; backbone dominates overall | contradicts universal harness-dominance claims and motivates boundary/transport analysis |
| [Agent Psychometrics](https://arxiv.org/abs/2604.00594) | IRT decomposition into LLM and scaffold ability; predicts unseen combinations when parts were observed | blocks a basic latent-ability decomposition; enables comparison with causal transport |
| [Efficient Benchmarking](https://arxiv.org/abs/2603.23749) | rank fidelity can survive large task-subset reductions under scaffold shift | any new evaluation suite should consider a compact subset and ranking fidelity |
| [STABLEVAL](https://arxiv.org/abs/2605.02122) | uncertainty-aware, disagreement-aware ranking stability | supplies stable evaluation machinery; does not address adaptive harness selection |
| [Cross-Component Interference](https://arxiv.org/abs/2605.05716) | full `2^5` component factorial, Shapley values, non-submodularity, greedy failure | blocks static component-Shapley work; explicitly leaves mechanism unresolved and motivates active selection |
| [AgentSpec](https://arxiv.org/abs/2606.14674) | controlled component composition and compatibility interactions in embodied agents | reinforces that component interactions are known; transport and time-varying effects remain open |
| [Inside the Scaffold](https://arxiv.org/abs/2604.03515) | source-code taxonomy across 13 coding-agent scaffolds | supplies intervention vocabulary; blocks another descriptive taxonomy |
| [Meta-Harness](https://arxiv.org/abs/2603.28052) | outer-loop search over executable harness code; some held-out-model transfer | blocks “automatically search harness code” as novelty |
| [AHE](https://arxiv.org/abs/2604.25850) | trace-driven autonomous harness evolution and cross-family transfer | admits portability is confounded with operating-point coupling; directly motivates transportability |
| [Self-Harness](https://arxiv.org/abs/2606.09498) | model-specific self-improvement with repeatedly used regression gates | motivates an adaptive-validity audit; its gate is not an untouched final test |
| [Life-Harness](https://arxiv.org/abs/2605.22166) | deterministic environment-side interventions transfer from one source model to 17 others | strong positive transfer evidence, but limited to stable rule-governed interfaces |
| [Harness Updating Is Not Harness Benefit](https://arxiv.org/abs/2605.30621) | update quality is flat across tiers while harness benefit is non-monotone | blocks monotone capability stories and motivates moderator-based transport models |
| [AdaCoM](https://arxiv.org/abs/2605.30785) | learned context manager transfers best among similarly capable agents | supplies a concrete capability-similarity moderator for transport |
| [Adaptive Auto-Harness](https://arxiv.org/abs/2606.01770) | harness construction and routing for open-ended shifting streams | blocks “handle distribution shift with a harness tree” as novelty; evaluation validity remains separate |
| [RHO](https://arxiv.org/abs/2606.05922) | label-free retrospective harness optimization using self-preference | adds another evolver baseline and highlights proxy-selection risk |
| [TTHE](https://arxiv.org/abs/2607.08124) | unlabeled test-time harness evolution; transductive gains; proxy failures | explicitly leaves next-batch prequential scoring to future work; strongest opening for Idea 1 |
| [Offline Harness Control](https://arxiv.org/abs/2607.05458) | Harness MDP and offline RL controller | blocks a generic learned dynamic controller; causal identification at decision points remains open |
| [HarnessFix](https://arxiv.org/abs/2606.06324) | trace IR, harness-layer diagnosis, repair operators, held-out gains | blocks diagnosis-plus-patching as novelty; offers a reproducible evolution baseline |
| [`tau`-bench](https://arxiv.org/abs/2406.12045) | deterministic state grading and `pass^k` reliability | suitable task surface and reliability metric for transport/MRT experiments |
| [Beyond pass@1](https://arxiv.org/abs/2603.29231) | long-horizon reliability needs more than one-run success | reinforces repeated trials and trajectory-level uncertainty |
| [Long-Horizon Task Mirage](https://arxiv.org/abs/2604.11978) | cross-domain long-horizon failure trajectories and attribution | supplies failure states; blocks another broad failure taxonomy |
| [StaminaBench](https://arxiv.org/abs/2606.19613) | stress tests coding agents over 100 interaction turns | candidate external-validity surface for sequential effects |
| [HarnessAudit](https://arxiv.org/abs/2605.14271) | trajectory-level safety audit over harness configurations | blocks output-only safety evaluation; supports layer-specific corruptions |
| [Harness-Induced Belief Divergence](https://arxiv.org/abs/2607.04528) | alternative harness views change elicited multi-step beliefs under fixed model/task | nearest collision for stability work; Harness-C must use standardized corruptions and severity curves |
| [Effective Feedback Compute](https://arxiv.org/abs/2605.29682) | matched-budget feedback quality predicts success better than raw expenditure | kills a raw equal-budget paper; resource controls remain necessary in every experiment |
| [Equal-Token Single vs Multi-Agent](https://arxiv.org/abs/2604.02460) | multi-agent gains can vanish when reasoning tokens are matched | reinforces budget confounding and paired resource accounting |
| [Governance Decay](https://arxiv.org/abs/2606.22528) | compaction can erase constraints and create later tool violations | useful corruption/intervention primitive; component-specific paper slot is occupied |
| [Plans Don't Persist](https://arxiv.org/abs/2606.22953) | plan information is context-resident and decays after actions | motivates state features for context-refresh interventions |
| [Proactive Memory Agent](https://arxiv.org/abs/2607.08716) | selective reminder injection beats passive and always-on memory | blocks a generic “inject memory when needed” algorithm; provides an MRT treatment primitive |
| [Causal Memory Intervention](https://arxiv.org/abs/2605.17641) | selects memories by estimated causal usefulness | blocks causal language around memory selection alone |
| [ToolMenuBench](https://arxiv.org/abs/2606.15508) | tool visibility changes success, risk, and cost across models | supplies a tool-interface corruption/intervention family |
| [AgenTracer](https://arxiv.org/abs/2509.03312) | counterfactual replay and fault injection for failure attribution | blocks naive replay-based blame assignment |
| [Causal Agent Replay](https://arxiv.org/abs/2606.08275) | step interventions, forward re-execution, Shapley credit, confidence intervals | blocks retrospective causal replay; leaves direct-effect coupling unresolved |
| [Micro-Randomized Trial](https://arxiv.org/abs/2107.03544) | identifies time-varying causal excursion effects under repeated randomization | methodological foundation for prospective sequential harness experiments |
| [Generalization in Adaptive Data Analysis](https://arxiv.org/abs/1506.02629) | repeated adaptive holdout reuse can overfit; reusable validation can preserve guarantees | methodological foundation for PREQ-Harness |

