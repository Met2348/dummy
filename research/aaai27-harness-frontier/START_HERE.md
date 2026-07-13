# Start Here

## Bottom Line

Do not submit the original `model x harness x task` factorial attribution idea in
its simple form. That contribution is already occupied by 2026 work.

After the 1,150-paper full-corpus audit, the recommended AAAI-27 direction is:

> **Prospective Harness Transport: predict which atomic harness interventions
> preserve their sign and value after a model or task shift.**

The high-upside pivot is:

> **MRT-Harness: micro-randomized identification of when verification or context
> refresh helps during an agent trajectory.**

Do not lead with the original PREQ-Harness or ActiveHarness claims. SEA now
implements anytime-valid gates for self-evolving harnesses, TTHE explicitly names
prequential scoring, and HARBOR already formulates cost-aware Bayesian harness
configuration search.

## Reading Order

1. [`proposals/README.md`](proposals/README.md): five detailed proposals and branching rule.
2. [`notes/full-corpus-reading-report.md`](notes/full-corpus-reading-report.md): final evidence and changed decisions.
3. [`ideas/idea-ranking.md`](ideas/idea-ranking.md): updated sprint ranking.
4. [`proposals/02-prospective-harness-transport-proposal.md`](proposals/02-prospective-harness-transport-proposal.md): primary proposal.
5. [`proposals/03-mrt-harness-proposal.md`](proposals/03-mrt-harness-proposal.md): high-upside pivot.
6. [`notes/direct-competitor-dossier.md`](notes/direct-competitor-dossier.md): 108 core/direct papers with full-text evidence.
7. [`notes/all-library-index.md`](notes/all-library-index.md): all 1,150 papers and reading cards.

## First 72 Hours

1. Freeze two source models, two deterministic task families, and one untouched target model-task slice.
2. Implement three atomic interventions: verification, context refresh, and retry/recovery, without changing tools or budgets between arms.
3. Run a 72-96 episode kill test and estimate paired intervention effects.
4. Fit a deliberately small moderator model using intervention type, model tier, task horizon, and interface-error rate; predict the sealed target effects.
5. Continue only if at least one effect changes materially across cells and the target sign or magnitude is predictable above a constant-effect baseline.
6. If transport is flat, run a 200-decision-point MRT pilot for verification and continue only if proximal outcomes are dense and state-conditional effects vary.

No large benchmark sweep or paper writing should start before one of these kill
tests produces a defensible central figure.
