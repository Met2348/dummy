# Search Log

## 2026-07-11: Local Full-Text Mining

Source corpus: `research/aaai27-harness-frontier/`.

- 1,150 verified full-text records searched.
- Initial loose ranking returned 833 records and was rejected as too noisy because generic `source model`, `target model`, and `prediction` phrases dominated.
- Revised context-aware screening requires an explicit transport phrase near a harness or component object.
- Revised result: 98 candidates, including 35 machine-labeled direct harness records and 63 component-transfer records.
- Manual adjudication selected 44 source-corpus papers for the isolated library.

Literal full-text searches included:

- `cross-model transfer`
- `held-out model`
- `model replacement`
- `ranking reversal`
- `negative transfer`
- `model-specific harness`
- `leave-one-model-out`
- `unseen model-scaffold`

## 2026-07-11: Primary-Source Web Search

Search was restricted to official arXiv records for technical claims. Query families included:

- harness/scaffold plus cross-model transfer;
- held-out models plus harness;
- model upgrade/replacement/substitution plus agent harness;
- negative transfer/ranking reversal plus LLM agents;
- prospective/treatment-effect/atomic intervention plus harness;
- target-conditioned or utility prediction plus agent skills.

Confirmed high-pressure records:

- [TTHE: Test-Time Harness Evolution](https://arxiv.org/abs/2607.08124), submitted 2026-07-09.
- [AgentTether](https://arxiv.org/abs/2607.06273), submitted 2026-07-07.
- [Life-Harness](https://arxiv.org/abs/2605.22166).
- [Meta-Harness](https://arxiv.org/abs/2603.28052).
- [AutoRISE](https://arxiv.org/abs/2604.22871).
- [FinHarness](https://arxiv.org/abs/2605.27333).
- [Harness as an Asset / CAAF](https://arxiv.org/abs/2604.17025).

The exact conjunction `prospective + target-model-conditioned + atomic harness effect prediction` did not return a direct match. This is a search result, not proof of absence.

## 2026-07-11: Reproducible arXiv Incremental Search

`scripts/search_incremental_arxiv.py` executed 19 transport-specific arXiv API queries through a UTC cutoff of `202607101825`.

- 144 unique API hits.
- 19 were already in the 1,150-paper source manifest.
- 125 were nominally new.
- Most new hits were lexical false positives from physical transport or generic "harnessing" usage.
- Abstract adjudication retained 14 papers for download.
- All 19 queries completed without API failure.

The retained set added the following especially relevant records:

- MemDelta;
- What Should a Skill Remember?;
- From Raw Experience to Skill Consumption;
- Natural-Language Agent Harnesses;
- AutoRISE;
- Memory Transfer Learning;
- BenchTrace;
- Harness VLA.

`Terminus-4B` is withdrawn in arXiv v2 because of product IP issues. Its publicly available v1 PDF is archived locally and the record is marked withdrawn; it is not used as stable core evidence.

## 2026-07-11: Backward Citation Chaining

The two closest new skill papers were used for backward chaining. Ten missing or out-of-manifest papers were fetched and downloaded:

- Trace2Skill;
- CoEvoSkills;
- SWE-Skills-Bench;
- Skills in the Wild;
- SkillCraft;
- AutoRefine;
- ProcMem;
- SkillRL;
- MemP;
- AgentSkillOS.

The final isolated corpus contains:

- 68 curated records;
- 68 validated PDFs;
- 68 extracted full texts;
- 68 transport-specific evidence cards;
- 44 records materialized from the source corpus;
- 24 records added through incremental search and citation chaining.

## Saturation Status

**Not saturated.** The field is moving on a daily cadence, and TTHE was one day old at search time.

Search may be treated as provisionally saturated only after:

1. the incremental script returns no new direct candidate on three runs separated by at least 48 hours;
2. backward references of every priority-1 paper are screened;
3. exact-title and author searches find no newer versions or follow-up records;
4. the collision stop condition in `collision-map.md` remains unmet.

## Rerun Command

```powershell
python research\aaai27-harness-transport\scripts\search_incremental_arxiv.py `
  --source-manifest research\aaai27-harness-frontier\metadata\all_candidates.jsonl `
  --output research\aaai27-harness-transport\metadata\incremental_arxiv_hits.jsonl `
  --new-output research\aaai27-harness-transport\metadata\incremental_arxiv_new.jsonl `
  --log research\aaai27-harness-transport\metadata\incremental_arxiv_run.json
```

