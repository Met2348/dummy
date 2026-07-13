#!/usr/bin/env python3
"""Build the manually adjudicated local and incremental transport manifests."""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


LOCAL_SELECTION: dict[str, tuple[int, str, str]] = {
    "2605.22166": (1, "direct-positive", "Bundled Life-Harness evolved on one model and evaluated on 17 targets."),
    "2604.25850": (1, "direct-positive", "Evolved coding harness transfer with source-tuned operating budgets."),
    "2603.28052": (1, "direct-positive", "Discovered harness evaluated on five held-out models."),
    "2606.17546": (1, "direct-mixed", "Cross-backend harness transfer includes sign reversals."),
    "2606.12882": (1, "direct-positive", "Learned interface controller tested on unseen generators."),
    "2606.06324": (1, "direct-positive", "Harness repair transferred without target-model-specific repair."),
    "2606.20512": (1, "direct-negative", "Repository guidance collapses under Qwen-to-Nemotron transfer."),
    "2607.06273": (1, "direct-positive", "Runtime repair layer transferred from Qwen to GPT-5.4."),
    "2606.09498": (1, "direct-adaptation", "Motivates model-specific self-adapting harnesses."),
    "2607.08124": (1, "direct-adaptation", "Test-time harness evolution is a newest direct collision."),
    "2605.23950": (1, "interaction-evidence", "Harness variance and model ranking reversals."),
    "2605.26731": (1, "interaction-evidence", "Harness sensitivity is non-monotone across model tiers."),
    "2604.00594": (1, "predictive-collision", "Predicts unseen model-scaffold combinations, but not atomic effects."),
    "2607.02032": (1, "predictive-collision", "Calibrates a capability proxy on models and evaluates held-out models."),
    "2605.05716": (1, "atomicity-collision", "Controlled component combinations reveal scaffold interference."),
    "2605.30621": (1, "evaluation-collision", "Separates harness benefit from harness updating ability."),
    "2604.20938": (1, "optimization-collision", "Model-conditional Bayesian harness optimization; transfer left open."),
    "2603.14987": (1, "transport-method", "Tests intervention transfer across models in representative evaluation."),
    "2606.07711": (2, "component-transport", "Adaptive memory profiles support unseen-model replacement."),
    "2606.23127": (2, "component-transport", "Procedural memories tested for cross-model generalization."),
    "2605.26275": (2, "component-transport", "Prompt optimizer reports positive retention on target models."),
    "2606.12387": (2, "component-transport", "Hint bank transfers to weaker backbones without retraining."),
    "2606.06741": (2, "component-transport", "Skill libraries are deployed across target models."),
    "2606.11182": (2, "component-transport", "Test-time prompt learning includes cross-model evaluation."),
    "2606.06079": (2, "component-transport", "Evolved agent skills are executed by a different-size model."),
    "2606.16420": (2, "component-transport", "Security playbooks include teacher-to-target transfer."),
    "2607.02911": (2, "component-transport", "Observation compression studies cross-model transfer."),
    "2605.18401": (2, "negative-transfer", "Skill governance explicitly targets negative transfer."),
    "2606.05684": (2, "negative-transfer", "Adaptive memory retrieval corrects negative transfer."),
    "2606.08151": (2, "negative-transfer", "Memory-card utility includes a negative-transfer term."),
    "2606.25115": (2, "negative-transfer", "Memory curation trades future value against harm."),
    "2602.22953": (2, "evaluation-method", "General agent evaluation measures cross-model gains and interactions."),
    "2606.12344": (2, "benchmark-method", "Harness benchmark supports model replacement studies."),
    "2606.14249": (2, "optimization-collision", "Composable harness foundry claims cross-model generalization."),
    "2604.01235": (2, "factorial-method", "Full-factorial cross-backend methodology."),
    "2604.11061": (2, "heldout-method", "Held-out-model prediction protocol in an adjacent setting."),
    "2606.30566": (2, "heldout-method", "Leave-one-model-out evaluation in an adjacent setting."),
    "2604.08224": (2, "survey", "Survey explicitly names fixed-harness cross-model tests as an open method."),
    "2606.10209": (2, "component-transport", "Context engineering reports cross-model generalization."),
    "2606.29354": (2, "component-transport", "Cross-model transfer of induced symbolic communication."),
    "2606.20333": (3, "boundary-negative", "Names cross-model execution as future work rather than evidence."),
    "2605.26302": (3, "interaction-evidence", "Deployment mechanisms induce recurring rank reversals."),
    "2605.27333": (3, "atomic-harness", "Inline verification and routing components provide atomic candidates."),
    "2602.12670": (2, "skill-benchmark", "Measures marginal utility of agent skills across diverse tasks."),
}

NEW_SELECTION: dict[str, tuple[int, str, str]] = {
    "2607.08448": (3, "adjacent-harness", "Newest memory-guided harness in frozen VLA deployment."),
    "2606.29914": (1, "interaction-evidence", "Controlled memory baselines reverse ranking across models."),
    "2606.17591": (2, "negative-transfer", "Insight governance decides when experience should be applied or withheld."),
    "2606.09421": (1, "component-transport", "Frozen cross-model skill rewriting preserves quality and lowers cost."),
    "2605.29225": (1, "negative-transfer", "Controlled self-evolution benchmark observes cross-context negative transfer."),
    "2605.27621": (2, "attribution-method", "Removal-based attribution and intervention via model replacement."),
    "2605.23899": (1, "component-transport", "Skill extractor-consumer study exposes non-uniform utility and negative transfer."),
    "2605.03195": (3, "model-replacement", "Replaces frontier execution subagents with a small specialized model."),
    "2604.27003": (1, "negative-transfer", "Memory representation changes forward transfer and forgetting."),
    "2604.22871": (1, "direct-positive", "Executable strategy programs are evaluated on held-out model families."),
    "2604.14004": (1, "component-transport", "Memory abstraction controls cross-domain and cross-model transfer."),
    "2603.25723": (1, "representation-collision", "Makes harness policy an executable, transferable, ablatable object."),
    "2601.07470": (2, "component-transport", "Learns when memory is transferable and when to abstain."),
    "2604.17025": (2, "atomic-harness", "Deterministic assertion interface is isolated by ablation."),
    "2603.25158": (2, "citation-chain", "Distills trajectory-local lessons into transferable agent skills."),
    "2604.01687": (2, "citation-chain", "Co-evolves multi-file skills with verification."),
    "2603.15401": (2, "skill-benchmark", "Tests whether skills help on real software-engineering tasks."),
    "2604.04323": (2, "skill-benchmark", "Evaluates skill usage in realistic settings."),
    "2603.00718": (2, "citation-chain", "Studies extraction and reuse of executable tool compositions."),
    "2601.22758": (3, "citation-chain", "Refines reusable expertise from agent trajectories."),
    "2602.01869": (3, "citation-chain", "Learns procedural memory via non-parametric optimization."),
    "2602.08234": (3, "citation-chain", "Evolves agents through recursive skill-augmented learning."),
    "2508.06433": (3, "citation-chain", "Foundational procedural-memory build-retrieve-update study."),
    "2603.02176": (3, "skill-ecosystem", "Organizes and benchmarks agent skills at ecosystem scale."),
}

API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def text_of(parent: ET.Element, path: str) -> str:
    node = parent.find(path, NS)
    return re.sub(r"\s+", " ", node.text or "").strip() if node is not None else ""


def fetch_missing(ids: list[str]) -> list[dict[str, object]]:
    if not ids:
        return []
    url = f"{API}?{urllib.parse.urlencode({'id_list': ','.join(ids), 'max_results': len(ids)})}"
    request = urllib.request.Request(url, headers={"User-Agent": "aaai27-harness-transport/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        root = ET.fromstring(response.read())
    rows: list[dict[str, object]] = []
    for entry in root.findall("atom:entry", NS):
        paper_id = re.sub(r"v\d+$", "", text_of(entry, "atom:id").rsplit("/", 1)[-1])
        primary = entry.find("arxiv:primary_category", NS)
        rows.append({
            "paper_id": paper_id,
            "arxiv_id": paper_id,
            "title": text_of(entry, "atom:title"),
            "abstract": text_of(entry, "atom:summary"),
            "authors": [text_of(author, "atom:name") for author in entry.findall("atom:author", NS)],
            "published": text_of(entry, "atom:published"),
            "updated": text_of(entry, "atom:updated"),
            "categories": [node.attrib.get("term", "") for node in entry.findall("atom:category", NS)],
            "primary_category": primary.attrib.get("term", "") if primary is not None else "",
            "abs_url": f"https://arxiv.org/abs/{paper_id}",
            "pdf_url": f"https://arxiv.org/pdf/{paper_id}",
        })
    return rows


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def annotate(row: dict[str, object], selection: dict[str, tuple[int, str, str]], source: str) -> dict[str, object]:
    paper_id = str(row["paper_id"])
    priority, evidence_role, adjudication = selection[paper_id]
    result = dict(row)
    result.update({
        "curation_priority": priority,
        "evidence_role": evidence_role,
        "adjudication": adjudication,
        "curation_source": source,
        "deep_read": priority == 1,
    })
    if paper_id == "2605.03195":
        result["withdrawn_current_version"] = True
        result["archived_version"] = "v1"
        result["pdf_url"] = "https://arxiv.org/pdf/2605.03195v1"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--incremental", type=Path, required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    parser.add_argument("--new-output", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    args = parser.parse_args()

    source_by_id = {str(row["paper_id"]): row for row in load_jsonl(args.source_manifest)}
    incremental_by_id = {str(row["paper_id"]): row for row in load_jsonl(args.incremental)}
    local_missing = sorted(set(LOCAL_SELECTION) - set(source_by_id))
    if local_missing:
        raise SystemExit(f"selected local IDs missing from source corpus: {local_missing}")

    missing_new = sorted(set(NEW_SELECTION) - set(incremental_by_id))
    for row in fetch_missing(missing_new):
        incremental_by_id[str(row["paper_id"])] = row
    still_missing = sorted(set(NEW_SELECTION) - set(incremental_by_id))
    if still_missing:
        raise SystemExit(f"selected incremental IDs missing after metadata fetch: {still_missing}")

    local_rows = [annotate(source_by_id[paper_id], LOCAL_SELECTION, "source-corpus") for paper_id in LOCAL_SELECTION]
    new_rows = [annotate(incremental_by_id[paper_id], NEW_SELECTION, "incremental-search") for paper_id in NEW_SELECTION]
    local_rows.sort(key=lambda row: (int(row["curation_priority"]), str(row["published"])), reverse=False)
    new_rows.sort(key=lambda row: (int(row["curation_priority"]), str(row["published"])), reverse=False)
    combined = sorted(local_rows + new_rows, key=lambda row: (int(row["curation_priority"]), str(row["published"])), reverse=False)
    write_jsonl(args.local_output, local_rows)
    write_jsonl(args.new_output, new_rows)
    write_jsonl(args.combined_output, combined)
    print(f"curated local={len(local_rows)} incremental={len(new_rows)} combined={len(combined)}")


if __name__ == "__main__":
    main()
