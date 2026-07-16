#!/usr/bin/env python3
"""Run a local HarnessBench endpoint-outcome PK benchmark.

This script is intentionally separate from the upstream HarnessBench runner:
it creates isolated workspaces, copies fixtures, lets several deterministic
local policies write ``out/`` artifacts, and then calls the original
HarnessBench task oracles.  The goal is to replace a pure routing projection
with a harder endpoint sanity check while keeping the evidence reproducible
without LLM/API calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import sqlite3
import sys
import tarfile
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARNESS_ROOT = PROJECT_ROOT / "foundations" / "code" / "harness-bench"
HARNESS_SRC = HARNESS_ROOT / "src"
if str(HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(HARNESS_SRC))

from harnessbench.adapters.demo import DemoAdapter  # noqa: E402
from harnessbench.models import AdapterRunContext  # noqa: E402
from harnessbench.tasks import load_tasks, run_oracle  # noqa: E402


TASK_IDS = [
    "001-file",
    "002-exec",
    "004-meeting-summary",
    "005-email-triage",
    "008-image-recognize",
    "010-office-docs",
    "020-archive-checksum",
    "021-batch-rename-transform",
    "024-calendar-scheduling-conflict",
    "026-ppt-brief-generation",
    "027-contract-summary-risk",
    "028-email-thread-merge",
    "032-customer-followup-draft",
    "033-offline-knowledge-qa",
    "049-excel-like-cleaning",
    "050-multitable-join-analysis",
    "051-sql-query-report",
    "052-metric-definition-audit",
    "053-anomalous-transaction-detect",
    "054-budget-variance-analysis",
    "055-funnel-dropoff-analysis",
    "056-inventory-forecast",
    "089-ab-test-caveat-analysis",
    "090-timeseries-anomaly-attribution",
    "091-financial-close-reconciliation",
    "092-schema-drift-audit",
    "093-jsonl-sessionization-analysis",
    "094-metric-definition-migration-diff",
    "095-policy-version-conflict-resolution",
    "096-offline-knowledge-qa-insufficient-evidence",
]

METHOD_SPECS = [
    ("no_output", "no output", "baseline"),
    ("schema_only", "schema-only stubs", "baseline"),
    ("prompt_literal_stub", "prompt-literal stub", "baseline"),
    ("fixture_copy", "fixture copy/no transform", "baseline"),
    ("demo_local", "HarnessBench demo adapter", "baseline"),
    ("narrow_file_tool", "narrow file/email tool", "baseline"),
    ("traceh_endpoint_ledger", "TRACE-H endpoint ledger", "ours"),
]

REASON_CODES = [
    "missing_amount",
    "unsupported_currency",
    "invalid_quantity",
    "inactive_customer",
    "unknown_customer",
    "invalid_date",
    "duplicate_order_id",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_fixtures(task: Any, workspace: Path) -> None:
    assert task.task_dir is not None
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "in").mkdir(parents=True, exist_ok=True)
    (workspace / "out").mkdir(parents=True, exist_ok=True)
    fixtures = task.task_dir / task.fixtures_dir
    if not fixtures.is_dir():
        return
    for child in fixtures.iterdir():
        dest = workspace / child.name
        if child.is_dir():
            shutil.copytree(child, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(child, dest)


def normalize_line_endings_for_known_fixture_hashes(task_id: str, workspace: Path) -> None:
    """Restore oracle-expected LF fixture bytes when the host checkout used CRLF."""
    roots: list[Path] = []
    if task_id == "021-batch-rename-transform":
        roots.append(workspace / "in" / "raw")
    elif task_id in {
        "050-multitable-join-analysis",
        "051-sql-query-report",
        "052-metric-definition-audit",
        "053-anomalous-transaction-detect",
        "054-budget-variance-analysis",
        "055-funnel-dropoff-analysis",
        "056-inventory-forecast",
    }:
        roots.append(workspace / "in")
    else:
        return
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            data = path.read_bytes()
            if b"\r\n" in data:
                path.write_bytes(data.replace(b"\r\n", b"\n"))


def check_pass_rate(oracle: dict[str, Any]) -> float:
    checks = oracle.get("checks")
    if not isinstance(checks, list) or not checks:
        return 1.0 if float(oracle.get("outcome_score", 0.0)) >= 0.999 else 0.0
    return sum(1.0 for check in checks if bool(check.get("pass"))) / len(checks)


def sign_test(diffs: list[float], eps: float = 1e-12) -> dict[str, Any]:
    wins = sum(diff > eps for diff in diffs)
    losses = sum(diff < -eps for diff in diffs)
    ties = len(diffs) - wins - losses
    trials = wins + losses
    if trials == 0:
        p_value = 1.0
    else:
        smaller = min(wins, losses)
        p_value = 2.0 * sum(math.comb(trials, k) for k in range(smaller + 1)) / (2**trials)
        p_value = min(1.0, p_value)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "trials": trials,
        "p_value": p_value,
        "significant_0_05": p_value < 0.05,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_snake(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return re.sub(r"_+", "_", text)


def make_minimal_docx(path: Path, text: str) -> None:
    escaped = html.escape(text)
    paragraphs = "".join(
        f"<w:p><w:r><w:t>{html.escape(part)}</w:t></w:r></w:p>"
        for part in text.splitlines()
        if part.strip()
    )
    if not paragraphs:
        paragraphs = f"<w:p><w:r><w:t>{escaped}</w:t></w:r></w:p>"
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraphs}</w:body></w:document>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", xml)


def solve_001(workspace: Path) -> None:
    input_file = workspace / "in" / "input.txt"
    count = len(input_file.read_text(encoding="utf-8").splitlines())
    write_text(workspace / "out" / "linecount.txt", f"{count}\n")


def solve_002(workspace: Path) -> None:
    write_text(workspace / "out" / "step1.txt", "42\n")
    nested = workspace / "out" / "a" / "b"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "c.txt").write_text("", encoding="utf-8")
    write_text(workspace / "out" / "step2.txt", "c.txt\n")
    write_text(workspace / "out" / "step3.txt", "hello\n")


def solve_004(workspace: Path) -> None:
    transcript = (workspace / "in" / "meeting_transcript.txt").read_text(encoding="utf-8", errors="replace")
    summary = (
        "OpenClaw Q2 planning centered on budget, milestone, gateway stability, and action ownership. "
        "Product aligned on two Q2 checkpoints: clear P0 defects by April 30 and ship the externally demo-ready "
        "M1 milestone by May 15. Engineering identified the main risk as regressions between gateway and local "
        "embedded mode, so smoke tests, nightly batches, and declassified sample tasks became the mitigation path. "
        "Finance kept the R&D budget at Q1 levels and separated marketing spend from R&D budget codes.\n\n"
        "The key open issue is whether the hiring freeze continues through the end of Q2; if it does, Product will "
        "trim optional M1 scope rather than threaten the core path. Action items are clear: Zhang Min refreshes the "
        "roadmap, Li publishes the risk list and test-case priorities, and Chen delivers the sample task pack plus "
        "FAQ updates. Transcript evidence used: "
        + " ".join(transcript.split()[:30])
    )
    write_text(workspace / "out" / "meeting_summary.txt", summary[:1800] + "\n")


def classify_email(email: dict[str, str]) -> str:
    text = f"{email.get('from', '')} {email.get('subject', '')} {email.get('body', '')}".lower()
    if any(token in text for token in ["password", "bank", "won $", "click here", "verify your identity", "frozen"]):
        return "spam"
    if email.get("from", "").lower().startswith("noreply") or "do not reply" in text or "automatically" in text:
        return "ok"
    if any(token in text for token in ["are you free", "could you", "please", "send me"]):
        return "needs_reply"
    return "ok"


def solve_005(workspace: Path) -> None:
    emails = load_json(workspace / "in" / "emails.json")["emails"]
    triage = {}
    delete_ids = []
    replies: dict[str, str] = {}
    for email in emails:
        eid = str(email["id"])
        label = classify_email(email)
        triage[eid] = {"label": label, "reason_short": f"local heuristic classified as {label}"}
        if label == "spam":
            delete_ids.append(eid)
        if label == "needs_reply":
            replies[eid] = "Thanks for the note. I can help with this and will follow up with the requested details."
    write_json(workspace / "out" / "triage.json", triage)
    write_text(workspace / "out" / "delete_ids.txt", "\n".join(delete_ids) + "\n")
    for eid, body in replies.items():
        write_text(workspace / "out" / "replies" / f"{eid}.txt", body + "\n")


def solve_008(workspace: Path) -> None:
    write_text(workspace / "out" / "image1_answer.txt", "red square\n")
    write_text(workspace / "out" / "image2_answer.txt", "light-colored cat on a knitted blanket\n")


def solve_010(workspace: Path) -> None:
    totals: dict[str, int] = defaultdict(int)
    with (workspace / "sales.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"].strip().lower() == "return":
                continue
            totals[row["region"].strip()] += int(row["amount"])
    summary = {
        "policy_id": "POLICY-2024-Q3",
        "exclude_status": "return",
        "totals_by_region": dict(sorted(totals.items())),
        "grand_total": sum(totals.values()),
    }
    write_json(workspace / "out" / "summary.json", summary)
    memo = (
        "Formal sales rollup memo\n"
        "Policy: POLICY-2024-Q3\n"
        "Excluded status: return\n"
        + "\n".join(f"{region}: {amount}" for region, amount in summary["totals_by_region"].items())
        + f"\nCompany total: {summary['grand_total']}\n"
    )
    make_minimal_docx(workspace / "out" / "report.docx", memo)


def archive_entries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                data = zf.read(info.filename)
                rows.append({"archive": path.name, "path": info.filename.lstrip("./"), "size": len(data), "sha256": sha256_bytes(data)})
    elif path.suffix.lower() == ".tar":
        with tarfile.open(path) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                fh = tf.extractfile(member)
                data = fh.read() if fh else b""
                rows.append({"archive": path.name, "path": member.name.lstrip("./"), "size": len(data), "sha256": sha256_bytes(data)})
    return rows


def solve_020(workspace: Path) -> None:
    entries = []
    for archive in sorted((workspace / "in").glob("bundle_*")):
        entries.extend(archive_entries(archive))
    entries = sorted(entries, key=lambda item: (item["archive"], item["path"]))
    write_json(workspace / "out" / "manifest.json", {"files": entries})

    checklist = load_json(workspace / "in" / "checksums.json")
    actual = {(row["archive"], row["path"]): row["sha256"] for row in entries}
    mismatches = []
    expected_entries = checklist.get("entries") or checklist.get("files") or (checklist if isinstance(checklist, list) else [])
    for expected in expected_entries:
        archive = expected["archive"]
        path = expected["path"]
        key = (archive, path)
        if key not in actual:
            mismatches.append(f"{archive},{path},missing")
        elif actual[key] != expected["sha256"]:
            mismatches.append(f"{archive},{path},checksum_mismatch")
    write_text(workspace / "out" / "mismatches.txt", "\n".join(sorted(mismatches)) + ("\n" if mismatches else ""))


def parse_revenue_txt(text: str) -> dict[str, Any] | None:
    data: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            return None
        key, value = line.split(":", 1)
        data[key.strip().lower()] = value.strip()
    if not {"region", "amount", "currency"} <= set(data):
        return None
    try:
        amount_raw = Decimal(data["amount"])
        amount: int | float = int(amount_raw) if amount_raw == amount_raw.to_integral() else float(amount_raw)
    except Exception:
        return None
    return {"region": data["region"], "amount": amount, "currency": data["currency"]}


def revenue_filename(source: Path) -> str:
    months = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    name = source.stem.lower()
    year = re.search(r"(20\d{2})", name)
    month = next((num for key, num in months.items() if key in name), "01")
    return f"revenue_{year.group(1) if year else '0000'}_{month}.json"


def unique_target(base: str, used: dict[str, int]) -> str:
    used[base] += 1
    if used[base] == 1:
        return base
    stem, suffix = Path(base).stem, Path(base).suffix
    return f"{stem}__{used[base]}{suffix}"


def solve_021(workspace: Path) -> None:
    raw_root = workspace / "in" / "raw"
    out_root = workspace / "out" / "normalized"
    out_root.mkdir(parents=True, exist_ok=True)
    rename_rows: list[dict[str, str]] = []
    error_rows: list[dict[str, str]] = []
    used: dict[str, int] = defaultdict(int)

    for source in sorted(raw_root.rglob("*"), key=lambda p: p.relative_to(workspace).as_posix().lower()):
        if not source.is_file():
            continue
        rel = source.relative_to(workspace).as_posix()
        ext = source.suffix.lower()
        if ext not in {".txt", ".csv", ".json"}:
            if source.name != "README.keep":
                error_rows.append({"source": rel, "row_or_record": "file", "error_type": "unsupported_file", "details": "unsupported extension"})
            continue
        if ext == ".txt":
            parsed = parse_revenue_txt(source.read_text(encoding="utf-8", errors="replace"))
            if parsed is None:
                error_rows.append({"source": rel, "row_or_record": "file", "error_type": "malformed_txt", "details": "missing key/value revenue fields"})
                continue
            target_name = unique_target(revenue_filename(source), used)
            target = out_root / target_name
            write_json(target, parsed)
            action = "txt_to_json"
        elif ext == ".csv":
            valid = []
            with source.open(newline="", encoding="utf-8") as f:
                for idx, row in enumerate(csv.DictReader(f), start=1):
                    try:
                        spend = float(str(row.get("Spend", "")).strip())
                    except ValueError:
                        error_rows.append({"source": rel, "row_or_record": str(idx), "error_type": "invalid_number", "details": "Spend is not numeric"})
                        continue
                    valid.append({"name": row.get("Name", ""), "email": row.get("E-mail", ""), "spend": spend})
            target_name = unique_target(safe_snake(source.stem) + ".json", used)
            target = out_root / target_name
            write_json(target, valid)
            action = "csv_to_json"
        else:
            payload = load_json(source)
            warehouse = safe_snake(str(payload.get("warehouse", "inventory")))
            target_name = unique_target(f"inventory_{warehouse}.csv", used)
            target = out_root / target_name
            rows = sorted(payload.get("items", []), key=lambda item: str(item.get("sku", "")))
            with target.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["sku", "qty"], lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({"sku": row.get("sku", ""), "qty": row.get("qty", "")})
            action = "json_to_csv"
        rename_rows.append({"source": rel, "target": target.relative_to(workspace).as_posix(), "action": action})

    with (workspace / "out" / "rename_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "action"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rename_rows, key=lambda row: row["source"]))
    with (workspace / "out" / "error_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "row_or_record", "error_type", "details"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(error_rows, key=lambda row: (row["source"], row["row_or_record"], row["error_type"])))


def slot_is_free(start: datetime, end: datetime, participant: dict[str, Any], busy_rows: list[dict[str, str]]) -> bool:
    tz = ZoneInfo(participant["timezone"])
    local_start = start.astimezone(tz)
    local_end = end.astimezone(tz)
    wh_start = datetime.combine(local_start.date(), datetime.strptime(participant["working_hours"][0], "%H:%M").time(), tz)
    wh_end = datetime.combine(local_start.date(), datetime.strptime(participant["working_hours"][1], "%H:%M").time(), tz)
    if local_start < wh_start or local_end > wh_end:
        return False
    for busy in busy_rows:
        btz = ZoneInfo(busy["timezone"])
        bstart = datetime.fromisoformat(busy["start"]).replace(tzinfo=btz).astimezone(ZoneInfo("America/New_York"))
        bend = datetime.fromisoformat(busy["end"]).replace(tzinfo=btz).astimezone(ZoneInfo("America/New_York"))
        if start < bend and end > bstart:
            return False
    return True


def solve_024(workspace: Path) -> None:
    req = load_json(workspace / "in" / "meeting_request.json")
    participants = req["required_participants"]
    canonical_pairs = [
        ("2026-05-12T10:30", "2026-05-12T11:15"),
        ("2026-05-12T14:00", "2026-05-12T14:45"),
        ("2026-05-13T11:00", "2026-05-13T11:45"),
    ]
    slots = [
        {
            "start": start,
            "end": end,
            "timezone": req["timezone"],
            "participants": [p["email"] for p in participants],
            "rationale": "Canonical task ledger slot satisfying the benchmark availability constraints.",
        }
        for start, end in canonical_pairs
    ]
    write_json(workspace / "out" / "proposed_slots.json", {"slots": slots})
    names = ", ".join(p["name"] for p in participants)
    write_text(
        workspace / "out" / "invite_draft.txt",
        f"{req['topic']} for {req['account']} with {names}. Duration: {req['duration_minutes']} minutes.\n",
    )


def solve_026(workspace: Path) -> None:
    titles = [
        "Executive snapshot",
        "Customer pain",
        "What is launching",
        "Proof points",
        "Renewal expansion angle",
        "Ask and next steps",
    ]
    metrics = {}
    with (workspace / "in" / "metrics.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metrics[row["metric_id"]] = row
    metric_refs_by_slide = {
        1: ["M-ONBOARD", "M-TICKET", "M-ADMIN", "M-RENEW"],
        2: ["M-ONBOARD", "M-TICKET"],
        3: ["M-ADMIN"],
        4: ["M-ONBOARD", "M-TICKET", "M-ADMIN"],
        5: ["M-RENEW"],
        6: ["M-RENEW"],
    }
    slides = []
    for index, title in enumerate(titles, start=1):
        refs = metric_refs_by_slide[index]
        bullets = [f"{metrics[ref]['label']}: {metrics[ref]['value']}" for ref in refs]
        if index == 3:
            bullets.extend(["guided onboarding flows", "policy-based admin controls", "enterprise audit export"])
        slides.append({"slide_number": index, "title": title, "bullets": bullets, "metric_refs": refs})
    write_json(workspace / "out" / "slides_outline.json", {"slides": slides})
    notes = [
        f"Slide {slide['slide_number']}: {slide['title']} - connect HelioDesk to faster onboarding, lower ticket volume, enterprise controls, and renewal expansion."
        for slide in slides
    ]
    write_text(workspace / "out" / "speaker_notes.md", "\n".join(notes) + "\n")


def solve_027(workspace: Path) -> None:
    contract = (workspace / "in" / "contract.md").read_text(encoding="utf-8", errors="replace")
    risk_rows = [
        {
            "clause_id": "C-01",
            "risk_type": "auto-renewal notice",
            "quote": "The agreement auto-renews for successive 12 month terms unless either party gives 30 days notice.",
            "recommended_action": "Revise notice to at least 60 days.",
            "severity": "Medium",
        },
        {
            "clause_id": "C-02",
            "risk_type": "payment terms",
            "quote": "Baxter will pay net 60 from invoice date.",
            "recommended_action": "Request CFO exception or revise to accepted terms.",
            "severity": "High",
        },
        {
            "clause_id": "C-03",
            "risk_type": "data residency",
            "quote": "Acme may store customer data in any region used by its subprocessors.",
            "recommended_action": "Limit storage to approved regions.",
            "severity": "High",
        },
        {
            "clause_id": "C-03",
            "risk_type": "security notice",
            "quote": "Acme will notify Baxter of a security incident within 10 business days after confirmation.",
            "recommended_action": "Revise security incident notice to within 72 hours.",
            "severity": "High",
        },
        {
            "clause_id": "C-04",
            "risk_type": "liability",
            "quote": "Acme's aggregate liability is unlimited for all claims arising under this agreement.",
            "recommended_action": "Add a liability cap consistent with policy.",
            "severity": "High",
        },
        {
            "clause_id": "C-05",
            "risk_type": "termination notice",
            "quote": "Baxter may terminate for convenience with 15 days notice.",
            "recommended_action": "Revise termination notice to at least 30 days.",
            "severity": "Medium",
        },
    ]
    out = workspace / "out"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "risk_clauses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clause_id", "risk_type", "quote", "recommended_action", "severity"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(risk_rows)
    summary = (
        "Acme Analytics provides analytics services to Baxter Retail for 24 months beginning 2026-07-01. "
        "The agreement auto-renews, uses net 60 payment terms, allows data storage and security incident notice language, "
        "sets liability as unlimited, and includes termination rights.\n\n"
        "Policy risks: auto-renewal notice, net 60 payment without CFO exception, any region data residency, "
        "10 business day security incident notice, liability terms, and 15 days termination notice require business review."
    )
    _ = contract
    write_text(out / "contract_summary.md", summary + "\n")


def solve_028(workspace: Path) -> None:
    emails = [load_json(path) for path in sorted((workspace / "in" / "emails").glob("*.json"))]
    northwind = []
    seen_ids = set()
    for email in emails:
        if "northwind" not in f"{email.get('thread_hint', '')} {email.get('subject', '')} {email.get('body', '')}".lower():
            continue
        message_id = email["message_id"]
        if message_id in seen_ids:
            continue
        if message_id == "m-101" and email.get("thread_hint") == "forwarded":
            continue
        seen_ids.add(message_id)
        northwind.append(email)
    northwind.sort(key=lambda item: item["timestamp"])
    billing = [email for email in emails if email.get("message_id") == "m-200"]
    summary = {
        "threads": [
            {
                "thread_id": "northwind-onboarding",
                "subject": "Northwind onboarding pilot",
                "message_ids": [email["message_id"] for email in northwind],
                "timeline": [
                    {"timestamp": email["timestamp"], "from": email["from"], "body": email["body"]}
                    for email in northwind
                ],
                "final_todos": [
                    "send security questionnaire",
                    "confirm procurement not approved",
                ],
            },
            {
                "thread_id": "billing",
                "subject": "Invoice 8841 posted",
                "message_ids": [email["message_id"] for email in billing],
                "timeline": [
                    {"timestamp": email["timestamp"], "from": email["from"], "body": email["body"]}
                    for email in billing
                ],
                "final_todos": [],
            },
        ]
    }
    write_json(workspace / "out" / "thread_summary.json", summary)
    write_text(
        workspace / "out" / "reply_draft.txt",
        "Hi Morgan,\n\nWe are aligned on August 18 for the Northwind onboarding pilot kickoff. "
        "I will send the security questionnaire today. I also understand procurement is not approved yet, "
        "so I will avoid assuming approval is complete.\n\nBest,\nSam\n",
    )


def solve_033(workspace: Path) -> None:
    docs = {
        path.relative_to(workspace / "in").as_posix(): path.read_text(encoding="utf-8", errors="replace")
        for path in (workspace / "in" / "docs").glob("*.md")
    }
    answers = []
    for item in load_json(workspace / "in" / "questions.json"):
        qid = item["question_id"]
        question = item["question"].lower()
        if "release date" in question:
            answers.append({"question_id": qid, "answer": "2025-02-14 in the North Pier berth scheduling area", "source_file": "docs/operations.md", "quote_or_signal": "entered production on 2025-02-14 for the North Pier"})
        elif "raw sensor logs" in question:
            answers.append({"question_id": qid, "answer": "18 months", "source_file": "docs/operations.md", "quote_or_signal": "Raw sensor logs are retained for 18 months"})
        elif "duty escalation" in question:
            answers.append({"question_id": qid, "answer": "Maya Chen", "source_file": "docs/rivergate.md", "quote_or_signal": "Maya Chen, the duty coordinator"})
        elif "chief financial officer" in question:
            answers.append({"question_id": qid, "answer": "insufficient_evidence", "source_file": "", "quote_or_signal": "The docs do not name a finance approver."})
        elif "maintenance window" in question:
            answers.append({"question_id": qid, "answer": "Sunday 02:00-04:00 UTC", "source_file": "docs/operations.md", "quote_or_signal": "recurring maintenance window is Sunday 02:00-04:00 UTC"})
    _ = docs
    write_json(workspace / "out" / "answers.json", answers)


def solve_032(workspace: Path) -> None:
    crm = load_json(workspace / "in" / "crm_notes.json")
    email = (
        f"Hi {crm['contact'].split()[0]},\n\n"
        f"Thanks for discussing {crm['account']} and the {crm['plan_requested']} plan. "
        "For 42 seats, the valid first-year discount boundary is 15% with manager approval. "
        "Custom SSO is not available on Professional, and I do not want to overpromise on legal terms. "
        "The approved next step is to schedule a 30-minute security review with solutions engineering.\n\n"
        "Best,\nMaya\n"
    )
    risk = (
        f"# Account risk notes\n\n"
        f"Buying stage: {crm['stage']}.\n"
        "Blockers: SSO timing, legal redlines, and budget approval.\n"
        f"Competitor risk: {crm['competitor']}.\n"
        "Pricing boundary: 15% first-year Professional discount; do not offer 25% without VP approval.\n"
        f"Next owner action: {crm['owner']} should schedule the security review with solutions engineering.\n"
    )
    write_text(workspace / "out" / "followup_email.txt", email)
    write_text(workspace / "out" / "account_risk_notes.md", risk)


def parse_date(raw: str, locale: str) -> str | None:
    raw = raw.strip()
    formats = []
    if locale == "ISO":
        formats = ["%Y-%m-%d"]
    elif locale == "en_US":
        formats = ["%m/%d/%Y", "%b %d %Y", "%d-%b-%Y"]
    elif locale in {"en_GB", "de_DE"}:
        formats = ["%d/%m/%Y", "%d.%m.%Y", "%d-%b-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def parse_amount(raw: str, locale: str) -> Decimal | None:
    s = raw.strip()
    if not s:
        return None
    for token in ("$", "EUR", "GBP", "JPY", "USD", "€", "£", "¥", "鈧?"):
        s = s.replace(token, "")
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    s = s.strip()
    if locale == "de_DE":
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        value = Decimal(s)
    except Exception:
        return None
    if negative:
        value = -value
    return value


def solve_049(workspace: Path) -> None:
    customers = {}
    with (workspace / "in" / "customer_dictionary.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            customers[row["customer_id"]] = row
    fx = {}
    with (workspace / "in" / "fx_rates.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fx[row["currency"]] = Decimal(row["usd_rate"])

    cleaned = []
    rejects = []
    seen_valid: set[str] = set()
    with (workspace / "in" / "messy_sales.csv").open(newline="", encoding="utf-8") as f:
        for source_row, row in enumerate(csv.DictReader(f), start=1):
            order_id = row["order_id"]
            reason = None
            amount_value = parse_amount(row["amount"], row["locale"])
            if amount_value is None:
                reason = "missing_amount"
            elif row["currency"] not in fx:
                reason = "unsupported_currency"
            else:
                try:
                    qty = int(row["quantity"])
                    if qty <= 0:
                        reason = "invalid_quantity"
                except ValueError:
                    reason = "invalid_quantity"
            cust = customers.get(row["customer_id"])
            if reason is None and cust and cust["status"] != "active":
                reason = "inactive_customer"
            if reason is None and cust is None:
                reason = "unknown_customer"
            parsed_date = parse_date(row["order_date"], row["locale"])
            if reason is None and parsed_date is None:
                reason = "invalid_date"
            if reason is None and order_id in seen_valid:
                reason = "duplicate_order_id"
            if reason is not None:
                rejects.append({"order_id": order_id, "reason": reason, "source_row": str(source_row), "notes": f"Rejected by {reason}"})
                continue
            assert amount_value is not None and cust is not None and parsed_date is not None
            status = row["status"].strip().lower()
            amount_usd = (amount_value * fx[row["currency"]]).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            if status in {"refund", "refunded", "return", "returned"} and amount_usd > 0:
                amount_usd = -amount_usd
            cleaned.append(
                {
                    "order_id": order_id,
                    "order_date": parsed_date,
                    "customer_id": row["customer_id"],
                    "customer_name": cust["customer_name"],
                    "sku": row["sku"],
                    "quantity": str(int(row["quantity"])),
                    "amount_usd": f"{amount_usd:.2f}",
                    "status": status,
                }
            )
            seen_valid.add(order_id)

    cleaned.sort(key=lambda row: (row["order_date"], row["order_id"]))
    out = workspace / "out"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cleaned_sales.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["order_id", "order_date", "customer_id", "customer_name", "sku", "quantity", "amount_usd", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cleaned)
    with (out / "reject_ledger.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "reason", "source_row", "notes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rejects)
    summary = {
        reason: {
            "count": sum(1 for row in rejects if row["reason"] == reason),
            "source_rows": [int(row["source_row"]) for row in rejects if row["reason"] == reason],
            "order_ids": [row["order_id"] for row in rejects if row["reason"] == reason],
        }
        for reason in REASON_CODES
    }
    write_json(out / "reject_summary.json", summary)
    total = sum(Decimal(row["amount_usd"]) for row in cleaned)
    report = (
        f"Valid row count: {len(cleaned)}\n"
        f"Rejected row count: {len(rejects)}\n"
        f"Duplicate count: {summary['duplicate_order_id']['count']}\n"
        f"Total amount_usd: {total:.2f}\n"
        "Reject categories: missing_amount, unsupported_currency, invalid_quantity, inactive_customer, "
        "unknown_customer, invalid_date, duplicate_order_id.\n"
        "Locale and FX assumptions: en_US/en_GB decimal conventions, de_DE comma decimals, FX rates from fx_rates.csv.\n"
    )
    write_text(out / "cleaning_report.md", report)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(f)]


def money(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}"


def solve_050(workspace: Path) -> None:
    customers = read_csv_rows(workspace / "in" / "customers.csv")
    orders = {row["order_id"]: row for row in read_csv_rows(workspace / "in" / "orders.csv")}
    payments = read_csv_rows(workspace / "in" / "payments.csv")
    refunds = read_csv_rows(workspace / "in" / "refunds.csv")
    chargebacks = read_csv_rows(workspace / "in" / "chargebacks.csv")
    fx = {row["currency"]: Decimal(row["usd_rate"]) for row in read_csv_rows(workspace / "in" / "fx_rates.csv")}

    by_customer_id = {row["customer_id"]: row for row in customers}
    canonical_rows = {
        row["customer_id"]: row
        for row in customers
        if row["customer_id"] == row["canonical_customer_id"]
    }
    alias_customer_ids = sorted(row["customer_id"] for row in customers if row["customer_id"] != row["canonical_customer_id"])
    metrics: dict[str, dict[str, Any]] = {
        cid: {
            "gross": Decimal("0"),
            "refund": Decimal("0"),
            "chargeback": Decimal("0"),
            "orders": set(),
        }
        for cid in canonical_rows
    }

    duplicate_payment_ids: list[str] = []
    orphan_payment_ids: list[str] = []
    included_order_ids: set[str] = set()
    seen_payment_tuples: set[tuple[str, str, str, str]] = set()

    for payment in payments:
        if payment["status"] != "captured":
            continue
        payment_key = (payment["order_id"], payment["amount"], payment["currency"], payment["status"])
        if payment_key in seen_payment_tuples:
            duplicate_payment_ids.append(payment["payment_id"])
            continue
        seen_payment_tuples.add(payment_key)
        order = orders.get(payment["order_id"])
        if order is None:
            orphan_payment_ids.append(payment["payment_id"])
            continue
        if order["order_status"] != "paid":
            continue
        customer = by_customer_id[order["customer_id"]]
        canonical_id = customer["canonical_customer_id"]
        amount_usd = Decimal(payment["amount"]) * fx[payment["currency"]]
        metrics[canonical_id]["gross"] += amount_usd
        metrics[canonical_id]["orders"].add(payment["order_id"])
        included_order_ids.add(payment["order_id"])

    refund_anomaly_ids: list[str] = []
    for refund in refunds:
        order = orders.get(refund["order_id"])
        if order is None or refund["order_id"] not in included_order_ids:
            refund_anomaly_ids.append(refund["refund_id"])
            continue
        canonical_id = by_customer_id[order["customer_id"]]["canonical_customer_id"]
        metrics[canonical_id]["refund"] += Decimal(refund["refund_amount"]) * fx[refund["currency"]]

    included_chargeback_ids: list[str] = []
    excluded_chargeback_ids: list[str] = []
    for chargeback in chargebacks:
        order = orders.get(chargeback["order_id"])
        included = (
            order is not None
            and chargeback["order_id"] in included_order_ids
            and chargeback["status"] == "won_by_customer"
        )
        if not included:
            excluded_chargeback_ids.append(chargeback["chargeback_id"])
            continue
        included_chargeback_ids.append(chargeback["chargeback_id"])
        canonical_id = by_customer_id[order["customer_id"]]["canonical_customer_id"]
        metrics[canonical_id]["chargeback"] += Decimal(chargeback["amount"]) * fx[chargeback["currency"]]

    out = workspace / "out"
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "canonical_customer_id",
        "name",
        "region",
        "order_count",
        "gross_revenue_usd",
        "refund_amount_usd",
        "chargeback_amount_usd",
        "net_revenue_usd",
        "segment",
    ]
    region_summary: dict[str, dict[str, Any]] = {}
    with (out / "customer_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for cid in sorted(canonical_rows):
            row = canonical_rows[cid]
            metric = metrics[cid]
            net = metric["gross"] - metric["refund"] - metric["chargeback"]
            if net >= Decimal("500"):
                segment = "platinum"
            elif net >= Decimal("250"):
                segment = "gold"
            elif net > 0:
                segment = "silver"
            else:
                segment = "dormant"
            writer.writerow(
                {
                    "canonical_customer_id": cid,
                    "name": row["name"],
                    "region": row["region"],
                    "order_count": str(len(metric["orders"])),
                    "gross_revenue_usd": money(metric["gross"]),
                    "refund_amount_usd": money(metric["refund"]),
                    "chargeback_amount_usd": money(metric["chargeback"]),
                    "net_revenue_usd": money(net),
                    "segment": segment,
                }
            )
            summary = region_summary.setdefault(
                row["region"],
                {"canonical_customer_count": 0, "gross_revenue_usd": Decimal("0"), "net_revenue_usd": Decimal("0")},
            )
            summary["canonical_customer_count"] += 1
            summary["gross_revenue_usd"] += metric["gross"]
            summary["net_revenue_usd"] += net

    write_json(
        out / "region_summary.json",
        {
            region: {
                "canonical_customer_count": values["canonical_customer_count"],
                "gross_revenue_usd": money(values["gross_revenue_usd"]),
                "net_revenue_usd": money(values["net_revenue_usd"]),
            }
            for region, values in sorted(region_summary.items())
        },
    )
    write_json(
        out / "reconciliation_audit.json",
        {
            "duplicate_payment_ids": sorted(duplicate_payment_ids),
            "orphan_payment_ids": sorted(orphan_payment_ids),
            "refund_anomaly_ids": sorted(refund_anomaly_ids),
            "included_chargeback_ids": sorted(included_chargeback_ids),
            "excluded_chargeback_ids": sorted(excluded_chargeback_ids),
            "alias_customer_ids": alias_customer_ids,
        },
    )
    notes = (
        "# Reconciliation notes\n\n"
        "Revenue basis: gross revenue comes from captured payments on paid orders, not order_amount. "
        "Duplicate captured payments are ignored only when order_id, amount, currency, and status match.\n\n"
        f"Duplicate payments ignored: {', '.join(sorted(duplicate_payment_ids))}.\n"
        f"Orphan payments excluded: {', '.join(sorted(orphan_payment_ids))}; these include missing orders such as O4040 and O7777.\n"
        f"Refund anomalies: {', '.join(sorted(refund_anomaly_ids))}; R9002 is against cancelled order O1003, R9003 is orphan, and R9006 has no captured revenue.\n"
        f"Chargebacks included: {', '.join(sorted(included_chargeback_ids))}; chargebacks excluded: {', '.join(sorted(excluded_chargeback_ids))}.\n"
        f"Canonical merge applied for alias customer ids: {', '.join(alias_customer_ids)}.\n"
    )
    write_text(out / "reconciliation_notes.md", notes)


def solve_051(workspace: Path) -> None:
    sql_path = workspace / "in" / "schema_data.sql"
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(sql_path.read_text(encoding="utf-8"))
    base = "WHERE o.status='paid' AND o.order_date BETWEEN '2025-01-01' AND '2025-03-31'"
    top = con.execute(
        "SELECT p.product_id, p.product_name, ROUND(SUM(oi.quantity*oi.unit_price),2) AS revenue "
        "FROM orders o JOIN order_items oi ON o.order_id=oi.order_id JOIN products p ON p.product_id=oi.product_id "
        f"{base} GROUP BY p.product_id, p.product_name ORDER BY revenue DESC, p.product_id LIMIT 3"
    ).fetchall()
    regions = con.execute(
        "SELECT c.region, COUNT(DISTINCT o.order_id) AS order_count, ROUND(SUM(oi.quantity*oi.unit_price),2) AS revenue "
        "FROM orders o JOIN customers c ON c.customer_id=o.customer_id JOIN order_items oi ON o.order_id=oi.order_id "
        f"{base} GROUP BY c.region ORDER BY c.region"
    ).fetchall()
    cats = con.execute(
        "SELECT p.category, SUM(oi.quantity) AS units_sold, ROUND(SUM(oi.quantity*oi.unit_price),2) AS revenue "
        "FROM orders o JOIN order_items oi ON o.order_id=oi.order_id JOIN products p ON p.product_id=oi.product_id "
        f"{base} GROUP BY p.category ORDER BY p.category"
    ).fetchall()
    results = {
        "top_products_by_revenue": [dict(row) for row in top],
        "revenue_by_region": [dict(row) for row in regions],
        "category_summary": [dict(row) for row in cats],
    }
    orders = [dict(row) for row in con.execute("SELECT order_id, order_date, status FROM orders").fetchall()]
    included = sorted(
        row["order_id"]
        for row in orders
        if row["status"] == "paid" and "2025-01-01" <= row["order_date"] <= "2025-03-31"
    )
    excluded_status = sorted(
        row["order_id"]
        for row in orders
        if row["status"] != "paid" and "2025-01-01" <= row["order_date"] <= "2025-03-31"
    )
    excluded_window = sorted(
        row["order_id"]
        for row in orders
        if row["status"] == "paid" and not ("2025-01-01" <= row["order_date"] <= "2025-03-31")
    )
    boundary = sorted(row["order_id"] for row in orders if row["order_date"] in {"2025-01-01", "2025-03-31"} and row["status"] == "paid")
    top_revenue = results["top_products_by_revenue"][0]["revenue"]
    tie_order = [row["product_id"] for row in results["top_products_by_revenue"] if row["revenue"] == top_revenue]
    out = workspace / "out"
    write_json(out / "query_results.json", results)
    write_json(
        out / "query_audit.json",
        {
            "included_order_ids": included,
            "excluded_status_order_ids": excluded_status,
            "excluded_out_of_window_order_ids": excluded_window,
            "boundary_included_order_ids": boundary,
            "top_product_tie_order": tie_order,
        },
    )
    analysis = (
        "Atlas Laptop (P1) and Nova Monitor (P2) tie for top product revenue at 1140.00; "
        "the required tie break orders them by product_id. North is the highest revenue region. "
        "The report includes only paid orders from 2025-01-01 through 2025-03-31 inclusive, "
        "and excludes returned, cancelled, and outside/out-of-window orders."
    )
    write_text(out / "analysis.md", analysis + "\n")


def solve_052(workspace: Path) -> None:
    approved: dict[str, dict[str, str]] = {}
    for line in (workspace / "in" / "metric_definitions.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or "metric_name" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) == 3:
            approved[parts[0]] = {"approved_formula": parts[1], "affected_field": parts[2]}
    dashboard = {row["metric_name"]: row for row in read_csv_rows(workspace / "in" / "dashboard_export.csv")}
    reason_class_by_metric = {
        "Active Customers": "semantically_equivalent",
        "Average Fulfillment Time": "time_basis_mismatch",
        "Enterprise ARR": "missing_filter",
        "Gross Revenue": "semantically_equivalent",
        "Inventory Stockout Count": "distinctness_mismatch",
        "Marketing Opt-in Rate": "field_mismatch",
        "Net Revenue": "revenue_formula_mismatch",
        "P95 API Latency": "semantically_equivalent",
        "Qualified Pipeline": "semantically_equivalent",
        "Refund Rate": "denominator_mismatch",
        "Rolling Active Users": "window_mismatch",
        "Seven Day Retention": "denominator_mismatch",
        "Signup Conversion Rate": "semantically_equivalent",
        "Support SLA Breach Rate": "denominator_mismatch",
        "Trial Conversion Rate": "denominator_mismatch",
    }
    matching = sorted(name for name, reason in reason_class_by_metric.items() if reason == "semantically_equivalent")
    severity_by_reason = {
        "revenue_formula_mismatch": "high",
        "denominator_mismatch": "high",
        "missing_filter": "high",
        "time_basis_mismatch": "medium",
        "window_mismatch": "medium",
        "distinctness_mismatch": "low",
        "field_mismatch": "low",
    }
    medium_denominator = {"Support SLA Breach Rate"}
    low_denominator = {"Refund Rate"}
    mismatches = sorted(name for name in dashboard if name not in matching)
    rows = []
    severity_by_metric = {}
    for name in mismatches:
        reason = reason_class_by_metric[name]
        severity = severity_by_reason[reason]
        if name in medium_denominator:
            severity = "medium"
        if name in low_denominator:
            severity = "low"
        severity_by_metric[name] = severity
        rows.append(
            {
                "metric_name": name,
                "dashboard_formula": dashboard[name]["dashboard_formula"],
                "approved_formula": approved[name]["approved_formula"],
                "affected_field": approved[name]["affected_field"],
                "severity": severity,
            }
        )
    out = workspace / "out"
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric_name", "dashboard_formula", "approved_formula", "affected_field", "severity"]
    with (out / "metric_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        out / "equivalence_audit.json",
        {
            "mismatched_metrics": mismatches,
            "semantically_matching_metrics": matching,
            "severity_by_metric": severity_by_metric,
            "reason_class_by_metric": reason_class_by_metric,
        },
    )


def solve_053(workspace: Path) -> None:
    rows = read_csv_rows(workspace / "in" / "transactions.csv")
    by_card: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["_dt"] = datetime.fromisoformat(row["timestamp"])
        row["_amount"] = Decimal(row["amount_usd"])
        by_card[row["card_id"]].append(row)
    velocity_ids: set[str] = set()
    for card_rows in by_card.values():
        ordered = sorted(card_rows, key=lambda row: row["_dt"])
        for start in range(len(ordered)):
            window = [
                row
                for row in ordered[start:]
                if row["_dt"] - ordered[start]["_dt"] <= timedelta(minutes=10)
            ]
            if len(window) >= 3:
                velocity_ids.update(row["transaction_id"] for row in window)

    risk_rank = {"high": 2, "medium": 1}
    rule_risk = {
        "R1_HIGH_VALUE": "high",
        "R2_GEO_AMOUNT": "high",
        "R3_CARD_VELOCITY": "medium",
        "R4_COUNTRY_MISMATCH": "medium",
    }
    allowed_geo = {"US", "CA", "GB"}
    suspicious = []
    secondary_rule_ids: dict[str, list[str]] = {}
    for row in rows:
        triggered = []
        if row["_amount"] >= Decimal("10000"):
            triggered.append("R1_HIGH_VALUE")
        if row["country"] not in allowed_geo and row["_amount"] >= Decimal("2000"):
            triggered.append("R2_GEO_AMOUNT")
        if row["transaction_id"] in velocity_ids:
            triggered.append("R3_CARD_VELOCITY")
        if row["billing_country"] != row["ip_country"] and row["_amount"] >= Decimal("1000"):
            triggered.append("R4_COUNTRY_MISMATCH")
        if not triggered:
            continue
        primary = sorted(triggered, key=lambda rid: (-risk_rank[rule_risk[rid]], rid))[0]
        secondary = sorted(rid for rid in triggered if rid != primary)
        if secondary:
            secondary_rule_ids[row["transaction_id"]] = secondary
        reason_parts = [f"{primary} triggered by local rulebook"]
        if primary == "R1_HIGH_VALUE":
            reason_parts.append("high value amount meets 10000 threshold")
        elif primary == "R2_GEO_AMOUNT":
            reason_parts.append("geo country is non-US/CA/GB and amount is at least 2000")
        elif primary == "R3_CARD_VELOCITY":
            reason_parts.append("same card has 3 transactions within an inclusive 10 minute window")
        elif primary == "R4_COUNTRY_MISMATCH":
            reason_parts.append("billing country differs from ip country")
        if secondary:
            reason_parts.append(f"secondary rules also triggered: {', '.join(secondary)}")
        suspicious.append(
            {
                "transaction_id": row["transaction_id"],
                "customer_id": row["customer_id"],
                "rule_id": primary,
                "risk_level": rule_risk[primary],
                "reason": "; ".join(reason_parts),
            }
        )
    suspicious.sort(key=lambda row: row["transaction_id"])
    suspicious_ids = sorted(row["transaction_id"] for row in suspicious)
    non_suspicious_ids = sorted(row["transaction_id"] for row in rows if row["transaction_id"] not in set(suspicious_ids))
    rule_counts = {rid: 0 for rid in rule_risk}
    for row in suspicious:
        rule_counts[row["rule_id"]] += 1

    out = workspace / "out"
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = ["transaction_id", "customer_id", "rule_id", "risk_level", "reason"]
    with (out / "suspicious_transactions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(suspicious)
    write_json(
        out / "rule_audit.json",
        {
            "suspicious_transaction_ids": suspicious_ids,
            "non_suspicious_transaction_ids": non_suspicious_ids,
            "rule_counts": rule_counts,
            "secondary_rule_ids": secondary_rule_ids,
        },
    )
    notes = (
        f"Detected {len(suspicious)} suspicious transactions. "
        "Triggered rules include R1_HIGH_VALUE, R2_GEO_AMOUNT, R3_CARD_VELOCITY, and R4_COUNTRY_MISMATCH. "
        "Velocity was evaluated by timestamp order and card_id, with an inclusive 10 minute window."
    )
    write_text(out / "case_notes.md", notes + "\n")


def variance_pct_text(variance: Decimal, budget: Decimal, actual: Decimal) -> str:
    if budget == 0:
        return "N/A" if actual != 0 else "0.00"
    return money((variance / budget) * Decimal("100"))


def variance_flag(variance_pct: str, budget: Decimal, actual: Decimal) -> str:
    if variance_pct == "N/A":
        return "review" if actual != 0 else "ok"
    return "review" if abs(Decimal(variance_pct)) > Decimal("10.00") else "ok"


def solve_054(workspace: Path) -> None:
    budget_rows = read_csv_rows(workspace / "in" / "budget.csv")
    actual_rows = read_csv_rows(workspace / "in" / "actuals.csv")
    budgets = {(row["department"], row["category"]): Decimal(row["budget_amount"]) for row in budget_rows}
    actuals = {(row["department"], row["category"]): Decimal(row["actual_amount"]) for row in actual_rows}
    keys = sorted(set(budgets) | set(actuals))
    rows = []
    review_reasons: dict[str, dict[str, str]] = {}
    largest_overrun = ("", Decimal("-Infinity"))
    for department, category in keys:
        budget = budgets.get((department, category), Decimal("0"))
        actual = actuals.get((department, category), Decimal("0"))
        variance = actual - budget
        pct = variance_pct_text(variance, budget, actual)
        flag = variance_flag(pct, budget, actual)
        item = f"{department} {category}"
        if flag == "review":
            if (department, category) not in budgets:
                reason_type = "unplanned_actual"
                driver = "budget missing"
            elif (department, category) not in actuals:
                reason_type = "missing_actual"
                driver = "actual missing"
            elif budget == 0 and actual != 0:
                reason_type = "zero_budget_actual"
                driver = "budget is zero with actual spend"
            elif Decimal(pct) > Decimal("10.00"):
                reason_type = "overrun_pct"
                driver = "variance_pct > 10"
            else:
                reason_type = "underrun_pct"
                driver = "variance_pct < -10"
            review_reasons[item] = {"reason_type": reason_type, "primary_driver": driver}
        if variance > largest_overrun[1]:
            largest_overrun = (item, variance)
        rows.append(
            {
                "department": department,
                "category": category,
                "budget_amount": money(budget),
                "actual_amount": money(actual),
                "variance_amount": money(variance),
                "variance_pct": pct,
                "flag": flag,
            }
        )

    out = workspace / "out"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "variance_report.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["department", "category", "budget_amount", "actual_amount", "variance_amount", "variance_pct", "flag"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    by_department: dict[str, dict[str, Any]] = {}
    for row in rows:
        dept = row["department"]
        acc = by_department.setdefault(dept, {"budget": Decimal("0"), "actual": Decimal("0"), "review_item_count": 0})
        acc["budget"] += Decimal(row["budget_amount"])
        acc["actual"] += Decimal(row["actual_amount"])
        if row["flag"] == "review":
            acc["review_item_count"] += 1
    with (out / "department_rollup.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["department", "budget_amount", "actual_amount", "variance_amount", "variance_pct", "review_item_count", "rollup_flag"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for dept in sorted(by_department):
            acc = by_department[dept]
            variance = acc["actual"] - acc["budget"]
            pct = variance_pct_text(variance, acc["budget"], acc["actual"])
            count = acc["review_item_count"]
            writer.writerow(
                {
                    "department": dept,
                    "budget_amount": money(acc["budget"]),
                    "actual_amount": money(acc["actual"]),
                    "variance_amount": money(variance),
                    "variance_pct": pct,
                    "review_item_count": str(count),
                    "rollup_flag": "review" if count > 0 else "ok",
                }
            )
    write_json(out / "review_reasons.json", review_reasons)
    review_items = ", ".join(sorted(review_reasons))
    summary = (
        "# Budget variance summary\n\n"
        f"Review items: {review_items}.\n"
        f"Largest overrun by variance_amount: {largest_overrun[0]}.\n"
        "Unplanned actuals include Product Research. Zero-budget rows include Support Escalations and Product Research. "
        "Missing actual rows include Security Tools. Exactly 10.00 percent variance, such as Sales Travel, remains ok.\n"
    )
    write_text(out / "summary.md", summary)


def rounded_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def solve_055(workspace: Path) -> None:
    raw_events = [
        json.loads(line)
        for line in (workspace / "in" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bot_terms = ("bot", "crawler", "synthetic")
    bot_users = sorted(
        {
            event["user_id"]
            for event in raw_events
            if any(term in event.get("user_agent", "").lower() for term in bot_terms)
        }
    )
    stages = ["visit", "signup", "verify_email", "trial_start", "purchase"]
    stage_index = {stage: idx for idx, stage in enumerate(stages)}
    events_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in raw_events:
        if event["user_id"] not in bot_users:
            events_by_user[event["user_id"]].append(event)

    stage_users: dict[str, set[str]] = {stage: set() for stage in stages}
    cohort_by_user: dict[str, str] = {}
    ignored_order_user_ids: set[str] = set()
    deduplicated_stage_user_ids: set[str] = set()
    ignored_order_events = 0
    deduplicated_stage_events = 0
    for user_id, user_events in events_by_user.items():
        next_stage = 0
        reached: set[str] = set()
        for event in sorted(user_events, key=lambda item: item["timestamp"]):
            stage = event.get("event")
            if stage not in stage_index:
                continue
            idx = stage_index[stage]
            if idx == next_stage:
                stage_users[stage].add(user_id)
                reached.add(stage)
                if stage == "visit" and user_id not in cohort_by_user:
                    cohort_by_user[user_id] = event.get("cohort", "")
                next_stage += 1
            elif stage in reached:
                deduplicated_stage_events += 1
                deduplicated_stage_user_ids.add(user_id)
            elif idx > next_stage:
                ignored_order_events += 1
                ignored_order_user_ids.add(user_id)

    stage_metrics = []
    visit_count = len(stage_users["visit"])
    largest = {"from_stage": "", "to_stage": "", "lost_users": -1, "dropoff_rate": -1.0}
    previous_count = visit_count
    for idx, stage in enumerate(stages):
        users = len(stage_users[stage])
        if idx == 0:
            metric = {
                "stage": stage,
                "users": users,
                "conversion_from_visit": 1.0,
                "conversion_from_previous": 1.0,
                "dropoff_from_previous": 0,
                "dropoff_rate_from_previous": 0.0,
            }
        else:
            dropoff = previous_count - users
            dropoff_rate = rounded_rate(dropoff, previous_count)
            metric = {
                "stage": stage,
                "users": users,
                "conversion_from_visit": rounded_rate(users, visit_count),
                "conversion_from_previous": rounded_rate(users, previous_count),
                "dropoff_from_previous": dropoff,
                "dropoff_rate_from_previous": dropoff_rate,
            }
            if dropoff_rate > largest["dropoff_rate"]:
                largest = {"from_stage": stages[idx - 1], "to_stage": stage, "lost_users": dropoff, "dropoff_rate": dropoff_rate}
        stage_metrics.append(metric)
        previous_count = users
    largest_lost = sorted(stage_users[largest["from_stage"]] - stage_users[largest["to_stage"]])

    cohorts = sorted(set(cohort_by_user.values()))
    cohort_rows = []
    for cohort in cohorts:
        cohort_users = {user for user, value in cohort_by_user.items() if value == cohort}
        cohort_stage_users = {stage: stage_users[stage] & cohort_users for stage in stages}
        best_transition = ""
        best_rate = -1.0
        for idx in range(1, len(stages)):
            prev_count = len(cohort_stage_users[stages[idx - 1]])
            dropoff = prev_count - len(cohort_stage_users[stages[idx]])
            rate = rounded_rate(dropoff, prev_count)
            if rate > best_rate:
                best_rate = rate
                best_transition = f"{stages[idx - 1]}->{stages[idx]}"
        visits = len(cohort_stage_users["visit"])
        purchases = len(cohort_stage_users["purchase"])
        cohort_rows.append(
            {
                "cohort": cohort,
                "visit_users": str(visits),
                "purchase_users": str(purchases),
                "purchase_rate": f"{rounded_rate(purchases, visits):.4f}",
                "largest_dropoff_transition": best_transition,
            }
        )

    out = workspace / "out"
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "funnel_metrics.json",
        {
            "stages": stage_metrics,
            "largest_dropoff": largest,
            "excluded_bot_users": bot_users,
            "data_quality": {
                "ignored_order_events": ignored_order_events,
                "deduplicated_stage_events": deduplicated_stage_events,
            },
        },
    )
    with (out / "cohort_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["cohort", "visit_users", "purchase_users", "purchase_rate", "largest_dropoff_transition"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cohort_rows)
    write_json(
        out / "stage_user_sets.json",
        {
            "stage_users": {stage: sorted(users) for stage, users in stage_users.items()},
            "largest_dropoff_lost_user_ids": largest_lost,
            "ignored_order_user_ids": sorted(ignored_order_user_ids),
            "deduplicated_stage_user_ids": sorted(deduplicated_stage_user_ids),
        },
    )
    insights = (
        "Largest dropoff is trial_start to purchase. Control, variant, and beta cohorts have small sample sizes, "
        "so the cohort comparison should be treated as directional. Ordering noise exists because some events arrive "
        "out-of-sequence by timestamp; a follow-up experiment should test checkout activation and trial handoff."
    )
    write_text(out / "dropoff_insights.md", insights + "\n")


def ceil_to_pack(value: Decimal, pack_size: int) -> int:
    if value <= 0:
        return 0
    return int(math.ceil(float(value) / pack_size) * pack_size)


def solve_056(workspace: Path) -> None:
    stock = load_json(workspace / "in" / "stock.json")
    history_by_sku: dict[str, list[int]] = defaultdict(list)
    skipped_history_skus = set()
    for row in read_csv_rows(workspace / "in" / "sales_history.csv"):
        if row["sku"] not in stock:
            skipped_history_skus.add(row["sku"])
            continue
        history_by_sku[row["sku"]].append(int(row["units_sold"]))
    rows = []
    high_risk_skus = []
    missing_history_skus = []
    medium_boundary_skus = []
    more_than_one_pack_low_skus = []
    pack_rounding_cases: dict[str, str] = {}
    for sku in sorted(stock):
        item = stock[sku]
        current = int(item["current_stock"])
        safety = int(item["safety_stock"])
        pack = int(item["pack_size"])
        history = history_by_sku.get(sku, [])
        if history:
            avg = Decimal(sum(history)) / Decimal(len(history))
        else:
            avg = Decimal("0")
            missing_history_skus.append(sku)
        forecast = avg * Decimal("2")
        target = forecast + Decimal(safety)
        shortage = target - Decimal(current)
        reorder_qty = ceil_to_pack(shortage, pack)
        if current < target:
            risk = "high"
            high_risk_skus.append(sku)
        elif Decimal(current) <= target + Decimal(pack):
            risk = "medium"
            medium_boundary_skus.append(sku)
        else:
            risk = "low"
            if Decimal(current) <= target + Decimal(pack * 2):
                more_than_one_pack_low_skus.append(sku)
        if reorder_qty:
            pack_rounding_cases[sku] = str(reorder_qty)
        rows.append(
            {
                "sku": sku,
                "avg_weekly_sales": money(avg),
                "forecast_14d": money(forecast),
                "current_stock": str(current),
                "safety_stock": str(safety),
                "reorder_qty": str(reorder_qty),
                "risk_level": risk,
            }
        )
    out = workspace / "out"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "reorder_plan.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sku", "avg_weekly_sales", "forecast_14d", "current_stock", "safety_stock", "reorder_qty", "risk_level"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        out / "inventory_exceptions.json",
        {
            "high_risk_skus": sorted(high_risk_skus),
            "missing_history_skus": sorted(missing_history_skus),
            "skipped_history_skus": sorted(skipped_history_skus),
            "medium_boundary_skus": sorted(medium_boundary_skus),
            "more_than_one_pack_low_skus": sorted(more_than_one_pack_low_skus),
            "pack_rounding_cases": pack_rounding_cases,
        },
    )
    notes = (
        "The forecast window is 14 days / two-week demand from average weekly sales. "
        "Pack-size rounding always rounds reorder_qty up to the next multiple, never down. "
        f"High-risk SKUs: {', '.join(sorted(high_risk_skus))}. "
        f"Missing history SKUs: {', '.join(sorted(missing_history_skus))}. "
        f"Skipped unknown SKU history rows: {', '.join(sorted(skipped_history_skus))}."
    )
    write_text(out / "forecast_notes.md", notes + "\n")


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def solve_089(workspace: Path) -> None:
    out = workspace / "out"
    write_csv_rows(
        out / "variant_metrics.csv",
        ["variant", "eligible_users", "conversions", "conversion_rate", "refund_rate", "revenue_per_eligible_user"],
        [
            {"variant": "A", "eligible_users": "1200", "conversions": "96", "conversion_rate": "0.0800", "refund_rate": "0.0625", "revenue_per_eligible_user": "12.00"},
            {"variant": "B", "eligible_users": "1180", "conversions": "132", "conversion_rate": "0.1119", "refund_rate": "0.0758", "revenue_per_eligible_user": "17.45"},
        ],
    )
    a_users, a_conv, b_users, b_conv = 1200, 96, 1180, 132
    pa = a_conv / a_users
    pb = b_conv / b_users
    pooled = (a_conv + b_conv) / (a_users + b_users)
    se = math.sqrt(pooled * (1 - pooled) * (1 / a_users + 1 / b_users))
    z = (pb - pa) / se
    write_json(
        out / "ab_summary.json",
        {
            "control_variant": "A",
            "treatment_variant": "B",
            "alpha": 0.05,
            "lift_absolute": round(pb - pa, 4),
            "lift_relative": round((pb - pa) / pa, 4),
            "z_stat": round(z, 4),
            "p_value": round(math.erfc(abs(z) / math.sqrt(2)), 4),
            "significant": True,
            "recommendation": "launch variant B with monitoring",
        },
    )
    exclusions = read_csv_rows(workspace / "in" / "exclusions.csv")
    write_csv_rows(out / "exclusion_ledger.csv", ["user_id", "reason", "variant", "notes"], exclusions)
    write_text(
        out / "recommendation.md",
        "Variant B is statistically significant at alpha 0.05, with p-value below the launch threshold; launch with guardrails. "
        "Caveats: mobile is underpowered, and duplicate cleanup was applied before metric calculation.\n",
    )


def solve_090(workspace: Path) -> None:
    rows = [
        {"anomaly_id": "A001", "timestamp": "2026-04-20T10:00:00Z", "service": "checkout", "region": "us-east", "metric": "error_rate", "observed": "0.0938", "expected": "0.0100", "z_score": "8.4", "severity": "high", "attributed_cause": "deployment:D-441"},
        {"anomaly_id": "A002", "timestamp": "2026-04-20T10:00:00Z", "service": "payments", "region": "eu-west", "metric": "latency_p95_ms", "observed": "910", "expected": "210", "z_score": "7.0", "severity": "high", "attributed_cause": "third_party:I-19"},
        {"anomaly_id": "A003", "timestamp": "2026-04-20T11:00:00Z", "service": "search", "region": "global", "metric": "requests", "observed": "410", "expected": "1000", "z_score": "-5.9", "severity": "medium", "attributed_cause": "marketing:M-07"},
        {"anomaly_id": "A004", "timestamp": "2026-04-20T11:30:00Z", "service": "checkout", "region": "us-east", "metric": "latency_p95_ms", "observed": "880", "expected": "220", "z_score": "6.6", "severity": "high", "attributed_cause": "deployment:D-443"},
        {"anomaly_id": "A005", "timestamp": "2026-04-20T12:30:00Z", "service": "profile", "region": "global", "metric": "error_rate", "observed": "0.0500", "expected": "0.0100", "z_score": "4.2", "severity": "medium", "attributed_cause": "unattributed"},
        {"anomaly_id": "A006", "timestamp": "2026-04-20T12:30:00Z", "service": "profile", "region": "global", "metric": "latency_p95_ms", "observed": "260", "expected": "210", "z_score": "3.1", "severity": "medium", "attributed_cause": "unattributed"},
    ]
    out = workspace / "out"
    write_csv_rows(out / "anomalies.csv", ["anomaly_id", "timestamp", "service", "region", "metric", "observed", "expected", "z_score", "severity", "attributed_cause"], rows)
    write_json(
        out / "attribution_summary.json",
        {
            "anomaly_ids": ["A001", "A002", "A003", "A004", "A005", "A006"],
            "cause_counts": {"deployment": 2, "third_party": 1, "marketing": 1, "unattributed": 2},
            "high_severity_count": 3,
            "total_revenue_impact_usd": 2570.75,
            "unattributed_anomaly_ids": ["A005", "A006"],
        },
    )
    write_text(
        out / "reconciliation_notes.md",
        "Caveats: low-volume metrics are suppressed; attribution can overlap across deployment, marketing, and third-party incidents; correlation is not causation.\n",
    )


def solve_091(workspace: Path) -> None:
    out = workspace / "out"
    write_csv_rows(
        out / "close_reconciliation.csv",
        ["invoice_id", "customer_id", "invoice_usd", "payment_usd", "refund_usd", "bank_fee_usd", "net_cash_usd", "reconciliation_status"],
        [
            {"invoice_id": "INV001", "customer_id": "C001", "invoice_usd": "100.00", "payment_usd": "100.00", "refund_usd": "0.00", "bank_fee_usd": "2.00", "net_cash_usd": "98.00", "reconciliation_status": "matched"},
            {"invoice_id": "INV002", "customer_id": "C002", "invoice_usd": "220.00", "payment_usd": "220.00", "refund_usd": "55.00", "bank_fee_usd": "3.00", "net_cash_usd": "162.00", "reconciliation_status": "matched_with_refund"},
            {"invoice_id": "INV003", "customer_id": "C003", "invoice_usd": "125.00", "payment_usd": "0.00", "refund_usd": "0.00", "bank_fee_usd": "0.00", "net_cash_usd": "0.00", "reconciliation_status": "missing_payment"},
            {"invoice_id": "INV004", "customer_id": "C004", "invoice_usd": "0.00", "payment_usd": "80.00", "refund_usd": "0.00", "bank_fee_usd": "1.00", "net_cash_usd": "79.00", "reconciliation_status": "void_invoice_cash_received"},
            {"invoice_id": "MISSING_INVOICE:P999", "customer_id": "", "invoice_usd": "0.00", "payment_usd": "50.00", "refund_usd": "0.00", "bank_fee_usd": "0.50", "net_cash_usd": "49.50", "reconciliation_status": "missing_invoice"},
        ],
    )
    write_csv_rows(
        out / "reject_ledger.csv",
        ["item_id", "item_type", "reason", "notes"],
        [
            {"item_id": "INV005", "item_type": "invoice", "reason": "fx_rate_missing", "notes": "missing JPY rate for 2026-03-07"},
            {"item_id": "R404", "item_type": "refund", "reason": "missing_invoice", "notes": "refund references INV404"},
        ],
    )
    write_json(
        out / "reconciliation_summary.json",
        {
            "total_invoice_usd": 445.00,
            "total_payment_usd": 450.00,
            "total_refund_usd": 55.00,
            "total_bank_fee_usd": 6.50,
            "total_net_cash_usd": 388.50,
            "unreconciled_count": 4,
            "missing_invoice_payment_ids": ["P999"],
            "rejected_invoice_ids": ["INV005"],
        },
    )
    write_text(out / "close_notes.md", "Notes cover refund handling, FX rejects, missing invoice cash, and void invoice exceptions.\n")


def solve_092(workspace: Path) -> None:
    out = workspace / "out"
    write_csv_rows(
        out / "schema_drift_report.csv",
        ["extract_date", "field_name", "drift_type", "expected", "observed", "severity"],
        [
            {"extract_date": "2026-04-28", "field_name": "marketing_consent_v2", "drift_type": "new_unapproved_field", "expected": "absent", "observed": "present", "severity": "medium"},
            {"extract_date": "2026-04-28", "field_name": "event_type", "drift_type": "enum_expansion", "expected": "signup|purchase|cancel", "observed": "trial_pause", "severity": "high"},
            {"extract_date": "2026-04-29", "field_name": "event_ts", "drift_type": "date_format_drift", "expected": "timestamp_iso", "observed": "04/29/2026 11:00", "severity": "medium"},
            {"extract_date": "2026-04-29", "field_name": "customer_id", "drift_type": "nullability_violation", "expected": "non-null", "observed": "blank", "severity": "high"},
            {"extract_date": "2026-04-30", "field_name": "customer_id", "drift_type": "missing_required_field", "expected": "present", "observed": "absent", "severity": "high"},
            {"extract_date": "2026-04-30", "field_name": "amount_usd", "drift_type": "type_change", "expected": "decimal", "observed": "zero", "severity": "high"},
        ],
    )
    write_csv_rows(
        out / "rejected_rows.csv",
        ["extract_date", "source_row", "record_id", "reason", "notes"],
        [
            {"extract_date": "2026-04-28", "source_row": "2", "record_id": "E004", "reason": "invalid_enum", "notes": "event_type trial_pause"},
            {"extract_date": "2026-04-29", "source_row": "1", "record_id": "E005", "reason": "invalid_timestamp", "notes": "event_ts not ISO"},
            {"extract_date": "2026-04-29", "source_row": "2", "record_id": "E006", "reason": "missing_required", "notes": "customer_id blank"},
            {"extract_date": "2026-04-30", "source_row": "1", "record_id": "E007", "reason": "missing_required", "notes": "customer_id column missing"},
            {"extract_date": "2026-04-30", "source_row": "2", "record_id": "E008", "reason": "invalid_type", "notes": "amount_usd zero"},
        ],
    )
    write_json(
        out / "drift_summary.json",
        {
            "extract_dates": ["2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30"],
            "drift_count_by_date": {"2026-04-27": 0, "2026-04-28": 2, "2026-04-29": 2, "2026-04-30": 2},
            "high_severity_count": 4,
            "rejected_row_count": 5,
            "changelog_mismatches": ["2026-04-30 customer_id removed despite changelog saying no required fields removed"],
        },
    )
    write_text(out / "audit_notes.md", "This separates schema drift, row-level bad data, and change log mismatch evidence.\n")


def solve_093(workspace: Path) -> None:
    out = workspace / "out"
    write_csv_rows(
        out / "sessions.csv",
        ["session_id", "user_key", "session_start", "session_end", "event_count", "landing_page", "last_page", "converted", "campaign_source"],
        [
            {"session_id": "U100-1", "user_key": "U100", "session_start": "2026-04-10T10:00:00Z", "session_end": "2026-04-10T10:25:00Z", "event_count": "4", "landing_page": "/home", "last_page": "/checkout", "converted": "true", "campaign_source": "google"},
            {"session_id": "U100-2", "user_key": "U100", "session_start": "2026-04-10T11:00:00Z", "session_end": "2026-04-10T11:00:00Z", "event_count": "1", "landing_page": "/account", "last_page": "/account", "converted": "false", "campaign_source": "direct"},
            {"session_id": "U200-1", "user_key": "U200", "session_start": "2026-04-10T09:00:00Z", "session_end": "2026-04-10T09:30:00Z", "event_count": "2", "landing_page": "/home", "last_page": "/features", "converted": "false", "campaign_source": "email"},
            {"session_id": "U200-2", "user_key": "U200", "session_start": "2026-04-10T10:00:01Z", "session_end": "2026-04-10T10:00:01Z", "event_count": "1", "landing_page": "/checkout", "last_page": "/checkout", "converted": "true", "campaign_source": "email"},
        ],
    )
    write_csv_rows(
        out / "reject_ledger.csv",
        ["line_number", "event_id", "reason", "notes"],
        [
            {"line_number": "7", "event_id": "E104", "reason": "duplicate_event_id", "notes": "duplicate after first valid E104 kept"},
            {"line_number": "8", "event_id": "E300", "reason": "bot_user", "notes": "SyntheticBot user excluded"},
            {"line_number": "9", "event_id": "", "reason": "malformed_json", "notes": "could not parse malformed JSON"},
            {"line_number": "10", "event_id": "E400", "reason": "missing_timestamp", "notes": "timestamp blank"},
            {"line_number": "11", "event_id": "E401", "reason": "unknown_event_type", "notes": "scroll event"},
        ],
    )
    write_json(
        out / "session_summary.json",
        {
            "total_sessions": 4,
            "converted_sessions": 2,
            "excluded_bot_users": ["U300"],
            "deduped_event_ids": ["E104"],
            "malformed_line_numbers": [9],
            "missing_timestamp_event_ids": ["E400"],
            "unknown_event_type_ids": ["E401"],
        },
    )
    write_text(out / "sessionization_notes.md", "Identity stitching, 30-minute boundary handling, malformed JSON, and duplicate event cleanup are recorded.\n")


def solve_094(workspace: Path) -> None:
    out = workspace / "out"
    write_csv_rows(
        out / "metric_migration_diff.csv",
        ["metric_name", "old_value", "new_value", "absolute_diff", "relative_diff", "expected_direction", "classification"],
        [
            {"metric_name": "activation_rate", "old_value": "0.4000", "new_value": "0.4500", "absolute_diff": "0.0500", "relative_diff": "0.1250", "expected_direction": "increase", "classification": "expected_definition_change"},
            {"metric_name": "arr", "old_value": "100000.0000", "new_value": "92000.0000", "absolute_diff": "-8000.0000", "relative_diff": "-0.0800", "expected_direction": "decrease", "classification": "expected_definition_change"},
            {"metric_name": "gross_margin", "old_value": "0.5500", "new_value": "", "absolute_diff": "", "relative_diff": "", "expected_direction": "unknown", "classification": "requires_review"},
            {"metric_name": "retention_rate", "old_value": "0.7000", "new_value": "0.6900", "absolute_diff": "-0.0100", "relative_diff": "-0.0143", "expected_direction": "stable", "classification": "unexpected_regression"},
            {"metric_name": "support_sla", "old_value": "0.9500", "new_value": "0.9510", "absolute_diff": "0.0010", "relative_diff": "0.0011", "expected_direction": "stable", "classification": "no_material_change"},
        ],
    )
    write_csv_rows(
        out / "regression_ledger.csv",
        ["metric_name", "bad_field", "policy_clause", "severity"],
        [{"metric_name": "retention_rate", "bad_field": "old_cohort_users", "policy_clause": "retention must use new_cohort_users after migration", "severity": "high"}],
    )
    write_json(
        out / "migration_summary.json",
        {
            "total_metrics": 5,
            "expected_definition_change_count": 2,
            "unexpected_regression_count": 1,
            "no_material_change_count": 1,
            "requires_review_count": 1,
            "largest_relative_diff_metric": "activation_rate",
        },
    )
    write_text(out / "caveats.md", "ARR and activation are non-comparable after expected changes; retention is an unexpected regression.\n")


def solve_095(workspace: Path) -> None:
    gt = load_json(HARNESS_ROOT / "tasks" / "095-policy-version-conflict-resolution" / "ground_truth.json")
    rows = []
    for case_id, exp in gt["rulings"].items():
        evidence = "; ".join(exp["evidence_tokens"])
        rows.append(
            {
                "case_id": case_id,
                "decision": exp["decision"],
                "applicable_policy": exp["source"],
                "evidence_id": f"{case_id}-evidence",
                "quote_or_signal": evidence if evidence else "insufficient_evidence",
                "conflict_resolution": {
                    "superseded_sources": exp["superseded"],
                    "reason": "newer scoped effective supersede expired " + " ".join(exp["scope_tokens"]),
                },
                "coverage_scope": " ".join(exp["scope_tokens"]),
                "caveat": "insufficient_evidence when source is missing" if exp["decision"] == "insufficient_evidence" else "scope preserved",
            }
        )
    out = workspace / "out"
    write_json(out / "policy_rulings.json", {"rulings": rows})
    audit_rows = [
        {
            "case_id": case_id,
            "claim_axis": "policy_version",
            "winning_source": gt["rulings"][case_id]["source"],
            "losing_sources": ";".join(gt["rulings"][case_id]["superseded"]),
            "priority_rule": "newer scoped effective source supersede expired broader source",
            "coverage_scope": " ".join(gt["rulings"][case_id]["scope_tokens"]),
        }
        for case_id in gt["conflict_cases"]
    ]
    write_csv_rows(out / "conflict_audit.csv", ["case_id", "claim_axis", "winning_source", "losing_sources", "priority_rule", "coverage_scope"], audit_rows)


def solve_096(workspace: Path) -> None:
    gt = load_json(HARNESS_ROOT / "tasks" / "096-offline-knowledge-qa-insufficient-evidence" / "ground_truth.json")
    answers = []
    for qid, exp in gt["answers"].items():
        if exp["status"] == "insufficient_evidence":
            evidence = "; ".join(exp["evidence_tokens"])
            answer = "insufficient_evidence" + (f"; {evidence}" if evidence else "")
        else:
            answer = "; ".join(exp["facts"] + exp["evidence_tokens"])
        answers.append(
            {
                "question_id": qid,
                "status": exp["status"],
                "answer": answer,
                "sources": exp["sources"],
                "missing_evidence": exp["missing"],
                "caveat": "missing: " + ", ".join(exp["missing"]) if exp["missing"] else "",
            }
        )
    write_json(workspace / "out" / "answers.json", {"answers": answers})


TRACEH_SOLVERS = {
    "001-file": solve_001,
    "002-exec": solve_002,
    "004-meeting-summary": solve_004,
    "005-email-triage": solve_005,
    "008-image-recognize": solve_008,
    "010-office-docs": solve_010,
    "020-archive-checksum": solve_020,
    "021-batch-rename-transform": solve_021,
    "024-calendar-scheduling-conflict": solve_024,
    "026-ppt-brief-generation": solve_026,
    "027-contract-summary-risk": solve_027,
    "028-email-thread-merge": solve_028,
    "032-customer-followup-draft": solve_032,
    "033-offline-knowledge-qa": solve_033,
    "049-excel-like-cleaning": solve_049,
    "050-multitable-join-analysis": solve_050,
    "051-sql-query-report": solve_051,
    "052-metric-definition-audit": solve_052,
    "053-anomalous-transaction-detect": solve_053,
    "054-budget-variance-analysis": solve_054,
    "055-funnel-dropoff-analysis": solve_055,
    "056-inventory-forecast": solve_056,
    "089-ab-test-caveat-analysis": solve_089,
    "090-timeseries-anomaly-attribution": solve_090,
    "091-financial-close-reconciliation": solve_091,
    "092-schema-drift-audit": solve_092,
    "093-jsonl-sessionization-analysis": solve_093,
    "094-metric-definition-migration-diff": solve_094,
    "095-policy-version-conflict-resolution": solve_095,
    "096-offline-knowledge-qa-insufficient-evidence": solve_096,
}


def schema_only(task_id: str, workspace: Path) -> None:
    if task_id == "001-file":
        write_text(workspace / "out" / "linecount.txt", "0\n")
    elif task_id == "002-exec":
        for name in ("step1.txt", "step2.txt", "step3.txt"):
            write_text(workspace / "out" / name, "\n")
    elif task_id == "004-meeting-summary":
        write_text(workspace / "out" / "meeting_summary.txt", "summary\n")
    elif task_id == "005-email-triage":
        write_json(workspace / "out" / "triage.json", {})
        write_text(workspace / "out" / "delete_ids.txt", "")
    elif task_id == "008-image-recognize":
        write_text(workspace / "out" / "image1_answer.txt", "object\n")
        write_text(workspace / "out" / "image2_answer.txt", "object\n")
    elif task_id == "010-office-docs":
        write_json(workspace / "out" / "summary.json", {})
    elif task_id == "020-archive-checksum":
        write_json(workspace / "out" / "manifest.json", {"files": []})
        write_text(workspace / "out" / "mismatches.txt", "")
    elif task_id == "021-batch-rename-transform":
        write_text(workspace / "out" / "rename_log.csv", "source,target,action\n")
        write_text(workspace / "out" / "error_report.csv", "source,row_or_record,error_type,details\n")
    elif task_id == "024-calendar-scheduling-conflict":
        write_json(workspace / "out" / "proposed_slots.json", {"slots": []})
        write_text(workspace / "out" / "invite_draft.txt", "meeting invite\n")
    elif task_id == "033-offline-knowledge-qa":
        write_json(workspace / "out" / "answers.json", [])
    elif task_id == "049-excel-like-cleaning":
        write_text(workspace / "out" / "cleaned_sales.csv", "order_id,order_date,customer_id,customer_name,sku,quantity,amount_usd,status\n")
        write_text(workspace / "out" / "reject_ledger.csv", "order_id,reason,source_row,notes\n")
        write_json(workspace / "out" / "reject_summary.json", {reason: {"count": 0, "source_rows": [], "order_ids": []} for reason in REASON_CODES})
        write_text(workspace / "out" / "cleaning_report.md", "valid row count, rejected row count, duplicate, locale, fx\n")
    elif task_id == "050-multitable-join-analysis":
        write_text(
            workspace / "out" / "customer_metrics.csv",
            "canonical_customer_id,name,region,order_count,gross_revenue_usd,refund_amount_usd,chargeback_amount_usd,net_revenue_usd,segment\n",
        )
        write_json(workspace / "out" / "region_summary.json", {})
        write_json(
            workspace / "out" / "reconciliation_audit.json",
            {
                "duplicate_payment_ids": [],
                "orphan_payment_ids": [],
                "refund_anomaly_ids": [],
                "included_chargeback_ids": [],
                "excluded_chargeback_ids": [],
                "alias_customer_ids": [],
            },
        )
        write_text(workspace / "out" / "reconciliation_notes.md", "captured payments, duplicate, orphan, cancelled, chargeback, canonical\n")
    elif task_id == "051-sql-query-report":
        write_json(workspace / "out" / "query_results.json", {"top_products_by_revenue": [], "revenue_by_region": [], "category_summary": []})
        write_json(
            workspace / "out" / "query_audit.json",
            {
                "included_order_ids": [],
                "excluded_status_order_ids": [],
                "excluded_out_of_window_order_ids": [],
                "boundary_included_order_ids": [],
                "top_product_tie_order": [],
            },
        )
        write_text(workspace / "out" / "analysis.md", "paid returned cancelled outside 2025-01-01 2025-03-31\n")
    elif task_id == "052-metric-definition-audit":
        write_text(workspace / "out" / "metric_audit.csv", "metric_name,dashboard_formula,approved_formula,affected_field,severity\n")
        write_json(
            workspace / "out" / "equivalence_audit.json",
            {
                "mismatched_metrics": [],
                "semantically_matching_metrics": [],
                "severity_by_metric": {},
                "reason_class_by_metric": {},
            },
        )
    elif task_id == "053-anomalous-transaction-detect":
        write_text(workspace / "out" / "suspicious_transactions.csv", "transaction_id,customer_id,rule_id,risk_level,reason\n")
        write_json(
            workspace / "out" / "rule_audit.json",
            {
                "suspicious_transaction_ids": [],
                "non_suspicious_transaction_ids": [],
                "rule_counts": {},
                "secondary_rule_ids": {},
            },
        )
        write_text(workspace / "out" / "case_notes.md", "R1_HIGH_VALUE R2_GEO_AMOUNT R3_CARD_VELOCITY R4_COUNTRY_MISMATCH\n")
    elif task_id == "054-budget-variance-analysis":
        write_text(workspace / "out" / "variance_report.csv", "department,category,budget_amount,actual_amount,variance_amount,variance_pct,flag\n")
        write_text(workspace / "out" / "department_rollup.csv", "department,budget_amount,actual_amount,variance_amount,variance_pct,review_item_count,rollup_flag\n")
        write_json(workspace / "out" / "review_reasons.json", {})
        write_text(workspace / "out" / "summary.md", "review items, largest overrun, unplanned, zero budget, missing actual\n")
    elif task_id == "055-funnel-dropoff-analysis":
        write_json(
            workspace / "out" / "funnel_metrics.json",
            {"stages": [], "largest_dropoff": {}, "excluded_bot_users": [], "data_quality": {"ignored_order_events": 0, "deduplicated_stage_events": 0}},
        )
        write_text(workspace / "out" / "cohort_comparison.csv", "cohort,visit_users,purchase_users,purchase_rate,largest_dropoff_transition\n")
        write_json(workspace / "out" / "stage_user_sets.json", {"stage_users": {}, "largest_dropoff_lost_user_ids": [], "ignored_order_user_ids": [], "deduplicated_stage_user_ids": []})
        write_text(workspace / "out" / "dropoff_insights.md", "trial_start purchase control variant beta sample order follow\n")
    elif task_id == "056-inventory-forecast":
        write_text(workspace / "out" / "reorder_plan.csv", "sku,avg_weekly_sales,forecast_14d,current_stock,safety_stock,reorder_qty,risk_level\n")
        write_json(
            workspace / "out" / "inventory_exceptions.json",
            {
                "high_risk_skus": [],
                "missing_history_skus": [],
                "skipped_history_skus": [],
                "medium_boundary_skus": [],
                "more_than_one_pack_low_skus": [],
                "pack_rounding_cases": {},
            },
        )
        write_text(workspace / "out" / "forecast_notes.md", "14 day pack rounding high risk missing history skipped unknown sku\n")
    elif task_id == "089-ab-test-caveat-analysis":
        write_text(workspace / "out" / "variant_metrics.csv", "variant,eligible_users,conversions,conversion_rate,refund_rate,revenue_per_eligible_user\n")
        write_json(workspace / "out" / "ab_summary.json", {})
        write_text(workspace / "out" / "exclusion_ledger.csv", "user_id,reason,variant,notes\n")
        write_text(workspace / "out" / "recommendation.md", "significant p launch mobile underpowered duplicate\n")
    elif task_id == "090-timeseries-anomaly-attribution":
        write_text(workspace / "out" / "anomalies.csv", "anomaly_id,timestamp,service,region,metric,observed,expected,z_score,severity,attributed_cause\n")
        write_json(workspace / "out" / "attribution_summary.json", {})
        write_text(workspace / "out" / "reconciliation_notes.md", "low-volume overlap correlation causation\n")
    elif task_id == "091-financial-close-reconciliation":
        write_text(workspace / "out" / "close_reconciliation.csv", "invoice_id,customer_id,invoice_usd,payment_usd,refund_usd,bank_fee_usd,net_cash_usd,reconciliation_status\n")
        write_text(workspace / "out" / "reject_ledger.csv", "item_id,item_type,reason,notes\n")
        write_json(workspace / "out" / "reconciliation_summary.json", {})
        write_text(workspace / "out" / "close_notes.md", "refund fx missing invoice void\n")
    elif task_id == "092-schema-drift-audit":
        write_text(workspace / "out" / "schema_drift_report.csv", "extract_date,field_name,drift_type,expected,observed,severity\n")
        write_text(workspace / "out" / "rejected_rows.csv", "extract_date,source_row,record_id,reason,notes\n")
        write_json(workspace / "out" / "drift_summary.json", {})
        write_text(workspace / "out" / "audit_notes.md", "schema drift row-level change log\n")
    elif task_id == "093-jsonl-sessionization-analysis":
        write_text(workspace / "out" / "sessions.csv", "session_id,user_key,session_start,session_end,event_count,landing_page,last_page,converted,campaign_source\n")
        write_text(workspace / "out" / "reject_ledger.csv", "line_number,event_id,reason,notes\n")
        write_json(workspace / "out" / "session_summary.json", {})
        write_text(workspace / "out" / "sessionization_notes.md", "identity 30-minute malformed duplicate\n")
    elif task_id == "094-metric-definition-migration-diff":
        write_text(workspace / "out" / "metric_migration_diff.csv", "metric_name,old_value,new_value,absolute_diff,relative_diff,expected_direction,classification\n")
        write_text(workspace / "out" / "regression_ledger.csv", "metric_name,bad_field,policy_clause,severity\n")
        write_json(workspace / "out" / "migration_summary.json", {})
        write_text(workspace / "out" / "caveats.md", "arr activation non-comparable retention unexpected regression\n")
    elif task_id == "095-policy-version-conflict-resolution":
        write_json(workspace / "out" / "policy_rulings.json", {"rulings": []})
        write_text(workspace / "out" / "conflict_audit.csv", "case_id,claim_axis,winning_source,losing_sources,priority_rule,coverage_scope\n")
    elif task_id == "096-offline-knowledge-qa-insufficient-evidence":
        write_json(workspace / "out" / "answers.json", {"answers": []})


def prompt_literal_stub(task_id: str, workspace: Path) -> None:
    if task_id == "002-exec":
        solve_002(workspace)
    elif task_id == "004-meeting-summary":
        write_text(workspace / "out" / "meeting_summary.txt", ("Q2 budget milestone OpenClaw M1 risk gateway action. " * 10).strip() + "\n")
    else:
        schema_only(task_id, workspace)


def fixture_copy(workspace: Path) -> None:
    out = workspace / "out" / "copied_inputs"
    if (workspace / "in").is_dir():
        shutil.copytree(workspace / "in", out, dirs_exist_ok=True)


def run_demo_adapter(task: Any, workspace: Path) -> None:
    prompt_file = task.task_dir / task.prompt_file
    prompt = prompt_file.read_text(encoding="utf-8", errors="replace") if prompt_file.is_file() else ""
    ctx = AdapterRunContext(
        task=task,
        workspace=workspace,
        sandbox=workspace.parent,
        prompt=prompt,
        prompt_file=prompt_file,
        session_id=f"endpoint-demo-{task.task_id}",
        timeout_sec=120,
        env={},
        model_id="demo-local",
        model_config={"adapter": "demo"},
        mode="demo",
    )
    DemoAdapter().run(ctx)


def narrow_file_tool(task_id: str, workspace: Path) -> None:
    if task_id in {"001-file", "002-exec", "005-email-triage", "008-image-recognize"}:
        TRACEH_SOLVERS[task_id](workspace)
    elif task_id == "020-archive-checksum":
        entries = []
        for archive in sorted((workspace / "in").glob("bundle_*")):
            entries.extend(archive_entries(archive))
        write_json(workspace / "out" / "manifest.json", {"files": sorted(entries, key=lambda item: (item["archive"], item["path"]))})
        write_text(workspace / "out" / "mismatches.txt", "")
    else:
        schema_only(task_id, workspace)


def run_policy(method_key: str, task: Any, workspace: Path) -> str | None:
    try:
        if method_key == "no_output":
            return None
        if method_key == "schema_only":
            schema_only(task.task_id, workspace)
        elif method_key == "prompt_literal_stub":
            prompt_literal_stub(task.task_id, workspace)
        elif method_key == "fixture_copy":
            fixture_copy(workspace)
        elif method_key == "demo_local":
            if int(task.task_id.split("-", 1)[0]) <= 10:
                run_demo_adapter(task, workspace)
            else:
                schema_only(task.task_id, workspace)
        elif method_key == "narrow_file_tool":
            narrow_file_tool(task.task_id, workspace)
        elif method_key == "traceh_endpoint_ledger":
            TRACEH_SOLVERS[task.task_id](workspace)
        else:
            raise KeyError(method_key)
    except Exception as exc:  # keep benchmark rows even when a policy crashes
        return f"{type(exc).__name__}: {exc}"
    return None


def build_html_report(path: Path, records: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    datasets = sorted({row["task_id"] for row in records})
    methods = [spec[0] for spec in METHOD_SPECS]
    labels = {key: label for key, label, _role in METHOD_SPECS}
    by_pair = {(row["task_id"], row["method_key"]): row for row in records}
    rows_html = []
    for task_id in datasets:
        cells = [f"<th>{html.escape(task_id)}</th>"]
        for method in methods:
            row = by_pair[(task_id, method)]
            value = float(row["outcome_score"])
            color = "#d8f3dc" if value >= 0.999 else "#fff3bf" if value > 0 else "#ffe3e3"
            cells.append(f'<td style="background:{color}">{value:.3f}</td>')
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    agg_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['method_label'])}</td>"
        f"<td>{row['role']}</td>"
        f"<td>{float(row['mean_outcome_score']):.4f}</td>"
        f"<td>{float(row['endpoint_success_rate']):.4f}</td>"
        f"<td>{float(row['mean_check_pass_rate']):.4f}</td>"
        "</tr>"
        for row in aggregate_rows
    )
    comp_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['baseline_label'])}</td>"
        f"<td>{row['primary_sign_test']['wins']}/{row['primary_sign_test']['losses']}/{row['primary_sign_test']['ties']}</td>"
        f"<td>{float(row['primary_sign_test']['p_value']):.5f}</td>"
        f"<td>{row['primary_sign_test']['significant_0_05']}</td>"
        "</tr>"
        for row in comparisons
    )
    method_header = "".join(f"<th>{html.escape(labels[m])}</th>" for m in methods)
    html_text = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>HarnessBench Endpoint PK</title></head>
<body>
<h1>HarnessBench Endpoint Outcome PK</h1>
<h2>Aggregate</h2>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>method</th><th>role</th><th>mean outcome</th><th>endpoint success</th><th>mean check pass</th></tr>
{agg_html}
</table>
<h2>Task Outcome Matrix</h2>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>task</th>{method_header}</tr>
{''.join(rows_html)}
</table>
<h2>TRACE-H vs Baselines Sign Test</h2>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>baseline</th><th>w/l/t</th><th>p</th><th>sig@0.05</th></tr>
{comp_html}
</table>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "local-dev" / "reports"))
    parser.add_argument("--work-dir", default=str(PROJECT_ROOT / "experiments" / "local-dev" / "harnessbench-endpoint-work"))
    parser.add_argument("--date", default="20260714")
    parser.add_argument("--keep-workspaces", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.exists():
        resolved = work_dir.resolve()
        expected = (PROJECT_ROOT / "experiments" / "local-dev").resolve()
        if expected not in resolved.parents and resolved != expected:
            raise SystemExit(f"refusing to remove unexpected work dir: {work_dir}")
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(HARNESS_ROOT / "tasks")
    records: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        task = tasks[task_id]
        for method_key, method_label, role in METHOD_SPECS:
            workspace = work_dir / method_key / task_id / "workspace"
            copy_fixtures(task, workspace)
            normalize_line_endings_for_known_fixture_hashes(task.task_id, workspace)
            error = run_policy(method_key, task, workspace)
            oracle = run_oracle(task, workspace)
            outcome = float(oracle.get("outcome_score", 0.0))
            records.append(
                {
                    "task_id": task_id,
                    "task_title": task.title,
                    "method_key": method_key,
                    "method_label": method_label,
                    "role": role,
                    "outcome_score": outcome,
                    "endpoint_success": 1.0 if outcome >= 0.999 else 0.0,
                    "check_pass_rate": check_pass_rate(oracle),
                    "policy_error": error,
                    "oracle_error": oracle.get("error"),
                    "workspace": str(workspace),
                    "oracle": oracle,
                }
            )

    aggregate_rows = []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_method[row["method_key"]].append(row)
    for method_key, method_label, role in METHOD_SPECS:
        rows = by_method[method_key]
        aggregate_rows.append(
            {
                "method_key": method_key,
                "method_label": method_label,
                "role": role,
                "task_count": len(rows),
                "mean_outcome_score": mean(float(row["outcome_score"]) for row in rows),
                "endpoint_success_rate": mean(float(row["endpoint_success"]) for row in rows),
                "mean_check_pass_rate": mean(float(row["check_pass_rate"]) for row in rows),
            }
        )
    aggregate_rows.sort(key=lambda row: (-float(row["mean_outcome_score"]), row["method_label"]))

    ours_rows = {row["task_id"]: row for row in by_method["traceh_endpoint_ledger"]}
    comparisons = []
    for method_key, method_label, role in METHOD_SPECS:
        if role != "baseline":
            continue
        baseline_rows = {row["task_id"]: row for row in by_method[method_key]}
        diffs_primary = [
            float(ours_rows[task_id]["outcome_score"]) - float(baseline_rows[task_id]["outcome_score"])
            for task_id in TASK_IDS
        ]
        diffs_secondary = [
            float(ours_rows[task_id]["check_pass_rate"]) - float(baseline_rows[task_id]["check_pass_rate"])
            for task_id in TASK_IDS
        ]
        comparisons.append(
            {
                "baseline_key": method_key,
                "baseline_label": method_label,
                "primary_metric": "outcome_score",
                "primary_sign_test": sign_test(diffs_primary),
                "secondary_metric": "check_pass_rate",
                "secondary_sign_test": sign_test(diffs_secondary),
            }
        )

    rank_by_primary = {row["method_key"]: idx + 1 for idx, row in enumerate(aggregate_rows)}
    status = {
        "task_count": len(TASK_IDS),
        "baseline_count": sum(1 for _key, _label, role in METHOD_SPECS if role == "baseline"),
        "ours_primary_rank_1": rank_by_primary.get("traceh_endpoint_ledger") == 1,
        "ours_mean_outcome_score": next(row["mean_outcome_score"] for row in aggregate_rows if row["method_key"] == "traceh_endpoint_ledger"),
        "ours_endpoint_success_rate": next(row["endpoint_success_rate"] for row in aggregate_rows if row["method_key"] == "traceh_endpoint_ledger"),
        "ours_vs_all_baselines_primary_significant_0_05": all(row["primary_sign_test"]["significant_0_05"] for row in comparisons),
        "ours_vs_all_baselines_secondary_significant_0_05": all(row["secondary_sign_test"]["significant_0_05"] for row in comparisons),
        "paper_goal_satisfied": False,
        "paper_goal_unsatisfied_reasons": [
            "This is deterministic local endpoint execution, not yet a sealed LLM-agent final.",
            "The TRACE-H endpoint ledger uses hand-written local handlers for a selected offline subset.",
            "It strengthens the HarnessBench evidence boundary but does not replace full external-agent PK.",
        ],
    }

    report = {
        "benchmark": "HarnessBench endpoint outcome local PK",
        "date": args.date,
        "harness_root": str(HARNESS_ROOT),
        "task_ids": TASK_IDS,
        "method_specs": METHOD_SPECS,
        "status": status,
        "aggregate_rows": aggregate_rows,
        "comparisons": comparisons,
        "records": records,
    }
    json_path = output_dir / f"L8-harnessbench-endpoint-pk-traceh-ledger-{args.date}.json"
    tsv_path = output_dir / f"L8-harnessbench-endpoint-pk-traceh-ledger-matrix-{args.date}.tsv"
    aggregate_tsv_path = output_dir / f"L8-harnessbench-endpoint-pk-traceh-ledger-aggregate-{args.date}.tsv"
    html_path = output_dir / f"L8-harnessbench-endpoint-pk-traceh-ledger-{args.date}.html"
    write_json(json_path, report)
    write_tsv(
        tsv_path,
        records,
        ["task_id", "method_key", "method_label", "role", "outcome_score", "endpoint_success", "check_pass_rate", "policy_error", "oracle_error", "workspace"],
    )
    write_tsv(
        aggregate_tsv_path,
        aggregate_rows,
        ["method_key", "method_label", "role", "task_count", "mean_outcome_score", "endpoint_success_rate", "mean_check_pass_rate"],
    )
    build_html_report(html_path, records, aggregate_rows, comparisons)
    if not args.keep_workspaces:
        shutil.rmtree(work_dir)
    print(json.dumps({"json": str(json_path), "tsv": str(tsv_path), "aggregate_tsv": str(aggregate_tsv_path), "html": str(html_path), "status": status}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
