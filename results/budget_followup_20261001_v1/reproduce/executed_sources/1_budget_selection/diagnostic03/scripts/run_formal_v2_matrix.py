"""Run the complete formal-v2 matrix sequentially with durable status logs."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_full_experiments.py"
RESULT_ROOT = ROOT / "results" / "formal_v2"
STATUS_PATH = RESULT_ROOT / "matrix_status.json"
LOG_PATH = RESULT_ROOT / "matrix_run.log"

CONFIGS = [
    "configs/formal_v2/sst2_bert_base_alpha_0_3.json",
    "configs/formal_v2/qnli_bert_base_alpha_0_3.json",
    "configs/formal_v2/ag_news_bert_base_alpha_0_3.json",
    "configs/formal_v2/sst2_roberta_base_alpha_0_3.json",
    "configs/formal_v2/qnli_roberta_base_alpha_0_3.json",
    "configs/formal_v2/ag_news_roberta_base_alpha_0_3.json",
    "configs/formal_v2/ag_news_bert_base_alpha_0_1.json",
    "configs/formal_v2/ag_news_bert_base_alpha_1_0.json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_status() -> dict:
    if STATUS_PATH.is_file():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {"started_at": utc_now(), "configs": {}}


def save_status(status: dict) -> None:
    STATUS_PATH.write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )


def append_log(line: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    status = load_status()
    for relative_config in CONFIGS:
        entry = status["configs"].setdefault(relative_config, {})
        if entry.get("status") == "completed":
            print(f"=== {relative_config} already completed ===", flush=True)
            continue
        entry.update({"status": "running", "started_at": utc_now()})
        save_status(status)
        header = f"=== formal-v2 start {relative_config} {utc_now()} ==="
        print(header, flush=True)
        append_log(header)
        process = subprocess.Popen(
            [
                sys.executable,
                str(RUNNER),
                "--config",
                relative_config,
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            append_log(line)
        return_code = process.wait()
        entry["finished_at"] = utc_now()
        entry["return_code"] = return_code
        entry["status"] = "completed" if return_code == 0 else "failed"
        save_status(status)
        if return_code != 0:
            raise SystemExit(
                f"{relative_config} failed with exit code {return_code}"
            )
    status["finished_at"] = utc_now()
    status["status"] = "completed"
    save_status(status)
    print("formal-v2 matrix completed", flush=True)


if __name__ == "__main__":
    main()
