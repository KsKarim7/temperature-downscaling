import argparse
import json
import os
import re
import subprocess
import sys
import time

from openai import OpenAI

from downscale.checks import check_region
from downscale.config import ROOT, load_config
from downscale.data import region_list

SYSTEM = """You check and report on a temperature downscaling project.

Work in this order.
1. Call check_data for every region. If any check fails, call write_report with a short
   failure report naming the failed checks, and stop.
2. Otherwise call run_evaluation once.
3. Read metrics_overall.csv, metrics_by_hour.csv, stations_summary.csv and
   stations_bias_drivers.csv with read_csv.
4. Call write_report with a report under 400 words in markdown.

The report covers, in this order: the data checks; test_blocks results for each region for
bilinear, lgbm and unet_main; the val to test_blocks gap; the worst local hours for unet_main;
the station results with ERA5-Land as the reference; and the caveats.

Use only numbers that a tool returned to you, written exactly at the precision the tool gave.
Never calculate a new number and never estimate one. If you want a number no tool gave you,
leave it out. Plain wording, no emoji, no bold, short paragraphs.
"""

WAITS = [30, 60, 120]
REGIONS = ("uae", "bangladesh")


def reports_dir():
    return ROOT / load_config()["paths"]["reports"]


def check_data(region):
    r = check_region(region)
    bad = sorted(k for k, v in r.items() if k.endswith("_ok") and not v)
    return {"region": region, "passed": not bad, "failed": bad, "detail": r}


def list_reports():
    d = reports_dir()
    return {"files": sorted(p.name for p in d.glob("*.csv"))}


def read_csv(name, max_rows=60):
    if "/" in name or "\\" in name or name.startswith("."):
        return {"error": f"refused {name}, pass a plain file name inside reports/"}
    p = (reports_dir() / name).resolve()
    if p.parent != reports_dir().resolve():
        return {"error": f"refused {name}, outside reports/"}
    if not p.exists():
        return {"error": f"no such file {name}"}
    lines = p.read_text(encoding="utf-8").splitlines()
    return {"name": name, "rows": len(lines) - 1, "text": "\n".join(lines[: max_rows + 1])}


def run_evaluation():
    r = subprocess.run(
        [sys.executable, "-m", "downscale.evaluate"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = (r.stdout or "").strip().splitlines()[-3:]
    return {"ok": r.returncode == 0, "output": "\n".join(tail), "error": (r.stderr or "").strip()[-500:]}


def write_report(markdown):
    p = reports_dir() / "agent_report.md"
    p.write_text(markdown, encoding="utf-8")
    return {"written": str(p.relative_to(ROOT)), "words": len(markdown.split())}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_data",
            "description": "Run the data checks for one region and report which ones failed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "enum": list(REGIONS), "description": "region name"}
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reports",
            "description": "List the CSV files available in the reports directory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_csv",
            "description": "Read a CSV from the reports directory as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "file name inside reports/"},
                    "max_rows": {"type": "integer", "description": "how many data rows to return"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_evaluation",
            "description": "Run the evaluation and rewrite the metrics CSVs.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": "Write the final markdown report to reports/agent_report.md.",
            "parameters": {
                "type": "object",
                "properties": {"markdown": {"type": "string", "description": "the report"}},
                "required": ["markdown"],
            },
        },
    },
]

FUNCS = {
    "check_data": check_data,
    "list_reports": list_reports,
    "read_csv": read_csv,
    "run_evaluation": run_evaluation,
    "write_report": write_report,
}

NUM = re.compile(r"-?\d+\.?\d*")


def numbers_in(text):
    out = []
    for m in NUM.finditer(text):
        s = m.group(0).rstrip(".")
        if s and s not in ("-",):
            out.append(s)
    return out


def trace(report, sources):
    pool = set()
    for text in sources:
        pool.update(numbers_in(text))
    traced = []
    untraced = []
    for s in numbers_in(report):
        (traced if s in pool or s.lstrip("-") in pool else untraced).append(s)
    return traced, untraced


def guardrail(report_text, seen):
    traced, untraced = trace(report_text, seen)
    p = reports_dir() / "agent_report_check.txt"
    lines = [f"traced {len(traced)} of {len(traced) + len(untraced)} numbers"]
    if untraced:
        lines.append("untraced:")
        lines += [f"  {s}" for s in untraced]
    else:
        lines.append("untraced: none")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return traced, untraced


def client_for(provider):
    cfg = load_config()["agent"]
    c = cfg[provider]
    key = os.environ.get(c["key_env"], "") if c["key_env"] else "ollama"
    if c["key_env"] and not key:
        raise SystemExit(
            f'set {c["key_env"]} for this session with: $env:{c["key_env"]}="your-key-here"'
        )
    return OpenAI(base_url=c["base_url"], api_key=key or "none"), c


def pick_model(client, c):
    if c["model"]:
        return c["model"]
    names = [m.id.split("/")[-1] for m in client.models.list()]
    flash = [n for n in names if "flash" in n and "preview" not in n and "image" not in n]
    full = [n for n in flash if "lite" not in n]
    if not flash:
        raise SystemExit("no flash model at this endpoint, set agent.gemini.model in config.yaml")
    pick = sorted(full or flash)[-1]
    print(f"model {pick}, chosen from {len(flash)} flash models at the endpoint")
    return pick


def call(client, model, messages, tries=WAITS):
    for w in list(tries) + [None]:
        try:
            return client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, tool_choice="auto"
            )
        except Exception as e:
            if "429" not in str(e) and "rate" not in str(e).lower():
                raise
            if w is None:
                raise SystemExit("rate limited by the endpoint after three retries, try again later")
            print(f"rate limited, waiting {w}s", flush=True)
            time.sleep(w)


def run(provider, turns):
    client, c = client_for(provider)
    model = pick_model(client, c)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Do the review."}]
    seen = []
    report_text = ""
    for _ in range(turns):
        m = call(client, model, messages).choices[0].message
        messages.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            break
        for tc in m.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            print(f"tool {tc.function.name} {json.dumps(args)[:80]}", flush=True)
            out = FUNCS[tc.function.name](**args)
            if tc.function.name == "read_csv" and "text" in out:
                seen.append(out["text"])
            if tc.function.name == "check_data":
                seen.append(json.dumps(out["detail"]))
            if tc.function.name == "write_report":
                report_text = args.get("markdown", "")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, default=str)}
            )
    if not report_text:
        raise SystemExit("the model stopped without writing a report")
    traced, untraced = guardrail(report_text, seen)
    print(f"traced {len(traced)} of {len(traced) + len(untraced)} numbers")
    if untraced:
        print("untraced numbers in the report: " + ", ".join(untraced))
        raise SystemExit(1)
    print("every number in the report traces to a csv the agent read")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default=None, choices=["gemini", "ollama"])
    p.add_argument("--checks-only", action="store_true")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--turns", type=int, default=15)
    a = p.parse_args()
    cfg = load_config()["agent"]
    provider = a.provider or cfg["provider"]

    if a.checks_only:
        bad = 0
        for region in region_list():
            r = check_data(region)
            print(f"{region} {'ok' if r['passed'] else 'failed: ' + ', '.join(r['failed'])}")
            bad += 0 if r["passed"] else 1
        raise SystemExit(1 if bad else 0)

    if a.list_models:
        client, c = client_for(provider)
        for m in client.models.list():
            print(m.id)
        return

    run(provider, a.turns)


if __name__ == "__main__":
    main()
