import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from openai import OpenAI

from downscale.checks import check_region
from downscale.config import ROOT, load_config
from downscale.data import region_list
from downscale.splits import load_splits, split_masks

SYSTEM = """You review a temperature downscaling project and write the report an ML lead will read.
They know the method. They want to know what the numbers mean, not to be read a table aloud.

Work in this order.
1. Call get_run_info. It gives the run context: which year is training, which is test, how many
   hours the evaluation covers, how many cells and stations each region has, and what each region
   box really covers. Use these facts when you describe the scope of anything.
2. Call check_data for every region. If any check fails, call write_report with a short failure
   report naming the failed checks, and stop.
3. Call run_evaluation once.
4. Read metrics_overall.csv, model_margins.csv, worst_hours.csv, stations_summary.csv,
   stations_bias_drivers.csv and kaggle_runs.csv with read_csv. If a result ends with a line
   starting "truncated", you have not seen every row: call read_csv again with a filter such as
   {"region": "bangladesh", "method": "unet_main"} rather than guessing at the missing rows.
5. Call write_report. Under 400 words, markdown, in these sections.

Call the two regions "UAE region" and "Bangladesh region" in every heading and sentence.

Setup: one short line. Say what each region box also covers, from get_run_info, and name any
station get_run_info lists as excluded, with its reason in a few words.

Data: one or two sentences. Say which year the evaluation covers and how many hours that is.

Results: a markdown table per region for the test_blocks split, columns method, rmse_c, mae_c,
bias_c, skill, rows bilinear, lgbm and unet_main. Then two or three sentences saying which method
generalises best to held-out blocks, and comparing the val to test_blocks gap for lgbm against
unet_main. Take both gaps from the lgbm_val_to_test_gap_c and unet_main_val_to_test_gap_c columns
of model_margins.csv, never by subtracting the table values yourself.

Hours: take every number here from worst_hours.csv and nowhere else. For each region give the
unet_main rows, quoting local_hour and rmse_c exactly as that file gives them. Then one sentence
noting that the two regions degrade at different times of day. If a worst hour you want is not in
worst_hours.csv, say so rather than supplying a number.

Stations: the stations are the observations, the thing being predicted. The era5land row is not a
reference method and it is not a bound. It is how the training target itself scores against those
observations. A model trained on that target can land slightly above or below it, because part of
the station error is representativeness: a station is a point and a cell is about 10 km across.
Never say any row bounds or limits what a model can achieve. Cover both regions: the number of
stations, and median_rmse and mean_bias for era5land, lgbm and unet_main. Then state that in the
UAE region era5land and the models share a cold bias, quoting the mean_bias numbers.

Caveats: at most four bullets, drawn only from this list, each quoting the numbers it needs.
- The test_blocks margin between unet_main and lgbm, set against run to run noise. Quote
  unet_minus_lgbm_test_blocks_c, run_to_run_noise_c and margin_over_noise from model_margins.csv.
- One block partition and one split seed, so these are a single draw, not a distribution.
- ERA5-Land is a model product, not observations, so agreement with it is not accuracy.
- The UAE region station cold bias is inherited from the training target.
Nothing about region boxes or excluded stations here; those belong in Setup.

Never do arithmetic yourself. Do not subtract, add, average or round anything in your head, not
even a difference between two numbers in the same table. Every gap, margin and noise figure is
already computed in model_margins.csv: read it and quote its columns. For anything else, call calc
and quote its result string. A number you worked out yourself will be rejected even if it is right.

Use only numbers a tool returned, written exactly at the precision the tool gave. Never estimate a
number. If you want one no tool gave you, leave it out. Do not describe a number as
covering the evaluation when get_run_info shows it covers both years. Plain wording, no emoji, no
bold, short paragraphs.
"""

WAITS = [30, 60, 120, 180, 300]
REGIONS = ("uae", "bangladesh")


def reports_dir():
    return ROOT / load_config()["paths"]["reports"]


COVERS = {
    "uae": "UAE region, a box that also covers Qatar and northern Oman",
    "bangladesh": "Bangladesh region, a box that also covers Kolkata and parts of West Bengal and Meghalaya",
}


def get_run_info():
    cfg = load_config()
    sp = reports_dir() / "stations_summary.csv"
    counts = {}
    if sp.exists():
        s = pd.read_csv(sp)
        counts = {r: int(s[s.region == r].stations.max()) for r in s.region.unique()}
    ex = reports_dir() / "stations_excluded.csv"
    excluded = []
    if ex.exists():
        e = pd.read_csv(ex)
        excluded = [{"region": r.region, "id": str(r.id), "name": r["name"], "reason": r.reason} for _, r in e.iterrows()]

    out = {"train_year": cfg["years"]["train"], "test_year": cfg["years"]["test"], "regions": {}}
    for region in region_list():
        times = pd.DatetimeIndex(load_splits(region)["times"])
        m = split_masks(region)
        out["regions"][region] = {
            "label": COVERS[region],
            "hours_in_test_year": int(np.sum(np.asarray(times.year) == cfg["years"]["test"])),
            "hours_in_store_both_years": int(times.size),
            "test_block_cells": int(m["test_blocks"][1].sum()),
            "val_block_cells": int(m["val"][1].sum()),
            "all_land_cells": int(m["all_land"][1].sum()),
            "stations_after_exclusions": counts.get(region, 0),
        }
    out["excluded_stations"] = excluded
    return out


def check_data(region):
    r = check_region(region)
    bad = sorted(k for k, v in r.items() if k.endswith("_ok") and not v)
    return {"region": region, "passed": not bad, "failed": bad, "detail": r}


def list_reports():
    d = reports_dir()
    return {"files": sorted(p.name for p in d.glob("*.csv"))}


def read_csv(name, max_rows=200, filter=None):
    if "/" in name or "\\" in name or name.startswith("."):
        return {"error": f"refused {name}, pass a plain file name inside reports/"}
    p = (reports_dir() / name).resolve()
    if p.parent != reports_dir().resolve():
        return {"error": f"refused {name}, outside reports/"}
    if not p.exists():
        return {"error": f"no such file {name}"}

    lines = p.read_text(encoding="utf-8").splitlines()
    head, rows = lines[0], [r for r in lines[1:] if r.strip()]
    cols = next(csv.reader([head]))
    total = len(rows)
    if filter:
        missing = [k for k in filter if k not in cols]
        if missing:
            return {"error": f"no column {missing[0]} in {name}, columns are {cols}"}
        idx = {k: cols.index(k) for k in filter}
        rows = [
            r
            for r in rows
            for cells in [next(csv.reader([r]))]
            if all(len(cells) > i and cells[i] == str(filter[k]) for k, i in idx.items())
        ]

    kept = len(rows)
    text = "\n".join([head] + rows[:max_rows])
    if kept > max_rows:
        text += (
            f"\ntruncated: showing {max_rows} of {kept} rows; "
            "use the filter argument or a summary file"
        )
    return {"name": name, "rows": total, "matched": kept, "text": text}


METRICS = (
    "metrics_overall.csv",
    "metrics_by_hour.csv",
    "metrics_by_month.csv",
    "metrics_by_landcover.csv",
    "metrics_by_geohash.csv",
)
FORCE_EVAL = False


def store_mtime(p):
    m = p.stat().st_mtime
    z = p / ".zmetadata"
    return max(m, z.stat().st_mtime) if z.exists() else m


def eval_fresh(rep, out_dir, src):
    files = [rep / n for n in METRICS]
    if not all(f.exists() for f in files):
        return False, []
    deps = [store_mtime(s) for s in sorted(out_dir.glob("*/pred_*.zarr"))]
    if src.exists():
        deps.append(src.stat().st_mtime)
    oldest = min(f.stat().st_mtime for f in files)
    return oldest > max(deps, default=0.0), [f.name for f in files]


def run_evaluation():
    cfg = load_config()
    rep = ROOT / cfg["paths"]["reports"]
    fresh, files = eval_fresh(rep, ROOT / cfg["paths"]["outputs"], ROOT / "downscale" / "evaluate.py")
    if fresh and not FORCE_EVAL:
        return {"ok": True, "output": "evaluation up to date, using existing reports", "files": files}
    r = subprocess.run(
        [sys.executable, "-m", "downscale.evaluate"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = (r.stdout or "").strip().splitlines()[-3:]
    return {"ok": r.returncode == 0, "output": "\n".join(tail), "error": (r.stderr or "").strip()[-500:]}


OPS = {
    "sub": lambda a, b: a - b,
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b if b else float("nan"),
}


def calc(a, b, op):
    if op not in OPS:
        return {"error": f"unknown op {op}, use one of sub, add, mul, div"}
    return {"a": a, "b": b, "op": op, "result": f"{round(OPS[op](float(a), float(b)), 4)}"}


SEP = re.compile(r"^\s*\|?(\s*:?-+:?\s*\|)+\s*:?-*:?\s*\|?\s*$")


def is_row(line):
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def fix_tables(markdown):
    lines = markdown.splitlines()
    out = []
    for i, line in enumerate(lines):
        out.append(line)
        if not is_row(line) or (i and is_row(lines[i - 1])):
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if SEP.match(nxt):
            continue
        n = len(line.strip().strip("|").split("|"))
        out.append("|" + "|".join([" --- "] * n) + "|")
    text = "\n".join(out)
    return text + "\n" if markdown.endswith("\n") else text


RUN = {"model": "", "provider": ""}


def provenance():
    day = datetime.date.today().isoformat()
    return f"model {RUN['model']}, provider {RUN['provider']}, run {day}"


def write_report(markdown):
    md = fix_tables(markdown).rstrip() + "\n\n---\n\n" + provenance() + "\n"
    p = reports_dir() / "agent_report.md"
    p.write_text(md, encoding="utf-8")
    return {"written": f"{p.parent.name}/{p.name}", "words": len(md.split())}


def stamp(traced, total):
    p = reports_dir() / "agent_report.md"
    if not p.exists():
        return
    body = p.read_text(encoding="utf-8").rstrip()
    p.write_text(f"{body}\n\nguardrail: traced {traced} of {total} numbers\n", encoding="utf-8")


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
            "name": "get_run_info",
            "description": "Run context: train and test year, hours in the test year, cell counts, "
            "station counts after exclusions, and what each region box covers.",
            "parameters": {"type": "object", "properties": {}},
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
                    "filter": {
                        "type": "object",
                        "description": "keep only rows matching all of these columns",
                        "properties": {
                            "region": {"type": "string", "description": "uae or bangladesh"},
                            "method": {"type": "string", "description": "bilinear, lgbm or unet_main"},
                            "split": {"type": "string", "description": "val, test_blocks or all_land"},
                        },
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Arithmetic on two numbers. Use this instead of working anything out yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "first number"},
                    "b": {"type": "number", "description": "second number"},
                    "op": {
                        "type": "string",
                        "enum": ["sub", "add", "mul", "div"],
                        "description": "operation to apply as a op b",
                    },
                },
                "required": ["a", "b", "op"],
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
    "get_run_info": get_run_info,
    "check_data": check_data,
    "list_reports": list_reports,
    "read_csv": read_csv,
    "calc": calc,
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


BOTH_YEARS = "17544"
BOTH_YEARS_NEEDS = "both years"


def misused_both_years(report):
    out = []
    for m in re.finditer(BOTH_YEARS, report):
        a = max(0, m.start() - 90)
        if BOTH_YEARS_NEEDS not in report[a : m.end() + 90].lower():
            out.append(report[a : m.end() + 90].strip())
    return out


def quarantine():
    src = reports_dir() / "agent_report.md"
    dst = reports_dir() / "agent_report_failed.md"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)
    return dst.name


def guardrail(report_text, seen):
    traced, untraced = trace(report_text, seen)
    misused = misused_both_years(report_text)
    p = reports_dir() / "agent_report_check.txt"
    lines = [f"traced {len(traced)} of {len(traced) + len(untraced)} numbers"]
    lines.append("untraced: none" if not untraced else "untraced:")
    lines += [f"  {s}" for s in untraced]
    if misused:
        lines.append(f"{BOTH_YEARS} used without '{BOTH_YEARS_NEEDS}':")
        lines += [f"  ...{s}..." for s in misused]
    else:
        lines.append(f"{BOTH_YEARS} misuse: none")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return traced, untraced, misused


def read_key(key_env):
    if not key_env:
        return "ollama"
    return os.environ.get(key_env, "").strip().strip("\"'").strip()


def client_for(provider):
    cfg = load_config()["agent"]
    c = cfg[provider]
    key = read_key(c["key_env"])
    if c["key_env"] and not key:
        raise SystemExit(
            f'set {c["key_env"]} for this session with: $env:{c["key_env"]}="your-key-here"'
        )
    if c["key_env"]:
        print(f"{c['key_env']} {len(key)} chars, starts {key[:4]}", flush=True)
    return OpenAI(base_url=c["base_url"], api_key=key), c


SKIP = (
    "omni",
    "live",
    "tts",
    "image",
    "audio",
    "transcribe",
    "robotics",
    "preview",
    "computer-use",
    "embedding",
    "deep-research",
)
PLAIN = re.compile(r"^gemini-[\d.]+-flash$")


def choose(names):
    flash = [n.split("/")[-1] for n in names]
    flash = [n for n in flash if "flash" in n and not any(k in n for k in SKIP)]
    plain = [n for n in flash if PLAIN.match(n)]
    full = [n for n in flash if "lite" not in n]
    return sorted(plain or full or flash)[-1] if flash else ""


def pick_model(client, c):
    if c["model"]:
        return c["model"]
    pick = choose([m.id for m in client.models.list()])
    if not pick:
        raise SystemExit("no flash model at this endpoint, set agent.gemini.model in config.yaml")
    print(f"model {pick}, chosen from the endpoint listing")
    return pick


def busy(e):
    s = str(e).lower()
    return "429" in s or "503" in s or "rate" in s or "overloaded" in s or "unavailable" in s


def show(e):
    code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", "?")
    body = getattr(getattr(e, "response", None), "text", "") or str(e)
    print(f"api error http {code}", flush=True)
    print(body.strip()[:1200], flush=True)


def call(client, model, messages, tries=WAITS):
    for w in list(tries) + [None]:
        try:
            return client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, tool_choice="auto"
            )
        except Exception as e:
            if not busy(e):
                show(e)
                raise SystemExit(1)
            if w is None:
                raise SystemExit(
                    f"endpoint still busy or rate limited after {len(WAITS)} retries, try again later"
                )
            print(f"endpoint busy, waiting {w}s", flush=True)
            time.sleep(w)


def probe(provider):
    client, c = client_for(provider)
    model = pick_model(client, c)
    u = [{"role": "user", "content": "reply with ok"}]
    s = [{"role": "system", "content": SYSTEM}] + u
    cases = [
        ("user only, no tools", u, None),
        ("system only, no tools", s, None),
        ("user plus tools", u, TOOLS),
        ("system plus tools", s, TOOLS),
    ]
    for label, msgs, tools in cases:
        kw = {"model": model, "messages": msgs}
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        try:
            client.chat.completions.create(**kw)
            print(f"ok    {label}")
        except Exception as e:
            code = getattr(e, "status_code", "?")
            print(f"fail  {label}  http {code}  {str(e)[:200]}")
    for t in TOOLS:
        try:
            client.chat.completions.create(model=model, messages=u, tools=[t], tool_choice="auto")
            print(f"ok    tool {t['function']['name']}")
        except Exception as e:
            print(f"fail  tool {t['function']['name']}  {str(e)[:200]}")


def run(provider, turns):
    client, c = client_for(provider)
    model = pick_model(client, c)
    RUN["model"] = model
    RUN["provider"] = provider
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Do the review."}]
    seen = []
    done = {}
    report_text = ""
    for _ in range(turns):
        m = call(client, model, messages).choices[0].message
        messages.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            break
        for tc in m.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            print(f"tool {tc.function.name} {json.dumps(args)[:80]}", flush=True)
            key = (tc.function.name, json.dumps(args, sort_keys=True))
            done[key] = done.get(key, 0) + 1
            if tc.function.name == "read_csv" and done[key] > 2:
                out = {"note": "already read; use filter or a summary file"}
            else:
                out = FUNCS[tc.function.name](**args)
            if tc.function.name == "read_csv" and "text" in out:
                seen.append(out["text"])
            if tc.function.name == "check_data":
                seen.append(json.dumps(out["detail"]))
            if tc.function.name == "get_run_info":
                seen.append(json.dumps(out, default=str))
            if tc.function.name == "calc" and "result" in out:
                seen.append(out["result"])
            if tc.function.name == "write_report":
                report_text = fix_tables(args.get("markdown", ""))
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, default=str)}
            )
    if not report_text:
        raise SystemExit("the model stopped without writing a report")
    traced, untraced, misused = guardrail(report_text, seen)
    stamp(len(traced), len(traced) + len(untraced))
    print(f"traced {len(traced)} of {len(traced) + len(untraced)} numbers")
    if untraced:
        print("untraced numbers in the report: " + ", ".join(untraced))
    for s in misused:
        print(f"{BOTH_YEARS} used without '{BOTH_YEARS_NEEDS}': ...{s}...")
    if untraced or misused:
        print(f"report kept as {quarantine()} so a failed report is not left in place")
        raise SystemExit(1)
    print("every number in the report traces to a csv the agent read")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default=None, choices=["gemini", "ollama"])
    p.add_argument("--checks-only", action="store_true")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--turns", type=int, default=15)
    a = p.parse_args()
    cfg = load_config()["agent"]
    provider = a.provider or cfg["provider"]
    global FORCE_EVAL
    FORCE_EVAL = a.force_eval

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

    if a.probe:
        probe(provider)
        return

    run(provider, a.turns)


if __name__ == "__main__":
    main()
