import json
import os

from downscale.agent import (
    FUNCS,
    TOOLS,
    choose,
    get_run_info,
    misused_both_years,
    numbers_in,
    read_csv,
    trace,
)

FAKE = [
    "models/gemini-3.5-flash",
    "models/gemini-3.5-flash-lite",
    "models/gemini-3.5-flash-preview-09-2026",
    "models/gemini-3.5-flash-image",
    "models/gemini-3.5-flash-live-001",
    "models/gemini-2.0-flash",
    "models/gemini-3.5-pro",
    "models/text-embedding-004",
    "models/gemini-robotics-er-1.5-flash",
    "models/gemini-2.5-flash-tts",
]


def test_choose_prefers_plain_flash():
    assert choose(FAKE) == "gemini-3.5-flash"


def test_choose_rejects_every_skipped_kind():
    for bad in ("preview", "image", "live", "embedding", "robotics", "tts", "lite", "pro"):
        assert bad not in choose(FAKE)


def test_choose_falls_back_to_lite_when_only_lite():
    assert choose(["models/gemini-3.5-flash-lite"]) == "gemini-3.5-flash-lite"


def test_choose_takes_newest_plain_version():
    assert choose(["models/gemini-2.0-flash", "models/gemini-3.5-flash"]) == "gemini-3.5-flash"


def test_choose_empty_when_nothing_suitable():
    assert choose(["models/gemini-3.5-pro", "models/text-embedding-004"]) == ""


def test_run_info_has_the_context_fields():
    info = get_run_info()
    assert info["train_year"] == 2023 and info["test_year"] == 2024
    for region, r in info["regions"].items():
        assert r["hours_in_test_year"] == 8784
        assert r["hours_in_store_both_years"] == 17544
        assert r["test_block_cells"] > 0 and r["val_block_cells"] > 0
        assert region.split("_")[0][:3].lower() in r["label"].lower() or "region" in r["label"]


def test_run_info_labels_name_what_the_box_covers():
    info = get_run_info()
    assert "Oman" in info["regions"]["uae"]["label"]
    assert "Kolkata" in info["regions"]["bangladesh"]["label"]


def test_both_years_number_flagged_without_the_words():
    bad = "Data checks passed with 17544 total hours evaluated."
    assert misused_both_years(bad)


def test_both_years_number_allowed_with_the_words():
    ok = "The store holds 17544 hours across both years; the evaluation covers 8784."
    assert misused_both_years(ok) == []


def test_both_years_rule_ignores_other_numbers():
    assert misused_both_years("The evaluation covers 8784 hours of 2024.") == []


def test_every_tool_has_a_function():
    names = [t["function"]["name"] for t in TOOLS]
    assert sorted(names) == sorted(FUNCS)


def test_tool_schemas_are_well_formed():
    for t in TOOLS:
        assert t["type"] == "function"
        f = t["function"]
        assert f["name"] and isinstance(f["description"], str) and f["description"]
        p = f["parameters"]
        assert p["type"] == "object"
        assert isinstance(p["properties"], dict)
        for k in p.get("required", []):
            assert k in p["properties"]
        for v in p["properties"].values():
            assert v["type"] in ("string", "integer", "number", "boolean", "array", "object")
        json.dumps(t)


def test_required_args_match_the_functions():
    import inspect

    for t in TOOLS:
        f = t["function"]
        sig = inspect.signature(FUNCS[f["name"]])
        for k in f["parameters"].get("required", []):
            assert k in sig.parameters
        for k in f["parameters"]["properties"]:
            assert k in sig.parameters


REJECTED = (
    "additionalProperties",
    "default",
    "$schema",
    "$ref",
    "$defs",
    "title",
    "examples",
    "const",
    "oneOf",
    "allOf",
    "patternProperties",
)


def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def test_no_rejected_schema_keywords():
    keys = set(walk(TOOLS))
    assert not keys & set(REJECTED), f"gemini rejects these schema keywords: {sorted(keys & set(REJECTED))}"


def test_schemas_serialise_without_rejected_text():
    blob = json.dumps(TOOLS)
    for k in REJECTED:
        assert f'"{k}"' not in blob


def test_no_empty_required_list():
    for t in TOOLS:
        p = t["function"]["parameters"]
        assert "required" not in p or len(p["required"]) > 0


def test_no_arg_tools_all_share_one_form():
    forms = {
        t["function"]["name"]: t["function"]["parameters"]
        for t in TOOLS
        if not t["function"]["parameters"].get("properties")
    }
    assert set(forms) == {"get_run_info", "list_reports", "run_evaluation"}
    assert all(f == {"type": "object", "properties": {}} for f in forms.values())


def test_every_property_is_a_flat_scalar():
    for t in TOOLS:
        for v in t["function"]["parameters"]["properties"].values():
            assert set(v) <= {"type", "description", "enum", "properties"}
            for inner in v.get("properties", {}).values():
                assert set(inner) <= {"type", "description", "enum"}
            if v["type"] == "object":
                assert v.get("properties"), "gemini rejects an object with no declared properties"


def test_client_gets_the_value_of_the_named_env_var(monkeypatch):
    from downscale.agent import client_for
    from downscale.config import load_config

    name = load_config()["agent"]["gemini"]["key_env"]
    monkeypatch.setenv(name, "test-key-abcdef")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-key-should-not-be-used")
    client, c = client_for("gemini")
    assert c["key_env"] == name
    assert client.api_key == "test-key-abcdef"
    assert client.auth_headers["Authorization"] == "Bearer test-key-abcdef"


def test_client_uses_base_url_from_config(monkeypatch):
    from downscale.agent import client_for
    from downscale.config import load_config

    cfg = load_config()["agent"]["gemini"]
    monkeypatch.setenv(cfg["key_env"], "test-key-abcdef")
    client, _ = client_for("gemini")
    assert str(client.base_url).rstrip("/") == cfg["base_url"].rstrip("/")


def test_missing_key_exits_before_any_request(monkeypatch):
    import pytest

    from downscale.agent import client_for
    from downscale.config import load_config

    monkeypatch.delenv(load_config()["agent"]["gemini"]["key_env"], raising=False)
    with pytest.raises(SystemExit):
        client_for("gemini")


def test_key_is_stripped_of_quotes_and_whitespace(monkeypatch):
    from downscale.agent import read_key

    for raw in (' abc123 ', '"abc123"', "'abc123'", 'abc123\n', '  "abc123"  '):
        monkeypatch.setenv("SOME_KEY_ENV", raw)
        assert read_key("SOME_KEY_ENV") == "abc123"


def test_ollama_needs_no_key(monkeypatch):
    from downscale.agent import client_for

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client, c = client_for("ollama")
    assert c["key_env"] == ""
    assert client.api_key


def build(tmp_path, metric_time, store_time, src_time):
    from downscale.agent import METRICS

    rep = tmp_path / "reports"
    out = tmp_path / "outputs" / "uae"
    rep.mkdir(parents=True)
    out.mkdir(parents=True)
    src = tmp_path / "evaluate.py"
    src.write_text("x", encoding="utf-8")
    os.utime(src, (src_time, src_time))
    store = out / "pred_unet_main.zarr"
    store.mkdir()
    (store / ".zmetadata").write_text("{}", encoding="utf-8")
    os.utime(store / ".zmetadata", (store_time, store_time))
    os.utime(store, (store_time, store_time))
    for n in METRICS:
        f = rep / n
        f.write_text("a,b\n1,2\n", encoding="utf-8")
        os.utime(f, (metric_time, metric_time))
    return rep, tmp_path / "outputs", src


def test_eval_fresh_when_metrics_are_newest(tmp_path):
    from downscale.agent import METRICS, eval_fresh

    rep, out, src = build(tmp_path, 3000, 2000, 1000)
    fresh, files = eval_fresh(rep, out, src)
    assert fresh is True
    assert files == list(METRICS)


def test_eval_stale_when_a_store_is_newer(tmp_path):
    from downscale.agent import eval_fresh

    rep, out, src = build(tmp_path, 2000, 3000, 1000)
    assert eval_fresh(rep, out, src)[0] is False


def test_eval_stale_when_the_source_is_newer(tmp_path):
    from downscale.agent import eval_fresh

    rep, out, src = build(tmp_path, 2000, 1000, 3000)
    assert eval_fresh(rep, out, src)[0] is False


def test_eval_stale_when_a_metric_file_is_missing(tmp_path):
    from downscale.agent import METRICS, eval_fresh

    rep, out, src = build(tmp_path, 3000, 2000, 1000)
    (rep / METRICS[-1]).unlink()
    assert eval_fresh(rep, out, src) == (False, [])


def test_eval_stale_when_only_one_metric_is_old(tmp_path):
    from downscale.agent import METRICS, eval_fresh

    rep, out, src = build(tmp_path, 3000, 2000, 1000)
    os.utime(rep / METRICS[2], (1500, 1500))
    assert eval_fresh(rep, out, src)[0] is False


def test_eval_fresh_with_no_prediction_stores(tmp_path):
    from downscale.agent import eval_fresh

    rep, out, src = build(tmp_path, 3000, 2000, 1000)
    for p in (out / "uae").glob("pred_*.zarr"):
        (p / ".zmetadata").unlink()
        p.rmdir()
    assert eval_fresh(rep, out, src)[0] is True


def test_eval_uses_inner_zmetadata_mtime(tmp_path):
    from downscale.agent import eval_fresh

    rep, out, src = build(tmp_path, 2500, 2000, 1000)
    z = out / "uae" / "pred_unet_main.zarr" / ".zmetadata"
    os.utime(z, (4000, 4000))
    assert eval_fresh(rep, out, src)[0] is False


def test_force_eval_flag_exists():
    import argparse
    import inspect

    import downscale.agent as agent

    assert "--force-eval" in inspect.getsource(agent.main)
    assert agent.FORCE_EVAL is False


def test_waits_are_five_increasing_steps():
    from downscale.agent import WAITS

    assert WAITS == [30, 60, 120, 180, 300]
    assert WAITS == sorted(WAITS)


def test_read_csv_truncation_is_explicit():
    out = read_csv("metrics_by_hour.csv", max_rows=60)
    last = out["text"].splitlines()[-1]
    assert last.startswith("truncated: showing 60 of 144 rows")
    assert "use the filter argument or a summary file" in last
    assert out["rows"] == 144


def test_read_csv_default_shows_every_hour_row():
    out = read_csv("metrics_by_hour.csv")
    assert "truncated" not in out["text"]
    assert len(out["text"].splitlines()) == 145


def test_read_csv_filter_reaches_the_rows_that_were_cut():
    out = read_csv("metrics_by_hour.csv", filter={"region": "bangladesh", "method": "unet_main"})
    assert out["matched"] == 24
    assert "truncated" not in out["text"]
    assert "bangladesh,Bangladesh region,unet_main,12," in out["text"]


def test_read_csv_filter_all_conditions_must_match():
    a = read_csv("metrics_by_hour.csv", filter={"region": "uae"})
    b = read_csv("metrics_by_hour.csv", filter={"region": "uae", "method": "lgbm"})
    assert a["matched"] == 72 and b["matched"] == 24


def test_read_csv_filter_on_unknown_column():
    out = read_csv("metrics_by_hour.csv", filter={"nope": "x"})
    assert "error" in out and "no column nope" in out["error"]


def test_read_csv_filter_with_no_matches():
    out = read_csv("metrics_by_hour.csv", filter={"region": "atlantis"})
    assert out["matched"] == 0
    assert out["text"].splitlines() == [out["text"].splitlines()[0]]


def test_read_csv_handles_quoted_commas():
    out = read_csv("stations_excluded.csv", filter={"region": "bangladesh"})
    assert out["matched"] == 1
    assert "Jorhat" in out["text"]


def test_worst_hours_recomputed_from_metrics_by_hour():
    import pandas as pd

    from downscale.config import ROOT, load_config

    rep = ROOT / load_config()["paths"]["reports"]
    w = pd.read_csv(rep / "worst_hours.csv")
    h = pd.read_csv(rep / "metrics_by_hour.csv")
    assert len(w) == 12
    for region in h.region.unique():
        for method in ("lgbm", "unet_main"):
            s = h[(h.region == region) & (h.method == method)]
            top = s.sort_values(["rmse_c", "local_hour"], ascending=[False, True]).head(3)
            got = w[(w.region == region) & (w.method == method)].sort_values("rank")
            assert list(got.local_hour) == list(top.local_hour)
            assert list(got.rmse_c) == [round(float(x), 4) for x in top.rmse_c]
            assert list(got["rank"]) == [1, 2, 3]


def test_worst_hours_has_the_number_the_model_invented():
    import pandas as pd

    from downscale.config import ROOT, load_config

    rep = ROOT / load_config()["paths"]["reports"]
    w = pd.read_csv(rep / "worst_hours.csv")
    bd = w[(w.region == "bangladesh") & (w.method == "unet_main") & (w["rank"] == 1)].iloc[0]
    assert bd.local_hour == 12
    assert bd.rmse_c == 1.0538


def test_prompt_points_the_hours_section_at_worst_hours():
    from downscale.agent import SYSTEM

    assert "worst_hours.csv" in SYSTEM
    assert "truncated" in SYSTEM


def test_calc_returns_a_rounded_string():
    from downscale.agent import calc

    assert calc(0.8277, 0.6576, "sub")["result"] == "0.1701"
    assert calc(0.7815, 0.7048, "sub")["result"] == "0.0767"
    assert calc(1.0, 3.0, "div")["result"] == "0.3333"
    assert calc(2.5, 4.0, "mul")["result"] == "10.0"
    assert calc(1.5, 2.25, "add")["result"] == "3.75"


def test_calc_rejects_unknown_op():
    from downscale.agent import calc

    assert "error" in calc(1.0, 2.0, "pow")


def test_calc_div_by_zero_is_not_an_exception():
    from downscale.agent import calc

    assert calc(1.0, 0.0, "div")["result"] == "nan"


def test_calc_result_traces():
    from downscale.agent import calc, trace

    r = calc(0.9437, 0.6483, "sub")["result"]
    traced, untraced = trace(f"The gap is {r} C.", [r])
    assert untraced == [] and r in traced


def test_calc_is_registered_as_a_tool():
    names = [t["function"]["name"] for t in TOOLS]
    assert "calc" in names and "calc" in FUNCS


def test_prompt_forbids_arithmetic():
    from downscale.agent import SYSTEM

    low = SYSTEM.lower()
    assert "never do arithmetic yourself" in low
    assert "model_margins.csv" in low


def test_quarantine_renames_a_failed_report(tmp_path, monkeypatch):
    import downscale.agent as agent

    monkeypatch.setattr(agent, "reports_dir", lambda: tmp_path)
    (tmp_path / "agent_report.md").write_text("bad report", encoding="utf-8")
    assert agent.quarantine() == "agent_report_failed.md"
    assert not (tmp_path / "agent_report.md").exists()
    assert (tmp_path / "agent_report_failed.md").read_text(encoding="utf-8") == "bad report"


def test_quarantine_overwrites_an_older_failed_report(tmp_path, monkeypatch):
    import downscale.agent as agent

    monkeypatch.setattr(agent, "reports_dir", lambda: tmp_path)
    (tmp_path / "agent_report_failed.md").write_text("older", encoding="utf-8")
    (tmp_path / "agent_report.md").write_text("newer", encoding="utf-8")
    agent.quarantine()
    assert (tmp_path / "agent_report_failed.md").read_text(encoding="utf-8") == "newer"


def test_quarantine_with_no_report_is_harmless(tmp_path, monkeypatch):
    import downscale.agent as agent

    monkeypatch.setattr(agent, "reports_dir", lambda: tmp_path)
    assert agent.quarantine() == "agent_report_failed.md"
    assert not (tmp_path / "agent_report_failed.md").exists()


def test_margins_match_the_metrics_table():
    import pandas as pd

    from downscale.config import ROOT, load_config

    rep = ROOT / load_config()["paths"]["reports"]
    m = pd.read_csv(rep / "model_margins.csv")
    o = pd.read_csv(rep / "metrics_overall.csv")
    for _, r in m.iterrows():
        def v(method, split):
            s = o[(o.region == r.region) & (o.method == method) & (o.split == split)]
            return round(float(s.rmse_c.iloc[0]), 4)

        assert r.lgbm_val_to_test_gap_c == round(v("lgbm", "test_blocks") - v("lgbm", "val"), 4)
        assert r.unet_main_val_to_test_gap_c == round(
            v("unet_main", "test_blocks") - v("unet_main", "val"), 4
        )
        assert r.unet_minus_lgbm_test_blocks_c == round(
            v("unet_main", "test_blocks") - v("lgbm", "test_blocks"), 4
        )


def test_margins_noise_from_two_runs():
    from downscale.margins import noise
    import pandas as pd

    runs = pd.DataFrame({"best_val_rmse_c": [0.6667, 0.6623]})
    assert noise(runs) == 0.0044


def test_tables_get_a_separator_row():
    from downscale.agent import fix_tables

    md = "## UAE region\n\n| method | rmse_c |\n| bilinear | 1.1883 |\n| lgbm | 0.8277 |\n"
    out = fix_tables(md).splitlines()
    assert out[2] == "| method | rmse_c |"
    assert out[3] == "| --- | --- |"
    assert out[4] == "| bilinear | 1.1883 |"


def test_existing_separator_is_left_alone():
    from downscale.agent import fix_tables

    md = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert fix_tables(md) == md


def test_fix_tables_is_idempotent():
    from downscale.agent import fix_tables

    md = "| a | b | c |\n| 1 | 2 | 3 |\n"
    once = fix_tables(md)
    assert fix_tables(once) == once
    assert once.splitlines()[1].count("---") == 3


def test_fix_tables_matches_column_count():
    from downscale.agent import fix_tables

    md = "| method | rmse_c | mae_c | bias_c | skill |\n| lgbm | 0.8277 | 0.6263 | 0.0564 | 0.3035 |\n"
    assert fix_tables(md).splitlines()[1] == "| --- | --- | --- | --- | --- |"


def test_fix_tables_handles_two_tables():
    from downscale.agent import fix_tables

    md = "| a | b |\n| 1 | 2 |\n\ntext\n\n| c | d |\n| 3 | 4 |\n"
    out = fix_tables(md).splitlines()
    assert out[1] == "| --- | --- |"
    assert out[7] == "| --- | --- |"
    assert out.count("| --- | --- |") == 2


def test_fix_tables_leaves_prose_alone():
    from downscale.agent import fix_tables

    md = "The U-Net reached 0.7815 c.\n\n- a bullet\n- another\n"
    assert fix_tables(md) == md


def test_write_report_writes_the_fixed_table(tmp_path, monkeypatch):
    import downscale.agent as agent

    monkeypatch.setattr(agent, "reports_dir", lambda: tmp_path)
    agent.write_report("| method | rmse_c |\n| lgbm | 0.8277 |\n")
    assert "| --- | --- |" in (tmp_path / "agent_report.md").read_text(encoding="utf-8")


def test_prompt_never_claims_a_bound():
    from downscale.agent import SYSTEM

    low = SYSTEM.lower()
    assert "bounds what" not in low
    assert "representativeness" in low
    assert "kaggle_runs.csv" in low


def test_read_csv_can_read_kaggle_runs():
    out = read_csv("kaggle_runs.csv")
    assert "text" in out and "best_val_rmse_c" in out["text"]


def test_read_csv_refuses_parent_paths():
    for bad in ("../config.yaml", "..\\config.yaml", "../../secrets.csv", ".git/config"):
        out = read_csv(bad)
        assert "error" in out and "refused" in out["error"]


def test_read_csv_refuses_absolute_path():
    out = read_csv("g:/temperature-downscaling/config.yaml")
    assert "error" in out


def test_read_csv_reads_a_real_report():
    out = read_csv("metrics_overall.csv", max_rows=5)
    assert "text" in out
    lines = out["text"].splitlines()
    assert lines[0].startswith("region,")
    assert len(lines) == 7
    assert lines[-1].startswith("truncated: showing 5 of 18 rows")


def test_read_csv_missing_file():
    out = read_csv("not_a_real_file.csv")
    assert "error" in out and "no such file" in out["error"]


def test_numbers_in_extracts_decimals_and_negatives():
    got = numbers_in("rmse 0.7815 c, bias -0.131, n 2178432")
    assert "0.7815" in got and "-0.131" in got and "2178432" in got


def test_trace_passes_real_numbers():
    src = ["region,method,rmse_c\nuae,unet_main,0.7815\n"]
    traced, untraced = trace("The U-Net reaches 0.7815 C.", src)
    assert untraced == []
    assert "0.7815" in traced


def test_trace_flags_a_planted_number():
    src = ["region,method,rmse_c\nuae,unet_main,0.7815\n"]
    traced, untraced = trace("The U-Net reaches 0.7815 C and lgbm 0.6123 C.", src)
    assert untraced == ["0.6123"]
    assert "0.7815" in traced


def test_trace_flags_a_rounded_number():
    src = ["0.7815"]
    traced, untraced = trace("about 0.78 C", src)
    assert untraced == ["0.78"]


def test_trace_on_empty_sources_flags_everything():
    traced, untraced = trace("rmse 1.23", [])
    assert traced == [] and untraced == ["1.23"]
