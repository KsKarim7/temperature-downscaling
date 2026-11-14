import json

from downscale.agent import FUNCS, TOOLS, numbers_in, read_csv, trace


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
    assert out["text"].splitlines()[0].startswith("region,")
    assert len(out["text"].splitlines()) == 6


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
