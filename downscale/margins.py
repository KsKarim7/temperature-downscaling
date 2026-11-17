import pandas as pd

from downscale.config import ROOT, load_config

LABEL = {"uae": "UAE region", "bangladesh": "Bangladesh region"}


def noise(runs):
    v = runs.best_val_rmse_c.to_numpy()
    return round(float(abs(v[0] - v[1])), 4) if v.size >= 2 else float("nan")


def rmse(o, region, method, split):
    s = o[(o.region == region) & (o.method == method) & (o.split == split)]
    return round(float(s.rmse_c.iloc[0]), 4)


def main():
    cfg = load_config()
    rep = ROOT / cfg["paths"]["reports"]
    o = pd.read_csv(rep / "metrics_overall.csv")
    n = noise(pd.read_csv(rep / "kaggle_runs.csv"))

    rows = []
    for region in sorted(o.region.unique()):
        lv = rmse(o, region, "lgbm", "val")
        lt = rmse(o, region, "lgbm", "test_blocks")
        uv = rmse(o, region, "unet_main", "val")
        ut = rmse(o, region, "unet_main", "test_blocks")
        margin = round(ut - lt, 4)
        rows.append(
            {
                "region": region,
                "label": LABEL[region],
                "lgbm_val_rmse_c": lv,
                "lgbm_test_blocks_rmse_c": lt,
                "lgbm_val_to_test_gap_c": round(lt - lv, 4),
                "unet_main_val_rmse_c": uv,
                "unet_main_test_blocks_rmse_c": ut,
                "unet_main_val_to_test_gap_c": round(ut - uv, 4),
                "unet_minus_lgbm_test_blocks_c": margin,
                "run_to_run_noise_c": n,
                "margin_over_noise": round(margin / n, 4) if n else float("nan"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(rep / "model_margins.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
