import argparse
import csv
from argparse import Namespace

import torch

from downscale.config import ROOT, load_config
from downscale.train import train


def run(name, lr, loss, drop, epochs, batch, crops, vcrops):
    args = Namespace(
        epochs=epochs,
        lr=lr,
        loss=loss,
        drop=drop,
        batch=batch,
        no_amp=True,
        name=name,
        train_crops=crops,
        val_crops=vcrops,
    )
    print(f"run {name} lr {lr} loss {loss} drop '{drop}'", flush=True)
    r = train(args)
    torch.cuda.empty_cache()
    return {
        "name": name,
        "lr": lr,
        "loss": loss,
        "drop": drop,
        "best_val_rmse_c": round(r["best_val"], 4),
        "best_epoch": r["best_epoch"],
        "minutes": round(r["minutes"], 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--train-crops", type=int, default=20000)
    p.add_argument("--val-crops", type=int, default=4000)
    a = p.parse_args()
    cfg = load_config()
    rep = ROOT / cfg["paths"]["reports"]
    rep.mkdir(parents=True, exist_ok=True)

    rows = []
    for lr in (1e-3, 3e-4):
        for loss in ("l1", "mse"):
            tag = f"sweep_lr{lr:g}_{loss}"
            rows.append(run(tag, lr, loss, "", a.epochs, a.batch, a.train_crops, a.val_crops))

    best = min(rows, key=lambda r: r["best_val_rmse_c"])
    print(f"stage 1 best lr {best['lr']} loss {best['loss']} val {best['best_val_rmse_c']}")

    for drop in ("position", "landcover", "terrain"):
        tag = f"sweep_drop_{drop}"
        rows.append(run(tag, best["lr"], best["loss"], drop, a.epochs, a.batch, a.train_crops, a.val_crops))

    cols = ["name", "lr", "loss", "drop", "best_val_rmse_c", "best_epoch", "minutes"]
    with open(rep / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    top = min(rows, key=lambda r: r["best_val_rmse_c"])
    print("--- sweep ---")
    for r in rows:
        print(
            f"{r['name']:24s} lr {r['lr']:<7g} loss {r['loss']:4s} drop {r['drop'] or 'none':10s} "
            f"val {r['best_val_rmse_c']:.4f} c  epoch {r['best_epoch']}  {r['minutes']:.1f} min"
        )
    print(
        f"best config lr {top['lr']} loss {top['loss']} drop '{top['drop']}' "
        f"val rmse {top['best_val_rmse_c']:.4f} c"
    )


if __name__ == "__main__":
    main()
