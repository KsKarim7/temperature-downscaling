import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from downscale.config import ROOT, load_config
from downscale.data import build
from downscale.unet import Unet


def masked_loss(pred, y, m, kind):
    e = (pred - y).abs() if kind == "l1" else (pred - y) ** 2
    return (e * m).sum() / m.sum().clamp(min=1.0)


def validate(model, dl, dev):
    model.eval()
    sse = 0.0
    n = 0.0
    with torch.no_grad():
        for x, y, m in dl:
            x = x.to(dev)
            y = y.to(dev)
            m = m.to(dev)
            p = model(x)
            sse += float((((p - y) ** 2) * m).sum())
            n += float(m.sum())
    model.train()
    return float(np.sqrt(sse / max(n, 1.0)))


def train(args):
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    cuda = torch.cuda.is_available()
    dev = torch.device(f"cuda:{local}" if cuda else "cpu")
    name = torch.cuda.get_device_name(local) if cuda else "cpu"
    print(f"rank {rank} of {world} on {dev} {name}", flush=True)

    cfg = load_config()
    drop = tuple(s for s in args.drop.split(",") if s)
    tr, va, norm, names = build(drop, args.train_crops, args.val_crops, cfg["seed"])

    sampler = DistributedSampler(tr, shuffle=True) if world > 1 else None
    dl = DataLoader(
        tr,
        batch_size=args.batch,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    vdl = DataLoader(va, batch_size=args.batch, num_workers=0) if rank == 0 else None

    model = Unet(len(names)).to(dev)
    if world > 1:
        model = DDP(model, device_ids=[local] if cuda else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    amp = (not args.no_amp) and cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out = ROOT / cfg["paths"]["outputs"]
    ck = ROOT / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)

    hist = []
    best = float("inf")
    best_ep = 0
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        tot = 0.0
        nb = 0
        t0 = time.time()
        for x, y, m in dl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            m = m.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = masked_loss(model(x), y, m, args.loss)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
            nb += 1
        sched.step()
        if rank == 0:
            vr = validate(model, vdl, dev)
            dt = time.time() - t0
            print(f"epoch {ep} train {tot / max(nb, 1):.4f} val rmse {vr:.4f} c {dt:.0f}s", flush=True)
            hist.append((ep, tot / max(nb, 1), vr, dt))
            if vr < best:
                best = vr
                best_ep = ep
                torch.save(
                    {
                        "model": (model.module if world > 1 else model).state_dict(),
                        "channels": names,
                        "drop": list(drop),
                        "norm": norm,
                        "args": vars(args),
                        "val_rmse": vr,
                        "epoch": ep,
                    },
                    ck / f"{args.name}.pt",
                )

    if rank == 0:
        with open(out / f"{args.name}_history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_rmse", "seconds"])
            w.writerows(hist)
        print(f"best val rmse {best:.4f} c at epoch {best_ep}")
    if world > 1:
        dist.destroy_process_group()
    return {"best_val": best, "best_epoch": best_ep, "minutes": (time.time() - t_start) / 60.0}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", default="l1", choices=["l1", "mse"])
    p.add_argument("--drop", default="")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--name", default="unet_main")
    p.add_argument("--train-crops", type=int, default=20000)
    p.add_argument("--val-crops", type=int, default=4000)
    return p.parse_args()


if __name__ == "__main__":
    train(parse())
