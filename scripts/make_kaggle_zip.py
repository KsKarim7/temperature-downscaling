import zipfile

from downscale.config import ROOT

TREES = ["data/processed/uae.zarr", "data/processed/bangladesh.zarr"]
FILES = ["outputs/uae/splits.npz", "outputs/bangladesh/splits.npz", "outputs/norm.json"]


def items():
    out = []
    for t in TREES:
        for p in sorted((ROOT / t).rglob("*")):
            if p.is_file():
                out.append(p)
    for f in FILES:
        out.append(ROOT / f)
    return out


def main():
    dst = ROOT / "kaggle_upload.zip"
    paths = items()
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"{len(paths)} files")
    print(f"{dst.name} {dst.stat().st_size / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
