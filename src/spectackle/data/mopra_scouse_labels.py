### Scouse/Henshaw .dat labels: parse, spatial split, spectrum cache for fine-tuning.
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.wcs import WCS

from .mopra_preprocess import MOPRA_BLANK_VALUE, prepare_mopra_input


def parse_scouse_dat_positions(
    dat_path: Path | str,
    *,
    use_row_count: bool = False,
) -> list[dict]:
    """
    Parse final_fits_updated.dat into one record per unique (l, b).

    K_true uses column 0 (ncomps) by default; set use_row_count=True for len(rows).
    """
    arr = np.loadtxt(dat_path)
    by_pos: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for row in arr:
        key = (round(float(row[1]), 5), round(float(row[2]), 5))
        by_pos[key].append(row)

    records: list[dict] = []
    for (l, b), rows in by_pos.items():
        if use_row_count:
            k = len(rows)
        else:
            k = int(rows[0][0])
        records.append({"l": float(l), "b": float(b), "K_true": int(k), "n_rows": len(rows)})
    return records


def assign_spatial_split(
    records: list[dict],
    *,
    val_frac: float = 0.2,
    cell_deg: float = 0.08,
    seed: int = 42,
) -> dict[tuple[float, float], str]:
    """
    Assign train/val by spatial cell (avoids leaking correlated neighbors into val).

    Returns map (l_rounded, b_rounded) -> "train" | "val".
    """
    cells: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for rec in records:
        l, b = rec["l"], rec["b"]
        ci = int(np.floor(l / cell_deg))
        cj = int(np.floor(b / cell_deg))
        cells[(ci, cj)].append((round(l, 5), round(b, 5)))

    rng = np.random.default_rng(seed)
    cell_ids = sorted(cells.keys())
    rng.shuffle(cell_ids)
    n_val_cells = max(1, int(round(len(cell_ids) * val_frac)))
    val_cells = set(cell_ids[:n_val_cells])

    split_map: dict[tuple[float, float], str] = {}
    for cell in cell_ids:
        tag = "val" if cell in val_cells else "train"
        for key in cells[cell]:
            split_map[key] = tag
    return split_map


def build_scouse_labeled_cache(
    *,
    dat_path: Path | str,
    cube_path: Path | str,
    out_path: Path | str,
    val_frac: float = 0.2,
    cell_deg: float = 0.08,
    seed: int = 42,
    use_row_count: bool = False,
    blank_value: float = MOPRA_BLANK_VALUE,
) -> dict:
    """
    Extract smooth60 spectra at Scouse (l,b) positions; save NPZ + sidecar JSON metadata.

    NPZ keys: spec_norm, valid_mask, K_true, l, b, yi, xi, split (0=train, 1=val).
    """
    from spectral_cube import SpectralCube

    dat_path = Path(dat_path)
    cube_path = Path(cube_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = parse_scouse_dat_positions(dat_path, use_row_count=use_row_count)
    split_map = assign_spatial_split(records, val_frac=val_frac, cell_deg=cell_deg, seed=seed)

    cube = SpectralCube.read(str(cube_path.resolve()), use_dask=False)
    arr = np.asarray(cube.filled(np.nan), dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    nv, ny, nx = arr.shape
    wcs = WCS(cube.header).celestial

    n = len(records)
    spec_norm = np.zeros((n, nv), dtype=np.float32)
    valid_mask = np.zeros((n, nv), dtype=np.float32)
    K_true = np.zeros((n, 1), dtype=np.int64)
    l_arr = np.zeros(n, dtype=np.float64)
    b_arr = np.zeros(n, dtype=np.float64)
    yi_arr = np.zeros(n, dtype=np.int32)
    xi_arr = np.zeros(n, dtype=np.int32)
    split_arr = np.zeros(n, dtype=np.int8)

    n_skip = 0
    for i, rec in enumerate(records):
        l, b = rec["l"], rec["b"]
        key = (round(l, 5), round(b, 5))
        xp, yp = wcs.all_world2pix([[l, b]], 0)[0]
        xi, yi = int(round(xp)), int(round(yp))
        if not (0 <= xi < nx and 0 <= yi < ny):
            n_skip += 1
            continue
        spec = arr[:, yi, xi].astype(np.float64)
        sn, vm = prepare_mopra_input(spec, blank_value=blank_value)
        spec_norm[i] = sn
        valid_mask[i] = vm
        K_true[i, 0] = int(rec["K_true"])
        l_arr[i] = l
        b_arr[i] = b
        yi_arr[i] = yi
        xi_arr[i] = xi
        split_arr[i] = 1 if split_map.get(key, "train") == "val" else 0

    if n_skip:
        print(f"  skipped {n_skip} positions off cube grid", flush=True)

    np.savez_compressed(
        out_path,
        spec_norm=spec_norm,
        valid_mask=valid_mask,
        K_true=K_true,
        l=l_arr,
        b=b_arr,
        yi=yi_arr,
        xi=xi_arr,
        split=split_arr,
        n_channels=np.array([nv], dtype=np.int32),
    )

    n_train = int((split_arr == 0).sum())
    n_val = int((split_arr == 1).sum())
    meta = {
        "dat_path": str(dat_path.resolve()),
        "cube_path": str(cube_path.resolve()),
        "n_positions": n,
        "n_train": n_train,
        "n_val": n_val,
        "n_channels": int(nv),
        "val_frac": float(val_frac),
        "cell_deg": float(cell_deg),
        "seed": int(seed),
        "use_row_count": bool(use_row_count),
        "K_hist": {str(k): int((K_true[:, 0] == k).sum()) for k in range(1, int(K_true.max()) + 1)},
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["cache_path"] = str(out_path.resolve())
    return meta


def load_scouse_labeled_cache(cache_path: Path | str) -> dict:
    """Load NPZ cache; returns dict of arrays + metadata JSON if present."""
    cache_path = Path(cache_path)
    data = dict(np.load(cache_path, allow_pickle=False))
    meta_path = cache_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return {"arrays": data, "meta": meta}


__all__ = [
    "parse_scouse_dat_positions",
    "assign_spatial_split",
    "build_scouse_labeled_cache",
    "load_scouse_labeled_cache",
]
