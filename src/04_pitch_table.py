"""
Every pitch on the Pitch Plane, for the peak finder (05) and the valley check (06).

Each pitch (横, 縦, (球速 − v0) × 0.63) is laid into the plane: u along 横, w up the plane. Also
relative to the pitcher's primary fastball (the most-thrown of FF/SI/FC): ur, wr. half splits
each pitcher's pitches alternately, for split-half checks.

Writes output/pitches_pp_2026.parquet.

Usage:  python src/04_pitch_table.py
"""
import sys, os
import numpy as np, pandas as pd
import common as sph
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

YEAR = 2026
z = np.load(os.path.join(ROOT, "output", "plane_%d.npz" % YEAR))
n, S, V0 = z["normal"], float(z["s"]), float(z["v0"])
F1 = np.array([1., 0., 0.]) - n[0] * n; F1 /= np.linalg.norm(F1)       # 横, laid in the plane
F2 = np.cross(n, F1); F2 *= np.sign(F2[1])                             # up, in the plane


def load():
    d = pd.read_parquet(sph.SNAP[YEAR], columns=["game_type", "pitcher", "player_name",
                        "pitch_type", "p_throws", "pfx_x", "pfx_z", "release_speed"])
    d = d[(d.game_type == "R") & d.pitch_type.isin(sph.TYPES)].dropna().copy()
    cnt = d.groupby("pitcher").size(); d = d[d.pitcher.isin(cnt[cnt >= sph.MIN_PITCHES].index)]
    for c in ["pfx_x", "pfx_z", "release_speed"]:
        d[c] = d[c].astype(float)
    X = np.c_[np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12, d.pfx_z * 12,
              (d.release_speed - V0) * S]
    d["u"], d["w"] = X @ F1, X @ F2
    # primary fastball = the most-thrown of FF/SI/FC
    f = d[d.pitch_type.isin(["FF", "SI", "FC"])]
    top = f.groupby("pitcher").pitch_type.agg(lambda s: s.value_counts().index[0])
    ref = f[f.pitch_type.values == f.pitcher.map(top).values].groupby("pitcher")[["u", "w"]].mean()
    d = d[d.pitcher.isin(ref.index)]
    d["ur"] = d.u - d.pitcher.map(ref.u)
    d["wr"] = d.w - d.pitcher.map(ref.w)
    d["half"] = d.groupby("pitcher").cumcount() % 2
    return d


d = load()
path = os.path.join(ROOT, "output", "pitches_pp_%d.parquet" % YEAR)
d[["pitcher", "player_name", "pitch_type", "u", "w", "ur", "wr", "half"]].to_parquet(path, index=False)
print("%d年 %d投手 / %d球  wrote %s" % (YEAR, d.pitcher.nunique(), len(d), path))
