"""
Each pitcher's plane height C.

On the Pitch Plane (01), speed = A·縦 + B·横 + C, and only C differs between pitchers. For each
pitcher: his type means (types with at least 3% use and 30 pitches), the speed each would have
with its movement taken out (speed − A·縦 − B·横), and C = the median of those over his types,
so one outlying pitch (a very slow changeup) does not move it.

Writes output/plane_height_2026.csv (pitcher, name, hand, c).

Usage:  python src/02_plane_height.py
"""
import sys, os
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import common as sph

YEAR = 2026
MIN_SHARE, MIN_N = .03, 30
DOTS = 60
SEED = 11

z = np.load(os.path.join(ROOT, "output", "plane_%d.npz" % YEAR))
n, S = z["normal"], float(z["s"])
A, B = -n[1] / (n[2] * S), -n[0] / (n[2] * S)        # mph per inch of 縦, of 横
V0 = float(z["v0"])
F1 = np.array([1., 0., 0.]) - n[0] * n; F1 /= np.linalg.norm(F1)   # 横, laid into the plane
F2 = np.cross(n, F1); F2 *= np.sign(F2[1])                          # up, in the plane

d = pd.read_parquet(sph.SNAP[YEAR], columns=["game_type", "pitcher", "player_name", "pitch_type",
                    "p_throws", "pfx_x", "pfx_z", "release_speed"])
d = d[(d.game_type == "R") & d.pitch_type.isin(sph.TYPES)].dropna().copy()
cnt = d.groupby("pitcher").size()
d = d[d.pitcher.isin(cnt[cnt >= sph.MIN_PITCHES].index)]
for c in ["pfx_x", "pfx_z", "release_speed"]:
    d[c] = d[c].astype(float)
d["x"] = np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12
d["y"] = d.pfx_z * 12
Wd = np.c_[d.x, d.y, (d.release_speed - V0) * S]
d["p1"], d["p2"] = Wd @ F1, Wd @ F2
d["eq"] = d.release_speed - (A * d.y + B * d.x)

g = d.groupby(["pitcher", "pitch_type"])
m = g[["x", "y", "release_speed"]].mean()
m["n"] = g.size()
m["share"] = m.n / m.groupby(level=0).n.transform("sum")
m = m[(m.share >= MIN_SHARE) & (m.n >= MIN_N)].reset_index()
m["pred0"] = A * m.y + B * m.x
W = np.c_[m.x, m.y, (m.release_speed - V0) * S]
m["p1"], m["p2"] = W @ F1, W @ F2
m["eq"] = m.release_speed - m.pred0                  # speed with the movement taken out
# median, not mean: one far-off pitch (a very slow changeup) must not drag the others
m["c"] = (m.release_speed - m.pred0).groupby(m.pitcher).transform("median")
m["resid"] = m.release_speed - m.pred0 - m.c
m["pct"] = m.groupby("pitch_type").resid.rank(pct=True)
m["peers"] = m.groupby("pitch_type").resid.transform("size")

info = d.groupby("pitcher").agg(name=("player_name", "first"), hand=("p_throws", "first"))
cs = m.groupby("pitcher").c.first()
out = info.join(cs.rename("c"), how="inner")
path = os.path.join(ROOT, "output", "plane_height_%d.csv" % YEAR)
out.to_csv(path)
print("%d投手  A=%.3f B=%.3f  C の中央 %.1f（10%%点 %.1f、90%%点 %.1f）  wrote %s"
      % (len(out), A, -B, cs.median(), cs.quantile(.1), cs.quantile(.9), path))
