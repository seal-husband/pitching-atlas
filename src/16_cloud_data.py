"""
Data for the rotatable cloud of every pitch (web/cloud.html): figure 1 of the note article, to turn.

The same 30,000 pitches as note_1.png (15, same seed), each with its pitch type, 横 (glove side +,
lefties mirrored), 縦, speed, and its speed with the pitcher's plane moved to the league height
(speed − (C − median C), as in note_3.png). Also the plane itself and three camera directions in
the plane-fit coordinates (横, 縦, 球速 × 0.63): the oblique view of note_1, Savant's front view, and
the edge-on view of note_2/3.

Writes web/cloud_data.js as window.CLOUD (the artifact sandbox cannot fetch).

Usage:  python src/16_cloud_data.py
"""
import sys, os, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from paths import SNAP

TYPES = ["FF", "SI", "FC", "FA", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO"]
MIN_PITCHES, SEED, N = 400, 32, 30000

z = np.load(os.path.join(ROOT, "output", "plane_2026.npz"))
n, S = z["normal"], float(z["s"])
A, B = -n[1] / (n[2] * S), -n[0] / (n[2] * S)
F1 = np.array([1., 0., 0.]) - n[0] * n; F1 /= np.linalg.norm(F1)
F2 = np.cross(n, F1); F2 *= np.sign(F2[1])
CP = pd.read_csv(os.path.join(ROOT, "output", "plane_height_2026.csv"), index_col=0).c.to_dict()
C_MED = float(np.median(list(CP.values())))

# exactly as 15, so the sample is note_1's
d = pd.read_parquet(SNAP[2026], columns=["game_type", "pitcher", "player_name", "pitch_type", "p_throws",
                                         "pfx_x", "pfx_z", "release_speed"])
d = d[(d.game_type == "R") & d.pitch_type.isin(TYPES)].dropna().copy()
cnt = d.groupby("pitcher").size()
d = d[d.pitcher.isin(cnt[cnt >= MIN_PITCHES].index)]
for c in ["pfx_x", "pfx_z", "release_speed"]:
    d[c] = d[c].astype(float)
d["x"] = np.where(d.p_throws.eq("L"), -d.pfx_x, d.pfx_x) * 12
d["y"] = d.pfx_z * 12
d["v"] = d.release_speed
d["c"] = d.pitcher.map(CP)
d = d[d.c.notna()]
rng = np.random.default_rng(SEED)
bg = d.iloc[rng.choice(len(d), N, replace=False)]
bg = bg.sample(frac=1, random_state=SEED)          # draw order of note_1


def eye(v):
    v = np.asarray(v, float)
    return np.round(v / np.linalg.norm(v), 4).tolist()


e, a = np.radians(14), np.radians(-62)
edge = np.cos(np.radians(20)) * F1 + np.sin(np.radians(20)) * F2
out = {"types": [t for t in TYPES if (bg.pitch_type == t).any()],
       "t": [], "x": np.round(bg.x.values, 1).tolist(), "y": np.round(bg.y.values, 1).tolist(),
       "v": np.round(bg.v.values, 1).tolist(), "va": np.round((bg.v - (bg.c - C_MED)).values, 1).tolist(),
       "A": round(A, 4), "B": round(B, 4), "cmed": round(C_MED, 2), "s": S, "n": len(bg),
       "cam": {"free": eye([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]),
               "front": [0, 0, 1], "edge": eye(edge)}}
code = {t: i for i, t in enumerate(out["types"])}
out["t"] = [code[t] for t in bg.pitch_type]
path = os.path.join(ROOT, "web", "cloud_data.js")
with open(path, "w", encoding="utf-8") as f:
    f.write("window.CLOUD=" + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";")
print("%d球（%s）  PP 球速 = %.3f×縦 %+.3f×横 + C、C の中央 %.1f  wrote %s (%.0f KB)"
      % (len(bg), " ".join(out["types"]), A, B, C_MED, path, os.path.getsize(path) / 1024))
