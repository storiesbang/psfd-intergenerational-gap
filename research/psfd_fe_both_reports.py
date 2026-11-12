# 圖 4 同口徑：雙向 FE，說心事用子女自陳 vs 父母回報，對親評/子評/溫差；另算兩版 Δ 的相關
import io, contextlib, numpy as np, pandas as pd
from linearmodels.panel import PanelOLS
with contextlib.redirect_stdout(io.StringIO()):
    from psfd_engine import build_panel
    p = build_panel((2020, 2022, 2024))
p["hh"] = p.hh_24
NUM = ["aff_p", "aff_k", "gap", "k2p_k", "k2p_p"]
long = pd.concat([p[[f"{v}_{w % 100:02d}" for v in NUM]].rename(columns=lambda c: c[:-3])
                  .assign(wave=w, hh=p.hh.values, dyad_id=p.index) for w in (2020, 2022, 2024)], ignore_index=True)
for v in ("k2p_k", "k2p_p"): long[v] = 6 - long[v]
print("口徑：三波交集、雙向 FE、戶層叢集｜說心事反向（大＝越常說）")
for x, xl in (("k2p_k", "子女自陳"), ("k2p_p", "父母回報")):
    for y in ("aff_p", "aff_k", "gap"):
        s = long[[x, y, "wave", "hh", "dyad_id"]].dropna().set_index(["dyad_id", "wave"])
        r = PanelOLS(s[y], s[[x]], entity_effects=True, time_effects=True).fit(cov_type="clustered", clusters=s[["hh"]])
        print(f"{xl} → {y:5}: b={r.params[x]:+.3f} SE={r.std_errors[x]:.3f} p={r.pvalues[x]:.3g} n={int(r.nobs)}")
dm = long.set_index(["dyad_id", "wave"])[["k2p_k", "k2p_p"]].dropna()
dm = dm - dm.groupby(level=0).transform("mean")
print(f"組內變化（去個體平均）的相關 r(子陳, 親報) = {dm.k2p_k.corr(dm.k2p_p):.3f}")
