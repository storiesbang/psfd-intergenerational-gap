# 去年同住 vs 幾乎每天通話：兩代評分與溫差的兩組比較（圖 3b 同口徑，RR2024）
import io, contextlib, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
with contextlib.redirect_stdout(io.StringIO()):
    from psfd_engine import load, build_dyads, V
    D = load(2024); dy = build_dyads(D)[0].copy()
K = D.set_index("x01").loc[dy.kid].reset_index(drop=True)
dy["tel_raw"] = [K.at[i, V[2024]["k"]["tel"].format(s="f1" if f == 1 else "m1")] for i, f in enumerate(dy.father)]
s = dy[dy.tel_raw.isin([0, 1])][["tel_raw", "aff_p", "aff_k", "gap", "hh"]].dropna().copy()
s["cohab"] = (s.tel_raw == 0).astype(int)
print("口徑：RR2024｜去年同住(tel=0) vs 幾乎每天(tel=1)｜n =", s.cohab.sum(), "/", (1 - s.cohab).sum())
for y in ("aff_p", "aff_k", "gap"):
    a, b = s[s.cohab == 1][y], s[s.cohab == 0][y]
    t = stats.ttest_ind(a, b, equal_var=False)
    m = smf.ols(f"{y} ~ cohab", s).fit(cov_type="cluster", cov_kwds={"groups": s.hh})
    print(f"{y:6} 同住 {a.mean():.3f} 每天 {b.mean():.3f} 差 {a.mean()-b.mean():+.3f} | Welch p={t.pvalue:.3g} | 叢集 p={m.pvalues['cohab']:.3g}")
