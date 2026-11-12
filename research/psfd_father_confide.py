# 子女對父親 vs 母親：說心事頻率與評分（RR2024，build_dyads 口徑，戶層叢集）
import io, contextlib, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
with contextlib.redirect_stdout(io.StringIO()):
    from psfd_engine import load, build_dyads
    dy = build_dyads(load(2024))[0].copy()
for v in ("k2p_k", "k2p_p"): dy[v + "_f"] = 6 - dy[v]   # 反向：1=從未 … 5=總是
print("口徑：RR2024｜說心事反向為 1從未…5總是｜father=1 父親、0 母親｜p＝OLS 戶層叢集 SE；MW＝Mann-Whitney")
for y, lab in (("aff_k", "子代評分"), ("aff_p", "親代評分"), ("k2p_k_f", "說心事(子報)"), ("k2p_p_f", "說心事(親報)")):
    s = dy[[y, "father", "hh"]].dropna()
    f, m = s[s.father == 1][y], s[s.father == 0][y]
    r = smf.ols(f"{y} ~ father", s).fit(cov_type="cluster", cov_kwds={"groups": s.hh})
    lo, hi = r.conf_int().loc["father"]
    print(f"{lab:9} 父 {f.mean():.3f}(n={len(f)}) 母 {m.mean():.3f}(n={len(m)}) 差 {r.params['father']:+.3f} [{lo:+.3f},{hi:+.3f}] p={r.pvalues['father']:.2g} MW p={stats.mannwhitneyu(f, m).pvalue:.2g}")
s = dy[["k2p_k_f", "father"]].dropna()
tab = pd.crosstab(s.father, s.k2p_k_f, normalize="index").round(3) * 100
tab.index = ["母親", "父親"]; tab.columns = ["從未", "偶爾", "有時", "經常", "總是"]
print("說心事(子報) 分布 %：\n" + tab.to_string())
s = dy[["aff_k", "father", "k2p_k_f", "hh"]].dropna()
r = smf.ols("aff_k ~ father + k2p_k_f", s).fit(cov_type="cluster", cov_kwds={"groups": s.hh})
print(f"子代評分 ~ 父親 + 說心事：父親 b={r.params['father']:+.3f} p={r.pvalues['father']:.2g}｜說心事 b={r.params['k2p_k_f']:+.3f}｜n={int(r.nobs)}")
