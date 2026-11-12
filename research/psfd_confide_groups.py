# -*- coding: utf-8 -*-
"""圖 3a 各說心事組的溫差是否 ≠ 0（RR2024）。
樣本與分組同 psfd_v3_figs.py 圖 3(a)：engine.build_dyads(2024)，k2p_k（子女報告，1總是…5從未）與兩代感情皆有值，n=988。
檢定：各組 溫差 ~ 1 的 OLS，戶層叢集 SE（同 3.2／第 2 章口徑）；另列未叢集單樣本 t 對照；BH–FDR 對 5 組。
跑法：cd research && uv run --with pandas --with pyreadstat --with statsmodels --with scipy python psfd_confide_groups.py
"""
import numpy as np, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from psfd_engine import load, build_dyads

print("=" * 70)
print("口徑：PSFD RR2024，engine.build_dyads 配對；溫差＝親代−子代感情（1–5）")
print("分組：子女報告說心事 k2p_k（同圖 3a）；H0：該組平均溫差＝0；主檢定＝戶層叢集 SE，FDR＝BH 對 5 組")
print("=" * 70)

dy = build_dyads(load(2024))[0]
s = dy[["k2p_k", "aff_p", "aff_k", "hh"]].dropna().copy()
s["gap"] = s.aff_p - s.aff_k
GROUPS = [(5, "從未"), (4, "偶爾"), (3, "有時"), (2, "經常"), (1, "總是")]
ns = [int((s.k2p_k == v).sum()) for v, _ in GROUPS]
assert len(s) == 988 and ns == [378, 406, 126, 59, 19], (len(s), ns)  # 對回圖 3a


def test(sub):
    r = smf.ols("gap ~ 1", sub).fit(cov_type="cluster", cov_kwds={"groups": sub.hh})
    return r.params.iloc[0], r.bse.iloc[0], r.pvalues.iloc[0], stats.ttest_1samp(sub.gap, 0).pvalue, sub.hh.nunique()


res = [(lab, *test(s[s.k2p_k == v])) for v, lab in GROUPS]
q = multipletests([r[3] for r in res], method="fdr_bh")[1]
print(f"\n{'組':4s} {'n':>4s} {'戶':>4s} {'平均溫差':>8s} {'叢集SE':>7s} {'95% CI':>17s} {'p(叢集)':>9s} {'q':>9s} {'p(未叢集)':>9s}")
for (lab, m, se, p, p0, hh), n, qq in zip(res, ns, q):
    print(f"{lab:4s} {n:4d} {hh:4d} {m:+8.3f} {se:7.3f} [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}] {p:9.2g} {qq:9.2g} {p0:9.2g}")

hi = s[s.k2p_k <= 2]
m, se, p, p0, hh = test(hi)
print(f"\n合併「經常＋總是」n={len(hi)}（{hh} 戶）：平均溫差 {m:+.3f} [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}]，p(叢集)={p:.2g}")
m, se, p, _, _ = test(s)
print(f"全樣本 n={len(s)}：平均溫差 {m:+.3f}，p(叢集)={p:.2g}")
print("\n結論：q<0.05 的組 →", [lab for (lab, *_), qq in zip(res, q) if qq < .05])
