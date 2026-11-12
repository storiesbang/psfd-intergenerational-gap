# -*- coding: utf-8 -*-
"""通話頻率對溫差：單獨放 vs 與說心事等因素同時放，顯著性是否改變。

沿用 psfd_controls_robust.py 的圖 2 資料與設計矩陣（同樣本、同標準化、戶層叢集）。
單因素模型一律附「同住」：同住者的通話/見面被設為 0，不附同住會把同住效果混進來。
跑法：cd research && uv run --with pandas --with pyreadstat --with statsmodels --with scipy python psfd_call_alone_vs_joint.py
"""
import os, io, contextlib
import statsmodels.api as sm

src = open("psfd_controls_robust.py", encoding="utf-8").read().split("# ── 3.")[0]
ns = {"__file__": os.path.abspath("psfd_controls_robust.py")}
with contextlib.redirect_stdout(io.StringIO()):
    exec(src, ns)
s, X, ok = ns["s"], ns["X"], ns["okA"]


def ols(cols):
    m = sm.OLS(s.gap[ok], sm.add_constant(X.loc[ok, cols])).fit(
        cov_type="cluster", cov_kwds={"groups": s.x01d[ok]})
    return m


print(f"口徑：圖 2 樣本 n={int(ok.sum())}｜依變項＝溫差(親評−子評)｜自變項標準化(1 SD)｜戶層叢集標準誤")
print(f"{'因素':<14}{'單獨(+同住) b':>14}{'p':>8}{'全模型 b':>12}{'p':>8}")
full = ols(list(X.columns))
for v in ["通話頻率", "見面頻率", "子女說心事（子報）", "金錢往來", "家務協助天數"]:
    m1 = ols([v, "同住"])
    print(f"{v:<14}{m1.params[v]:>+14.3f}{m1.pvalues[v]:>8.3f}{full.params[v]:>+12.3f}{full.pvalues[v]:>8.3f}")

# 通話係數是被誰吸走的：只加說心事 vs 只加見面
print("\n通話頻率 b（p）：")
for extra in [[], ["子女說心事（子報）"], ["見面頻率"], ["子女說心事（子報）", "見面頻率"]]:
    m = ols(["通話頻率", "同住"] + extra)
    print(f"  +{'、'.join(extra) or '無'}：{m.params['通話頻率']:+.3f}（{m.pvalues['通話頻率']:.3f}）")
print(f"通話 × 說心事 相關 r = {X.loc[ok, '通話頻率'].corr(X.loc[ok, '子女說心事（子報）']):.3f}")

# 中介（係數乘積法 a×b，戶層 cluster bootstrap 2000 次）；兩個方向都跑，看橫斷面能否分辨方向
import numpy as np
CALL, CONF = "通話頻率", "子女說心事（子報）"
D = X.loc[ok].assign(gap=s.gap[ok], hh=s.x01d[ok]).reset_index(drop=True)


def ind(df, x, m):
    a = sm.OLS(df[m], sm.add_constant(df[[x, "同住"]])).fit().params[x]
    bm = sm.OLS(df.gap, sm.add_constant(df[[x, m, "同住"]])).fit().params
    return a * bm[m], bm[x]


rng = np.random.default_rng(0)
groups = {h: g.index.values for h, g in D.groupby("hh")}
keys = np.array(list(groups))
print("\n中介（a×b；95% CI＝戶層 bootstrap 2000 次）")
for x, m in [(CALL, CONF), (CONF, CALL)]:
    ie, de = ind(D, x, m)
    bs = [ind(D.loc[np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])], x, m)[0] for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {x} → {m} → 溫差：間接 {ie:+.3f} [{lo:+.3f}, {hi:+.3f}]｜直接 {de:+.3f}")
