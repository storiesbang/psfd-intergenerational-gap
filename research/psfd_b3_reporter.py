# -*- coding: utf-8 -*-
"""B3 換報告人的 FE（2×2）：吐露（子報 k2p_k／親報 k2p_p）× 評分（子代 aff_k／親代 aff_p）。
主軸「父母對訊號反應不足」要在交叉報告的兩格也成立：子報吐露→親代評分、親報吐露→子代評分。
跑法：cd research && uv run --with pandas --with pyreadstat --with statsmodels --with scipy --with linearmodels python psfd_b3_reporter.py | tee out/psfd/b3_output.txt
"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from psfd_engine import build_panel, header

WAVES = (2020, 2022, 2024)
p = build_panel(WAVES)
assert (p.hh_20 == p.hh_24).all() and (p.hh_22 == p.hh_24).all(), "戶號應跨波穩定"
p["hh"] = p.hh_24
header(p, "PSFD 三波 · B3 換報告人的固定效果（2×2）")

NUM = ["aff_p", "aff_k", "k2p_k", "k2p_p"]
long = pd.concat([p[[f"{v}_{w % 100:02d}" for v in NUM]].rename(columns=lambda c: c[:-3])
                  .assign(wave=w, hh=p.hh.values, dyad_id=p.index) for w in WAVES], ignore_index=True)
for v in ("k2p_k", "k2p_p"):  # 量表 1=最常說；同尺度才能比
    assert long[v].dropna().between(1, 5).all(), f"{v} 超出 1–5"
print(f"長格式 {len(long)} 觀測／{long.dyad_id.nunique()} 組／{long.hh.nunique()} 戶")
print("吐露量表 1=最常說 ⇒ 係數為負＝吐露變多、評分變高\n")

CELLS = [("k2p_k", "aff_k", "子報吐露→子代評分", "同報告人（原主菜）"),
         ("k2p_k", "aff_p", "子報吐露→親代評分", "交叉"),
         ("k2p_p", "aff_k", "親報吐露→子代評分", "交叉"),
         ("k2p_p", "aff_p", "親報吐露→親代評分", "同報告人")]


def old_way(x, y):
    """原 psfd_panel.py 口徑：全欄先去中心再 dropna，事後加 C(wave)。用來對回 −0.118。"""
    w = long[NUM] - long.groupby("dyad_id")[NUM].transform("mean")
    s = long.assign(**{v: w[v] for v in NUM})[[x, y, "wave", "hh"]].dropna()
    r = smf.ols(f"{y} ~ {x} + C(wave)", data=s).fit(cov_type="cluster", cov_kwds={"groups": s.hh})
    return r.params[x], r.bse[x], r.tvalues[x], len(s)


def twfe(x, y):
    """正規雙向 FE：估計樣本內去中心、dyad＋波次效果、戶層叢集（dyad 巢套於戶，不另扣自由度）。"""
    s = long[[x, y, "wave", "hh", "dyad_id"]].dropna().set_index(["dyad_id", "wave"])
    r = PanelOLS(s[y], s[[x]], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", clusters=s[["hh"]])
    return r.params[x], r.std_errors[x], r.tstats[x], r.pvalues[x], int(r.nobs)


print("【1】四格（雙向 FE＋戶層叢集）｜括號＝原口徑對照")
print(f"  {'格':20s} {'類型':12s} {'n':>5s} {'b':>8s} {'SE':>6s} {'t':>7s} {'p':>8s}   原口徑 b (t)")
res = {}
for x, y, nm, kind in CELLS:
    b, se, t, pv, n = twfe(x, y)
    ob, _, ot, _ = old_way(x, y)
    res[(x, y)] = (b, t, pv)
    print(f"  {nm:20s} {kind:12s} {n:5d} {b:+8.3f} {se:6.3f} {t:+7.2f} {pv:8.2g}   {ob:+.3f} ({ot:+.2f})")


def side_gap(x):
    """同一個吐露來源下，子代側 − 親代側 的係數差：疊兩個評分、各自 dyad×側 固定效果、側別專屬波次效果。"""
    rows = []
    for side, y in (("k", "aff_k"), ("p", "aff_p")):
        s = long[[x, y, "wave", "hh", "dyad_id"]].dropna().rename(columns={y: "aff"})
        rows.append(s.assign(ent=s.dyad_id + "|" + side, kside=float(side == "k")))
    s = pd.concat(rows, ignore_index=True)
    s["x_k"] = s[x] * s.kside
    for w in WAVES[1:]:
        s[f"w{w}_k"] = ((s.wave == w) & (s.kside == 1)).astype(float)
    s = s.set_index(["ent", "wave"])
    exog = [x, "x_k"] + [f"w{w}_k" for w in WAVES[1:]]
    r = PanelOLS(s.aff, s[exog], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", clusters=s[["hh"]])
    return r.params["x_k"], r.std_errors["x_k"], r.tstats["x_k"], r.pvalues["x_k"]


print("\n【2】側別差檢定（子代側係數 − 親代側係數，負＝子代反應較大）")
for x, nm in (("k2p_k", "子報吐露"), ("k2p_p", "親報吐露")):
    d, se, t, pv = side_gap(x)
    print(f"  用{nm}：差 {d:+.3f} (SE {se:.3f}, t={t:+.2f}, p={pv:.2g})")

kk, kp, pk, pp = (res[c] for c in [("k2p_k", "aff_k"), ("k2p_k", "aff_p"), ("k2p_p", "aff_k"), ("k2p_p", "aff_p")])
print("\n【3】判準：交叉報告兩格——親報吐露→子代評分 顯著、子報吐露→親代評分 不顯著")
print(f"  親報→子代 t={pk[1]:+.2f} p={pk[2]:.2g}｜子報→親代 t={kp[1]:+.2f} p={kp[2]:.2g}")
ok = pk[2] < .05 and kp[2] >= .05
print("  ⇒ " + ("通過：子代側在換報告人後仍在動、親代側仍不動，主軸不是同報告人偏誤" if ok
               else "未通過：主軸要重新評估，見上表哪一格翻了"))

# 交叉格不顯著有兩種解釋：①同報告人偏誤 ②親報吐露的「變化」根本沒對到子報的變化（衰減到測不出）
W = long[NUM] - long.groupby("dyad_id")[NUM].transform("mean")
cs = [long[long.wave == w][["k2p_k", "k2p_p"]].corr().iloc[0, 1] for w in WAVES]
rw = W[["k2p_k", "k2p_p"]].corr().iloc[0, 1]
print("\n【4】兩個報告人對吐露的一致度")
print(f"  橫斷面 r(子報, 親報)：" + "／".join(f"{w} {c:.3f}" for w, c in zip(WAVES, cs)))
print(f"  組內變化 r(Δ子報, Δ親報)：{rw:.3f}｜SD {W.k2p_k.std():.3f} vs {W.k2p_p.std():.3f}")
print(f"  若真效果是 {kk[0]:+.3f} 且只透過兩邊共享的變化傳遞，交叉格預期約 "
      f"{kk[0] * rw * W.k2p_k.std() / W.k2p_p.std():+.3f} ⇒ 交叉格檢定力不足，不能當反證也不能當正證")

