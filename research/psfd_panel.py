# -*- coding: utf-8 -*-
"""PSFD 代間溫差 · 三波追蹤（RR2020 × RR2022 × RR2024）：橫斷面的結論在時間上站不站得住。
兩波只能說「擴大」；三波才能分辨「單調趨勢」與「2024 的期間效果」。
跑法：cd research && uv run --with pandas --with pyreadstat --with statsmodels --with scipy python psfd_panel.py
"""
import re
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from psfd_engine import load, build_dyads, build_panel, header, V

WAVES = (2020, 2022, 2024)
S = [w % 100 for w in WAVES]
ITEM = dict(aff="感情", dis="距離", tel="通話", met="見面", p2k="親→子心事", k2p="子→親心事")

D = {w: load(w) for w in WAVES}
DY = {w: build_dyads(D[w])[0] for w in WAVES}
p = build_panel(WAVES)
assert (p.hh_20 == p.hh_24).all() and (p.hh_22 == p.hh_24).all(), "戶號應跨波穩定"
p["hh"] = p.hh_24
p["father"], p["son"] = p.father_20, p.son_20
header(p, "PSFD RR2020×RR2022×RR2024 代間溫差 · 三波追蹤")


def fit(f, data=None):
    data = p if data is None else data
    cols = {c for c in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f) if c in data.columns}
    s = data[sorted(cols | {"hh"})].dropna()
    return smf.ols(f, data=s).fit(cov_type="cluster", cov_kwds={"groups": s.hh}), s


def bt(r, x):
    return f"b={r.params[x]:+.3f} (SE {r.bse[x]:.3f}, t={r.tvalues[x]:+.2f}, p={r.pvalues[x]:.2g})"


# ── 樣本流 ───────────────────────────────────────────────────────────────
print("【樣本流】三波怎麼串起來")
for w in WAVES:
    k = D[w][D[w].x01a == 3]
    print(f"  {w}：受訪 {len(D[w])} 人（主樣本 {(D[w].x01a==1).sum()}／子女 {len(k)}）"
          f"｜dyad {len(DY[w])} 組（{DY[w].hh.nunique()} 戶）｜子女樣本出生年上限 "
          f"{int((k[V[w]['byear']] + 1911).max())}")
for a, b in [(2020, 2022), (2022, 2024), (2020, 2024)]:
    print(f"  個人串接 {a}∩{b}：{len(set(D[a].pid) & set(D[b].pid))} 人")
print(f"  個人三波都在：{len(set(D[2020].pid) & set(D[2022].pid) & set(D[2024].pid))} 人")
ids = {w: set(DY[w].dyad_id) for w in WAVES}
print(f"  dyad 三波都配得上：{len(p)} 組（{p.hh.nunique()} 戶）"
      f"｜僅 22∩24：{len(ids[2022] & ids[2024])}｜僅 20∩22：{len(ids[2020] & ids[2022])}")
p2 = build_panel((2022, 2024))
g2 = p2[["gap_22", "gap_24"]].dropna()
print(f"  ＊對照：22∩24 兩波樣本（{len(p2)} 組）的溫差 {g2.gap_22.mean():+.3f} → {g2.gap_24.mean():+.3f}"
      f" Δ={g2.gap_24.mean()-g2.gap_22.mean():+.3f}（下面三波樣本是它的子集，看有沒有被選擇性縮小）")

print("\n【0】發現1 的三波複製：每一波各自做一次兩代配對比較（橫斷面，各波全樣本）")
print("     要看的是「客觀題兩代對得上、主觀題對不上」這個對照在三波都成立")
for w in WAVES:
    dy = DY[w]
    print(f"  {w}（{len(dy)} 組／{dy.hh.nunique()} 戶）")
    for v in ["dis", "met", "k2p", "aff"]:
        s = dy[[f"{v}_p", f"{v}_k"]].dropna()
        t = stats.ttest_rel(s.iloc[:, 0], s.iloc[:, 1])
        print(f"    {ITEM[v]:8s} n={len(s):4d} 親{s.iloc[:,0].mean():.3f} 子{s.iloc[:,1].mean():.3f}"
              f" 差{s.iloc[:,0].mean()-s.iloc[:,1].mean():+.3f} (t={t.statistic:+6.2f}, p={t.pvalue:.2g})"
              f" 一致率{(s.iloc[:,0]==s.iloc[:,1]).mean()*100:5.1f}% r={s.iloc[:,0].corr(s.iloc[:,1]):.3f}")

print("\n【反證0】三波都留下 vs 中途掉出：以 2020 為基期比較，檢查存活樣本有無選擇性")
base = DY[2020].set_index("dyad_id")
base["stay"] = base.index.isin(p.index)
for v, nm in [("gap", "基期溫差"), ("aff_p", "親代評分"), ("aff_k", "子代評分"), ("father", "父親比例"),
              ("p_age", "親代年齡"), ("k_age", "子代年齡"), ("cohab", "同住率"), ("p_happy", "親代快樂")]:
    x1, x0 = base[base.stay][v].dropna(), base[~base.stay][v].dropna()
    t = stats.ttest_ind(x1, x0)
    print(f"  {nm:6s} 留下 {x1.mean():+.3f} (n={len(x1)})｜掉出 {x0.mean():+.3f} (n={len(x0)})"
          f"｜差 {x1.mean()-x0.mean():+.3f} t={t.statistic:+.2f} p={t.pvalue:.2g}")

# ── 1 四年軌跡 ───────────────────────────────────────────────────────────
print("\n【1】同一批親子的四年軌跡（n 隨題目遺漏略異；單位＝級）")
for v, nm in [("gap", "溫差"), ("aff_p", "親代評分"), ("aff_k", "子代評分"),
              ("k2p_k", "子→親吐露(子代版)"), ("met_k", "見面(子代版)"), ("cohab", "同住率"),
              ("p_happy", "親代快樂"), ("k_happy", "子代快樂"), ("p_lonely", "親代孤獨"), ("k_lonely", "子代孤獨")]:
    x = p[[f"{v}_{s:02d}" for s in S]].dropna()
    m = x.mean()
    print(f"  {nm:14s} n={len(x):4d} " + " → ".join(f"{m.iloc[i]:+.3f}" for i in range(3))
          + f"｜Δ1(20→22) {m.iloc[1]-m.iloc[0]:+.3f}｜Δ2(22→24) {m.iloc[2]-m.iloc[1]:+.3f}"
          + f"｜四年 {m.iloc[2]-m.iloc[0]:+.3f}")

print("\n【2】是單調趨勢，還是 2024 的期間效果？（同一批 dyad，配對檢定）")
g = p[[f"gap_{s:02d}" for s in S]].dropna()
d1, d2 = g.iloc[:, 1] - g.iloc[:, 0], g.iloc[:, 2] - g.iloc[:, 1]
for nm, x in [("Δ1 20→22", d1), ("Δ2 22→24", d2), ("四年 20→24", g.iloc[:, 2] - g.iloc[:, 0])]:
    t = stats.ttest_1samp(x, 0)
    print(f"  {nm:12s} {x.mean():+.4f} (SE {x.sem():.4f}, t={t.statistic:+.2f}, p={t.pvalue:.2g}, n={len(x)})")
t = stats.ttest_rel(d2, d1)
print(f"  Δ2 − Δ1 = {(d2-d1).mean():+.4f} (SE {(d2-d1).sem():.4f}, t={t.statistic:+.2f}, p={t.pvalue:.2g})"
      f"  ⇒ {'兩段加速度不同＝不是等速趨勢' if t.pvalue < .05 else '兩段變化幅度無顯著差異'}")
r, s = fit("gap ~ year", pd.DataFrame({"gap": g.values.ravel(), "year": np.tile([2020, 2022, 2024], len(g)),
                                       "hh": np.repeat(p.loc[g.index, "hh"].values, 3)}))
print(f"  線性趨勢（長格式、戶層叢集）每 2 年 {r.params['year']*2:+.4f}（t={r.tvalues['year']:+.2f}）")

print("\n【3】穩定性：相鄰兩波 vs 相隔四年（衰減＝真的在變，不只是量表雜訊）")
for v, nm in [("gap", "溫差"), ("aff_p", "親代評分"), ("aff_k", "子代評分"), ("dis_k", "距離"), ("k2p_k", "吐露")]:
    rs = {}
    for a, b in [(20, 22), (22, 24), (20, 24)]:
        x = p[[f"{v}_{a:02d}", f"{v}_{b:02d}"]].dropna()
        rs[(a, b)] = x.iloc[:, 0].corr(x.iloc[:, 1])
    mk = rs[(20, 22)] * rs[(22, 24)]
    print(f"  {nm:6s} r(20,22)={rs[(20,22)]:.3f} r(22,24)={rs[(22,24)]:.3f} r(20,24)={rs[(20,24)]:.3f}"
          f"｜馬可夫預測 {mk:.3f}（實際{'高於' if rs[(20,24)] > mk else '低於'}預測"
          f" {rs[(20,24)]-mk:+.3f}＝{'有穩定的個體成分' if rs[(20,24)] > mk else '近似一階馬可夫'}）")

# ── 4 長格式固定效果 ─────────────────────────────────────────────────────
NUM = ["gap", "aff_p", "aff_k", "k2p_k", "p2k_k", "met_k", "tel_k", "cohab", "p_happy", "k_happy", "p_lonely", "k_lonely"]
long = pd.concat([p[[f"{v}_{s:02d}" for v in NUM]].rename(columns=lambda c: c[:-3])
                  .assign(wave=2000 + s, hh=p.hh, dyad_id=p.index, father=p.father) for s in S], ignore_index=True)
w_in = long[NUM] - long.groupby("dyad_id")[NUM].transform("mean")   # 個體去中心
fe = long.assign(**{v: w_in[v] for v in NUM})

print("\n【4】個體固定效果（三波長格式 n=%d 觀測／%d 組；個體去中心＋波次 FE，戶層叢集 SE）" % (len(long), len(p)))
print("     解讀：同一對親子自己前後比較，任何不變的東西（脾氣、家風、世代）都被吃掉了")
r, s = fit("gap ~ k2p_k + C(wave)", fe)
print(f"  Δ吐露→Δ溫差 n={len(s)}：{bt(r,'k2p_k')}；R²={r.rsquared:.3f}")
r, s = fit("gap ~ k2p_k + met_k + tel_k + C(wave)", fe[fe.cohab.notna()])
print(f"  ＋接觸量（非同住者）n={len(s)}：吐露 {bt(r,'k2p_k')}；見面 {bt(r,'met_k')}；通話 {bt(r,'tel_k')}")
for y, nm in [("aff_p", "親代評分"), ("aff_k", "子代評分")]:
    r, s = fit(f"{y} ~ k2p_k + C(wave)", fe)
    print(f"  拆哪一邊在動：Δ吐露→Δ{nm} n={len(s)}：{bt(r,'k2p_k')}")
r, s = fit("gap ~ k2p_k + C(wave)", long)
print(f"  ＊對照 pooled（沒去中心）n={len(s)}：{bt(r,'k2p_k')}（差額＝跨個體的組間成分）")
print("  ※ 去中心沒扣掉吃掉的自由度，t 略偏大；方向與量級可信，臨界顯著別當定論")

# ── 5 交叉延遲兩段複製 ───────────────────────────────────────────────────
print("\n【5】交叉延遲：溫差先動還是福祉先動（控制自身基期）；兩段各跑一次＝內部複製")
YS = [("p_happy", "親代快樂"), ("p_lonely", "親代孤獨"), ("k_happy", "子代快樂"), ("k_lonely", "子代孤獨")]
pv = []
for y, nm in YS:
    out = []
    for a, b in [(20, 22), (22, 24)]:
        r, s = fit(f"{y}_{b} ~ {y}_{a} + gap_{a}")
        rb, _ = fit(f"gap_{b} ~ gap_{a} + {y}_{a}")
        out.append(f"{a}→{b} n={len(s)} 溫差→福祉 {bt(r,f'gap_{a}')}｜福祉→溫差 {bt(rb,f'{y}_{a}')}")
        pv.append(r.pvalues[f"gap_{a}"])
    print(f"  {nm}\n    " + "\n    ".join(out))
q = multipletests(pv, method="fdr_bh")[1]
print(f"  溫差→福祉 8 條檢定 FDR-BH q：{np.round(q, 3).tolist()}")
print("  （困擾 distress 不列：2020 的 h03 問「平時」且缺 4 題，與 m02 不可比，見 SPEC）")
r, s = fit("p_distress_24 ~ p_distress_22 + gap_22", p2.assign(hh=p2.hh_24))
print(f"  ＊困擾只能用 22→24 兩波樣本 n={len(s)}：溫差22→親代困擾24 {bt(r,'gap_22')}")

# ── 6 父親效果三波複製 ──────────────────────────────────────────────────
print("\n【6】父親效果：橫斷面的招牌結論，三波各自複製一次")
for s_ in S:
    r, ss = fit(f"gap_{s_:02d} ~ father + son + p_age_{s_:02d}")
    print(f"  20{s_} n={len(ss)}：父親 {bt(r,'father')}｜兒子 {bt(r,'son')}")
for a, b in [(20, 22), (22, 24), (20, 24)]:
    r, ss = fit(f"I(gap_{b} - gap_{a}) ~ father + son")
    print(f"  Δ溫差 {a}→{b} ~ 父親 n={len(ss)}：{bt(r,'father')}")
print("  ＊各波全樣本對照（不限三波交集）：分辨『效果不存在』與『交集樣本檢定力不足』")
for w in WAVES:
    r, ss = fit("gap ~ father + son + p_age", DY[w])
    print(f"    {w} 全樣本 n={len(ss)}（{DY[w].hh.nunique()} 戶）：父親 {bt(r,'father')}"
          f"｜平均溫差 {DY[w].gap.mean():+.3f}")
print(f"    父親比例：三波交集 {p.father.mean():.3f}／2020 全樣本 {DY[2020].father.mean():.3f}"
      f"／2024 全樣本 {DY[2024].father.mean():.3f}"
      f" ⇒ 交集樣本父親偏少，而父親是高溫差側，故三波樣本會低估父親效果")

print("\n【7】年齡 vs 世代：橫斷面拆不開，四年追蹤可以")
r, _ = fit("gap_24 ~ p_age_24 + father")
cs = r.params["p_age_24"] * 4
dg = (p.gap_24 - p.gap_20).dropna()
print(f"  橫斷面斜率 {r.params['p_age_24']:+.4f}/歲 → 隱含 4 年變化 {cs:+.4f}")
print(f"  同一批人實際 4 年變化 {dg.mean():+.4f} (SE {dg.sem():.4f}, n={len(dg)})")
print(f"  差 {dg.mean()-cs:+.4f} ＝ 橫斷面年齡斜率裡的世代／期間成分")

# ── 反證 ─────────────────────────────────────────────────────────────────
print("\n【反證1】均值回歸：Δ 對基期回歸必然為負，不可當發現")
for v, nm in [("gap", "溫差"), ("aff_p", "親代評分"), ("dis_k", "距離")]:
    r, s = fit(f"I({v}_24 - {v}_20) ~ {v}_20")
    print(f"  Δ{nm}(四年) ~ 基期 n={len(s)}：{bt(r,f'{v}_20')}")

print("\n【反證2】客觀題安慰劑：客觀題的兩代落差不該隨時間系統性漂移（感情是焦點題，應該只有它動）")
for v in ["dis", "met", "tel", "p2k", "k2p", "aff"]:
    row = []
    for a, b in [(20, 22), (22, 24)]:
        x = ((p[f"{v}_p_{b}"] - p[f"{v}_k_{b}"]) - (p[f"{v}_p_{a}"] - p[f"{v}_k_{a}"])).dropna()
        row.append(f"{a}→{b} n={len(x):4d} {x.mean():+.4f} (SE {x.sem():.4f}, t={x.mean()/x.sem():+.2f})")
    print(f"  Δ(親−子) {ITEM[v]:8s} " + "｜".join(row))

print("\n【反證3】訪問模式：改用網路自填會不會製造假的溫差變化")
for a, b in [(20, 22), (22, 24)]:
    chg = (p[f"k_mode_{b}"] == 4).astype(float) - (p[f"k_mode_{a}"] == 4).astype(float)
    r, s = fit(f"I(gap_{b} - gap_{a}) ~ chg + father", p.assign(chg=chg))
    print(f"  Δ溫差 {a}→{b} ~ 子代改網路自填 n={len(s)}：{bt(r,'chg')}")

print("\n【反證4】衰減上限：gap 是兩個有誤差的題相減，效果量小是結構性的")
for v in ["aff_p", "aff_k", "gap"]:
    x = p[[f"{v}_20", f"{v}_24"]].dropna()
    print(f"  {v:6s} 四年 r={x.iloc[:,0].corr(x.iloc[:,1]):.3f}  SD(Δ)={(x.iloc[:,1]-x.iloc[:,0]).std():.3f}"
          f"  SD(基期)={x.iloc[:,0].std():.3f}")
print("=" * 78)
