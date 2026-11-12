# -*- coding: utf-8 -*-
"""PSFD 代間溫差提案 · 一次印出報告裡引用的每一個數字（含穩健性與反證）。
跑法：cd research && uv run --with pandas --with statsmodels python psfd_core.py
"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from psfd_engine import load, build_dyads, header

d = load()
dy, par = build_dyads(d)
header(dy)
rng = np.random.default_rng(20260913)


def fit(f, data=dy):
    cols = sorted({c for c in dy.columns if c in f.replace("(", " ").replace(")", " ").replace("+", " ").replace("~", " ").split()})
    s = data[cols + ["hh"]].dropna()
    return smf.ols(f, data=s).fit(cov_type="cluster", cov_kwds={"groups": s.hh}), s


def bt(r, x):
    return f"b={r.params[x]:+.3f} (t={r.tvalues[x]:+.2f}, p={r.pvalues[x]:.2g})"


# ── 樣本 ─────────────────────────────────────────────────────────────────
n_kid = (d.x01a == 3).sum()
print("【樣本】")
print(f"  原始 {len(d)} 人；主樣本 {(d.x01a==1).sum()}／子女樣本 {n_kid}")
print(f"  配上的 dyad {len(dy)} 組（{dy.hh.nunique()} 戶）；子女樣本有親代在樣本內但配不上：",
      d[(d.x01a == 3) & d.x01d.isin(par.index)].shape[0] - len(dy))
print(f"  親代：父 {int(dy.father.sum())}／母 {int((1-dy.father).sum())}，年齡 {dy.p_age.mean():.1f}±{dy.p_age.std():.1f}；"
      f"子代：兒 {int(dy.son.sum())}／女 {int((1-dy.son).sum())}，年齡 {dy.k_age.mean():.1f}±{dy.k_age.std():.1f}")
print(f"  每戶 dyad 數：{dy.groupby('hh').size().value_counts().sort_index().to_dict()}\n")

# ── 發現 1：客觀題對得上、主觀題對不上 ─────────────────────────────────────
print("【發現1】同一題兩代各答一次（配對 t 檢定）")
LAB = dict(dis="居住距離", met="見面頻率", tel="通話頻率", k2p="子→親說心事", p2k="親→子說心事", aff="感情好不好")
f1 = []
for v in ["dis", "met", "tel", "k2p", "p2k", "aff"]:
    s = dy[[f"{v}_p", f"{v}_k"]].dropna()
    t = stats.ttest_rel(s.iloc[:, 0], s.iloc[:, 1])
    agree = (s.iloc[:, 0] == s.iloc[:, 1]).mean()
    r = s.corr().iloc[0, 1]
    wk = stats.wilcoxon(s.iloc[:, 0], s.iloc[:, 1]).pvalue
    f1.append((v, len(s), agree, r))
    print(f"  {LAB[v]:6s} n={len(s):4d} 親{s.iloc[:,0].mean():.3f} 子{s.iloc[:,1].mean():.3f} 差{t.statistic and s.iloc[:,0].mean()-s.iloc[:,1].mean():+.3f}"
          f" t={t.statistic:+6.2f} p={t.pvalue:.2g}（Wilcoxon p={wk:.2g}）一致率{agree*100:5.1f}% r={r:.3f}")
g = dy.gap.dropna()
print(f"  溫差分布 n={len(g)}：高估 {(g>0).mean()*100:.1f}%／一致 {(g==0).mean()*100:.1f}%／低估 {(g<0).mean()*100:.1f}%；"
      f"高估≥2級 {(g>=2).mean()*100:.1f}%；分布 {g.value_counts().sort_index().to_dict()}")
r, s = fit("gap ~ 1")
print(f"  溫差平均（戶層叢集 SE）{bt(r, 'Intercept')}\n")

# ── 發現 2：親代性別 ──────────────────────────────────────────────────────
print("【發現2】溫差 × 親代性別 × 子代性別")
for fa, fn in [(1, "父親"), (0, "母親")]:
    cells = []
    for so, sn in [(1, "兒子"), (0, "女兒")]:
        x = dy[(dy.father == fa) & (dy.son == so)].gap.dropna()
        cells.append(f"{sn} {x.mean():+.3f} (SE {x.sem():.3f}, n={len(x)})")
    sub = dy[dy.father == fa]
    print(f"  {fn}：{'｜'.join(cells)}｜親代自評 {sub.aff_p.mean():.3f}、子代評此親 {sub.aff_k.mean():.3f}")
r, _ = fit("gap ~ father * son")
print(f"  gap ~ 父 × 兒：父 {bt(r,'father')}；兒 {bt(r,'son')}；交互 {bt(r,'father:son')}")
for y, nm in [("aff_p", "親代自評"), ("aff_k", "子代評分")]:
    r, _ = fit(f"{y} ~ father")
    print(f"  {nm} 父−母：{bt(r,'father')}")
print()

# ── 發現 3：APIM ─────────────────────────────────────────────────────────
print("【發現3】APIM：各自的幸福由誰的版本決定（戶層叢集 SE）")
apim, pp = [], []
for y, nm in [("p_happy", "親代快樂"), ("p_lonely", "親代孤獨"), ("p_distress", "親代困擾"),
              ("k_happy", "子代快樂"), ("k_lonely", "子代孤獨"), ("k_distress", "子代困擾")]:
    act, prt = ("aff_p", "aff_k") if y.startswith("p_") else ("aff_k", "aff_p")
    r, s = fit(f"{y} ~ aff_p + aff_k")
    rc, sc = fit(f"{y} ~ aff_p + aff_k + father + son + p_age + k_age")
    apim.append((nm, len(s), r.params[act], r.tvalues[act], r.params[prt], r.tvalues[prt], r.pvalues[prt]))
    pp.append(r.pvalues[prt])
    print(f"  {nm} n={len(s)} actor {bt(r,act)}｜partner {bt(r,prt)}｜加控制後 actor t={rc.tvalues[act]:+.2f} partner t={rc.tvalues[prt]:+.2f}")
q = multipletests(pp, method="fdr_bh")[1]
print(f"  partner 效果 FDR-BH q：{np.round(q, 3).tolist()}")
for nm, _, a, ta, p, tp, _ in apim:
    print(f"  {nm}: |actor|/|partner| = {abs(a)/max(abs(p),1e-9):.1f}")
print()

# ── 發現 4：機制 ─────────────────────────────────────────────────────────
print("【發現4】溫差的來源：接觸量 vs 吐露")
r, s = fit("gap ~ father + p_age + cohab")
print(f"  全樣本 n={len(s)}：父 {bt(r,'father')}；年齡 {bt(r,'p_age')}；同住 {bt(r,'cohab')}；R²={r.rsquared:.3f}")
r, s = fit("gap ~ father + p_age + met_k + tel_k", dy[dy.cohab == 0])
print(f"  非同住 n={len(s)}：父 {bt(r,'father')}；見面 {bt(r,'met_k')}；通話 {bt(r,'tel_k')}；R²={r.rsquared:.3f}")
r, s = fit("gap ~ father + p_age + met_k + tel_k + k2p_k", dy[dy.cohab == 0])
print(f"  非同住＋吐露 n={len(s)}：吐露(1總是→5從未) {bt(r,'k2p_k')}；見面 {bt(r,'met_k')}；通話 {bt(r,'tel_k')}；R²={r.rsquared:.3f}")
TALK = {1: "總是", 2: "經常", 3: "有時", 4: "偶爾", 5: "從未"}
tab = dy.groupby("k2p_k").gap.agg(["size", "mean", "sem"])
for k, row in tab.iterrows():
    print(f"  子代跟{('親代')}說心事＝{TALK[int(k)]}：n={int(row['size']):4d} 溫差 {row['mean']:+.3f} (SE {row['sem']:.3f})"
          f"｜親代自評 {dy[dy.k2p_k==k].aff_p.mean():.2f} 子代 {dy[dy.k2p_k==k].aff_k.mean():.2f}")
print(f"  從未說心事佔 {(dy.k2p_k==5).mean()/dy.k2p_k.notna().mean()*100:.1f}%；"
      f"父親 {(dy[dy.father==1].k2p_k==5).mean()*100:.1f}% vs 母親 {(dy[dy.father==0].k2p_k==5).mean()*100:.1f}%")

m = dy[["father", "k2p_k", "gap", "hh"]].dropna().reset_index(drop=True)


def med(s):
    X = np.column_stack([np.ones(len(s)), s.father])
    a = np.linalg.lstsq(X, s.k2p_k, rcond=None)[0][1]
    c = np.linalg.lstsq(X, s.gap, rcond=None)[0][1]
    bc = np.linalg.lstsq(np.column_stack([X, s.k2p_k]), s.gap, rcond=None)[0]
    return a, bc[2], c, bc[1]


a, b, c, cp = med(m)
ra, _ = fit("k2p_k ~ father", m)
rb, _ = fit("gap ~ father + k2p_k", m)
rc, _ = fit("gap ~ father", m)
hhs = m.hh.unique()
idx = {h: np.flatnonzero(m.hh.values == h) for h in hhs}
boot = []
for _ in range(2000):
    pick = rng.choice(hhs, len(hhs))
    s = m.iloc[np.concatenate([idx[h] for h in pick])]
    aa, bb, _, _ = med(s)
    boot.append(aa * bb)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"  中介 n={len(m)}：a 父→少吐露 {bt(ra,'father')}；b 少吐露→溫差 {bt(rb,'k2p_k')}")
print(f"        c 總效果 {bt(rc,'father')}；c' 直接 {bt(rb,'father')}")
print(f"        間接 a×b={a*b:+.4f}，佔 {a*b/c*100:.1f}%；戶層叢集 bootstrap 2000 次 95% CI [{lo:+.3f}, {hi:+.3f}]\n")

# ── 發現 5：年齡 ─────────────────────────────────────────────────────────
print("【發現5】溫差 × 親代年齡／世代群")
bins = pd.cut(dy.p_age, [0, 50, 60, 70, 120], right=False, labels=["<50", "50–59", "60–69", "70+"])
for k, x in dy.groupby(bins, observed=True).gap:
    print(f"  {k:6s} n={x.notna().sum():4d} 溫差 {x.mean():+.3f} (SE {x.sem():.3f})")
r, s = fit("gap ~ p_age + father")
print(f"  gap ~ 年齡 + 父：年齡每 10 歲 {r.params['p_age']*10:+.3f}（t={r.tvalues['p_age']:+.2f}）")
print(f"  （年齡與世代在單一橫斷面完全共線，無法拆開）\n")

# ── 穩健性 ───────────────────────────────────────────────────────────────
print("【穩健1】客觀題安慰劑：把 X 換成兩代一致的客觀題，own-report 優勢應消失")
for y, nm in [("p_happy", "親代快樂"), ("k_happy", "子代快樂")]:
    for v in ["aff", "met", "dis"]:
        r, s = fit(f"{y} ~ {v}_p + {v}_k")
        print(f"  {nm} ← {LAB[v]:6s} n={len(s)} 親代版 {bt(r,f'{v}_p')}｜子代版 {bt(r,f'{v}_k')}")
print()

print("【穩健2】手足安慰劑：親代對『其他子女』的評分能預測這個孩子的快樂嗎？")
r, s = fit("k_happy ~ aff_k + aff_p + aff_p_sib")
print(f"  n={len(s)}（{s.hh.nunique()} 戶）子代版 {bt(r,'aff_k')}｜親代對此子 {bt(r,'aff_p')}｜親代對手足 {bt(r,'aff_p_sib')}\n")

print("【穩健3】親代固定效果：戶內去中心後兩代評分的相關")
multi = dy[dy.groupby("hh").hh.transform("size") >= 2][["hh", "aff_p", "aff_k"]].dropna()
dm = multi[["aff_p", "aff_k"]] - multi.groupby("hh")[["aff_p", "aff_k"]].transform("mean")
print(f"  多子女戶 n={len(multi)}（{multi.hh.nunique()} 戶）原始 r={multi.aff_p.corr(multi.aff_k):.3f}；戶內 r={dm.aff_p.corr(dm.aff_k):.3f}")
print(f"  戶內親代評分的變異佔比 {dm.aff_p.var()/multi.aff_p.var()*100:.1f}%（親代會分辨孩子，不是一律給同分）\n")

print("【穩健4】FDR：見發現3 partner q 值\n")

print("【穩健5】感情二元化（≥4＝好）重跑 APIM")
dy["affb_p"], dy["affb_k"] = (dy.aff_p >= 4).astype(float).where(dy.aff_p.notna()), (dy.aff_k >= 4).astype(float).where(dy.aff_k.notna())
for y, nm in [("p_happy", "親代快樂"), ("k_happy", "子代快樂")]:
    r, _ = fit(f"{y} ~ affb_p + affb_k")
    print(f"  {nm}：親代版 {bt(r,'affb_p')}｜子代版 {bt(r,'affb_k')}")
print()

print("【穩健6】選樣偏誤：配上的子女 vs 同一批親代其他沒受訪的子女（親代版感情）")
m_slots = set(zip(dy.hh, dy.slot))
rows = [(hh, k, par.at[hh, f"g28c{k}"], (hh, k) in m_slots, par.at[hh, f"g32c{k}"])
        for hh in dy.hh.unique() for k in range(1, 7) if pd.notna(par.at[hh, f"g28c{k}"])]
sl = pd.DataFrame(rows, columns=["hh", "k", "aff", "matched", "dis"])
x1, x0 = sl[sl.matched].aff, sl[~sl.matched].aff
print(f"  配上 n={len(x1)} 平均 {x1.mean():.3f}｜同戶沒受訪 n={len(x0)} 平均 {x0.mean():.3f}｜差 {x1.mean()-x0.mean():+.3f}"
      f"（t={stats.ttest_ind(x1, x0).statistic:+.2f}）；同住率 配上 {(sl[sl.matched].dis==8).mean()*100:.1f}% vs 沒受訪 {(sl[~sl.matched].dis==8).mean()*100:.1f}%")

print("\n【穩健7】訪問模式：溫差是否因子代自填/電訪而不同（社會期許）")
kmode = d.set_index("x01").loc[dy.kid, "x01f"].values
dy["k_mode"] = kmode
dy["p_mode"] = par.loc[dy.hh, "x01f"].values
print("  子代 " + "｜".join(f"模式{int(k)} n={len(x)} {x.mean():+.3f}" for k, x in dy.groupby("k_mode").gap))
print("  親代 " + "｜".join(f"模式{int(k)} n={len(x)} {x.mean():+.3f}" for k, x in dy.groupby("p_mode").gap))
dy["k_web"] = (dy.k_mode == 4).astype(float)
for y, nm in [("gap", "溫差"), ("aff_k", "子代評分"), ("aff_p", "親代評分")]:
    r, s = fit(f"{y} ~ k_web + father + son + p_age + k_age", dy[dy.k_mode.isin([2, 4])])
    print(f"  {nm} ~ 子代網路自填(vs 面訪錄音)＋控制 n={len(s)}：{bt(r,'k_web')}")
r, s = fit("gap ~ father + k_web + p_age + k_age", dy[dy.k_mode.isin([2, 4])])
print(f"  控制自填後父親效果仍在：{bt(r,'father')}")
print("  （1=面訪 2=面訪錄音 3=視訊 4=網路自填 5=電訪 6=現場自填）")
