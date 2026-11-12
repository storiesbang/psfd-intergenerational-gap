# -*- coding: utf-8 -*-
"""初賽作品_v2.docx 內『現有輸出檔查不到出處』的數字，逐一重算。
其餘數字的出處：core_output.txt / panel_output.txt / factors_output.txt / b3_output.txt。
跑法：cd research && uv run --with pandas --with pyreadstat --with statsmodels --with scipy python psfd_v2_verify.py | tee out/psfd/v2_verify_output.txt
"""
import warnings; warnings.filterwarnings("ignore")
import itertools
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from psfd_engine import load, build_dyads, build_panel, header, V

WAVES = (2020, 2022, 2024)
D = {w: load(w) for w in WAVES}
dy, par = build_dyads(D[2024])
header(dy, "初賽作品 v2 · 無出處數字重算")


def fit(f, data, cols):
    s = data[cols + ["hh"]].dropna()
    return smf.ols(f, data=s).fit(cov_type="cluster", cov_kwds={"groups": s.hh}), s


def bt(r, x):
    return f"b={r.params[x]:+.3f} (t={r.tvalues[x]:+.2f}, p={r.pvalues[x]:.3f})"


# ── 2. 樣本描述（RR2024）：997 全部 vs 991 兩端感情皆有 ─────────────────
print("【2 分析樣本】")
P = par.loc[dy.hh].reset_index(drop=True)
K = D[2024].set_index("x01").loc[dy.kid].reset_index(drop=True)
dy = dy.reset_index(drop=True)
dy["p_univ"] = (P.b01.between(12, 15)).astype(float).where(P.b01.notna() & (P.b01 != 97))
dy["k_univ"] = (K.b01.between(12, 15)).astype(float).where(K.b01.notna() & (K.b01 != 97))
for tag, s in (("997 全部", dy), ("991 感情兩端皆有", dy[dy.aff_p.notna() & dy.aff_k.notna()])):
    hh = s.groupby("hh").size().value_counts().sort_index().to_dict()
    print(f"  [{tag}] n={len(s)} 戶={s.hh.nunique()} 每戶對數={hh}")
    print(f"    親代年齡 {s.p_age.mean():.1f} (SD {s.p_age.std():.1f}, {s.p_age.min():.0f}–{s.p_age.max():.0f})"
          f"｜父親 {s.father.mean()*100:.1f}%｜子代年齡 {s.k_age.mean():.1f} (SD {s.k_age.std():.1f}, "
          f"{s.k_age.min():.0f}–{s.k_age.max():.0f})｜兒子 {s.son.mean()*100:.1f}%")
    print(f"    同住 {s.cohab.mean()*100:.1f}%（有效 {s.cohab.notna().sum()}）｜大學以上（b01 12–15）"
          f" 親代 {s.p_univ.mean()*100:.1f}%／子代 {s.k_univ.mean()*100:.1f}%")
pp = dy.drop_duplicates("hh")
print(f"  親代以『人』計（{len(pp)} 人）：大學以上 {pp.p_univ.mean()*100:.1f}%｜父親 {pp.father.mean()*100:.1f}%")

# ── 2. 配對演算法：同戶同性別同出生年的子女欄位 ─────────────────────────
print("\n【2 配對】同戶親代子女欄位中 (性別, 出生年) 重複者")
for w in WAVES:
    c = V[w]; d = D[w]; pr = d[d.x01a == 1]
    kids = d[d.x01a == 3]
    n_grp = n_mon = n_hit = 0
    for _, r in pr.iterrows():
        sl = [(r[c["kid_sex"].format(k=k)], r[c["kid_byear"].format(k=k)], k) for k in range(1, 7)]
        sl = [x for x in sl if pd.notna(x[0]) and pd.notna(x[1])]
        for key, g in itertools.groupby(sorted(sl), key=lambda x: x[:2]):
            g = list(g)
            if len(g) < 2:
                continue
            n_grp += 1
            mcol = "g08z02c{k}" if w != 2020 else None
            if mcol:
                ms = [r.get(mcol.format(k=k)) for *_, k in g]
                n_mon += len(set(ms)) == 1 and pd.notna(ms[0])
            byc = c["byear"]
            n_hit += ((kids.x01d == r.x01d) & (kids.a01 == key[0]) & (kids[byc] == key[1])).any()
    print(f"  {w}：重複組 {n_grp}（出生月亦相同 {n_mon if w != 2020 else 'n/a'}）｜其中有子女受訪者 {n_hit}")
print("  ※ engine 的配對規則：同 (性別, 出生年) 依欄位順序取第一個未用者——**未使用出生月**（psfd_engine.py build_dyads）")

# ── 2. 跨波串接 ──────────────────────────────────────────────────────────
print("\n【2 跨波串接】")
x = {w: set(D[w].x01.dropna().astype(int)) for w in WAVES}
print(f"  直接 join x01：20∩22={len(x[2020] & x[2022])}｜22∩24={len(x[2022] & x[2024])}｜"
      f"20∩24={len(x[2020] & x[2024])}｜三波={len(x[2020] & x[2022] & x[2024])}")
for a, b in ((2020, 2022), (2022, 2024), (2020, 2024)):
    m = D[a].set_index("pid").join(D[b].set_index("pid"), how="inner", lsuffix="_a", rsuffix="_b")
    ya, yb = V[a]["byear"], V[b]["byear"]
    sex = (m.a01_a == m.a01_b).mean()
    by = (m[ya + "_a" if ya == yb else ya] == m[yb + "_b" if ya == yb else yb]).mean()
    gap = (m.x02_b - m.x02_a).value_counts().to_dict()
    print(f"  {a}→{b} n={len(m)}：性別一致 {sex*100:.2f}%｜出生年一致 {by*100:.2f}%｜訪問年差 {gap}")

# ── 2. ICC ──────────────────────────────────────────────────────────────
print("\n【2 為何需要三波】親代對同戶不同子女的評分")
s = dy[["hh", "aff_p"]].dropna(); m = s[s.groupby("hh").aff_p.transform("size") >= 2]
same = m.groupby("hh").aff_p.nunique().eq(1).mean()
k = m.groupby("hh").size(); N, a = len(m), len(k); gm = m.aff_p.mean(); mu = m.groupby("hh").aff_p.mean()
msb = (k * (mu - gm) ** 2).sum() / (a - 1)
msw = ((m.aff_p - m.groupby("hh").aff_p.transform("mean")) ** 2).sum() / (N - a)
k0 = (N - (k ** 2).sum() / N) / (a - 1)
print(f"  多子女戶 {a} 戶／{N} 對：全給同分 {same*100:.1f}%｜ICC(1) = {(msb - msw) / (msb + (k0 - 1) * msw):.3f}")

# ── 3.1 居住距離 r（2020） ───────────────────────────────────────────────
print("\n【3.1】兩代相關（各波全樣本）")
for w in WAVES:
    y, _ = build_dyads(D[w])
    print(f"  {w}：距離 r={y[['dis_p','dis_k']].dropna().corr().iloc[0,1]:.3f}｜感情 r={y[['aff_p','aff_k']].dropna().corr().iloc[0,1]:.3f}")

# ── 3.2 接觸量：水準 vs 落差（同一樣本、同一模型，gap 係數 ≡ 親代 − 子代） ─
print("\n【3.2 接觸量】")
dy["tell"] = dy.k2p_k
X = ["tel_k", "met_k", "cohab", "tell"]
base = dy[["aff_p", "aff_k", "gap", "hh"] + X].dropna()
print(f"  ① 單變項（各自最大樣本）")
for x in ("tel_k", "met_k", "cohab", "tell"):
    row = []
    for y in ("aff_p", "aff_k", "gap"):
        r, s = fit(f"{y} ~ {x}", dy, [y, x]); row.append(f"{y} {bt(r, x)} n={len(s)}")
    print(f"    {x:6s} " + "｜".join(row))
print(f"  ② 四個一起放（共同樣本 n={len(base)}，{base.hh.nunique()} 戶）")
for y in ("aff_p", "aff_k", "gap"):
    r, _ = fit(f"{y} ~ " + " + ".join(X), base, [y] + X)
    print(f"    {y:6s} " + "  ".join(f"{x} {bt(r, x)}" for x in X))
rp, _ = fit("aff_p ~ " + " + ".join(X), base, ["aff_p"] + X)
rk, _ = fit("aff_k ~ " + " + ".join(X), base, ["aff_k"] + X)
print(f"    吐露 子代/親代 = {rk.params['tell'] / rp.params['tell']:.2f} 倍")
ra, _ = fit("aff_p ~ tell", dy, ["aff_p", "tell", "aff_k"]); rb, _ = fit("aff_k ~ tell", dy, ["aff_k", "tell", "aff_p"])
print(f"    （單變項 991 樣本的吐露：親代 {ra.params['tell']:+.3f}／子代 {rb.params['tell']:+.3f}，{rb.params['tell']/ra.params['tell']:.2f} 倍）")

print("  ③ 通訊頻率分組（子代報；同住依子代報距離=同住）")
grp = np.where(dy.cohab == 1, "同住", dy.tel_k.map({1: "幾乎每天", 2: "每週", 3: "每月", 4: "少於每月", 5: "不通話"}))
t = dy.assign(g=grp)[["g", "aff_p", "aff_k", "gap"]].dropna()
t = t[t.g.astype(str) != "nan"]
for g in ["同住", "幾乎每天", "每週", "每月", "少於每月", "不通話"]:
    s = t[t.g == g]
    print(f"    {g:5s} n={len(s):3d} 親 {s.aff_p.mean():.3f} 子 {s.aff_k.mean():.3f} 溫差 {s.gap.mean():+.3f}")

# ── 3.2 「13 個人口與接觸變項」 ─────────────────────────────────────────
print("\n【3.2 13 變項直接預測溫差】（文稿列出的變項；子代就業＝c02r1 1–4 有工作）")
dy["k_work"] = K.c02r1.between(1, 4).astype(float).where(K.c02r1.notna()).values
dy["k_edu"], dy["k_married"] = K.b01.where(K.b01 != 97).values, (K.d02b == 2).astype(float).values
dy["p_health"], dy["k_nsib"] = (6 - P.a11).values, P.g02z01.values
slot = dy.slot.values
pick = lambda pre: np.array([P.at[i, f"{pre}c{k}"] for i, k in enumerate(slot)], float)
dy["give_care"], dy["get_care"] = pick("g24"), pick("g25")
X13 = ["father", "son", "p_age", "k_age", "met_k", "tel_k", "cohab", "k_edu", "k_married", "k_work",
       "p_health", "k_nsib", "give_care", "get_care"]
for xs in (X13, [x for x in X13 if x != "k_age"], [x for x in X13 if x != "son"]):
    r, s = fit("gap ~ " + " + ".join(xs), dy, ["gap"] + xs)
    sig = [f"{x}(p={r.pvalues[x]:.3f})" for x in xs if r.pvalues[x] < .05]
    print(f"  {len(xs)} 變項 n={len(s)}：顯著 {sig or '無'}｜father p={r.pvalues['father']:.3f} p_age p={r.pvalues['p_age']:.3f}"
          f" tel_k p={r.pvalues['tel_k']:.3f} met_k p={r.pvalues['met_k']:.3f}")

# ── 3.4 三波軌跡 ────────────────────────────────────────────────────────
print("\n【3.4 三波軌跡】")
p = build_panel(WAVES); p["hh"] = p.hh_24
for w in (20, 22, 24):  # 通話：同住者記 0（＝最頻繁）的版本，與只看非同住者的版本
    p[f"tel0_k_{w}"] = p[f"tel_k_{w}"].where(p[f"cohab_{w}"] != 1, 0)
nc = (p.cohab_20 == 0) & (p.cohab_22 == 0) & (p.cohab_24 == 0)
rows = [("通話（同住=0）", "tel0_k", p), ("通話（三波皆非同住）", "tel_k", p[nc]), ("通話（各波有值即可）", "tel_k", p),
        ("見面", "met_k", p), ("吐露 子報", "k2p_k", p), ("吐露 親報", "k2p_p", p), ("同住率", "cohab", p),
        ("溫差", "gap", p), ("親代評分", "aff_p", p), ("子代評分", "aff_k", p)]
for name, v, s in rows:
    c = [f"{v}_{w}" for w in (20, 22, 24)]
    s = s[c + ["hh"]].dropna()
    d4, d1 = s[c[2]] - s[c[0]], s[c[1]] - s[c[0]]
    t4, t1 = stats.ttest_1samp(d4, 0), stats.ttest_1samp(d1, 0)
    print(f"  {name:14s} n={len(s):3d} " + " → ".join(f"{s[x].mean():.3f}" for x in c)
          + f"｜四年 {d4.mean():+.3f} (t={t4.statistic:+.2f}, p={t4.pvalue:.3f})｜20→22 {d1.mean():+.3f} (p={t1.pvalue:.3f})")
s = p[[f"gap_{w}" for w in (20, 22, 24)]].dropna()
d4 = s.gap_24 - s.gap_20; se = d4.std(ddof=1) / np.sqrt(len(d4)); tc = stats.t.ppf(.975, len(d4) - 1)
print(f"  溫差四年 95% CI [{d4.mean()-tc*se:+.3f}, {d4.mean()+tc*se:+.3f}]｜上限/2024 溫差 = {(d4.mean()+tc*se)/s.gap_24.mean()*100:.0f}%")
for delta in (.10, .15):
    print(f"  檢定力 Δ={delta:.2f}：{stats.norm.cdf(delta/se - 1.96):.2f}")
s = p[[f"{v}_{w}" for v in ("aff_p", "aff_k", "gap") for w in (20, 22, 24)] + ["father_20"]].dropna()
print(f"  （同一批 n={len(s)}）兩代 r：" + "／".join(f"{s[f'aff_p_{w}'].corr(s[f'aff_k_{w}']):.3f}" for w in (20, 22, 24))
      + "｜溫差 父−母：" + "／".join(f"{s[s.father_20==1][f'gap_{w}'].mean()-s[s.father_20==0][f'gap_{w}'].mean():+.3f}" for w in (20, 22, 24)))

# ── 4. 子女給父親 vs 母親（同一子女組內） ────────────────────────────────
print("\n【4 子女評父 vs 評母】RR2024 全體子女樣本")
k = D[2024][D[2024].x01a == 3][["h19f1", "h19m1"]].dropna()
t = stats.ttest_rel(k.h19m1, k.h19f1)
print(f"  n={len(k)} 母 {k.h19m1.mean():.3f} 父 {k.h19f1.mean():.3f} 差 {(k.h19m1-k.h19f1).mean():.3f} "
      f"(t={t.statistic:.2f})｜母較高 {(k.h19m1>k.h19f1).mean()*100:.1f}% 父較高 {(k.h19f1>k.h19m1).mean()*100:.1f}%")

# ── 4. 父親效果中介：子報 vs 親報吐露 ────────────────────────────────────
print("\n【4 中介：父親 → 吐露 → 溫差】")
for v in ("k2p_k", "k2p_p"):
    m = dy[["father", v, "gap", "hh"]].dropna()
    rc, _ = fit("gap ~ father", m, ["gap", "father"]); rb, _ = fit(f"gap ~ father + {v}", m, ["gap", "father", v])
    ra, _ = fit(f"{v} ~ father", m, [v, "father"])
    ind = ra.params["father"] * rb.params[v]
    print(f"  {v} n={len(m)}：間接佔 {ind/rc.params['father']*100:.1f}%｜a {bt(ra,'father')}｜b {bt(rb, v)}")

# ── 補驗：通話題原始值（0＝去年同住）版本 ───────────────────────────────
print("\n【補 3.2 通訊頻率表】通話題原始值分組（0＝去年同住，子代報）")
sfx = np.where(dy.father == 1, "f1", "m1")
dy["tel_raw"] = [K.at[i, f"h13{s}"] for i, s in enumerate(sfx)]
t = dy[["tel_raw", "aff_p", "aff_k", "gap"]].dropna()
for v, g in zip(range(6), ["去年同住", "幾乎每天", "每週", "每月", "少於每月", "不通話"]):
    s = t[t.tel_raw == v]
    print(f"    {g:5s} n={len(s):3d} 親 {s.aff_p.mean():.3f} 子 {s.aff_k.mean():.3f} 溫差 {s.gap.mean():+.3f}")
print(f"  去年同住佔 {(dy.tel_raw == 0).sum()}/{len(dy[dy.aff_p.notna() & dy.aff_k.notna()])}（991 樣本）"
      f"＝{(dy[dy.aff_p.notna() & dy.aff_k.notna()].tel_raw == 0).mean()*100:.1f}%")

print("\n【補 3.4】通話原始值（0＝去年同住）三波平均，同一批 681 對")
raw = {}
for w in WAVES:
    y, _ = build_dyads(D[w]); y = y.set_index("dyad_id")
    Kw = D[w].set_index("x01").loc[y.kid]
    col = V[w]["k"]["tel"]
    raw[w % 100] = pd.Series([Kw.iloc[i][col.format(s="f1" if f == 1 else "m1")] for i, f in enumerate(y.father)], index=y.index)
s = pd.DataFrame(raw).loc[p.index]
s = s[p[[f"gap_{w}" for w in (20, 22, 24)]].notna().all(axis=1)].dropna()
print(f"  n={len(s)} " + " → ".join(f"{s[w].mean():.3f}" for w in (20, 22, 24))
      + f"｜2020 值域 {sorted(s[20].unique())}｜2024 值域 {sorted(s[24].unique())}")

print("\n【補 3.4】兩代 r 與 父−母溫差：各波 701 對中有值者")
for w in (20, 22, 24):
    s = p[[f"aff_p_{w}", f"aff_k_{w}", f"gap_{w}", f"father_{w}"]].dropna()
    print(f"  20{w} n={len(s)} r={s[f'aff_p_{w}'].corr(s[f'aff_k_{w}']):.3f}"
          f" 父−母 {s[s[f'father_{w}']==1][f'gap_{w}'].mean()-s[s[f'father_{w}']==0][f'gap_{w}'].mean():+.3f}")

print("\n【補 2 配對】RR2024 涉及受訪子女的重複組：出生月是否相同")
c = V[2024]; d = D[2024]; pr = d[d.x01a == 1]; kids = d[d.x01a == 3]
for _, r in pr.iterrows():
    sl = [(r[f"g07c{k}"], r[f"g08z01c{k}"], r[f"g08z02c{k}"], k) for k in range(1, 7) if pd.notna(r[f"g07c{k}"])]
    for key, g in itertools.groupby(sorted(sl, key=lambda x: x[:2]), key=lambda x: x[:2]):
        g = list(g)
        hit = kids[(kids.x01d == r.x01d) & (kids.a01 == key[0]) & (kids.a02z01 == key[1])]
        if len(g) >= 2 and len(hit):
            print(f"  戶 {int(r.x01d)}：欄位月 {[x[2] for x in g]}｜受訪子女 {len(hit)} 人 月 {hit.a02z02.tolist()}")
