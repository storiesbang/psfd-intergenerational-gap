# -*- coding: utf-8 -*-
"""初賽作品 v3 的四張圖（依 v2_數字校正with suggestion.docx 紅字修改）。
圖 1：(b) 改三波溫差分布＋配對 t／符號檢定 p；圖 2：q 值加顯著星號；圖 3：(b) 通話方向反轉（左=幾乎不通、右=去年同住）；
圖 4：雙向 FE 改成單一面板（三個結果 × 兩種報告人）。
圖 1/3/4 由 engine 重算並 assert 對回文中數字；圖 2 的係數抄自組員 joint_8.py 的 joint.csv（原始管線不在本機）。
跑法：cd research && uv run --with pandas --with pyreadstat --with statsmodels --with scipy --with linearmodels --with matplotlib python psfd_v3_figs.py
"""
import glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
from linearmodels.panel import PanelOLS
from psfd_engine import load, build_dyads, build_panel, V

for f in glob.glob("/System/Library/AssetsV2/*/*/AssetData/PingFang.ttc"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["PingFang TC", "PingFang HK", "Heiti TC"], "axes.unicode_minus": False})
OUT = "out/psfd/"
RED, BLUE, GRAY, MUTED = "#b85c45", "#3d6a9c", "#9a9fa5", "#777"
WAVES = (2020, 2022, 2024)

print("=" * 70)
print("口徑：PSFD RR2020/2022/2024，engine.build_dyads 配對；溫差＝親代−子代感情（1–5）；95% CI＝均值±1.96SE")
print("=" * 70)

D = {w: load(w) for w in WAVES}
DY = {w: build_dyads(D[w])[0] for w in WAVES}

# ── 圖 1 ─────────────────────────────────────────────
def r_of(dy, a, b):
    s = dy[[a, b]].dropna(); return stats.pearsonr(s[a], s[b])[0]
r_dis = [r_of(DY[w], "dis_p", "dis_k") for w in WAVES]
r_aff = [r_of(DY[w], "aff_p", "aff_k") for w in WAVES]
print("r 距離", np.round(r_dis, 3), "r 感情", np.round(r_aff, 3))
assert np.allclose(r_dis, [.894, .928, .930], atol=6e-4) and np.allclose(r_aff, [.268, .340, .285], atol=6e-4)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200, gridspec_kw={"width_ratios": [1, 1.15], "wspace": .3})
ys = [0, 1, 2, 3.6, 4.6, 5.6]
vals = r_dis + r_aff
a1.barh(ys, vals, height=.62, color=[GRAY] * 3 + [BLUE] * 3)
for y, v in zip(ys, vals):
    a1.text(v + .012, y, f"{v:.3f}", va="center", fontsize=11, fontweight="bold")
a1.set_yticks(ys, [f"{w}  居住距離" for w in WAVES] + [f"{w}  感情好不好" for w in WAVES], fontsize=11)
a1.text(1, -.7, "客觀題", ha="right", color=GRAY, fontsize=10, fontweight="bold")
a1.text(1, 2.9, "主觀題", ha="right", color=BLUE, fontsize=10, fontweight="bold")
a1.set_xlim(0, 1); a1.set_ylim(6.2, -1); a1.set_xlabel("兩代回答的相關係數 r", fontsize=11)
a1.set_title("(a) 客觀題對得上，主觀題對不上", fontsize=13)
a1.grid(axis="x", color="#e6e6e6"); a1.set_axisbelow(True)
for sp in ("top", "right"): a1.spines[sp].set_visible(False)
a1.tick_params(axis="y", length=0)

bins = np.arange(-3, 5)
shades = ["#c9ced4", "#8c949c", "#3f4750"]
share, tests = {}, {}
for i, w in enumerate(WAVES):
    g = DY[w].gap.dropna().astype(int)
    pct = g.value_counts(normalize=True).reindex(bins, fill_value=0) * 100
    a2.bar(bins + (i - 1) * .27, pct, width=.27, color=shades[i], label=f"{w}（n = {len(g)}）")
    share[w] = ((g < 0).mean() * 100, (g == 0).mean() * 100, (g > 0).mean() * 100)
    t = stats.ttest_1samp(g, 0)
    sign = stats.binomtest(int((g > 0).sum()), int((g != 0).sum()), .5)
    tests[w] = (g.mean(), t.statistic, t.pvalue, sign.pvalue)
    print(f"{w} n={len(g)} 子高/同/親高 = {share[w][0]:.1f}/{share[w][1]:.1f}/{share[w][2]:.1f}%  "
          f"平均 {g.mean():+.3f} t={t.statistic:.2f} p={t.pvalue:.1e}｜符號檢定 p={sign.pvalue:.1e}")
assert [len(DY[w].gap.dropna()) for w in WAVES][2] == 991 and abs(tests[2024][1] - 11.06) < .01
assert np.allclose(share[2024], (16.1, 43.8, 40.1), atol=.05)
for x, j, c in ((-2, 0, BLUE), (0, 1, "#555"), (2.2, 2, RED)):
    lab = ["子代較高", "一致", "親代較高"][j]
    a2.text(x, 56, lab + "\n" + "／".join(f"{share[w][j]:.1f}" for w in WAVES) + "%", ha="center", va="top",
            color=c, fontsize=10.5, fontweight="bold", linespacing=1.4)
a2.axvspan(-3.5, -.5, color=BLUE, alpha=.06, lw=0); a2.axvspan(.5, 4.5, color=RED, alpha=.06, lw=0)
a2.set_xlim(-3.5, 4.5); a2.set_ylim(0, 58); a2.set_xticks(bins)
a2.set_xlabel("溫差 = 親代評分 − 子代評分（分）", fontsize=11); a2.set_ylabel("親子對數占比（%）", fontsize=11)
a2.set_title("(b) 三波溫差分布：親代較高者明顯多於子代較高者", fontsize=13)
a2.legend(loc="upper left", bbox_to_anchor=(0, .8), frameon=False, fontsize=9.5)
a2.grid(axis="y", color="#e6e6e6"); a2.set_axisbelow(True)
for sp in ("top", "right"): a2.spines[sp].set_visible(False)
txt = "平均溫差（三波）\n" + "／".join(f"{tests[w][0]:+.3f}" for w in WAVES) + "\n配對 t 檢定、符號檢定\n三波皆 p < 0.001"
assert all(tests[w][2] < .001 and tests[w][3] < .001 for w in WAVES)
a2.text(4.4, 15, txt, ha="right", fontsize=9.5, color="#333", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=.4", fc="white", ec="#ccc"))
fig.text(.27, .01, "居住距離題的「同住」選項已改編碼為最近距離。資料：PSFD RR2020／RR2022／RR2024", ha="center", fontsize=9.5, color=MUTED)
fig.text(.73, .01, "符號檢定：親代較高 vs 子代較高（排除一致者）。資料：PSFD RR2020／RR2022／RR2024", ha="center", fontsize=9.5, color=MUTED)
fig.subplots_adjust(left=.11, right=.98, top=.9, bottom=.17)
fig.savefig(OUT + "v3_fig1.png", facecolor="white"); plt.close(fig)

# ── 圖 2：抄自組員 joint.csv（RR2024，n = 958，BH–FDR）──────────────
J = pd.DataFrame([("子女說心事（子報）", -.187, -.251, -.122, 9.8e-8), ("通話頻率†", -.058, -.148, .033, .247),
                  ("見面頻率†", -.079, -.169, .011, .148), ("同住", -.057, -.131, .017, .181),
                  ("金錢往來", .009, -.067, .084, .825), ("家務協助天數", .079, .002, .157, .148),
                  ("親代為父親", .071, -.005, .147, .148)], columns=["var", "b", "lo", "hi", "q"])
star = lambda q: "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""
fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
T = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
for i, r in J.iterrows():
    sig = r.q < .05; c = "#184f95" if sig else GRAY
    ax.plot([r.lo, r.hi], [i, i], color=c, lw=2.2)
    ax.scatter(r.b, i, s=60, zorder=3, facecolor=c if sig else "white", edgecolor=c, linewidth=1.6)
    if sig: ax.text(r.b, i - .3, star(r.q), ha="center", color=c, fontsize=12, fontweight="bold")
    col = "#222" if sig else "#555"
    ax.text(1.07, i, f"{r.b:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", va="center", fontsize=10.4, color=col, family="Menlo", transform=T)
    ax.text(2.22, i, (r"$9.8\times10^{-8}$" if r.q < .001 else f"{r.q:.3f}") + " " + star(r.q), va="center", fontsize=10.4,
            color=col, fontweight="bold" if sig else "normal", transform=T)
ax.text(1.07, -1.05, "係數 [95% CI]", fontsize=11.7, fontweight="bold", transform=T)
ax.text(2.22, -1.05, "q 值", fontsize=11.7, fontweight="bold", transform=T)
ax.axvline(0, color="#555", lw=.9)
ax.set_yticks(range(len(J)), J["var"], fontsize=12.3); ax.tick_params(axis="y", length=0)
ax.set_ylim(len(J) - .4, -1.4); ax.set_xlim(-.3, .3)
ax.set_xlabel("因素每增加 1 個標準差，溫差的變化（分）\n← 溫差縮小　　　　　　溫差擴大 →", fontsize=12.3)
ax.grid(axis="x", color="#e6e6e6", lw=.6); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="x", labelsize=9, colors="#444")
fig.text(.02, .02, "註：RR2024，n = 958 對，所有因素同時放入模型並標準化；誤差線為 95% CI。q 值為 BH–FDR 校正後之 p 值；\n"
         "實心＝q < 0.05，空心＝不顯著；* q < 0.05，** q < 0.01，*** q < 0.001。†僅非同住者有值。", fontsize=9.3, color=MUTED, linespacing=1.5)
fig.subplots_adjust(left=.21, right=.5, top=.93, bottom=.31)
fig.savefig(OUT + "v3_fig2.png", facecolor="white"); plt.close(fig)

# ── 圖 3 ─────────────────────────────────────────────
dy = DY[2024].copy()
K = D[2024].set_index("x01").loc[dy.kid].reset_index(drop=True)
dy["tel_raw"] = [K.at[i, V[2024]["k"]["tel"].format(s="f1" if f == 1 else "m1")] for i, f in enumerate(dy.father)]


def panel(ax, col, order, labels, title, xlab, shade=None):
    s = dy[[col, "aff_p", "aff_k"]].dropna()
    m = {y: [(s[s[col] == v][y].mean(), 1.96 * s[s[col] == v][y].sem()) for v in order] for y in ("aff_p", "aff_k")}
    ns = [int((s[col] == v).sum()) for v in order]
    x = np.arange(len(order))
    keep = [i for i in x if shade is None or i != shade]  # 同住點不跟頻率線連
    for y, c, mk in (("aff_p", RED, "o"), ("aff_k", BLUE, "s")):
        mu = np.array([a for a, _ in m[y]]); e = [b for _, b in m[y]]
        ax.plot(x[keep], mu[keep], color=c, lw=2.6)
        ax.errorbar(x, mu, yerr=e, fmt=mk, color=c, ms=9, capsize=4, lw=1.6, mec="white", zorder=3)
    p_, k_ = np.array([a for a, _ in m["aff_p"]]), np.array([a for a, _ in m["aff_k"]])
    ax.fill_between(x[keep], p_[keep], k_[keep], color="#ddd", alpha=.75, lw=0)
    if shade is not None:
        ax.axvspan(shade - .5, shade + .5, color="#eeeee8", zorder=0)
        ax.fill_between([shade - .08, shade + .08], p_[shade], k_[shade], color="#ddd", lw=0)
    ax.set_xticks(x, [f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=11)
    ax.set_title(title, loc="left", fontsize=14); ax.set_xlabel(xlab, fontsize=11.5)
    ax.set_ylim(3, 5.2); ax.set_yticks(np.arange(3, 5.01, .5)); ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    return ns


fig = plt.figure(figsize=(11, 8.6), dpi=200)
gs = fig.add_gridspec(2, 2, hspace=.5, wspace=.12)
a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
na = panel(a, "k2p_k", [5, 4, 3, 2, 1], ["從未", "偶爾", "有時", "經常", "總是"], "(a) 子女對父母說心事（子女報告）", "說心事頻率 → 越頻繁")
nb = panel(b, "tel_raw", [5, 4, 3, 2, 1, 0], ["幾乎\n不通", "每月\n<1次", "每月\n1–3次", "每週", "幾乎\n每天", "去年\n同住"],
           "(b) 通話頻率（子女報告）", "通話頻率 → 越頻繁", shade=5)
nc = panel(c, "father", [0, 1], ["母親", "父親"], "(c) 親代性別", "親代")
print("圖 3 n：", sum(na), sum(nb), sum(nc), "| (b) 各組", nb)
assert (sum(na), sum(nb), sum(nc)) == (988, 985, 991) and nb == [73, 69, 176, 223, 81, 363]
b.set_yticklabels([]); c.set_xlim(-.5, 1.5)
for ax in (a, c): ax.set_ylabel("「感情好不好」平均評分", fontsize=11.5, labelpad=8)
lg = fig.add_axes([.55, .08, .4, .35]); lg.axis("off")
lg.plot([], [], color=RED, marker="o", lw=2.6, ms=9, label="親代評分")
lg.plot([], [], color=BLUE, marker="s", lw=2.6, ms=9, label="子代評分")
lg.fill_between([], [], [], color="#ddd", label="溫差（兩線之間）")
lg.legend(loc="upper left", frameon=False, fontsize=13)
lg.text(0, .35, "資料：PSFD RR2024\nn = 988 對 (a)、985 對 (b)、991 對 (c)\n誤差線為 95% CI\n數值為未控制其他因素之原始平均",
        fontsize=11, color=MUTED, va="top", linespacing=1.8, transform=lg.transAxes)
fig.subplots_adjust(left=.08, right=.98, top=.95, bottom=.08)
fig.savefig(OUT + "v3_fig3.png", facecolor="white"); plt.close(fig)

# ── 圖 4：雙向 FE，單一面板 ─────────────────────────────
p = build_panel(WAVES); p["hh"] = p.hh_24
NUM = ["aff_p", "aff_k", "gap", "k2p_k", "k2p_p"]
long = pd.concat([p[[f"{v}_{w % 100:02d}" for v in NUM]].rename(columns=lambda c: c[:-3])
                  .assign(wave=w, hh=p.hh.values, dyad_id=p.index) for w in WAVES], ignore_index=True)
for v in ("k2p_k", "k2p_p"): long[v] = 6 - long[v]  # 反向：數值大＝說心事越頻繁，係數正＝說心事變多、評分變高


def twfe(x, y):
    s = long[[x, y, "wave", "hh", "dyad_id"]].dropna().set_index(["dyad_id", "wave"])
    r = PanelOLS(s[y], s[[x]], entity_effects=True, time_effects=True).fit(cov_type="clustered", clusters=s[["hh"]])
    return r.params[x], r.std_errors[x], r.pvalues[x]


R = {(x, y): twfe(x, y) for x in ("k2p_k", "k2p_p") for y in ("aff_p", "aff_k", "gap")}
for k, (bb, se, pv) in R.items(): print(f"FE {k}: b={bb:+.3f} SE={se:.3f} p={pv:.2g}")
assert np.allclose([R[("k2p_k", y)][0] for y in ("aff_p", "aff_k", "gap")], [.031, .118, -.090], atol=6e-4)

fig, ax = plt.subplots(figsize=(10, 3.4), dpi=200)
rows = [("aff_p", "親代評分"), ("aff_k", "子代評分"), ("gap", "溫差（親−子）")]
for i, (y, lab) in enumerate(rows):
    for x, dy_, c, nm in (("k2p_k", -.17, "#184f95", "子女自陳"), ("k2p_p", .17, "#c08a2e", "父母回報")):
        bb, se, pv = R[(x, y)]; sig = pv < .05
        ax.plot([bb - 1.96 * se, bb + 1.96 * se], [i + dy_] * 2, color=c, lw=2.4, alpha=1 if sig else .45)
        ax.scatter(bb, i + dy_, s=70, zorder=3, facecolor=c if sig else "white", edgecolor=c, lw=1.8)
        ps = "p < 0.001" if pv < .001 else f"p = {pv:.3f}"
        ax.text(.215, i + dy_, f"{bb:+.3f}（{ps}）" + ("  " + "*" * (3 if pv < .001 else 1) if sig else ""), va="center",
                fontsize=10, color=c if sig else "#888", fontweight="bold" if sig else "normal")
ax.axvline(0, color="#555", lw=1)
ax.set_yticks(range(3), [r[1] for r in rows], fontsize=12); ax.tick_params(axis="y", length=0)
ax.set_ylim(2.55, -.55); ax.set_xlim(-.2, .2)
ax.set_xlabel("子女說心事每多一分，同一對親子的評分變化（分）", fontsize=11)
ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.scatter([], [], s=70, color="#184f95", label="說心事＝子女自陳"); ax.scatter([], [], s=70, color="#c08a2e", label="說心事＝父母回報")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10.5)
fig.text(.01, .01, "註：RR2020／2022／2024 三波交集 701 對，雙向固定效果（個體＋波次），戶層叢集穩健標準誤，誤差線為 95% CI。"
         "實心＝p < 0.05。", fontsize=9, color=MUTED)
fig.subplots_adjust(left=.13, right=.78, top=.86, bottom=.24)
fig.savefig(OUT + "v3_fig4.png", facecolor="white"); plt.close(fig)
print("輸出：", ", ".join(OUT + f"v3_fig{i}.png" for i in range(1, 5)))
