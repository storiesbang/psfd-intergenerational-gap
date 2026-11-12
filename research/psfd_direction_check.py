# -*- coding: utf-8 -*-
"""頻率題方向核對：作品裡哪些數字是原始方向（大＝少）、哪些是反向（大＝多）。
A. 同一個模型，原始 vs 反向，係數只差正負號 → 看作品印的是哪一個。
B. 試著重建圖 2 七因素合併模型（n=958、說心事 −0.187），找出它用了什麼編碼。
跑法：uv run --with pandas --with pyreadstat --with statsmodels --with scipy python psfd_direction_check.py | tee out/psfd/direction_check_output.txt
"""
import warnings; warnings.filterwarnings("ignore")
import itertools
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from psfd_engine import load, build_dyads, header, CSV

d = load(2024)
dy, par = build_dyads(d)
header(dy, "頻率題方向核對（RR2024）")
P = par.loc[dy.hh].reset_index(drop=True)
dy = dy.reset_index(drop=True)
pick = lambda pre: np.array([P.at[i, f"{pre}c{k}"] for i, k in enumerate(dy.slot)], float)
raw = pd.read_csv(CSV[2024], low_memory=False).set_index("x01")
sfx = np.where(dy.p_sex == 1, "f1", "m1")
dy["tel_raw"] = [raw.at[k, f"h13{s}"] for k, s in zip(dy.kid, sfx)]   # 0=去年同住 保留
dy["tell"] = dy.k2p_k
SCALE = dict(tel_k=6, met_k=7, tell=6)   # 反向＝(最大值+1) − x


def fit(f, s):
    return smf.ols(f, data=s).fit(cov_type="cluster", cov_kwds={"groups": s.hh})


def bt(r, x):
    return f"b={r.params[x]:+.3f} (p={r.pvalues[x]:.3g})"


print("【原始選項順序】通話 1幾乎每天…5幾乎不通電話｜見面 1幾乎每天…6幾乎不碰面｜說心事 1總是…5從未 ⇒ 原始＝數值越大越『不』頻繁\n")

print("【A】作品第 3.2 節四變項聯合模型（n=614），原始 vs 反向")
X = ["tel_k", "met_k", "cohab", "tell"]
base = dy[["aff_p", "aff_k", "gap", "hh"] + X].dropna()
rev = base.assign(**{c: SCALE[c] - base[c] for c in ("tel_k", "met_k", "tell")})
for y, doc in (("aff_p", "通話 −0.095、說心事 −0.090"), ("aff_k", "通話 −0.163、說心事 −0.316")):
    a, b = fit(f"{y} ~ " + " + ".join(X), base), fit(f"{y} ~ " + " + ".join(X), rev)
    print(f"  {y}  原始：通話 {bt(a,'tel_k')}｜說心事 {bt(a,'tell')}")
    print(f"  {y}  反向：通話 {bt(b,'tel_k')}｜說心事 {bt(b,'tell')}   ← 作品印：{doc}")
print(f"  n={len(base)}\n")

print("【B】重建圖 2：gap ~ 7 因素（全部標準化），作品印 n=958、說心事 −0.187 [−0.251, −0.122]")
dy["money"] = ((pick("g22z01") == 1) | (pick("g23z01") == 1)).astype(float)   # 任一方向有金錢往來
dy["care"] = pd.Series(pick("g24")) + pd.Series(pick("g25"))                   # 雙向家務天數
dy["money_n"] = (pick("g22z01") == 1).astype(float) + (pick("g23z01") == 1)
# 通話：去年同住要不要放進來、放成哪一級
TEL = {"同住設空值": dy.tel_k, "去年同住＝最頻繁(0)": dy.tel_raw.where(dy.tel_raw >= 0)}
MET = {"同住設空值": dy.met_k, "同住者＝每天見面(1)": dy.met_k.where(dy.cohab != 1, 1)}
rows = []
for (tn, tv), (mn, mv), direction, care in itertools.product(TEL.items(), MET.items(), ("原始", "反向"), ("care", "money_n")):
    s = dy.assign(tel=tv, met=mv)
    if direction == "反向":
        s = s.assign(tel=6 - s.tel, met=7 - s.met, tell=6 - s.tell)
    cols = ["tell", "tel", "met", "cohab", "money", care, "father"]
    s = s[["gap", "hh"] + cols].dropna()
    z = s.assign(**{c: (s[c] - s[c].mean()) / s[c].std() for c in cols})
    r = fit("gap ~ " + " + ".join(cols), z)
    q = multipletests(r.pvalues[cols], method="fdr_bh")[1]
    ci = r.conf_int().loc["tell"]
    rows.append((f"通話:{tn}｜見面:{mn}｜{direction}｜{care}", len(s), r.params["tell"], ci[0], ci[1],
                 " ".join(f"{c}={qq:.3g}" for c, qq in zip(cols, q))))
for name, n, b, lo, hi, qs in sorted(rows, key=lambda x: (abs(x[1] - 958), abs(x[2] + 0.187))):
    flag = "  ◀ n 與係數都對上" if n == 958 and abs(b + 0.187) < 0.0015 else ""
    print(f"  n={n:4d} 說心事 {b:+.3f} [{lo:+.3f}, {hi:+.3f}]  {name}{flag}")
    print(f"         q: {qs}")
