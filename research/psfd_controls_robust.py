# -*- coding: utf-8 -*-
"""圖 2 七因素模型的穩健性檢查：加入已編碼控制變項後，「只有子女說心事顯著」是否成立。

基準＝組員管線 match.py → ext.py → coef.py（前段）→ joint_7.py，原封不動以 exec 執行，
只把組員機器的檔案路徑 U 換成本機路徑；不改任何計算。控制變項照搬 psfd_v2_verify.py 的編碼
（engine 資料，遺漏碼已處理），以子女受訪者 ID（c_id ≡ engine kid）併回。
跑法：cd research && uv run --with pandas --with pyreadstat --with statsmodels --with scipy python psfd_controls_robust.py
"""
import os, sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out", "psfd")
WORK = os.path.join(OUT, "team_rebuild")          # 組員管線的中間檔（dyad*.csv）
os.makedirs(WORK, exist_ok=True)
RAW = {w: os.path.join(ROOT, f"家庭動態調查：{w}追蹤問卷", f) for w, f in
       ((2020, "psfd_rr2020_v202208_csv.csv"), (2022, "psfd_rr2022_v202403_csv.csv"),
        (2024, "psfd_rr2024_v202603_csv.csv"))}
TEAM_U = {"7737d9a9-1791520839885_psfd_rr2020_v202208_csv.csv": RAW[2020],
          "6f51fc00-1791520817517_psfd_rr2022_v202403_csv.csv": RAW[2022],
          "d7cf9816-1791520728709_psfd_rr2024_v202603_csv.csv": RAW[2024]}


def team(src, ns=None, cut=None):
    """執行組員腳本：只替換檔案路徑，其餘一字不改。"""
    code = open(os.path.join(ROOT, src), encoding="utf-8").read()
    if cut:
        code = code.split(cut)[0]
    for k, v in TEAM_U.items():
        code = code.replace(f'U+"{k}"', repr(v))
    ns = ns if ns is not None else {}
    cwd = os.getcwd(); os.chdir(WORK)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, src, "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns


# ── 1. 組員管線：配對 → 擴充 → 因子欄位（與圖 2 同一份 d） ───────────────
team("match.py"); team("ext.py")
d = team("coef.py", cut="G=[")["d"]


# ── 2. 圖 2 模型：照抄 joint_7.py 的設計矩陣，另允許附加控制變項 ─────────
FOCAL = ["子女說心事（子報）", "通話頻率", "見面頻率", "同住", "金錢往來", "家務協助天數", "親代為父親"]


def design(s):
    keep = ['子女說心事（子報）', '通話頻率', '見面頻率', '金錢往來', '家務協助天數', '親代為父親']
    s = s.copy()
    s['金錢往來'] = s['親→子 金錢'] + s['子→親 金錢']
    s['家務協助天數'] = s['親→子 家務天數'] + s['子→親 家務天數']
    s['去年同住'] = (s.call == 0) * 1.0; s.loc[s.call.isna(), '去年同住'] = np.nan
    X = pd.DataFrame(index=s.index)
    for v in keep:
        x = s[v]; z = (x - x.mean()) / x.std()
        if v in ('通話頻率', '見面頻率'): z = z.where(s['去年同住'] == 0, 0.0)
        X[v] = z
    X['同住'] = (s['去年同住'] - s['去年同住'].mean()) / s['去年同住'].std()
    return s, X


def fit(s, X, ok):
    m = sm.OLS(s.gap[ok], sm.add_constant(X[ok])).fit(cov_type='cluster', cov_kwds={'groups': s.x01d[ok]})
    ci = m.conf_int()
    R = pd.DataFrame({'b': m.params[FOCAL], 'lo': ci.loc[FOCAL, 0], 'hi': ci.loc[FOCAL, 1], 'p': m.pvalues[FOCAL]})
    R['q'] = multipletests(R.p, method='fdr_bh')[1]          # 與圖 2 同：只對七個焦點因素做 BH
    return m, R


s, X = design(d)
okA = X.notna().all(axis=1) & s.gap.notna()
mA, RA = fit(s, X, okA)

# 基準必須逐項對回圖 2（joint.csv／docx 3.2）
assert int(mA.nobs) == 958, int(mA.nobs)
r = RA.loc['子女說心事（子報）']
assert (round(r.b, 3), round(r.lo, 3), round(r.hi, 3)) == (-0.187, -0.251, -0.122), r
FIG2_Q = {"通話頻率": .247, "見面頻率": .148, "同住": .181, "金錢往來": .825, "家務協助天數": .148, "親代為父親": .148}
for v, q in FIG2_Q.items():
    assert round(RA.q[v], 3) == q, (v, RA.q[v], q)


# ── 3. 控制變項：照搬 psfd_v2_verify.py 的編碼（engine 資料） ─────────────
sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    from psfd_engine import load, build_dyads
    D24 = load(2024)
    dy, par = build_dyads(D24)
P = par.loc[dy.hh].reset_index(drop=True)
K = D24.set_index("x01").loc[dy.kid].reset_index(drop=True)
dy = dy.reset_index(drop=True)
dy["k_work"] = K.c02r1.between(1, 4).astype(float).where(K.c02r1.notna()).values
dy["k_edu"], dy["k_married"] = K.b01.where(K.b01 != 97).values, (K.d02b == 2).astype(float).values
dy["p_health"], dy["k_nsib"] = (6 - P.a11).values, P.g02z01.values
# 註：v2_verify 的 give_care/get_care＝g24/g25 家務天數，與焦點因素「家務協助天數」同一題 ⇒ 不列入控制
CTRL = ["p_age", "k_age", "son", "k_married", "k_work", "p_health", "k_edu", "k_nsib"]
C = dy.set_index("kid")[CTRL].add_prefix("c_")   # 加前綴，避免與組員 d 的 p_age 撞名
CTRL = list(C.columns)
assert C.index.is_unique
s2 = s.join(C, on="c_id")
match_rate = s2.loc[okA, "c_p_age"].notna().mean()  # 併得上的比例（以 p_age 有值判斷）

XC = X.copy()
for v in CTRL:                                     # 與七因素相同：全樣本 z 分數
    x = s2[v]; XC[v] = (x - x.mean()) / x.std()
okC = XC.notna().all(axis=1) & s2.gap.notna()

mB, RB = fit(s2, X, okC)                           # (B) 七因素，限控制變項皆有值的樣本
mC, RC = fit(s2, XC, okC)                          # (C) 七因素＋控制變項
pall = mC.pvalues.drop("const")
q_all = pd.Series(multipletests(pall, method='fdr_bh')[1], index=pall.index)


# ── 4. 輸出：抬頭＋結論 ──────────────────────────────────────────────
NAME = {"子女說心事（子報）": "說心事", "通話頻率": "通話", "見面頻率": "見面", "同住": "同住",
        "金錢往來": "金錢", "家務協助天數": "家務", "親代為父親": "父親"}
lines = [
    "=" * 78,
    "圖 2 七因素模型 · 控制變項穩健性檢查（RR2024）",
    "樣本＝組員管線 match→ext→coef→joint_7（同一份 d，未改一行計算，只換本機檔案路徑）",
    "依變項＝親評 − 子評（未標準化）｜頻率題反向（6−x／7−x）｜同住＝通話題「去年同住」；同住者通話、見面 z 後設 0＋同住虛擬變項",
    "自變項全樣本 z 分數｜OLS＋戶層（x01d）叢集穩健 SE｜BH–FDR 族群＝七個焦點因素（同圖 2）",
    f"控制變項（v2_verify 編碼，engine 遺漏碼處理）：{', '.join(CTRL)}；以子女 ID 併回，焦點樣本中併得上 {match_rate:.1%}",
    "不列入：give_care/get_care（與焦點『家務協助天數』同一題 g24/g25）、親子衝突、經濟、是否有孩子",
    "=" * 78,
]


def block(tag, m, R, hh):
    out = [f"\n{tag}  n={int(m.nobs)}  戶={hh}"]
    for v in FOCAL:
        x = R.loc[v]
        out.append(f"  {NAME[v]:3s} b={x.b:+.3f} [{x.lo:+.3f},{x.hi:+.3f}] p={x.p:.3g} q={x.q:.3g}{'  ★' if x.q < .05 else ''}")
    sig = [NAME[v] for v in FOCAL if R.q[v] < .05]
    out.append(f"  q<0.05 的焦點因素：{sig or '無'}")
    return out


lines += [f"\n基準重現：n=958、說心事 −0.187 [−0.251,−0.122]、其餘六項 q 全部 assert 通過 ✓"]
lines += block("(A) 基準七因素", mA, RA, s.x01d[okA].nunique())
lines += block("(B) 七因素｜限控制變項皆有值樣本", mB, RB, s2.x01d[okC].nunique())
lines += block("(C) 七因素＋控制變項", mC, RC, s2.x01d[okC].nunique())
csig = [f"{v}(b={mC.params[v]:+.3f}, p={mC.pvalues[v]:.3g})" for v in CTRL if mC.pvalues[v] < .05]
lines.append(f"  控制變項 p<0.05：{csig or '無'}")
lines.append(f"  敏感度：全部 {len(pall)} 個係數一起 BH 時，說心事 q={q_all['子女說心事（子報）']:.3g}；"
             f"其他 q<0.05：{[v for v in q_all.index if q_all[v] < .05 and v != '子女說心事（子報）'] or '無'}")
lines.append(f"\n說心事 b：A {RA.b.iloc[0]:+.3f} → B {RB.b.iloc[0]:+.3f} → C {RC.b.iloc[0]:+.3f}"
             f"｜樣本 {int(mA.nobs)} → {int(mC.nobs)}")

txt = "\n".join(lines)
print(txt)
open(os.path.join(OUT, "controls_robust_output.txt"), "w", encoding="utf-8").write(txt + "\n")
