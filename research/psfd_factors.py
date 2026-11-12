# -*- coding: utf-8 -*-
"""組員提問：溫差到底被什麼決定？社經背景／相處時間／管教要求／孩子身心狀況。

口徑一律走 psfd_engine，不在這裡重定義配對、反向計分或遺漏碼。
每個區塊都印 N，因為收入題只覆蓋一半 dyad，跨區塊比係數必須看同一子樣本。
"""
import sys
import numpy as np, pandas as pd, statsmodels.api as sm
import psfd_engine as E

OUT = E.ROOT / "research" / "out" / "psfd" / "factors_output.txt"


def ols(y, X, hh, data):
    """戶層叢集穩健 SE。回傳 (res, 實際樣本 index)。"""
    df = data[[y] + X + [hh]].dropna()
    r = sm.OLS(df[y], sm.add_constant(df[X])).fit(
        cov_type="cluster", cov_kwds={"groups": df[hh]})
    return r, df.index


def show(tag, r, keys=None):
    print(f"  {tag:26s} n={int(r.nobs):4d}  R²={r.rsquared:.3f}")
    for k in (keys or [c for c in r.params.index if c != "const"]):
        if k in r.params.index:
            print(f"      {k:22s} b={r.params[k]:+.3f}  t={r.tvalues[k]:+.2f}  p={r.pvalues[k]:.3g}")


def main():
    d = E.load()
    dy, par = E.build_dyads(d)
    kid = d.set_index("x01")
    E.header(dy)

    P = par.loc[dy.hh].reset_index(drop=True)   # 親代原始列
    K = kid.loc[dy.kid].reset_index(drop=True)  # 子代原始列
    dy = dy.reset_index(drop=True)

    # ---- 候選因素 ------------------------------------------------------
    dy["p_edu"] = P.b01.values                  # 教育程度 1..(高)
    dy["k_edu"] = K.b01.values
    dy["edu_up"] = dy.k_edu - dy.p_edu          # 代間教育流動：子 − 親
    dy["p_health"] = 6 - P.a11.values           # 反向：高=健康好
    dy["k_health"] = 6 - K.a11.values
    dy["p_inc"] = np.log1p(P.k24e.values)       # 家戶總收入（覆蓋率低）
    dy["k_inc"] = np.log1p(K.k24e.values)
    dy["k_married"] = (K.d02b.values == 2).astype(float)
    dy["k_nsib"] = P.g02z01.values              # 親代的子女數＝子代的手足數
    slot = dy.slot.values
    pick = lambda pre: np.array([P.at[i, f"{pre}c{k}"] for i, k in enumerate(slot)], float)
    dy["k_grandkid"] = pick("g30z01") + pick("g30z02")       # 這個子女生了幾個
    dy["give_money"] = (pick("g22z01") == 1).astype(float)   # 親代→子代 金錢
    dy["get_money"] = (pick("g23z01") == 1).astype(float)    # 子代→親代 金錢
    dy["give_care"] = pick("g24")                            # 親代→子代 家務天數
    dy["get_care"] = pick("g25")                             # 子代→親代 家務天數
    dy["tell"] = dy.k2p_k                                    # 子代自報：跟親代說心事 1總是→5從未

    # ---- 1. 覆蓋率：先看哪些變項在 dyad 裡問得到 -------------------------
    print("【覆蓋率】各候選因素在 997 組 dyad 裡的有效 N")
    cands = ["p_edu", "k_edu", "edu_up", "p_health", "k_health", "p_inc", "k_inc",
             "k_married", "k_nsib", "k_grandkid", "give_money", "get_money",
             "give_care", "get_care", "cohab", "met_k", "tel_k", "tell",
             "k_distress", "k_lonely"]
    for c in cands:
        print(f"  {c:12s} {dy[c].notna().sum():4d} / {len(dy)}")
    print("  ※ 管教要求：本波無回溯教養題（g39 只問有學齡子女者、k03z04 問的是現任家戶決策）\n")

    # ---- 2. 逐一單變項：每個因素單獨解釋得了多少溫差 ---------------------
    print("【單變項】gap ~ 該因素（各自最大樣本，戶層叢集 SE）")
    rows = []
    for c in cands:
        if dy[c].notna().sum() < 120:
            continue
        r, _ = ols("gap", [c], "hh", dy)
        rows.append((c, r.nobs, r.params[c], r.tvalues[c], r.pvalues[c], r.rsquared))
    tab = pd.DataFrame(rows, columns=["因素", "n", "b", "t", "p", "R2"])
    print(tab.sort_values("p").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    # ---- 3. 分層加入：同一子樣本比 ΔR² --------------------------------
    print("【分層】固定同一子樣本（全部變項皆非遺漏）逐層加入")
    blocks = {
        "M0 基準（性別＋年齡）": ["father", "p_age"],
        "M1 ＋社經": ["father", "p_age", "p_edu", "k_edu"],
        "M2 ＋生命歷程": ["father", "p_age", "p_edu", "k_edu", "k_married", "k_nsib", "k_grandkid"],
        "M3 ＋資源交換": ["father", "p_age", "p_edu", "k_edu", "k_married", "k_nsib", "k_grandkid",
                      "give_money", "get_money", "give_care", "get_care"],
        "M4 ＋接觸量": ["father", "p_age", "p_edu", "k_edu", "k_married", "k_nsib", "k_grandkid",
                     "give_money", "get_money", "give_care", "get_care", "cohab", "met_k", "tel_k"],
        "M5 ＋吐露": ["father", "p_age", "p_edu", "k_edu", "k_married", "k_nsib", "k_grandkid",
                    "give_money", "get_money", "give_care", "get_care", "cohab", "met_k", "tel_k", "tell"],
    }
    full = sorted({v for vs in blocks.values() for v in vs})
    sub = dy[["gap", "hh"] + full].dropna()
    print(f"  共同子樣本 n={len(sub)}（{sub.hh.nunique()} 戶）")
    prev = 0.0
    for tag, xs in blocks.items():
        r, _ = ols("gap", xs, "hh", sub)
        print(f"  {tag:18s} R²={r.rsquared:.3f}  ΔR²={r.rsquared - prev:+.3f}")
        prev = r.rsquared
    print("\n  M5 完整模型係數：")
    r5, _ = ols("gap", blocks["M5 ＋吐露"], "hh", sub)
    show("M5", r5)
    print()

    # ---- 4. 收入：覆蓋率低，單獨一段 -----------------------------------
    print("【收入】k24e 只覆蓋約一半 dyad，單獨跑並附同子樣本基準")
    for xs in (["father", "p_age"], ["father", "p_age", "p_inc"], ["father", "p_age", "p_inc", "k_inc"]):
        r, _ = ols("gap", xs, "hh", dy[["gap", "hh", "father", "p_age", "p_inc", "k_inc"]].dropna())
        show("+".join(xs), r)
    print()

    # ---- 5. 身心狀況：測得動但內生，單獨標示 ----------------------------
    print("【身心】子代身心與健康（⚠️ 是 APIM 的依變項，與溫差同源，只能當描述）")
    r, _ = ols("gap", ["father", "p_age", "k_health", "k_distress", "k_lonely", "p_health"], "hh", dy)
    show("gap ~ 身心", r)
    print()

    # ---- 6. 打在哪一側：同一因素對親代版 vs 子代版 ------------------------
    print("【拆側】同一因素分別解釋 親代評分 / 子代評分（看溫差從哪一邊來）")
    print(f"  {'因素':12s} {'→親代版 b(t)':>20s} {'→子代版 b(t)':>20s}")
    for c in ["father", "p_age", "p_edu", "k_edu", "tell", "met_k", "tel_k", "cohab",
              "get_money", "get_care", "k_married", "k_grandkid"]:
        base = dy[["aff_p", "aff_k", "hh", c]].dropna()
        a, _ = ols("aff_p", [c], "hh", base)
        b, _ = ols("aff_k", [c], "hh", base)
        print(f"  {c:12s} {a.params[c]:+8.3f} ({a.tvalues[c]:+5.2f})   "
              f"{b.params[c]:+8.3f} ({b.tvalues[c]:+5.2f})   n={len(base)}")


if __name__ == "__main__":
    main()   # 存檔用 shell：python3 psfd_factors.py | tee out/psfd/factors_output.txt
