# -*- coding: utf-8 -*-
"""PSFD「代間溫差」提案的唯一口徑真相（RR2020 / RR2022 / RR2024 三波）。

任何腳本要用親子 dyad，一律 from psfd_engine import load, build_dyads, build_panel, SPEC，
不得自行重寫配對規則、變項名對照、量表反向計分、遺漏碼處理或跨波串接鍵。
自我檢核：python psfd_engine.py
"""
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STEM = {2020: "psfd_rr2020_v202208", 2022: "psfd_rr2022_v202403", 2024: "psfd_rr2024_v202603"}
CSV = {w: ROOT / f"家庭動態調查：{w}追蹤問卷" / f"{s}_csv.csv" for w, s in STEM.items()}
SAV = {w: ROOT / f"家庭動態調查：{w}追蹤問卷" / f"{s}_spss.sav" for w, s in STEM.items()}

SPEC = dict(
    source="家庭動態調查追蹤問卷 RR2020(v202208) / RR2022(v202403) / RR2024(v202603)；單波分析預設 2024",
    missing="負數一律遺漏；**2020 用正數遺漏碼**（0 跳答／6,96 不知道／8,98 拒答／9,99 遺漏值），"
            "遺漏碼一律由 .sav 選項標籤（跳答/不知道/拒答/遺漏值/不合理值）自動推出，不手抄",
    varmap="2020 的題目在 B 大題（親代 b12*／子代 b15b,b16*），2022+ 在 G/H 題組；對照表見 V",
    panel_key="跨波唯一人別＝(x01d 戶號, x01e 戶內序號)。x01 編碼規則不同"
              "（2020/2022=x01d*10+x01e，2024=x01d*100+x01e*10），**不得直接 join x01**",
    panel_dyad="dyad 跨波身分＝ppid|kpid；每波各自重跑配對再取交集（slot c{k} 會換波重排，不可沿用）",
    pairing="戶內以 子代(性別, 出生年) = 親代子女題組(性別, 出生年) 精確配對；x01e ≠ c{k}，不得直接對映",
    kid_view="親代性別 a01==1 → 子代用 ??f1（父親）；==2 → ??m1（母親）",
    recode="距離 8=同住 → 0（最近）；通話 0=去年同住／跳答 → NaN",
    happy="2022+ 是 a06；**2020 的 a06 是「第一份全職工作年份」，快樂題為 w01a**（1–7，高=快樂）",
    lonely="六題 1–5（倚靠,空虛,陪伴,信賴,排擠,親近），第 1/4/6 題反向(6−x)，取平均，高=孤獨；三波皆有",
    distress="m02 十題 1–4，第 4/5/6 題反向(5−x)，高=困擾；**2020 不可比**（h03 問「平時」非「過去一個月」"
             "且缺憂鬱/沮喪/睡眠/寂寞四題）→ 2020 一律 NaN，困擾只在 2022–2024 分析",
    gap="IWG = 親代版感情 − 子代版感情，1–5 量表，單位＝級",
    se="dyad 巢套於戶 → 一律戶層叢集穩健 SE（三波釋出皆無抽樣權重）",
)

# 同一題的變項名：p=親代版（對子女 c{k}），k=子代版（對父/母 {s}=f1/m1）
_G = dict(aff="g28c{k}", dis="g32c{k}", tel="g34c{k}", met="g35c{k}", p2k="g26c{k}", k2p="g27c{k}")
_H = dict(aff="h19{s}", dis="h11{s}", tel="h13{s}", met="h14{s}", p2k="h17{s}", k2p="h18{s}")
_LATER = dict(byear="a02z01", happy="a06", kid_sex="g07c{k}", kid_byear="g08z01c{k}",
              lonely=[f"m01z0{i}" for i in range(1, 7)],
              distress=[f"m02z{i:02d}" for i in range(1, 11)], p=_G, k=_H)
V = {
    2020: dict(byear="a02a", happy="w01a", kid_sex="b12bc{k}", kid_byear="b12c01z01c{k}",
               lonely=[f"h13{c}" for c in "abcdef"], distress=None,
               p=dict(aff="b12m04c{k}", dis="b12p01c{k}", tel="b12qc{k}", met="b12sc{k}",
                      p2k="b12m03c{k}", k2p="b12n03c{k}"),
               k=dict(aff="b16h{s}", dis="b15b{s}", tel="b16a{s}", met="b16b{s}",
                      p2k="b16e{s}", k2p="b16f{s}")),
    2022: _LATER,
    2024: _LATER,
}
PAIRS = tuple(_G)  # aff dis tel met p2k k2p


def _missing_codes(wave):
    """從 .sav 選項標籤推出每個變項的遺漏碼。2020 的遺漏碼是正數，mask(<0) 抓不到。"""
    import pyreadstat
    _, m = pyreadstat.read_sav(SAV[wave], metadataonly=True)
    bad = ("跳答", "不知道", "拒答", "遺漏值", "不合理值")
    out = {}
    for v, vl in m.variable_value_labels.items():
        codes = {c for c, lab in vl.items() if any(b in str(lab) for b in bad)}
        if codes:
            out[v] = codes
    return out


def load(wave=2024):
    d = pd.read_csv(CSV[wave], low_memory=False)
    num = d.select_dtypes("number").columns
    d[num] = d[num].mask(d[num] < 0)
    grp = defaultdict(list)                      # 相同遺漏碼的欄位一起 mask，省掉逐欄 insert
    for v, codes in _missing_codes(wave).items():
        if v in d.columns:
            grp[frozenset(codes)].append(v)
    for codes, cols in grp.items():
        d[cols] = d[cols].mask(d[cols].isin(list(codes)))
    # 跨波唯一人別：x01 編碼規則各波不同，只有 (戶號, 戶內序號) 串得起來
    return d.assign(pid=d.x01d.astype("Int64").astype(str) + "-" + d.x01e.astype("Int64").astype(str), wave=wave)


def _wellbeing(r, wave):
    c = V[wave]
    lone = pd.concat([(6 - r[v] if i in (0, 3, 5) else r[v]) for i, v in enumerate(c["lonely"])], axis=1)
    out = dict(happy=r[c["happy"]], lonely=lone.mean(axis=1),
               age=r["x02"] - (r[c["byear"]] + 1911), sex=r["a01"], x01b=r["x01b"])
    if c["distress"]:
        dist = pd.concat([(5 - r[v] if i in (3, 4, 5) else r[v]) for i, v in enumerate(c["distress"])], axis=1)
        out["distress"] = dist.mean(axis=1)
    else:
        out["distress"] = np.nan  # 2020 的 h03 與 m02 不可比，見 SPEC
    return pd.DataFrame(out)


def build_dyads(d):
    """回傳 (dyad DataFrame, 主樣本親代 DataFrame)。每列一組親子，欄位 *_p=親代版、*_k=子代版。"""
    wave = int(d.wave.iloc[0])
    c = V[wave]
    par = d[d.x01a == 1]
    assert par.x01d.is_unique, "每戶應只有一位主樣本親代"
    par = par.set_index("x01d")
    kid = d[d.x01a == 3]

    rows = []
    for hh, grp in kid.groupby("x01d"):
        if hh not in par.index:
            continue
        p = par.loc[hh]
        slots = {k: (p[c["kid_sex"].format(k=k)], p[c["kid_byear"].format(k=k)]) for k in range(1, 7)
                 if pd.notna(p[c["kid_sex"].format(k=k)]) and pd.notna(p[c["kid_byear"].format(k=k)])}
        used = set()
        for _, ch in grp.iterrows():
            hit = [k for k, v in slots.items() if v == (ch["a01"], ch[c["byear"]]) and k not in used]
            if hit:
                used.add(hit[0])
                rows.append((hh, ch["x01"], hit[0]))
    link = pd.DataFrame(rows, columns=["hh", "kid", "slot"])

    wp = _wellbeing(par.loc[link.hh].reset_index(drop=True), wave).add_prefix("p_")
    kd = d.set_index("x01").loc[link.kid].reset_index(drop=True)
    wk = _wellbeing(kd, wave).add_prefix("k_")
    out = pd.concat([link, wp, wk], axis=1)
    suf = np.where(out.p_sex == 1, "f1", "m1")
    for name in PAIRS:
        out[f"{name}_p"] = [par.at[hh, c["p"][name].format(k=k)] for hh, k in zip(out.hh, out.slot)]
        out[f"{name}_k"] = [kd.at[i, c["k"][name].format(s=s)] for i, s in enumerate(suf)]
    # 手足安慰劑用：親代對「同戶其他所有子女」的感情評分平均（不限有受訪者）
    sib = [[c["p"]["aff"].format(k=j) for j in range(1, 7) if j != k] for k in out.slot]
    out["aff_p_sib"] = [par.loc[hh, cols].astype(float).mean() for hh, cols in zip(out.hh, sib)]
    for s in ("p", "k"):
        out[f"dis_{s}"] = out[f"dis_{s}"].replace(8, 0)
        out[f"tel_{s}"] = out[f"tel_{s}"].mask(out[f"tel_{s}"] == 0)
    out["cohab"] = (out.dis_k == 0).astype(float).where(out.dis_k.notna())
    out["father"] = (out.p_sex == 1).astype(float)
    out["son"] = (out.k_sex == 1).astype(float)
    out["gap"] = out.aff_p - out.aff_k
    out["p_mode"], out["k_mode"] = par.loc[out.hh, "x01f"].values, kd["x01f"].values
    out["ppid"], out["kpid"] = par.loc[out.hh, "pid"].values, kd["pid"].values
    out["dyad_id"] = out.ppid + "|" + out.kpid
    assert out.dyad_id.is_unique, "dyad 跨波身分必須唯一"
    return out, par


def build_panel(waves=(2022, 2024)):
    """每一波都配得上的親子 dyad。每列一組親子，欄位加 _20/_22/_24 後綴，index=dyad_id。"""
    dys = {w: build_dyads(load(w))[0].set_index("dyad_id") for w in waves}
    ids = sorted(set.intersection(*(set(x.index) for x in dys.values())))
    wide = pd.concat({w % 100: dys[w].loc[ids] for w in waves}, axis=1)
    return wide.set_axis([f"{col}_{w:02d}" for w, col in wide.columns], axis=1)


def header(dy, title="PSFD RR2024 代間溫差"):
    print("=" * 78)
    print(title + " · 口徑")
    for k, v in SPEC.items():
        print(f"  {k:9s} {v}")
    hh = dy.hh if "hh" in dy else dy.filter(regex=r"^hh_").iloc[:, -1]
    print(f"  dyad      {len(dy)} 組／{hh.nunique()} 戶")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    # ponytail: 跨波串接鍵＋遺漏碼＋變項對照的最小哨兵，任何一條變了就當場 FAIL
    D = {w: load(w) for w in STEM}
    for w, r in [(2020, 10), (2022, 10), (2024, 100)]:
        e = D[w].x01e * (r // 10)
        assert (D[w].x01 == D[w].x01d * r + e).all(), f"{w} x01 編碼規則變了"
    cols = ["a01", "x02", "x01a", "x01b"]
    for w1, w2 in [(2020, 2022), (2022, 2024)]:
        a, b = D[w1], D[w2]
        m = a.set_index("pid")[cols + [V[w1]["byear"]]].join(
            b.set_index("pid")[cols + [V[w2]["byear"]]], lsuffix="_a", rsuffix="_b", how="inner")
        assert len(m) > 5000, f"{w1}→{w2} 只串上 {len(m)} 人，key 可能錯了"
        assert (m.a01_a == m.a01_b).all(), f"{w1}→{w2} 串到的人性別不一致"
        by = m[[c for c in m.columns if c.startswith("a02")]]
        assert (by.iloc[:, 0] == by.iloc[:, 1]).mean() > 0.99, f"{w1}→{w2} 出生年不一致"
        assert (m.x02_b - m.x02_a).eq(2).all(), f"{w1}→{w2} 訪問年差值不是 2 年"
        assert (m.x01a_a == m.x01a_b).all() and (m.x01b_a == m.x01b_b).all(), "身分/世代群不一致"
    for w in STEM:  # 遺漏碼沒清乾淨會在這裡爆（2020 的 6/8/9 會讓量表超出 1–5）
        dy, _ = build_dyads(D[w])
        for v in ("aff_p", "aff_k"):
            assert dy[v].min() >= 1 and dy[v].max() <= 5, f"{w} {v} 超出 1–5，遺漏碼沒清掉"
        assert dy.p_happy.max() <= 7 and dy.k_lonely.max() <= 5, f"{w} 福祉量表超出範圍"
        assert dy.p_distress.isna().all() if w == 2020 else dy.p_distress.notna().any()
        print(f"  {w}：dyad {len(dy):4d} 組／{dy.hh.nunique()} 戶  gap {dy.gap.mean():+.3f}")
    p3 = build_panel(tuple(STEM))
    assert p3.index.is_unique and len(p3) > 400
    print(f"OK 三波口徑檢核通過：三波都配得上的 dyad {len(p3)} 組")
