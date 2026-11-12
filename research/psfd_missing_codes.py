# -*- coding: utf-8 -*-
"""三波遺漏碼盤點：分析用到的變項，各波哪些值被 engine 當成遺漏。
遺漏碼的判定與 psfd_engine._missing_codes 同一套（.sav 選項標籤），這裡只負責列表。
跑法：uv run --with pandas --with pyreadstat python psfd_missing_codes.py | tee out/psfd/missing_codes_output.txt
"""
from collections import Counter
import pandas as pd, pyreadstat
from psfd_engine import SAV, CSV, V, PAIRS, _missing_codes

BAD = ("跳答", "不知道", "拒答", "遺漏值", "不合理值")
LAB = dict(aff="感情好不好", dis="居住距離", tel="通話頻率", met="見面頻率", p2k="親→子說心事", k2p="子→親說心事")

print("=" * 78)
print("PSFD 遺漏碼盤點 · 口徑：遺漏碼＝.sav 選項標籤含", "／".join(BAD), "；另外負數一律遺漏")
print("變項：親代版 c1（子女欄位第 1 欄，c2–c6 同碼）、子代版 f1（對父親，m1 同碼）")
print("=" * 78)

for w in V:
    _, meta = pyreadstat.read_sav(SAV[w], metadataonly=True)
    vl, miss = meta.variable_value_labels, _missing_codes(w)
    d = pd.read_csv(CSV[w], low_memory=False)
    print(f"\n【{w}】資料檔 {SAV[w].name}")
    # 全檔：每個遺漏碼數值 → 標籤（取最常見的寫法）
    allc = Counter((c, lab) for v, labs in vl.items() for c, lab in labs.items() if any(b in str(lab) for b in BAD))
    pos = sorted({c for (c, _) in allc if c >= 0}); neg = sorted({c for (c, _) in allc if c < 0})
    print(f"  全檔遺漏碼數值：正數 {pos or '無'}｜負數 {neg or '無'}")
    for c in pos + neg:
        labs = Counter({lab: n for (cc, lab), n in allc.items() if cc == c}).most_common(3)
        print(f"    {c:>4}: " + "；".join(f"{lab}（{n} 個變項）" for lab, n in labs))
    # 分析用變項
    print("  分析用變項（值：標籤，★＝engine 視為遺漏；n＝實際出現筆數）")
    for side, s in (("親代版", dict(k=1)), ("子代版", dict(s="f1"))):
        for name in PAIRS:
            v = V[w]["p" if side == "親代版" else "k"][name].format(**s)
            cnt = d[v].value_counts()
            cells = [f"{'★' if (c in miss.get(v, ()) or c < 0) else ''}{c:g}:{lab}(n={cnt.get(c, 0)})"
                     for c, lab in sorted(vl.get(v, {}).items())]
            extra = sorted(set(cnt.index) - set(vl.get(v, {})))
            print(f"    {side} {LAB[name]:6s} {v:11s} " + "｜".join(cells) + (f"｜無標籤值 {extra}" if extra else ""))
