# -*- coding: utf-8 -*-
"""PSFD RR2024 全部 1,353 變項清單 ＋ 代間溫差分析用了哪些。
輸出 out/psfd/變項清單.csv（每變項一列：題組／標籤／有效 N／是否使用／角色）。
跑法：cd research && uv run --with pandas python psfd_varlist.py
"""
import json
from pathlib import Path
import pandas as pd
from psfd_engine import load, PAIRS

OUT = Path(__file__).resolve().parent / "out" / "psfd"
SEG = dict(x="基本資料", a="A 生活與工作狀況", b="B 教育", c="C 工作", d="D 感情婚姻與伴侶工作",
           e="E 家庭價值與態度", f="F 生活經驗與居住安排", g="G 子女題組", h="H 父母題組",
           j="J 兄弟姊妹題組", k="K 家庭決策與收支", l="L 老年題組", m="M 個人題組", p="P 加掛題組")

KIDS = range(1, 7)
USED = {}
USED.update({v: "配對鍵" for v in ["x01", "x01a", "x01d"]})
USED.update({f"g07c{k}": "配對比對（子女性別）" for k in KIDS})
USED.update({f"g08z01c{k}": "配對比對（子女出生年）" for k in KIDS})
USED.update({"a01": "配對比對＋親代性別", "a02z01": "配對比對＋年齡", "x02": "年齡（訪問年）", "x01f": "穩健⑦ 訪問模式"})
ROLE = dict(aff="主結果 感情", dis="客觀對照 距離", tel="客觀對照 通話", met="客觀對照 見面",
            p2k="說心事（親→子）", k2p="說心事（子→親）")
for name, (g, h) in PAIRS.items():
    USED.update({f"{g}c{k}": f"親代版 {ROLE[name]}" for k in KIDS})
    USED.update({f"{h}{s}": f"子代版 {ROLE[name]}" for s in ("f1", "m1")})
USED.update({"a06": "福祉 快樂"})
USED.update({f"m01z0{i}": "福祉 孤獨感" for i in range(1, 7)})
USED.update({f"m02z{i:02d}": "福祉 心理困擾" for i in range(1, 11)})
TRAP = {"x01e": "刻意不用：子女編號≠G 題組序號，直接對映會配錯"}

d = load()   # 負數遺漏碼已遮成 NaN
lab = json.load(open(OUT / "psfd_meta.json"))["labels"]
main, kid = d.x01a == 1, d.x01a == 3
rows = []
for v in d.columns:
    seg = v[0] if v[0] in SEG else "其他"
    rows.append(dict(變項=v, 題組=SEG.get(seg, "其他（訪問紀錄等）"), 標籤=lab.get(v) or "",
                     有效N_全體=int(d[v].notna().sum()), 有效N_主樣本=int(d.loc[main, v].notna().sum()),
                     有效N_子女樣本=int(d.loc[kid, v].notna().sum()),
                     使用="是" if v in USED else ("刻意不用" if v in TRAP else ""),
                     角色=USED.get(v) or TRAP.get(v, "")))
t = pd.DataFrame(rows)
assert set(USED) <= set(d.columns), set(USED) - set(d.columns)
t.to_csv(OUT / "變項清單.csv", index=False, encoding="utf-8-sig")

print(f"總變項 {len(t)}；使用 {(t.使用=='是').sum()}；刻意不用 {(t.使用=='刻意不用').sum()}\n")
s = t.groupby("題組", sort=False).agg(變項數=("變項", "size"), 使用=("使用", lambda x: (x == "是").sum()),
                                     有效N中位數=("有效N_全體", "median"))
print(s.to_string(), "\n")
u = t[t.使用 != ""]
for role, g in u.groupby("角色", sort=False):
    print(f"{role:22s} {len(g):2d} 個  {' '.join(g.變項)}")
    print(f"{'':22s}      例：{g.標籤.iloc[0][:60]}（有效N 主{g.有效N_主樣本.max()}／子{g.有效N_子女樣本.max()}）")
