# 圖 2 七因素：各自變項原始尺度、標準差、係數換回原始單位（重用 psfd_controls_robust.py 前兩段的設計矩陣）
# 跑法：cd research && uv run --with pandas --with pyreadstat --with statsmodels --with scipy python psfd_fig2_units.py
import os
src=open("psfd_controls_robust.py",encoding="utf-8").read().split("# ── 3.")[0]
ns={"__file__":os.path.abspath("psfd_controls_robust.py")}; exec(src,ns)
s,X,okA,RA=ns["s"],ns["X"],ns["okA"],ns["RA"]
s=s.copy(); s["同住"]=s["去年同住"]
print("口徑：圖2 RR2024 n=",int(okA.sum()),"｜z 用全樣本 mean/std（非估計樣本）")
for v in RA.index:
    x=s[v]; sd=x.std(); vals=sorted(x.dropna().unique())
    sh=f"{len(vals)}個值 {vals[:8]}{'…' if len(vals)>8 else ''}"
    print(f"{v:<10} SD={sd:7.3f} 範圍[{x.min():g},{x.max():g}] {sh}\n   b/SD={RA.b[v]:+.3f}  每原始單位={RA.b[v]/sd:+.4f}  q={RA.q[v]:.3f}")
