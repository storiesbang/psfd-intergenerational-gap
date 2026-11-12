# 初賽作品圖 3：讀 out/psfd/b3_output.txt 的四格係數畫圖（不重算）
# 跑法：cd research && uv run --with matplotlib python psfd_fig3_b3.py → out/psfd/fig3_fe_2x2.png
import re, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
txt = open("out/psfd/b3_output.txt", encoding="utf-8").read()
rows = re.findall(r"^\s+(\S+→\S+)\s+(\S+)\s+(\d+)\s+([-+]\d\.\d+)\s+(\d\.\d+)\s+([-+]\d\.\d+)\s+(\S+)", txt, re.M)
assert len(rows) == 4, rows
rw = float(re.search(r"組內變化 r\(Δ子報, Δ親報\)：([\d.]+)", txt).group(1))
import glob
from matplotlib import font_manager
# ponytail: PingFang 在 AssetsV2 底下，matplotlib 不一定掃得到 → 直接載檔
for f in glob.glob("/System/Library/AssetsV2/*/*/AssetData/PingFang.ttc"):
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["PingFang TC", "PingFang HK", "Heiti TC"]  # 新版 matplotlib 讀得到 TC，舊版只讀到 TTC 第一個字面 HK
fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=200)
for i, (nm, kind, n, b, se, t, p) in enumerate(rows):
    b, se, t, p = float(b), float(se), float(t), float(p)
    yy = 3 - i
    col = "#3b6aa0" if p < .05 else "#9aa0a6"
    ax.errorbar(b, yy, xerr=1.96 * se, fmt="o", color=col, ms=9, capsize=5, lw=2.2,
                mfc=col if p < .05 else "white", mew=2)
    ps = "p < .001" if p < .001 else f"p = {p:.2f}".replace("0.", ".")
    ax.text(0.045, yy, f"{'顯著' if p < .05 else '不顯著'} (t = {t:.2f}, {ps})", va="center", fontsize=9, color=col)
ax.set_yticks(range(4)); ax.set_yticklabels([f"{r[0]}\n" + (f"（{r[1]}）" if "（" not in r[1] else r[1].replace("（", "・").replace("）", "").join("（）")) for r in rows][::-1], fontsize=9.5)
ax.axvline(0, color="#333", lw=1.2); ax.set_xlim(-0.2, 0.22); ax.set_ylim(-0.6, 3.6)
ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.set_xlabel("雙向固定效果迴歸係數（戶層叢集穩健標準誤）", fontsize=9.5)
ax.set_title("吐露變少的那一年：只有子女自陳的那一格在動\n三波雙向固定效果 · 2,103 筆觀測 / 701 對", fontsize=10.5)
fig.text(0.5, -0.16, f"橫線為 95% 信賴區間。吐露 1＝最常說，係數為負＝吐露增加、評分升高。\n兩份報告的吐露變化相關僅 r = {rw:.2f}。資料：PSFD RR2020×RR2022×RR2024",
         ha="center", fontsize=7.5, color="#777")
fig.savefig("out/psfd/fig3_fe_2x2.png", bbox_inches="tight", facecolor="white")
