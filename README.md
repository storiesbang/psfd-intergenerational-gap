# 成年親子的「代間溫差」：PSFD 三波統計分析

2026 資料科學漫步參賽作品《誰說我們關係好？成年親子的代間溫差》的統計程式碼。

**代間溫差**＝同一對親子中，親代評「和這個孩子感情好不好」減去子代評「和父／母感情好不好」（1–5 分）。
正值代表父母覺得關係比孩子覺得的好。程式從家庭動態調查（PSFD）把親代問卷與成年子女問卷配成親子對，
描述溫差的大小與方向、找出跟溫差有關的因素，並用三波追蹤做固定效果檢驗。

## 資料

使用中研院〈臺灣家庭動態調查〉追蹤問卷 RR2020 (v202208)、RR2022 (v202403)、RR2024 (v202603)。
原始資料須自行向 [學術調查研究資料庫 SRDA](https://srda.sinica.edu.tw/) 申請，**本 repo 不含任何資料**。
下載後依下列結構放在 repo 根目錄：

```
家庭動態調查：2020追蹤問卷/psfd_rr2020_v202208_csv.csv  (+ _spss.sav)
家庭動態調查：2022追蹤問卷/psfd_rr2022_v202403_csv.csv  (+ _spss.sav)
家庭動態調查：2024追蹤問卷/psfd_rr2024_v202603_csv.csv  (+ _spss.sav)
```

`.sav` 用來讀選項標籤、自動推出各變項的遺漏碼（2020 的遺漏碼是正數，不能只濾負值）。

## 程式結構

### `research/psfd_engine.py`：唯一口徑

所有分析共用的資料處理，其他腳本只 import、不重寫：

- `load(wave)`：讀資料並處理遺漏碼
- `build_dyads(d)`：戶內以「子代(性別, 出生年) = 親代子女題組(性別, 出生年)」配對親子
- `build_panel(waves)`：三波串接；跨波人別鍵＝(戶號 `x01d`, 戶內序號 `x01e`)，不可直接 join `x01`
- `V`：三波變項名對照（2020 題目在 B 大題，2022 後在 G/H 題組）
- `SPEC`：上述口徑的文字規格；`python psfd_engine.py` 會跑自我檢核

### `research/`：分析腳本

| 腳本 | 內容 |
|---|---|
| `psfd_core.py` | 橫斷面主結果：兩代評分相關（客觀題 vs 主觀題）、溫差分布、符號檢定、迴歸與穩健性／反證 |
| `psfd_panel.py` | 三波追蹤：溫差是否隨時間擴大、雙向固定效果（說心事變化 → 兩代評分變化） |
| `psfd_b3_reporter.py` | 換報告人 2×2 固定效果（子報／親報吐露 × 親代／子代評分） |
| `psfd_fe_both_reports.py` | 固定效果分別用子女自陳與父母回報的說心事 |
| `psfd_factors.py` | 社經背景、相處時間、管教、子女身心狀況與溫差的關係 |
| `psfd_controls_robust.py` | 七因素模型（圖 2）加控制變項的穩健性檢查 |
| `psfd_call_alone_vs_joint.py` | 通話頻率單獨放 vs 合併模型 |
| `psfd_aligned_diff.py` | 報告人對齊比較（親評←親報 vs 子評←子報）的差異檢定 |
| `psfd_confide_groups.py` | 各說心事組的溫差是否異於 0（BH–FDR） |
| `psfd_father_confide.py` | 子女對父親 vs 母親的說心事頻率與評分 |
| `psfd_cohab_vs_daily.py` | 去年同住 vs 幾乎每天通話的比較 |
| `psfd_direction_check.py` | 頻率題方向（大＝多／大＝少）核對 |
| `psfd_fig2_units.py` | 圖 2 係數換回原始單位 |
| `psfd_missing_codes.py` | 三波遺漏碼盤點 |
| `psfd_varlist.py` | RR2024 變項清單與分析用到哪些 |
| `psfd_v2_verify.py` | 作品中數字的逐項重算核對 |
| `psfd_v3_figs.py`, `psfd_fig3_b3.py` | 作品圖表 |

### 根目錄：組員的圖 2 管線

`match.py → ext.py → coef.py → joint_7.py / joint_8.py → jointplot_q.py`，`supp.py`、`supp2.py` 為補充檢定。
這組腳本是組員獨立寫的版本（七因素合併 OLS、標準化、戶層叢集 SE、BH–FDR q 值），檔案路徑 `U` 指向原作者機器。
`research/psfd_controls_robust.py` 會原封不動地 exec 這些腳本、只把 `U` 換成本機資料路徑，
用來確認兩套獨立實作得到同一結果。

## 統計方法共通設定

- 親子對巢套於戶 → 一律用**戶層叢集穩健標準誤**（PSFD 三波釋出皆無抽樣權重）
- 多重比較用 Benjamini–Hochberg FDR，以 q 值判斷
- 頻率題（說心事、通話、見面）原始編碼「1＝總是…5＝從未」，分析時反向為「大＝多」

## 執行

```bash
cd research
mkdir -p out/psfd
uv run --with pandas --with pyreadstat --with statsmodels --with scipy --with linearmodels --with matplotlib \
    python psfd_core.py
```

每支腳本開頭 docstring 有各自的跑法；輸出寫到 `research/out/psfd/`（不進版控）。
