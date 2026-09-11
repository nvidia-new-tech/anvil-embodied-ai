# `tools/`

跟主要 pipeline（資料轉換 → 訓練 → 評估 → 推論）無關、但實際會用到的輔助工具。

---

## Benchmark 計分工具（`tools/benchmark_web/`）

**要解決的問題**：每換一顆 checkpoint 就要手動計次「10 次抓幾次成功」，資料只留在腦子裡或聊天記錄，跨模型比較很痛苦。

**是什麼**：一個純 Python 標準庫寫的本機網頁（不需要 `pip install` 任何東西），會自動讀出目前 `.env` / 執行中容器所用的 `MODEL_PATH` 與 `CONFIG_FILE`，讓你邊測邊點選成功/失敗，存成 JSON。

**固定測試協定**（跟大家對過的版本）：
- 三種包裹：小黃包裹 / 小白包裹 / 大白包裹
- 每種各測 10 筆，拆成兩組各 5 筆：
  - 隨機放置干擾物（桌上還有其他包裹，位置隨機）
  - 無干擾物
- 失敗要選原因：approach 深度不夠 / 夾取後翻的不夠高 / 其他（可打字說明）
- 手臂測試間**不回原點**，連續執行下一筆 —— 這是死條件，工具會固定顯示，不用每次手打

**怎麼啟動**：
```bash
# 平常方式：inference 一起跑的時候會自動帶起來
MODEL_PATH=/path/to/checkpoint ./scripts/run_inference.sh up
# 畫面上會印出：Benchmark tool: http://127.0.0.1:8777

# 不想要它自動開就加這個
./scripts/run_inference.sh --no-benchmark up

# 也可以完全獨立跑，不用啟動 inference
python3 tools/benchmark_web/server.py            # 預設 port 8777
python3 tools/benchmark_web/server.py --port 9000
```

**資料存在哪**：`tools/benchmark_web/results/pack.json`（目前只給 flip-pack 任務用，所以檔名叫 pack）。歷史紀錄可以在網頁下方直接看到每個 checkpoint 的成功率、失敗原因分布，也能「載入」回來繼續編輯或「刪除」。

---

## Model zoo 盤點工具（`tools/model_zoo_index.py`）

```bash
./tools/model_zoo_index.py            # 直接印出目前所有 checkpoint 的表格
./tools/model_zoo_index.py --write    # 寫成 model_zoo/MODEL_ZOO.md（未進版控，本機自己看）
./tools/model_zoo_index.py --json     # 給程式用的原始資料
```
它會自動列出：policy 類型、訓練步數、action_type、大小、修改時間；還會用權重檔案的 hash 抓出「內容其實一樣的重複資料夾」，以及 `training_state/`、空資料夾這些可以清掉的東西——不用每次都手動 `du -sh` 一個個看。
