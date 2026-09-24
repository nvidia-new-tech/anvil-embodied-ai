# anvil-embodied-ai

推論與訓練側。機器人側在 `../anvil-loader`。完整交接文件：`../HANDOFF.md`。

## 先讀這個

**這套系統的失敗幾乎都是靜默的。** 沒有例外會跳出來，日誌看起來正常，但東西
不動或動錯。不要靠讀日誌找問題，去檢查具體項目 —— 見 `workcell-doctor` skill
（在 `../anvil-loader/.claude/skills/`）。

## 動手臂之前

推論會驅動真實手臂。啟動前**必須**先說明並確認周圍淨空、人在急停旁。

`docker compose up` 起 loader 會**自動 homing，手臂會動**。

`docker compose down` **不會**放下手臂 —— `pre_stop.sh` 只送分析事件。馬達通電
就一直用扭力鎖位置，這時候斷電手臂會掉。順序是：停推論 → Web UI 下 dehome →
才 down。

## 兩個常踩的坑

**1. 用 `./scripts/run_inference.sh`，不要 `docker compose up`。**
monitor 是 `monitor` compose profile 下的獨立服務，只有腳本會注入
`--profile monitor`。`MONITOR_ENABLE=true` 單獨設，推論會跑但 CSV 是空的。

**2. `preflight_checkpoint.py` 分不出 EE 和關節空間的 checkpoint。**
兩者都是 8 維、`config.json` 沒有 action 名稱，所以任何組合都回報
`All checks passed`。判別看 normalizer 的夾爪維度：`[-0.003, 0.05]` 在
**dim 0** 是關節空間，在 **dim 7** 是 EE。見 `checkpoint-intake` skill。

## 切換推論模式

只有兩個變數不同，**不用改 `.env`**（shell 變數優先）：

```bash
MODEL_PATH=$PWD/model_zoo/<ckpt> \
CONFIG_FILE=./configs/lerobot_control/shapes/<shape>.yaml \
CONTROL_FREQ=30.0 ./scripts/run_inference.sh --monitor-enable up
```

## 這台機器

- 對外頻寬約 **4 Mbit/s**，大下載以小時計
- HuggingFace 下載**必須設 `HF_HUB_DISABLE_XET=1`**（Xet 後端只有 1 KB/s）
- GPU 是 sm_120（Blackwell），**torch 必須 cu128 以上**
- `:latest` 映像**不能刪**，`:ee-space` 從它 commit 而來且共用層；那個 tag 從
  沒推上 ghcr，`docker compose pull` 會失敗

## Skills

`.claude/skills/` 裡有 `checkpoint-intake`、`run-inference`。
