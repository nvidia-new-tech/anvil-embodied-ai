# Checkpoint 驗收流程

新的權重檔案送到手上、還沒拿去驅動手臂之前，該做的事。

**為什麼需要這份流程**：checkpoint 跟 config 之間幾乎每一種對不上的方式都會**靜默失敗**——不會報錯。少一支相機會變成灰色畫面，多一支會讓啟動永遠卡住，錯誤的 action 切片會發布全零，壞掉的權重照樣能載入並產生垃圾輸出。所以我們要對照 checkpoint 自己的 metadata 來驗證，而不是相信資料夾名稱、備註，或是上次能跑的 config。

---

## 快速流程

```bash
# 1. 這是什麼 shape？該用哪個 config？
./.venv/bin/python scripts/suggest_shape_config.py <checkpoint_dir>

# 2. 確認選擇是對的（權重掃描 + 完整合約檢查）
./.venv/bin/python scripts/preflight_checkpoint.py <checkpoint_dir> \
  --config configs/lerobot_control/shapes/<picked>.yaml

# 3. 真的跑一次模型 —— 實際 forward pass，不用機器人（見第 5 點）
docker run --rm --gpus all -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
  -v ~/.cache/huggingface:/hf:ro -v "$PWD":/workspace/repo:ro \
  -v "$PWD/model_zoo":/model_zoo:ro -w /workspace/repo \
  ghcr.io/anvil-robotics/lerobot-inference:latest \
  python3 scripts/offline_inference_test.py /model_zoo/<ckpt> --config <picked>

# 4. 把 .env 指向這兩個檔案，然後啟動
```

第 2 步一定要 exit 0。如果不是，先修好它回報的問題——不要硬著頭皮啟動。

---

## Shape configs

`configs/lerobot_control/shapes/` 裡每個檔案對應一種**機器人 shape**，命名規則是
`<arms>arm_<cameras>cam[_variant]`。它們跟任務無關、可以重複使用：挑一個跟你的
checkpoint 相符的，不用另外編輯。

| Config | State | Cameras | 發布哪隻手 |
|---|---|---|---|
| `1arm_2cam` | 8 | chest, wrist_r | right `[0:8]` |
| `1arm_3cam` | 8 | chest, wrist_l, wrist_r | right `[0:8]` |
| `1arm_3cam_waist` | 8 | chest, waist, wrist_r | right `[0:8]` |
| `2arm_3cam` | 16 | chest, wrist_l, wrist_r | left `[0:8]`, right `[8:16]` |
| `2arm_3cam_rightonly` | 16 | chest, wrist_l, wrist_r | right `[8:16]` + left 釘住 |
| `2arm_4cam` | 16 | chest, waist, wrist_l, wrist_r | left `[0:8]`, right `[8:16]` |
| `2arm_4cam_rightonly` | 16 | chest, waist, wrist_l, wrist_r | right `[8:16]` + left 釘住 |

這些檔案刻意**省略 `model.task_description`**，所以會從 checkpoint 自己的
`anvil_config.json`（`inference_node.py:231-234`）讀取。這就是為什麼一份 config
可以服務多個任務。

> **不是每個 checkpoint 都有這欄位。** 截至 2026-09-11，`model_zoo/` 裡 27 個
> checkpoint 中有 11 個是在沒有 `task_description` 的情況下訓練的，包括
> `smolvla-flip/{005000,007500,010000,020000}`、
> `pi05-flip-pack-single-arm-20260827`（全部步數）、以及
> `pi05-flip-pack-combined-20260904/005000`。
>
> **節點並不會因此拒絕啟動 —— 它只會印一則警告**
> （`inference_node.py:413`），然後帶著空字串繼續跑下去。對 ACT/Diffusion
> 來說沒有影響；但對 VLA 來說，這代表 `_preprocess_vla_observation` 永遠不會
> 設定 `batch["task"]`，整個 run 會在推論途中才死掉，丟出
> `KeyError: 'task'`（在 `tokenizer_processor.py` 裡）。Preflight 也抓不到
> 這個問題——它只會警告「checkpoint 裡沒有 task_description 可以比對」，
> 不會讓檢查失敗。**任何 VLA checkpoint 遇到這個 preflight 警告都要當成阻擋性錯誤處理，不是純參考。**
>
> 遇到這種情況，把 shape config 裡那行取消註解，填上一字不差的字串：
>
> ```yaml
> model:
>   task_description: "Flip the package upside down."
> ```
>
> `suggest_shape_config.py` 遇到這種情況會回報 `task_description : MISSING`，
> `offline_inference_test.py` 也會用同樣的訊息失敗，而不是丟出 traceback。
> 字串可以從同一批訓練的其他 checkpoint 拿（`smolvla-flip/001500` 和
> `002500` 有存），或是從訓練指令裡找。

沒有 shape 對得上？複製最接近的那份，改 `arm_mapping` 和 `cameras.mapping`，
依照它的 shape 命名，然後在上面的表格加一列。

舊的、以任務命名的 config（`inference_flip.yaml`、`inference_default.yaml`、
……）還是能用、沒有變動。但新的東西一律優先用 shape config。

### 怎麼在 `_rightonly` 和一般版之間選

兩者的 state 寬度和相機都一樣，所以 `config.json` 沒辦法分辨。差別在於
**訓練資料本身的性質**：左手到底有沒有動？

`suggest_shape_config.py` 是從 normalizer 的統計量去判斷的。一個完全靜止的
關節，其分佈會退化成一個很小的值，而資料集的統計量下限會把它夾到一個固定的
最小值——而且這個「特徵」是**同一個**最小值反覆出現在多個維度上，獨立的關節
不可能剛好都落在同一個值。

- 左手是靜止的 → 用 **`_rightonly`**。`action[0:8]` 訓練時只是在重現「目前的
  觀測值」（也就是「保持不動」），不是一個真正的目標。如果照著追蹤，controller
  會去追一個統計上的平均姿勢，但那不是手臂實際所在的位置；`max_relative_target`
  雖然每一步都有限制幅度，但誤差會不斷累積，手臂就會慢慢漂走。
- 兩隻手都會動 → 用**一般版**。

---

## 七項檢查

### 1. 讀 checkpoint 自己宣告的內容

`config.json` —— policy 類型、相機的 key、`observation.state` / `action`
的寬度、`chunk_size`、`normalization_mapping`。`anvil_config.json` ——
`action_type` 和一字不差的 `task_description`。`train_config.json` ——
資料集和訓練步數。

不要相信資料夾名稱或是留傳下來的註解。曾經有個 checkpoint 在 `.env`
裡被標成「單手臂 8 DOF」，但 `config.json` 其實寫的是 16-DOF 雙手臂。
以檔案裡的內容為準。

### 2. 掃描權重有沒有 NaN/Inf

`preflight_checkpoint.py` 會做這件事。它已經抓到過兩次真實的損毀，兩次
都是**複製**過程出錯，而不是匯出出錯——而且兩次壞掉的複製，NaN 出現在
**不同的** tensor 裡，這就是我們判斷是傳輸問題的依據。

損毀的 checkpoint 照樣能無錯誤地載入。如果這一步失敗，重新複製再重新
掃描一次；如果反覆發生，檢查 `dmesg` 有沒有 I/O 錯誤。

### 3. 對照 config 跑 preflight

會交叉比對相機的 key、state 寬度 vs `arm_mapping × model_joint_order`、
`action_start:action_end` 切片，以及 `task_description`。Exit 0 才能繼續，
否則不要啟動。

### 4. 逐關節檢查 normalizer 的統計量

Preflight 沒有涵蓋這一步，任何雙手臂或新機型的 checkpoint 都值得手動做一次：

```bash
./.venv/bin/python -c "
from safetensors.numpy import load_file; import numpy as np
d=load_file('<ckpt>/policy_preprocessor_step_*_normalizer_processor.safetensors')
print(d['observation.state.std'])"
```

留意**退化的維度**——`std` 或 `q99-q01` 卡在統計量的下限上。這些是靜止的
關節被轉成了高增益的雜訊通道。詳見下方的「State pinning（狀態釘住）」小節。

### 5. 檢查執行期依賴

- 如果 `HF_HUB_OFFLINE=1`，確認 tokenizer 已經存在 `HF_CACHE` 裡
  （pi0/pi0.5 → PaliGemma；SmolVLA → SmolVLM2）。
- `LEROBOT_EXTRAS` 要包含對應的 policy family（`pi`、`smolvla`）。改這個
  值需要重新 `docker compose build`。

### 6. 設定 `.env`

`MODEL_PATH` 和 `CONFIG_FILE` 必須是 preflight 核可過的那一組配對。

### 7. Offline inference 測試

`preflight_checkpoint.py` 只比對 metadata。`offline_inference_test.py`
會真的載入權重、用合成輸入跑一次 policy，能抓到靜態檢查抓不到的問題：

- processor pipeline 建構失敗
- `HF_HUB_OFFLINE=1` 但 `HF_CACHE` 裡缺 tokenizer
- VRAM 和**主機 RAM** 裝不下
- 真實的每個 chunk 延遲
- 輸出死掉（每筆輸入都給出一樣的 action），或數值遠超出訓練範圍

要在 inference image 裡跑，不要用host 端的 venv —— host 的 `.venv`
沒有 `transformers`，VLA policy 連 import 都做不到。

在 RTX 5090 Laptop（24 GiB）上測到的數據，2026-09-10：

| Checkpoint | 載入時間 | 穩態延遲 | Chunk | VRAM |
|---|---|---|---|---|
| `smolvla_flip_pack_20260904_plus_005000` | 6.8 s | 128 ms | 50x8 | 0.91 GiB |
| `smolvla-flip/005000` | 7.2 s | 113 ms | 50x16 | 0.91 GiB |
| `pi05-flip/005000` | — | — | — | **OOM，見下方** |

**pi0.5 在這台機器上載入不了。** `from_pretrained` 在載入 9.3 GB 的
checkpoint 途中就被 SIGKILL 殺掉（exit 137），當時主機還有大約 25 GiB
的空閒 RAM。問題出在主機 RAM，不是 VRAM —— 權重會先在 CPU 上實體化，
再搬到 GPU。如果這台機器需要跑 pi0.5，得先釋放記憶體（觀察到的那次 run
swap 已經吃到 4 GiB），或是換一台 RAM 更多的機器。

### 8. 第一次跑要開監控

```bash
MONITOR_ENABLE=true
```

會寫出逐步的 CSV。Preflight 只是**靜態的**合約檢查——它只確認 checkpoint
和 config 互相吻合，不保證相機真的以 30 fps 在發布畫面，也不保證 policy
表現正常。留意 gripper 那個維度：卡在一個固定值是 config 出問題的典型症狀，
不是模型的問題。

---

## State pinning（狀態釘住）

`state_pinning` 會把靜止手臂的 state 維度凍結在訓練時的中位數，而不是餵進
即時的編碼器讀值。

**問題所在。** 一個從不移動的關節，分佈會退化，資料集統計量的下限會把它
夾到一個固定的最小值。正規化在這些維度上會用極大的增益運算——在
`QUANTILES`、下限 0.03 的情況下大約是 ~67 單位/rad，相對於一個會動的關節
大約是 ~2，差了大概 35 倍。訓練時的休息姿勢跟部署時只要差一點點角度，
正規化後的值就會完全跳出 `[-1, 1]`，而 `normalize_processor.py:377`
不會做 clamp。

**為什麼 pi0.5 特別容易受傷。** pi0.5 會把正規化後的 state 分箱成 256 個
等級，然後拼進**文字 prompt** 裡（`processor_pi05.py:74-82`）。超出範圍的
維度會飽和到 bin 0/255，或是產生訓練時從沒出現過的 token `-1`——這會污染
整個 action chunk（兩隻手臂）所依賴的 prefix。實際觀察到的現象：右手夾爪
卡在 0.048–0.053，而不是伸到 -0.003。

**為什麼 SmolVLA 面對同樣的資料沒事。** 它的 normalizer 是在 tokenizer
**之後**才跑，state 是透過 `nn.Linear`（`modeling_smolvla.py:693`）進到
模型裡的。一個分佈外的 state 只是一個偏大的浮點數，會平滑地劣化，而不是
變成一個錯誤的 token。這就是為什麼同一份資料下 `smolvla-flip` 沒事，
`pi05-flip` 卻出問題。

**修法。** 把這些維度釘在 `q50`（或 `mean`），讓 prompt 裡 state 的部分
每一步都完全一樣（正規化後正好是 0.0）。

```yaml
state_pinning:
  enabled: true
  arms: [l]
  source: q50    # QUANTILES 用 q50，MEAN_STD 用 mean
```

要做 A/B 比對、想改回餵即時 state 的話設 `enabled: false`。Pinning
同時也消除了另一個靜默失敗：`multi_process.py` 對任何 `/joint_states`
裡缺的關節（例如左手沒上電）會 fallback 成 `0.0`，如果統計量下限被
夾過，這個 0.0 正規化後可能是 -100 甚至更誇張，而且完全不會有警告。

長期的解法是重新訓練，把靜止的手臂從 state 和 action 裡拿掉。Pinning
只是推論端的緩解措施。

---

## 已知的靜默失敗

| 現象 | 原因 |
|---|---|
| Inference 完全沒啟動，也沒有有用的 log | Config 列了一支 checkpoint 沒宣告的相機 —— `has_complete_observation()` 永遠卡住 |
| 手臂動作看起來合理但是錯的；慢慢退化成一個固定姿勢 | 相機缺失或拼錯名字——被替換成全 -1 的空白影像，mask 為 0（`modeling_pi05.py:1195-1204`） |
| 手臂能正確伸過去，但夾爪永遠不合起來 | 舊版的 `min_position_delta` deadband。已經移除——上游沒有對應的東西，而且一個看似合理的值會讓夾爪凍結 |
| 手臂還沒伸到物體，夾爪就先合起來了 | 純量的 `max_relative_target` 限制了手臂（單位是弧度），但超過夾爪整個約 0.05 m 的行程，導致一步就關閉。改用逐關節的對應，`finger_joint1: 0.005` |
| 數值跑到錯的關節上 | State 寬度不匹配——pi0/pi05 會補到 `max_state_dim` 32 再截斷，把誤差吸收掉了 |
| 多出來的關節發布全零 | `action_end` 超過實際的 action 寬度 |
| 左手慢慢漂到一個從沒待過的姿勢 | 雙手臂 checkpoint 裡有一隻靜止的手臂被當成目標發布。改用 `_rightonly` |
| Checkpoint 能正常載入，但輸出全是垃圾 | 權重裡有 NaN/Inf —— 目前為止都是複製出錯造成的 |
| Action 微妙地變差，log 裡什麼都沒有 | `task_description` 被改寫過。VLA policy 是靠這個做 conditioning 的 |
| 節點沒有拒絕啟動，只印出「has no task_description」 | Checkpoint 訓練時沒存這欄位，config 也沒設。不是靜默失敗，但很常見——這裡 27 個裡有 11 個 |
| 容器在載入途中死掉，exit 137，沒有 traceback | 主機 RAM OOM。pi0.5 需要的空閒記憶體比這台機器現有的多 |

---

## 新增一個 shape config

`scripts/` 裡沒有現成的產生器；shape 檔案就是單純、自包含的 YAML
（config 是用單一檔案 bind-mount 進去的，沒有 include 機制）。要新增一個：

1. 複製最接近的現有 shape 檔案。
2. 改 `arm_mapping`（state 寬度 = `len(arm_mapping) × len(model_joint_order)`）
   和 `cameras.mapping`。
3. 把每隻手臂的 `action_start`/`action_end` 設成它在
   `sorted(arm_mapping)` 順序裡的位置——這必須跟 `multi_process.py:216-227`
   裡的組裝邏輯一致。
4. 命名成 `<arms>arm_<cameras>cam[_variant].yaml`，在上面的表格加一列。
5. 用 `preflight_checkpoint.py` 對照一個真實的 checkpoint 驗證。
