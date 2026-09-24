# 歡迎加入 Anvil Embodied AI

這份是在這個 repo 用 Claude Code 的快速上手指南。更深入的技術背景(架構、過去踩過的坑、驗證過的訓練參數)都整理在 `AGENT_HANDOFF.md`——Claude 需要時會自動去讀,你不用整份看完。

## 你最常會做的兩件事

**收了新的機器人錄製資料(USB 隨身碟、新 session)?**
直接請 Claude 幫你轉檔,它會照著 `/mcap-to-dataset` skill 走:先做 quality scan、跟你確認要用哪些相機/手臂/action space、轉成 LeRobot dataset、驗證、寫 `dataset_card.yaml`。

**想訓練一個 checkpoint?**
請 Claude 在某個 dataset 上訓練,它會照著 `/train-checkpoint` skill 走:用驗證過的 SmolVLA 或 pi0.5 recipe、啟動、監控訓練,等你在真實機器人上測過之後,再幫你 promote 到 shared model_zoo。

這兩個 skill 在關鍵決策點(要選哪些相機/手臂、action space 是 joint 還是 EE、訓練 steps、要不要 promote checkpoint)都會**主動停下來問你**——這是刻意設計的,不是 bug。請照實回答;除非你真的不在意結果,不然不要叫 Claude 自己隨便猜。

## 環境設定

- **Repo**:`github.com/anvil-robotics/anvil-embodied-ai`——push 到 `nvidia` remote(nvidia-new-tech fork),不是 `origin`。
- **共用儲存空間**:PRO6000 訓練伺服器上的 `/srv/shared/` 放正式的 dataset 跟已驗證的模型。還沒有權限的話跟同事要一下。

## 一些好習慣

- Checkpoint 值不值得信任,標準是實機測試過,不是 loss 曲線好看——val loss 曾經選錯過 checkpoint。
- 拿到不是自己做的 dataset/checkpoint,先看它旁邊的 `dataset_card.yaml` / `model_card.yaml`,裡面會記錄已知的注意事項。
- 遇到感覺「應該是已知的坑」的狀況(資料損毀、訓練結果莫名其妙變差、以前發生過的 bug),很可能真的是——先請 Claude 查一下 `AGENT_HANDOFF.md` 的 incident history,不要從零開始 debug。

## Skills & Commands

- `/mcap-to-dataset` —— 原始 MCAP session → 共用 LeRobot dataset
- `/train-checkpoint` —— dataset → 訓練出 checkpoint →(選擇性)promote 到 shared model_zoo

## 有問題?

問團隊——目前還沒有指定的頻道。
