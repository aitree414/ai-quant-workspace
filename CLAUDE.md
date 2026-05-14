# CLAUDE.md — AI 量化交易工作區行為規範

## 專案資訊
- **專案名稱**：AI 量化交易協作工作區
- **協作 AI**：Manus（遠端研究）+ Claude Code（本地開發）
- **語言**：繁體中文回應，程式碼英文註解

## 編碼原則
1. 所有程式碼必須有清楚的註解
2. 函數必須有 docstring
3. 所有 API 呼叫必須有 try/except
4. 使用 Python logging，不用 print
5. 加上 type hints

## Git 規範
- Commit message 格式：`[模組] 動作: 描述`
- 範例：`[backtest] feat: add RSI indicator`

## 工作流程
1. 檢查 TASKS.md 了解當前任務
2. git pull 取得 Manus 的更新
3. 實作功能
4. git commit + git push
5. 更新 TASKS.md
