# AI 量化交易協作工作區

> Manus + Claude Code 雙 AI 協作開發量化交易系統

## 協作模式

| AI | 角色 | 負責範圍 |
|----|------|---------|
| **Manus** | 研究員 + 架構師 | 策略研究、市場數據收集、架構設計 |
| **Claude Code** | 本地開發者 | 程式實作、本地測試、API 串接、部署 |

## 目錄結構


cat > README.md << 'EOF'
# AI 量化交易協作工作區

> Manus + Claude Code 雙 AI 協作開發量化交易系統

## 協作模式

| AI | 角色 | 負責範圍 |
|----|------|---------|
| **Manus** | 研究員 + 架構師 | 策略研究、市場數據收集、架構設計 |
| **Claude Code** | 本地開發者 | 程式實作、本地測試、API 串接、部署 |

## 目錄結構


ai-quant-workspace/
├── strategies/       # 交易策略模組
├── backtesting/      # 回測引擎
├── data/             # 市場數據（不進 git）
├── docs/             # 研究文件（Manus 撰寫）
├── utils/            # 共用工具
└── tests/            # 測試

## 開發階段

### Phase 1：基礎建設（當前）
- [x] 建立 GitHub 儲存庫
- [ ] 回測引擎核心
- [ ] 數據載入模組

### Phase 2：策略開發
- [ ] 網格交易策略
- [ ] 動量策略（RSI + MACD）

### Phase 3：即時交易
- [ ] 即時行情串接
- [ ] 風險管理模組
