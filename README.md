# 🤖 AI vs Human 多引擎文章偵測器（AI Text Detector — Multi-Engine Version）

本專案是一套以 **Streamlit** 建置的互動式 **AI 文章偵測器**，提供 **三種可切換偵測引擎**，能針對使用者輸入的文章進行分析並估算 **AI / Human 的可能性**。

本系統支持：

- **Stylometry 統計特徵分析（預設最快）**
- **DeBERTa AI Detector（Transformer 模型）**
- **GPT-2 Perplexity 語言模型（第二意見）**

使用者可以自由勾選要啟用的引擎，未勾選的模型不會下載 →  
**大幅提升啟動速度（特別是 Streamlit Cloud 上）。**

---

# 📌 系統特色（System Features）

### ✔ 1. 三引擎可切換架構（Modular Engine Selection）
使用者可在側邊欄選擇啟用：

| 引擎 | 速度 | 是否需下載模型 | 功能 | 適用語言 |
|------|------|----------------|-------|-----------|
| 🟢 Stylometry（預設） | ⭐⭐⭐⭐⭐（最快） | ❌ | 統計分析 + 啟發式 AI 評分 | 英文 / 中文皆可（統計特徵不分語言） |
| 🔵 DeBERTa AI Detector | ⭐⭐⭐ | ✔ | Transformer 專業分類器 | 英文（模型以英文訓練） |
| 🟣 GPT-2 Perplexity | ⭐⭐ | ✔ | 語言模型困惑度分析 | 英文 |

---

### ✔ 2. 多層級 AI 機率（AI Probability from 3 Engines）

每個引擎都能獨立輸出：

- AI 機率（AI%）
- Human 機率（Human%）

使用者可比較不同方法的結果，做最終研判。

---

### ✔ 3. 逐句 AI 分析 + 顏色標記（Engine 1）
DeBERTa 引擎可對文章逐句分析：

- 每句 AI%  
- 每句 Human%  
- 自動分類成：AI-like、Human-like、Uncertain  
- 色塊標註（紅 / 綠 / 黃）

並提供：

- 句子分類比例表  
- 每句 AI 機率長條圖  
- 句長 vs AI% 散佈圖  
- 最高風險句子摘要（可貼進報告）

---

### ✔ 4. Stylometry 統計特徵 + AI Score（Engine 0）
本系統計算下列特徵：

- 句子數量  
- 平均句長  
- 句長標準差  
- Burstiness（句子變異度）  
- TTR（詞彙多樣性）  
- 標點比例  
- Human-noise（haha、lol、XD、emoji 等）

並使用 **啟發式權重法（Heuristic Weighted Score）**  
輸出 Stylometry-based AI%（0～100%）。

---

### ✔ 5. 完整的資料視覺化（Visualization）
系統提供：

- 長條圖（每句 AI%）
- 散佈圖（句長 vs AI%）
- 統計特徵圖（Stylometry）
- 色彩標記句子（Highlight Mode）

---

# 🧠 三種偵測引擎的原理（Model Explanation）

## 🟢 1. Stylometry（文字統計特徵）
此方法不需任何模型，透過語言學統計特徵推斷 AI 風格。

AI 文常見特徵：

- 句子長度規律 → **Burstiness 偏低**
- 詞彙多樣性低 → **TTR 偏低**
- 人類噪音符號低 → **Human-noise = 0**
- 標點較一致

因此本系統根據以下特徵加權：

| 特徵 | 趨勢 | AI 判斷方式 |
|------|-------|--------------|
| Burstiness | AI ↓ | 越低越像 AI |
| TTR | AI ↓ | 越低越像 AI |
| Human-noise | AI ↓ | 越低越 AI |
| Punctuation ratio | 有參考 | 輔助指標 |

輸出平均 AI Score 作為 **Stylometry AI%**。

優點：  
✔ 速度極快  
✔ 不需模型、不需網路  
✔ 適用各語言  

缺點：  
✘ 可靠度低於深度模型  
✘ 容易被刻意調整風格欺騙  

---

## 🔵 2. DeBERTa AI Detector（Transformer 模型）
使用模型：

此模型基於 **DeBERTa-v3-large** 微調，用於判斷：

- 使用者輸入文章是否「AI生成」或「人類撰寫」

優點：  
✔ 深度模型，準確度最佳  
✔ 可逐句分析  
✔ 推論結果可靠  
✔ 有 logits → 可視覺化機率

缺點：  
✘ 模型大、載入較慢  
✘ 以英文訓練 → 中文準確度較差  

---

## 🟣 3. GPT-2 Perplexity（語言模型困惑度，第二意見）
Perplexity(PPL) 代表語言模型對文本的「預測困難度」：

- **AI 文通常 PPL 偏低 → 容易預測**
- **Human 文 PPL 較高 → 難預測**

本系統將 PPL 映射成 AI%（啟發式）：

- PPL ≤ 20 → AI% ≈ 90%  
- PPL ≥ 80 → AI% ≈ 10%  
- 中間線性插值

優點：  
✔ 適合作為第二意見  
✔ 不依賴分類器  

缺點：  
✘ 不適合判斷高品質 AI 文（PPL 可能很高）  
✘ 仍須下載語言模型  

---

# 🖥️ 系統 UI 頁面簡介

## 左側 Sidebar
- Engine 選擇（Stylometry / DeBERTa / GPT-2）
- DeBERTa 判斷閾值（Human-like ≤ x, AI-like ≥ y）

## 主畫面
1. 文字輸入框  
2. Stylometry 區域（統計特徵 + AI Score）  
3. DeBERTa 區域（整篇 + 逐句分析）  
4. GPT-2 區域（Loss, PPL, AI%）  
5. 圖表與色塊標註  

---

# 🚀 本機執行方式（Local Execution）

### 1. 下載專案
```bash
git clone https://github.com/<你的帳號>/ai-detector.git
cd ai-detector
