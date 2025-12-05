
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
