import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    AutoModelForCausalLM,
)

torch.set_grad_enabled(False)

# =========================
# Engine 1: Desklib DeBERTa AI Detector (英文)
# =========================

DET_MODEL_NAME = "desklib/ai-text-detector-v1.01"


class DesklibAIDetectionModel(PreTrainedModel):
    """
    依官方 model card 寫的 wrapper：
    - base: DeBERTa-v3
    - classifier: 線性層輸出 1 維 logit (AI-generated)
    """
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]                       # [B, T, H]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        ).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask            # masked mean pooling

        logits = self.classifier(pooled_output)              # [B, 1]

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output


@st.cache_resource
def load_detector():
    config = AutoConfig.from_pretrained(DET_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(DET_MODEL_NAME)
    model = DesklibAIDetectionModel.from_pretrained(DET_MODEL_NAME, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def detector_predict_prob_batch(
    sent_list, tokenizer, model, device, max_len: int = 256
):
    """
    一次對多個句子做 batch 推論，回傳 AI 機率 list（0~1）。
    空列表 → 回空 list。
    """
    if not sent_list:
        return []

    encoded = tokenizer(
        sent_list,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"].view(-1)  # [B]

    probs_ai = torch.sigmoid(logits).cpu().numpy().tolist()
    return probs_ai


def detector_predict_prob_doc(text: str, tokenizer, model, device, max_len: int = 512):
    """
    整篇文章丟一次模型，回傳 (ai_prob, human_prob)。
    """
    if not text.strip():
        return 0.5, 0.5

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]  # [1,1]
        ai_prob = torch.sigmoid(logits).item()

    ai_prob = float(ai_prob)
    human_prob = 1.0 - ai_prob
    return ai_prob, human_prob


# =========================
# Engine 2: GPT-2 Perplexity (英文)
# =========================

LM_MODEL_NAME = "distilgpt2"  # 比 gpt2 小一點，較快


@st.cache_resource
def load_language_model():
    tok = AutoTokenizer.from_pretrained(LM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LM_MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tok, model, device


def compute_perplexity(text: str, tok, model, device, max_length: int = 256):
    """
    使用 GPT-2 計算 cross-entropy loss 與 perplexity。
    只取前 max_length token。
    """
    if not text.strip():
        return 0.0, float("inf")

    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

    loss_val = loss.item()
    ppl = float(torch.exp(loss))
    return loss_val, ppl


def heuristic_ai_from_ppl(ppl: float) -> float:
    """
    根據 perplexity 給一個粗略 AI 機率 (0~1)：
      - ppl ≤ 20：非常好預測 → 很像 AI → ~0.9
      - ppl ≥ 80：很難預測 → 很像 Human → ~0.1
      - 中間線性插值
    純啟發式，當「第二意見」用。
    """
    if ppl <= 20:
        ai_prob = 0.9
    elif ppl >= 80:
        ai_prob = 0.1
    else:
        ai_prob = 0.9 - (ppl - 20) * (0.8 / 60.0)
    return max(0.0, min(1.0, ai_prob))


# =========================
# Stylometry + utils
# =========================

def split_sentences_en(text: str):
    """
    很簡單的英文斷句：以 . ? ! 和換行為主。
    """
    parts = re.split(r"[\.!?。\n\r]+", text)
    sents = [p.strip() for p in parts if p.strip()]
    return sents


def tokenize_for_stats(text: str):
    """
    給 Stylometry 用的 token。英文情境：\w+ 抓英文單字。
    """
    tokens = re.findall(r"\w+", text)
    return tokens


def compute_stylometry(text: str):
    chars = [c for c in text if not c.isspace()]
    n_chars = len(chars)

    sentences = split_sentences_en(text)
    sent_lens = [len(s) for s in sentences]
    avg_len = float(np.mean(sent_lens)) if sent_lens else 0.0
    std_len = float(np.std(sent_lens)) if sent_lens else 0.0
    burstiness = std_len / avg_len if avg_len > 0 else 0.0

    tokens = tokenize_for_stats(text)
    ttr = len(set(tokens)) / len(tokens) if tokens else 0.0

    puncts = re.findall(r"[^\w\s]", text)
    noise_ratio = len(puncts) / n_chars if n_chars > 0 else 0.0

    human_noise_patterns = [
        r"haha",
        r"lol",
        r"XD",
        r"\?{2,}",        # 兩個以上 ?
        r"!{2,}",        # 兩個以上 !
        r"[😂🤣😅😭😍🤔]",
    ]
    human_noise_hits = sum(
        len(re.findall(p, text, flags=re.IGNORECASE))
        for p in human_noise_patterns
    )

    return {
        "Characters": n_chars,
        "Sentences": len(sentences),
        "Avg sentence length": avg_len,
        "Sentence length std": std_len,
        "Burstiness (σ/μ)": burstiness,
        "Type-Token Ratio (TTR)": ttr,
        "Punctuation ratio": noise_ratio,
        "Human-noise count": human_noise_hits,
    }


def stylometry_ai_score(feat):
    """
    根據 stylometry 特徵輸出 AI 機率 (0~1)
    """
    burst = feat["Burstiness (σ/μ)"]
    ttr = feat["Type-Token Ratio (TTR)"]
    noise = feat["Human-noise count"]

    # Burstiness score（低 → AI）
    if burst <= 0.3:
        s_b = 1.0
    elif burst >= 0.8:
        s_b = 0.0
    else:
        s_b = 1 - (burst - 0.3) / (0.8 - 0.3)

    # TTR score（低 → AI）
    if ttr <= 0.35:
        s_t = 1.0
    elif ttr >= 0.7:
        s_t = 0.0
    else:
        s_t = 1 - (ttr - 0.35) / (0.7 - 0.35)

    # Noise score（少 → AI）
    if noise == 0:
        s_n = 1.0
    elif noise >= 3:
        s_n = 0.0
    else:
        s_n = 1 - (noise / 3)

    # 權重（可調）
    w1, w2, w3 = 0.4, 0.3, 0.3
    ai_score = (w1*s_b + w2*s_t + w3*s_n) / (w1 + w2 + w3)

    return float(ai_score)




def color_for_ai_prob(p: float, low_thr: float, high_thr: float) -> str:
    """
    根據 AI 機率上色：
      p >= high_thr → 紅
      p <= low_thr  → 綠
      其他         → 黃
    """
    if p >= high_thr:
        return "#ffb3b3"
    elif p <= low_thr:
        return "#b3ffb3"
    else:
        return "#fff4b3"


def render_colored_sentences(sent_df: pd.DataFrame, low_thr: float, high_thr: float):
    html_parts = []
    for _, row in sent_df.iterrows():
        p_ai = row["AI_prob"]
        sent = row["Sentence"]
        color = color_for_ai_prob(p_ai, low_thr, high_thr)
        html_parts.append(
            f"<span style='background-color:{color};"
            f"padding:2px 4px;margin:2px;display:inline-block;'>"
            f"[AI {p_ai*100:.1f}%] {sent}</span>"
        )
    html = "<div style='line-height:1.8;'>{}</div>".format("<br>".join(html_parts))
    st.markdown(html, unsafe_allow_html=True)


# =========================
# Streamlit App
# =========================

st.set_page_config(
    page_title="AI vs Human Detector (Multi-Engine)",
    layout="wide",
)

st.title("🤖 AI vs Human 文章偵測器（可選引擎版）")

st.markdown(
    """
本工具提供三種分析層級，**只有勾選的引擎才會載入模型**：

- 🟢 **Stylometry（預設啟用）**：  
  不下載任何模型，只計算文字統計特徵，速度最快。

- 🔵 **Engine 1 — DeBERTa AI Detector**（`desklib/ai-text-detector-v1.01`）：  
  針對 **英文文本** 的 AI / Human 分類器。

- 🟣 **Engine 2 — GPT-2 Perplexity**（`distilgpt2`）：  
  使用語言模型困惑度 (Perplexity) 粗略估計 AI 可能性（第二意見）。

> ⚠️ DeBERTa 與 GPT-2 都是針對英文訓練，用在中文或混合文本時準確度會下降。  
> 所有結果僅供參考，不應作為學術違規或法律判定的唯一依據。
"""
)

# ---- Sidebar: engine selection ----
st.sidebar.header("⚙️ Engine 選擇")
use_stylometry = st.sidebar.checkbox("Stylometry（文字統計，最快）", value=True)
use_deberta = st.sidebar.checkbox("Engine 1 — DeBERTa AI Detector", value=False)
use_gpt = st.sidebar.checkbox("Engine 2 — GPT-2 Perplexity", value=False)

st.sidebar.header("🎚️ DeBERTa 判斷閾值")
low_thr = st.sidebar.slider("Human-like 上限（AI 機率 ≤）", 0.0, 0.5, 0.3, 0.05)
high_thr = st.sidebar.slider("AI-like 下限（AI 機率 ≥）", 0.5, 1.0, 0.7, 0.05)

if high_thr <= low_thr:
    st.sidebar.warning("⚠️ 建議 AI-like 閾值要大於 Human-like 閾值。")

# ---- Lazy load engines ----
det_tokenizer = det_model = det_device = None
lm_tok = lm_model = lm_device = None

if use_deberta:
    with st.spinner("載入 Engine 1（DeBERTa AI Detector）..."):
        det_tokenizer, det_model, det_device = load_detector()
    st.success("Engine 1 載入完成：desklib/ai-text-detector-v1.01")

if use_gpt:
    with st.spinner("載入 Engine 2（GPT-2 Perplexity）..."):
        lm_tok, lm_model, lm_device = load_language_model()
    st.success(f"Engine 2 載入完成：{LM_MODEL_NAME}")

# ---- Text input ----
text = st.text_area(
    "✏️ 請輸入要偵測的文章（建議英文）：",
    height=220,
    placeholder="Paste an English paragraph (e.g., essay, report, blog post)...",
)

if text.strip():
    sentences = split_sentences_en(text)

    # =========================
    # Engine 1: DeBERTa (doc-level + sentence-level)
    # =========================
    if use_deberta and det_tokenizer is not None:
        st.subheader("📌 Engine 1：整篇文章判斷結果（DeBERTa AI Detector）")

        ai_prob_doc, human_prob_doc = detector_predict_prob_doc(
            text, det_tokenizer, det_model, det_device
        )
        c1, c2 = st.columns(2)
        with c1:
            st.metric("AI 機率（整篇）", f"{ai_prob_doc * 100:.1f}%")
            st.progress(ai_prob_doc)
        with c2:
            st.metric("Human 機率（整篇）", f"{human_prob_doc * 100:.1f}%")
            st.progress(human_prob_doc)

        st.subheader("🔍 Engine 1：句子級別 AI 偵測")

        ai_probs_sent = detector_predict_prob_batch(
            sentences, det_tokenizer, det_model, det_device
        )

        sent_rows = []
        for idx, (s, a_p) in enumerate(zip(sentences, ai_probs_sent), start=1):
            if a_p >= high_thr:
                label = "AI-like"
            elif a_p <= low_thr:
                label = "Human-like"
            else:
                label = "Uncertain"
            sent_rows.append(
                {
                    "Index": idx,
                    "Sentence": s,
                    "AI_prob": a_p,
                    "Human_prob": 1.0 - a_p,
                    "Label": label,
                    "Length": len(s),
                }
            )

        if sent_rows:
            sent_df = pd.DataFrame(sent_rows)

            # 類型比例
            st.markdown("**📊 句子類型比例（Engine 1）**")
            type_counts = (
                sent_df["Label"]
                .value_counts()
                .reindex(["AI-like", "Uncertain", "Human-like"])
                .fillna(0)
                .astype(int)
            )
            total_sents = len(sent_df)
            ratio_df = pd.DataFrame(
                {
                    "Type": type_counts.index,
                    "Count": type_counts.values,
                    "Ratio": [
                        f"{c/total_sents*100:.1f}%"
                        if total_sents > 0
                        else "0.0%"
                        for c in type_counts.values
                    ],
                }
            )
            st.table(ratio_df)

            # 自然語言摘要
            st.markdown("**📝 自然語言摘要（Engine 1，可貼到報告）**")
            ai_like_cnt = int(type_counts.get("AI-like", 0))
            human_like_cnt = int(type_counts.get("Human-like", 0))
            uncertain_cnt = int(type_counts.get("Uncertain", 0))
            avg_ai_sent = sent_df["AI_prob"].mean() if total_sents > 0 else 0.0
            max_row = (
                sent_df.loc[sent_df["AI_prob"].idxmax()] if total_sents > 0 else None
            )

            summary_lines = []
            summary_lines.append(
                f"- 整篇文字的 AI 機率約為 **{ai_prob_doc*100:.1f}%**，Human 機率約為 **{human_prob_doc*100:.1f}%**。"
            )
            summary_lines.append(
                f"- 共有 **{total_sents} 句**，其中 AI-like：**{ai_like_cnt}**，Human-like：**{human_like_cnt}**，不確定：**{uncertain_cnt}**。"
                f"（閾值：AI ≥ {high_thr:.2f}、Human ≤ {low_thr:.2f}）"
            )
            summary_lines.append(
                f"- 句子平均 AI 機率約為 **{avg_ai_sent*100:.1f}%**。"
            )
            if max_row is not None:
                summary_lines.append(
                    f"- AI 機率最高的句子是第 **{int(max_row['Index'])} 句**（約 **{max_row['AI_prob']*100:.1f}%**）："
                    f"「{max_row['Sentence'][:120]}{'...' if len(max_row['Sentence'])>120 else ''}」。"
                )
            st.markdown("\n".join(summary_lines))

            st.markdown("**📋 句子清單（可排序）**")
            st.dataframe(
                sent_df[["Index", "Label", "AI_prob", "Human_prob", "Sentence"]],
                use_container_width=True,
            )

            st.markdown("**📊 每句 AI 機率（bar chart）**")
            chart_df = sent_df.set_index("Index")[["AI_prob"]]
            st.bar_chart(chart_df)

            st.markdown("**📊 句長 vs AI 機率（scatter）**")
            scatter_df = sent_df[["Length", "AI_prob"]]
            st.scatter_chart(scatter_df)

            st.markdown("**🎨 句子視覺化（背景色代表 AI 可能性）**")
            st.caption(
                f"AI-like（AI≥{high_thr:.2f}）= 紅、Human-like（AI≤{low_thr:.2f}）= 綠、中間 = 黃。"
            )
            render_colored_sentences(sent_df, low_thr=low_thr, high_thr=high_thr)
        else:
            st.info("無法切出句子，請確認文字內容是否正確。")

    elif use_deberta:
        st.warning("Engine 1 選擇了，但模型尚未載入成功？請重新整理試試。")

    # =========================
    # Engine 2: GPT-2 Perplexity
    # =========================
    if use_gpt and lm_tok is not None:
        st.subheader("📉 Engine 2：Perplexity 分析（GPT-2）")
        loss_val, ppl = compute_perplexity(
            text, lm_tok, lm_model, lm_device, max_length=256
        )
        ai_prob_ppl = heuristic_ai_from_ppl(ppl)
        human_prob_ppl = 1.0 - ai_prob_ppl

        c3, c4, c5 = st.columns(3)
        with c3:
            st.metric("Cross-Entropy Loss", f"{loss_val:.3f}")
        with c4:
            st.metric("Perplexity (PP)", f"{ppl:.1f}")
        with c5:
            st.metric("AI 機率（依 PP 推估）", f"{ai_prob_ppl * 100:.1f}%")
        st.progress(ai_prob_ppl)
        st.caption(
            "直覺：Perplexity 越低 → 越像模型自己寫（AI）；"
            "Perplexity 越高 → 越像人類自然語言。此為啟發式估計。"
        )
    elif use_gpt:
        st.warning("Engine 2 選擇了，但模型尚未載入成功？請重新整理試試。")

    # =========================
    # Stylometry
    # =========================
    if use_stylometry:
        st.subheader("📈 Stylometry：文本統計特徵")
        feat = compute_stylometry(text)
        feat_df = pd.DataFrame.from_dict(feat, orient="index", columns=["Value"])
        st.table(feat_df)

        # Stylometry AI Score
        ai_prob_style = stylometry_ai_score(feat)
        human_prob_style = 1 - ai_prob_style

        st.markdown("### 🔍 Stylometry AI 判斷結果")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("AI 機率（Stylometry）", f"{ai_prob_style*100:.1f}%")
            st.progress(ai_prob_style)
        with c2:
            st.metric("Human 機率（Stylometry）", f"{human_prob_style*100:.1f}%")
            st.progress(human_prob_style)

        # bar chart
        st.markdown("**🔎 Stylometry 特徵圖**")
        numeric_feat_df = feat_df[
            feat_df["Value"].apply(lambda x: isinstance(x, (int, float)))
        ]
        st.bar_chart(numeric_feat_df)


else:
    st.info("請在上方輸入一段英文文字，再選擇左側要啟用的引擎，我會幫你做 AI / Human 分析。")
