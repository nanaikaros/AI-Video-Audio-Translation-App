# align_ocr.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 路径配置
OCR_PATH = "/Users/wang/code/test_video/.temp/subtitle_test/ocr.json"
CN_TXT = "/Users/wang/code/test/DOOKII从3分03秒cho回答是_开始.txt"

# 在线模型名（会自动下载到本地缓存）
MODEL_NAME = "intfloat/multilingual-e5-large"

# 匹配参数
MIN_SCORE = 0.6   # 根据效果调节
MIN_LEN = 4       # 短句过滤
BATCH_SIZE = 64

def load_cn_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines

def main():
    # 允许联网下载模型
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    with open(OCR_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    cn_lines = load_cn_lines(CN_TXT)
    if not cn_lines:
        print("no cn lines")
        return

    # 加载在线模型（会下载）
    model = SentenceTransformer(MODEL_NAME)

    # 为 E5 加前缀提升检索效果
    cn_inputs = [f"passage: {line}" for line in cn_lines]
    cn_emb = model.encode(cn_inputs, normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH_SIZE)
    cn_emb = np.array(cn_emb)  # shape (N, D)

    # 构建 queries（保留短句为空白）
    queries = []
    q_to_entry_idx = []
    for i, e in enumerate(entries):
        text = (e.get("text") or "").strip()
        if not text or len(text) < MIN_LEN:
            entries[i]["trans_text"] = ""
            continue
        # 拼接上下文
        ctx = text
        if i > 0:
            prev = (entries[i - 1].get("text") or "").strip()
            if prev:
                ctx = prev + " " + ctx
        if i + 1 < len(entries):
            nxt = (entries[i + 1].get("text") or "").strip()
            if nxt:
                ctx = ctx + " " + nxt

        queries.append(f"query: {ctx}")
        q_to_entry_idx.append(i)

    if queries:
        q_embs = model.encode(queries, normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH_SIZE)
        q_embs = np.array(q_embs)  # shape (M, D)

        # scores: (M, N) = q_embs @ cn_emb.T
        scores = q_embs @ cn_emb.T

        for qi, entry_idx in enumerate(q_to_entry_idx):
            row = scores[qi]
            best_idx = int(np.argmax(row))
            best_score = float(row[best_idx])
            if best_score >= MIN_SCORE:
                entries[entry_idx]["trans_text"] = cn_lines[best_idx]
            else:
                entries[entry_idx]["trans_text"] = ""

    # 保存结果
    with open(OCR_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("done:", OCR_PATH)

if __name__ == "__main__":
    main()