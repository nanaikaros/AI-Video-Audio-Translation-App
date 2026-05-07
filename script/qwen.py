import os
import json
from openai import OpenAI

try:
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
        api_key="sk-79c41f0c81654aa1b44073e5703d5cc2",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    ocr_path = "/Users/wang/Documents/娜英社长/5.5/.temp/video_[无字] 社长要去现场看LCK比..._0/ocr.json"
    rag_path = "/Users/wang/code/test/rag/qwer"

    def load_glossary(path):
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ";" not in line:
                    continue
                src, dst = [part.strip() for part in line.split(";", 1)]
                if src and dst:
                    pairs.append((src, dst))
        return pairs

    glossary_pairs = load_glossary(rag_path)
    if glossary_pairs:
        terminology_block = "Terminology constraints (must follow):\n"
        for src, dst in glossary_pairs:
            terminology_block += f"- {src} => {dst}\n"
        terminology_block += "\n"
    else:
        terminology_block = ""

    system_prompt = (
        "You are a subtitle translation engine.\n"
        "Automatically detect the source language of each sentence and translate everything into Simplified Chinese.\n"
        "Output only translated subtitle text.\n"
        "No explanation, no extra notes, no original text.\n"
        "If one subtitle line is long, insert natural line breaks for reading.\n"
        "Keep meaning accurate, concise, and subtitle-friendly.\n"
        "Do not add labels, explanations, prefixes, or notes.\n"
    )

    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    entries = ocr_data.get("entries", [])
    total_entries = len(entries)
    print(f"Loaded {total_entries} subtitle entries from OCR data.")

    batch_size = 5

    def iter_batches(entries_list, size):
        batch = []
        for entry in entries_list:
            text = (entry.get("text") or "").strip()
            if not text:
                continue
            batch.append(entry)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch
    processed = 0

    for batch in iter_batches(entries, batch_size):
        # 构造合并输入
        user_lines = []
        for i, entry in enumerate(batch, start=1):
            user_lines.append(f"{i}. {entry['text'].strip()}")
        user_prompt = (
            "请将下面每行翻译成简体中文，仅输出对应的译文，"
            "保持行数一致，不要编号：\n" +
            "\n".join(user_lines)
        )

        completion = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": system_prompt + "\n" + terminology_block},
                {"role": "user", "content": user_prompt},
            ],
        )

        # 解析输出：按行切分
        out_lines = [l.strip() for l in completion.choices[0].message.content.splitlines() if l.strip()]

        # 如果模型输出行数不匹配，建议加入兜底处理
        for entry, translated in zip(batch, out_lines):
            entry["trans_text"] = translated
            processed += 1
        
        print(f"Progress: {processed}/{total_entries}", end="\r", flush=True)


    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, ensure_ascii=False, indent=2)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")