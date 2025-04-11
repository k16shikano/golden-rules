import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

PATTERNS_FILE = "data/editing_patterns.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
sentence_model = SentenceTransformer(MODEL_NAME)
TOP_N_PATTERNS = 3  # Number of editing patterns to apply simultaneously

GOLDEN_RULES = """
以下の「黄金律」を必ず守って編集してください。

- 文章全体の意味は必ず維持すること。
- 入力の文章には論旨が欠けていることがあります。必要に応じて文脈を補完してください。
- 書き直しでは、元の文にある語彙だけを使い、お手本からは語彙を取り出さないでください。
- 送り仮名の使い方は、元の文ではなく、お手本の傾向に合わせてください。
- 「多いです」のような「イ形容詞＋です」は、お手本を参考にして別の表現に置き換えてください。
- 「非常に」や「かなり」のような形骸化した強調表現は、意味を変えずに取り除いてください。
- 文脈に依存せずに誤読しない表現へと置き換えてください。たとえば、目的でなく根拠を示すときは「～のため」を「～なので」に置き換えることで、文脈に依存せずに誤読を防げます。
- 入力の文章に含まれる「ことができます」、「すること」、「行う」といった冗長な表現は、お手本を参考にして、より簡潔な表現に変換してください。
- 入力の文章には目的語が省略されている場合があります。その場合は、入力から推測できる目的語を補ってください。お手本からは目的語を取り出さないでください。お手本は目的語をどのような文体で挿入するかの参考にしてください。
- 副詞などはひらがなにしてください。具体的には「毎」「全く」「無い」「1つ目」などの漢字はひらがなに置き換えてください。
- 日本語の文章における丸括弧では全角記号を使ってください。
- [@...]や[-@...]は、参考文献のcitationや相互参照です。書き替え後も該当箇所に残してください。
"""

def load_patterns():
    with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_category_embeddings(patterns):
    embeddings = {}
    for cat, ex in patterns.items():
        descriptions = [e["description"] for e in ex]
        embeddings[cat] = sentence_model.encode(descriptions).mean(axis=0)
    return embeddings

def find_top_patterns(input_text, category_embeddings, top_n=TOP_N_PATTERNS):
    input_emb = sentence_model.encode(input_text)
    similarities = []
    for cat, emb in category_embeddings.items():
        similarity = np.dot(input_emb, emb) / (np.linalg.norm(input_emb) * np.linalg.norm(emb))
        similarities.append((cat, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [cat for cat, _ in similarities[:top_n]]

def generate_prompt(input_text, selected_categories, patterns):
    instructions = []
    for cat in selected_categories:
        ex = patterns[cat][0]
        steps = "\n".join(f"- {s}" for s in ex["steps"])
        instructions.append(f"""
### 編集方針: {cat}
{ex["description"]}
編集手順:
{steps}
改善例:
編集前:
{ex["example_before"]}
編集後:
{ex["example_after"]}
""".strip())

    combined_instructions = "\n\n".join(instructions)

    prompt = f"""
あなたは日本語の文章を推敲しています。以下の黄金律と編集方針を守って改善してください。

## 黄金律:
{GOLDEN_RULES}

## 編集方針:
{combined_instructions}

## 改善対象の文章:
{input_text}

改善後の文章のみを出力してください。
"""
    return prompt.strip()

def refine_text_with_gpt4(prompt):
    response = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()
