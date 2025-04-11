import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

DATA_RAW = 'data/raw_markdown/'
DATA_REFINED = 'data/refined_markdown/'
OUTPUT_FILE = 'data/editing_patterns.json'

EDIT_CATEGORIES = [
    "構成の変更 (Reordering)",
    "表現の簡潔化 (Simplification)",
    "トーンや口調の調整 (Tone adjustment)",
    "明確化 (Clarity improvement)",
    "冗長表現の削除 (Removal of redundancy)"
]

def load_document_pairs():
    pairs = []
    filenames = [f for f in os.listdir(DATA_RAW) if f.endswith('.md')]
    for fname in filenames:
        with open(os.path.join(DATA_RAW, fname), 'r', encoding='utf-8') as f_raw, \
             open(os.path.join(DATA_REFINED, fname), 'r', encoding='utf-8') as f_refined:
            pairs.append({
                'filename': fname,
                'before': f_raw.read(),
                'after': f_refined.read()
            })
    return pairs

def classify_edit(pair):
    prompt = f"""
あなたは、以下の「編集前」と「編集後」の文章を比較して、どの編集カテゴリに最もよく当てはまるか判断してください。

編集カテゴリ：
{', '.join(EDIT_CATEGORIES)}

編集前：
{pair['before']}

編集後：
{pair['after']}

あなたの出力は以下のフォーマットで行ってください：
{{
  "category": "該当する編集カテゴリ（上記リストから1つ選択）",
  "description": "具体的にどのような編集が行われたかの簡潔な説明（日本語で）",
  "steps": ["この編集を実行する具体的な手順を箇条書きで（日本語で）"]
}}
"""

    response = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return json.loads(response.choices[0].message.content.strip())

def main():
    pairs = load_document_pairs()
    patterns = {}

    for idx, pair in enumerate(pairs):
        print(f"Processing {pair['filename']} ({idx+1}/{len(pairs)})...")
        try:
            edit_analysis = classify_edit(pair)
            category = edit_analysis['category']
            if category not in patterns:
                patterns[category] = []
            
            patterns[category].append({
                'filename': pair['filename'],
                'description': edit_analysis['description'],
                'steps': edit_analysis['steps'],
                'example_before': pair['before'],
                'example_after': pair['after']
            })
        except Exception as e:
            print(f"Error processing {pair['filename']}: {e}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)

    print("Semantic decomposition completed and editing patterns saved.")

if __name__ == "__main__":
    main()
