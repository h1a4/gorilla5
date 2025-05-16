#!/usr/bin/env python3
from pathlib import Path
import json
import sys

# ✅ 실제 경로 설정
IMAGES_DIR = Path("/workspaces/gorilla/raft/meme_data/memes/1_earth-chan_meme_pinterest")
DOCS_DIR = Path("/workspaces/gorilla/raft/meme_data/docs/environment")
OUTPUT_JSON = Path("/workspaces/gorilla/raft/meme_data/scripts/earth-chan.json")


def clean_keyword(folder_name):
    """
    폴더 이름에서 키워드만 추출하는 전처리 함수.
    예: '1_earth-chan_meme_pinterest' -> 'earth-chan'
    """
    # 숫자, 밈, 플랫폼 이름 등을 제거 (필요시 정교화 가능)
    for prefix in ["1_", "2_", "3_", "4_", "5_", "6_", "7_", "8_", "9_", "10_"]:
        if folder_name.startswith(prefix):
            folder_name = folder_name[len(prefix):]
    for suffix in ["_meme_pinterest", "_meme_reddit"]:
        if folder_name.endswith(suffix):
            folder_name = folder_name[:-len(suffix)]
    return folder_name


def main():
    if not IMAGES_DIR.is_dir() or not DOCS_DIR.is_dir():
        print("❌ Missing directories", file=sys.stderr)
        sys.exit(1)

    # 문서 로딩
    doc_map = {}
    for doc in DOCS_DIR.glob("*.txt"):
        keyword = doc.stem
        try:
            lines = doc.read_text(encoding='utf-8').splitlines()
            if not lines:
                continue
            doc_link = lines[0].strip()
            doc_text = "\n".join(lines[1:]).strip()
            doc_map[keyword] = (doc_text, doc_link)
        except Exception as e:
            print(f"⚠️ Skipping doc {doc.name}: {e}", file=sys.stderr)

    entries = []

    # 폴더 이름 → 키워드 추출
    keyword = clean_keyword(IMAGES_DIR.name)
    if keyword not in doc_map:
        print(f"❌ No matching document for keyword: {keyword}", file=sys.stderr)
        sys.exit(1)

    doc_text, doc_link = doc_map[keyword]

    # 이미지 파일 로딩
    image_files = sorted([
        f for f in IMAGES_DIR.glob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    for idx, img in enumerate(image_files, start=1):
        entries.append({
            "title": f"{keyword}_{idx}",
            "keyword": keyword,
            "image_path": str(img.resolve()),
            "doc_link": doc_link,
            "doc_text": doc_text
        })

    if not entries:
        print("❌ No entries found.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_JSON.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"✅ {len(entries)} image entries written to {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
