#!/usr/bin/env python3
from pathlib import Path
import json
import sys

# Fixed paths
IMAGES_DIR = Path("/workspaces/gorilla/raft/meme_data/memes")
DOCS_DIR = Path("/workspaces/gorilla/raft/meme_data/docs")
OUTPUT_JSON = Path("/workspaces/gorilla/raft/meme_data/scripts/meme_dataset.json")  # ✅ 저장 경로 변경


def main():
    if not IMAGES_DIR.is_dir() or not DOCS_DIR.is_dir():
        print("❌ Missing directories", file=sys.stderr)
        sys.exit(1)

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
    for folder in sorted(IMAGES_DIR.iterdir()):
        if not folder.is_dir():
            continue
        keyword = folder.name
        if keyword not in doc_map:
            continue

        doc_text, doc_link = doc_map[keyword]
        image_files = sorted([
            f for f in folder.glob("*")
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
