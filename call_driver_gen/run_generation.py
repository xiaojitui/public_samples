import os
import json
from glob import glob
from utils_token_chunking import chunk_by_tokens
from llm_pipeline import (
    extract_questions_from_chunk,
    consolidate_categories,
    generate_answers
)

DATA_FOLDERS = [
    "data/emails",
    "data/announcements",
    "data/training_docs"
]


def load_all_text_files(folder_paths):
    docs = []
    for folder in folder_paths:
        for file in glob(f"{folder}/**/*.txt", recursive=True):
            with open(file, "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs


def main():
    print("Loading documents...")
    docs = load_all_text_files(DATA_FOLDERS)

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_by_tokens(doc))

    print(f"Total chunks created: {len(all_chunks)}")

    # MAP STEP
    chunk_outputs = []
    for i, chunk in enumerate(all_chunks):
        result = extract_questions_from_chunk(chunk, i + 1)
        chunk_outputs.append(result)

    # REDUCE STEP
    consolidated = consolidate_categories(chunk_outputs)

    # FINAL ANSWERS
    final_qna = generate_answers(consolidated)

    with open("call_center_qna.json", "w", encoding="utf-8") as f:
        json.dump(final_qna, f, indent=2)

    print("🎉 Done! Output saved to call_center_qna.json")


if __name__ == "__main__":
    main()
