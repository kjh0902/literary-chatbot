import os, glob, json, re, uuid
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

DATA_DIR      = os.getenv("DATA_DIR", "rag/.data")
ARTIFACT_DIR  = os.getenv("ARTIFACT_DIR", "rag/.artifacts")
MAX_CHARS     = int(os.getenv("MAX_CHARS", "1200"))
OVERLAP       = int(os.getenv("OVERLAP", "150"))
EMB_MODEL     = os.getenv("EMB_MODEL", "text-embedding-3-small")
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "64"))

INCLUDE_SCENES     = os.getenv("INCLUDE_SCENES", "1") == "1"
INCLUDE_CHAPTERS   = os.getenv("INCLUDE_CHAPTERS", "1") == "1"
INCLUDE_CHARACTERS = os.getenv("INCLUDE_CHARACTERS", "1") == "1"
INCLUDE_META       = os.getenv("INCLUDE_META", "1") == "1"
INCLUDE_FULLTEXT   = os.getenv("INCLUDE_FULLTEXT", "1") == "1"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

def find_files(pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, pattern)))

def read_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def normalize_space(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = s.replace("\u201c","\"").replace("\u201d","\"").replace("\u2018","'").replace("\u2019","'")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sent_split(text: str) -> List[str]:
  import re

def sent_split(text: str):
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r'([.!?…])\s+', r'\1¶', text)
    text = re.sub(r'(다\.)\s+', r'\1¶', text)
    text = re.sub(r'\n{2,}', '¶', text)
    parts = [p.strip() for p in text.split('¶') if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
    return parts


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> List[str]:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return [text]
    sents, chunks, buf = sent_split(text), [], ""
    for s in sents:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf = buf + " " + s
        else:
            chunks.append(buf)
            buf = (buf[-overlap:] + " " + s) if overlap>0 and len(buf)>overlap else s
    if buf: chunks.append(buf)
    return chunks

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]

def load_all_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []

    if INCLUDE_SCENES:
        for p in find_files("*_scenes_*.jsonl"):
            for row in read_jsonl(p):
                txt = row.get("scene_full_text") or row.get("text") or ""
                if not txt: continue
                kind = "scene" if row.get("scene_id") != "_BLOCK_RAW_" else "scene_raw_block"
                meta = {
                    "work_id": row.get("work_id","unknown"),
                    "kind": kind,
                    "scene_id": row.get("scene_id"),
                    "scene_title": row.get("scene_title"),
                    "chapter_id": row.get("chapter_id"),
                    "chapter_label": row.get("chapter_label"),
                    "spoiler_level": row.get("spoiler_level", 3),
                    "source_file": os.path.basename(p),
                }
                for i, ch in enumerate(chunk_text(txt)):
                    chunks.append(Chunk(
                        id=f"{meta['work_id']}::{kind}::{row.get('scene_id','raw')}::{i}",
                        text=ch, metadata=meta
                    ))

    if INCLUDE_CHAPTERS:
        for p in find_files("*_chapters_*.jsonl"):
            for row in read_jsonl(p):
                txt = row.get("chapter_full_text") or row.get("text") or ""
                if not txt: continue
                meta = {
                    "work_id": row.get("work_id","unknown"),
                    "kind": "chapter",
                    "chapter_id": row.get("chapter_id"),
                    "chapter_label": row.get("chapter_label"),
                    "spoiler_level": row.get("spoiler_level", 3),
                    "source_file": os.path.basename(p),
                }
                for i, ch in enumerate(chunk_text(txt)):
                    chunks.append(Chunk(
                        id=f"{meta['work_id']}::chapter::{row.get('chapter_id','?')}::{i}",
                        text=ch, metadata=meta
                    ))

    if INCLUDE_CHARACTERS:
        for p in find_files("*_characters_*.jsonl"):
            for row in read_jsonl(p):
                txt = row.get("full_bio") or row.get("text") or ""
                if not txt: continue
                char = row.get("character") or "UNKNOWN"
                kind = "persona" if char != "_SECTION_RAW_" else "characters_raw"
                meta = {
                    "work_id": row.get("work_id","unknown"),
                    "kind": kind,
                    "character": char,
                    "source_file": os.path.basename(p),
                }
                for i, ch in enumerate(chunk_text(txt)):
                    chunks.append(Chunk(
                        id=f"{meta['work_id']}::{kind}::{char}::{i}",
                        text=ch, metadata=meta
                    ))

    if INCLUDE_META:
        for p in find_files("*_meta_*.jsonl"):
            for row in read_jsonl(p):
                work = row.get("work_id","unknown")
                for field in ["overview_raw","chapters_raw","scenes_raw","characters_raw"]:
                    txt = row.get(field) or ""
                    if not txt: continue
                    meta = {
                        "work_id": work,
                        "kind": f"meta_{field}",
                        "source_file": os.path.basename(p),
                    }
                    for i, ch in enumerate(chunk_text(txt)):
                        chunks.append(Chunk(
                            id=f"{work}::meta::{field}::{i}",
                            text=ch, metadata=meta
                        ))

    if INCLUDE_FULLTEXT:
        for p in find_files("*_fulltext*.jsonl"):
            for row in read_jsonl(p):
                txt = row.get("full_text") or row.get("text") or ""
                if not txt: continue
                work = row.get("work_id","unknown")
                meta = {
                    "work_id": work,
                    "kind": "fulltext",
                    "source_file": os.path.basename(p),
                }
                for i, ch in enumerate(chunk_text(txt)):
                    chunks.append(Chunk(
                        id=f"{work}::fulltext::{i}",
                        text=ch, metadata=meta
                    ))

    return chunks

def embed_texts(texts: List[str], model: str = EMB_MODEL, batch_size: int = BATCH_SIZE) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI()  # OPENAI_API_KEY 환경변수 필요
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out

def main():
    chunks = load_all_chunks()
    if not chunks:
        raise SystemExit(f"[ERROR] {DATA_DIR}에서 로드된 문서가 없습니다. JSONL 패턴을 확인하세요.")

    print(f"[LOAD] docs(chunks) = {len(chunks)}  | from: {DATA_DIR}")
    texts = [c.text for c in chunks]
    vecs  = embed_texts(texts, model=EMB_MODEL, batch_size=BATCH_SIZE)
    dim   = len(vecs[0]) if vecs else 0
    print(f"[EMBED] vectors = {len(vecs)}, dim = {dim}, model = {EMB_MODEL}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    chunks_path = os.path.join(ARTIFACT_DIR, f"chunks_{stamp}.jsonl")
    embs_path   = os.path.join(ARTIFACT_DIR, f"embeddings_{stamp}.jsonl")

    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c.id, "text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")
    with open(embs_path, "w", encoding="utf-8") as f:
        for c, v in zip(chunks, vecs):
            f.write(json.dumps({"id": c.id, "embedding": v}, ensure_ascii=False) + "\n")

    print(f"[SAVE] chunks     → {chunks_path}")
    print(f"[SAVE] embeddings → {embs_path}")
    print("[TIP] 다음 단계에서 Chroma DB에 적재할 때 kind/work_id로 필터링해서 컬렉션 분리 가능(예: persona 전용 컬렉션).")

if __name__ == "__main__":
    main()


