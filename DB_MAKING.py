from dotenv import load_dotenv
load_dotenv()

import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import glob, json, uuid
from typing import List, Dict, Any
import chromadb
from openai import OpenAI

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "rag/.artifacts").replace("\\","/")
PERSIST_DIR  = os.getenv("PERSIST_DIR", "rag/.chroma").replace("\\","/")
COLLECTION   = os.getenv("COLLECTION", "library-all")

CHUNKS_PATH  = os.getenv("CHUNKS_PATH", "")
EMBS_PATH    = os.getenv("EMBS_PATH", "")

FILTER_WORKS = [s.strip() for s in os.getenv("FILTER_WORKS","").split(",") if s.strip()]
FILTER_KINDS = [s.strip() for s in os.getenv("FILTER_KINDS","").split(",") if s.strip()]

EMB_MODEL = "text-embedding-3-small"   # OpenAI 임베딩 모델

def latest(path_glob: str) -> str:
    files = sorted(glob.glob(path_glob), key=lambda p: os.path.getmtime(p))
    return files[-1] if files else ""

def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def kind_matches(kind: str, patterns: List[str]) -> bool:
    if not patterns:
        return True
    for pat in patterns:
        if pat.endswith("*"):
            if kind.startswith(pat[:-1]): return True
        elif kind == pat:
            return True
    return False

def main():
    chunks_file = CHUNKS_PATH or latest(f"{ARTIFACT_DIR}/chunks_*.jsonl")
    embs_file   = EMBS_PATH   or latest(f"{ARTIFACT_DIR}/embeddings_*.jsonl")
    if not chunks_file or not embs_file:
        raise SystemExit(f"[ERROR] chunks/embeddings 파일을 찾을 수 없음.\n  chunks={chunks_file}\n  embs={embs_file}")

    chunks = load_jsonl(chunks_file)
    embs   = load_jsonl(embs_file)
    emb_map = {e["id"]: e["embedding"] for e in embs}

    ids, docs, metas, vecs = [], [], [], []
    seen = set()

    for rec in chunks:
        base_id  = rec["id"]
        text = rec["text"]
        meta = rec.get("metadata", {})
        work = meta.get("work_id","unknown")
        kind = meta.get("kind","unknown")

        if FILTER_WORKS and work not in FILTER_WORKS:
            continue
        if not kind_matches(kind, FILTER_KINDS):
            continue

        vec = emb_map.get(base_id)
        if vec is None:
            continue

        # 중복 방지
        _id = base_id
        if _id in seen:
            _id = f"{base_id}::{uuid.uuid4().hex[:6]}"
        seen.add(_id)

        ids.append(_id); docs.append(text); metas.append(meta); vecs.append(vec)

    if not ids:
        raise SystemExit("[WARN] 업서트할 레코드가 없습니다. 필터나 파일 경로를 확인하세요.")

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    # embedding_function=None → 사전 계산된 벡터만 사용
    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space":"cosine"},
        embedding_function=None
    )

    # 배치 업서트
    BATCH = 500
    for i in range(0, len(ids), BATCH):
        col.upsert(
            ids=ids[i:i+BATCH],
            documents=docs[i:i+BATCH],
            metadatas=metas[i:i+BATCH],
            embeddings=vecs[i:i+BATCH],
        )
    print(f"[UPSERT] {len(ids)} items → collection='{COLLECTION}' @ {PERSIST_DIR}")

    client_oa = OpenAI()
    q = "기억과 상실의 주제"
    emb = client_oa.embeddings.create(model=EMB_MODEL, input=[q]).data[0].embedding

    res = col.query(query_embeddings=[emb], n_results=3)
    print("[CHECK] sample query:", q)
    print("[CHECK] top ids:", (res.get("ids") or [[]])[0])

if __name__ == "__main__":
    main()


