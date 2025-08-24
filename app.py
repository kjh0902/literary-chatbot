import streamlit as st
st.set_page_config(page_title="📚 소설 캐릭터 챗봇", layout="centered")

try:
    import pysqlite3  # type: ignore
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
    
# RAG 검색 + 페르소나 주입 + 답변 생성
from dotenv import load_dotenv
load_dotenv()

import os, re
import chromadb
from rank_bm25 import BM25Okapi
from openai import OpenAI

# ================= 기본 설정 =================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR   = os.getenv("PERSIST_DIR") or os.path.join(BASE_DIR, "rag", ".chroma")
COLLECTION    = os.getenv("COLLECTION", "library-all")
MODEL         = os.getenv("MODEL", "gpt-4o")
TOP_K         = int(os.getenv("TOP_K", "6"))
EMB_MODEL     = "text-embedding-3-small"

# OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
oa = OpenAI()

WORK_ID_MAP = {
    "지구 끝의 온실": "jigu-ggut-onshil",
    "종의 기원": "jong-ui-giwon",
    "소년이 온다": "so-nyeon-i-onda"
}

# ================= 유틸/토큰화 =================
def tokenize(text: str):
    # 영문/숫자/한글만 추출 → 소문자 → 토큰 리스트
    return re.findall(r"[0-9A-Za-z가-힣]+", (text or "").lower())

def reciprocal_rank_fusion(results_lists, k=60):
    scores = {}
    for res in results_lists:
        for rank, doc_id in enumerate(res, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0/(k+rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ================= Chroma 로드 =================
client = chromadb.PersistentClient(path=PERSIST_DIR)
col = client.get_or_create_collection(name=COLLECTION, embedding_function=None)

results = col.get(include=["documents","metadatas"], limit=999_999)
all_ids   = results.get("ids", []) or []
all_docs  = results.get("documents", []) or []
all_metas = results.get("metadatas", []) or []

st.sidebar.write("loaded_from_chroma:", len(all_docs))

# 안전 필터링 + 토큰화 (bm25 ZeroDivisionError 방지)
filtered_ids, filtered_docs, filtered_metas, tokenized_docs = [], [], [], []
for doc_id, doc, meta in zip(all_ids, all_docs, all_metas):
    if not isinstance(doc, str):
        continue
    if not doc.strip():
        continue
    toks = tokenize(doc)
    if not toks:
        continue
    filtered_ids.append(doc_id)
    filtered_docs.append(doc)
    filtered_metas.append(meta or {})
    tokenized_docs.append(toks)

st.sidebar.write("after_filter:", len(filtered_docs))

bm25 = None
if tokenized_docs:
    bm25 = BM25Okapi(tokenized_docs)
else:
    st.warning("BM25 인덱스를 만들 문서가 없습니다. 데이터 경로 또는 문서 내용을 확인하세요.")

# id -> (text, meta)
id2doc = {i: (t, m) for i, t, m in zip(filtered_ids, filtered_docs, filtered_metas)}

# ================= 검색 함수 =================
def hybrid_retrieve(query, top_k, work_id=None):
    if not query or not query.strip():
        return []

    # 1) 벡터 검색
    emb = oa.embeddings.create(model=EMB_MODEL, input=query).data[0].embedding
    vec_res = col.query(
        query_embeddings=[emb],
        n_results=top_k * 3,
        where={"work_id": work_id} if work_id else None
    )
    vec_ids = vec_res["ids"][0] if vec_res.get("ids") else []

    # 2) BM25 검색 (있을 때만)
    bm25_ids = []
    if bm25 is not None:
        toks = tokenize(query)
        if toks:
            scores = bm25.get_scores(toks)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k*3]
            # ranked의 인덱스는 filtered_docs/tokenized_docs 기준
            bm25_ids = [
                filtered_ids[i] for i, _ in ranked
                if (not work_id or (filtered_metas[i].get("work_id") == work_id))
            ]

    # 3) RRF 융합
    fused = reciprocal_rank_fusion([vec_ids, bm25_ids])

    hits = []
    for did, _ in fused:
        if did in id2doc:
            txt, meta = id2doc[did]
            hits.append((did, txt, meta))
            if len(hits) >= top_k:
                break
    return hits

# ================= 프롬프트 생성 =================
def make_prompt(query, hits, work_id=None, speak_as=None, history=[]):
    persona_block = ""
    if speak_as and work_id:
        # id2doc를 순회해 페르소나/등장인물 원문을 찾음
        persos = []
        for _id, (txt, meta) in id2doc.items():
            if meta.get("work_id") == work_id and meta.get("kind") in ["persona", "characters_raw"]:
                ch = meta.get("character", "") or ""
                if speak_as in ch:
                    persos.append(txt)
        if persos:
            persona_block = f"[인물 페르소나: {speak_as}]\n{persos[0]}"

    context_cards = []
    for _, txt, meta in hits:
        title = meta.get("scene_title") or meta.get("chapter_label") or meta.get("kind")
        context_cards.append(f"### {title}\n{txt}")

    system = (
        "당신은 소설 속 인물의 말투를 재현하는 AI입니다.\n"
        "컨텍스트를 근거로 사용하세요.\n"
        "당신이 소설 속 등장인물이라고 생각하세요.\n"
        "대화할 때는 해당 인물의 말투/가치관을 반영해 1~2문장 이내로 대답하세요.\n"
        "답할때는 대화하듯이 자연스럽게 얘기해"
    )
    if persona_block:
        system += "\n\n" + persona_block

    msgs = [{"role": "system", "content": system}]
    if history:
        msgs.extend(history[-6:])   # 최근 6턴만 유지

    user = f"질문: {query}\n\n[컨텍스트]\n" + "\n\n".join(context_cards[:8])
    msgs.append({"role": "user", "content": user})
    return msgs

# ================= 답변 생성 =================
def generate(messages):
    try:
        resp = oa.responses.create(model=MODEL, input=messages)
        return getattr(resp, "output_text", "").strip()
    except Exception:
        comp = oa.chat.completions.create(model=MODEL, messages=messages)
        return comp.choices[0].message.content.strip()

# ================= Streamlit UI =================
st.set_page_config(page_title="📚 소설 캐릭터 챗봇", layout="centered")

# 👉 카톡 스타일 CSS
st.markdown("""
<style>
html, body, .stApp { background-color: #CFE7FF !important; }
.chat-container { display: flex; flex-direction: column; padding: 20px; }
.user-message {
  background-color: #FFEB00; color: #000;
  padding: 10px 14px; border-radius: 18px 0 18px 18px;
  max-width: 70%; font-size: 15px; line-height: 1.4;
  align-self: flex-end; margin: 6px 0 6px auto;
}
.bot-message {
  background-color: #FFFFFF; color: #000;
  padding: 10px 14px; border-radius: 0 18px 18px 18px;
  max-width: 70%; font-size: 15px; line-height: 1.4;
  align-self: flex-start; margin: 6px auto 6px 0;
}
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state.history = []
if "work_id" not in st.session_state:
    st.session_state.work_id = None
if "speak_as" not in st.session_state:
    st.session_state.speak_as = None

st.title("📚 소설 속 인물과 대화하기")

prev_work = st.session_state.get("work_id")
prev_speak = st.session_state.get("speak_as")

work_kor = st.selectbox("작품 선택", ["지구 끝의 온실", "종의 기원", "소년이 온다"])
st.session_state.work_id = WORK_ID_MAP.get(work_kor)
st.session_state.speak_as = st.text_input("인물 선택 (예: 유진, 동호, 아영 등)", "")

# 작품/인물이 바뀌면 대화 초기화
if (prev_work and prev_work != st.session_state.work_id) or \
   (prev_speak and prev_speak != st.session_state.speak_as):
    st.session_state.history = []
    st.rerun()

# 채팅 UI
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 입력창
query = st.text_input("메시지를 입력하세요", key="input")

if st.button("보내기", type="primary") and query.strip():
    hits = hybrid_retrieve(query, TOP_K, st.session_state.work_id)
    msgs = make_prompt(query, hits,
                       work_id=st.session_state.work_id,
                       speak_as=st.session_state.speak_as,
                       history=st.session_state.history)
    ans = generate(msgs)

    # 메모리에 기록
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": ans})

    st.rerun()

# 사이드바 진단
st.sidebar.write({
    "PERSIST_DIR": PERSIST_DIR,
    "COLLECTION": COLLECTION,
    "TOP_K": TOP_K,
    "has_bm25": bm25 is not None,
    "filtered_docs": len(filtered_docs),
})



