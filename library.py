from openai import OpenAI
import streamlit as st

# 👉 OpenAI 클라이언트
client = OpenAI(api_key="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")

# 👉 프롬프트 불러오기
def load_prompt(character):
    path = f"C:/Users/김준형/Documents/공모전 및 프로젝트/도서관/prompts/{character}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# 👉 1. 카톡 스타일 CSS
st.markdown("""
  <style>
    /* 전체 배경 */
    html, body, .stApp {
      background-color: #CFE7FF !important;
    }

    /* 채팅 컨테이너 */
    .chat-container {
      display: flex;
      flex-direction: column;
      padding: 20px;
    }

    /* 사용자 말풍선 (노랑, 오른쪽 정렬, auto width, 텍스트 우측 정렬) */
    .user-message {
      background-color: #FFEB00;
      color: #000;
      padding: 10px 14px;
      border-radius: 18px 0 18px 18px;
      display: inline-block;      /* 말풍선 폭을 내용에 맞춤 */
      width: auto;
      max-width: 70%;             /* 최대 폭 제한 */
      text-align: right;          /* 텍스트를 안에서 오른쪽 정렬 */
      align-self: flex-end;       /* 컨테이너 우측 끝에 붙임 */
      margin: 6px 0 6px auto;     /* 좌측 마진 자동으로 띄워 우측 정렬 유지 */
      font-size: 15px;
      line-height: 1.4;
    }

    /* 챗봇 말풍선 (흰색, 왼쪽 고정) */
    .bot-message {
      background-color: #FFFFFF;
      color: #000;
      padding: 10px 14px;
      border-radius: 0 18px 18px 18px;
      display: inline-block;
      width: auto;
      max-width: 70%;
      text-align: left;
      align-self: flex-start;
      margin: 6px auto 6px 0;
      font-size: 15px;
      line-height: 1.4;
    }
  </style>
""", unsafe_allow_html=True)




# 👉 2. UI: 타이틀 및 캐릭터 선택
st.title("📱 소설 인물 챗봇")
character = st.selectbox("🧑 대화할 등장인물을 선택하세요",
                         ["동호", "유진", "아영", "레이첼"])

# 👉 3. 캐릭터 변경 시 프롬프트 초기화
if "prev_character" not in st.session_state or st.session_state.prev_character != character:
    st.session_state.prev_character = character
    st.session_state.messages = [
        {"role": "system", "content": load_prompt(character)}
    ]

# 👉 4. 사용자 입력받기
user_input = st.text_input("질문을 입력하세요:")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=st.session_state.messages,
        max_tokens=150,
        temperature=0.7
    )
    reply = response.choices[0].message.content.strip()
    if len(reply) > 100:
        reply = reply[:100].rsplit('.', 1)[0] + "."
    st.session_state.messages.append({"role": "assistant", "content": reply})

# 👉 5. 채팅 렌더링
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages[1:]:
    cls = "user-message" if msg["role"] == "user" else "bot-message"
    st.markdown(f'<div class="{cls}">{msg["content"]}</div>',
                unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

