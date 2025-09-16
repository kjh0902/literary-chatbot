# literary-chatbot
This project delivers an AI-driven chatbot that lets users step inside the novels and chat directly with the book’s characters.

- RAG (Retrieval-Augmented Generation)
- Vector DB: Chroma 
- UI: Streamlit
- LLM: OpenAI API

- Review
처음엔 OpenAI API가 아니라 Plug-and-Play Language Model (PPLM)를 사용할 계획이었다. 구현은 성공했지만, 짧은 질의에도 이상한 답변을 내놓았다. 한국어 뉘앙스를 제대로 못 이해했다. 모델 자체의 언어 능력이 결과 품질에 직결된다는 걸 체감했다. 소설 속 인물과 대화하는 것을 구현하기 위해서는 대화 품질을 높일 다른 방법을 구상해야했다.

품질을 일정 수준 이상으로 끌어올리려면 토대부터 안정적이어야 한다고 보고 OpenAI API로 전환했다. 전환 효과는 즉각적이었다. 답변이 실제로 등장인물이 답하는 듯한 느낌을 주었고, 맥락도 얼추 맞았다. 하지만 여기서도 금방 한계가 드러났다. 모델이 자신 있게 틀리는 장면들(나중에 찾아보니 이런 문제를 환각이라고 한다)이 반복된 것이다. 특히 문학 작품과 같이 세부 묘사가 많은 텍스트에서는 장면을 섞거나, 등장인물 관계를 엇갈리게 설명하거나, 존재하지 않는 등장인물을 언급하는 실수가 나왔다. 출처가 없는 답변이 모델의 신뢰를 잃게 만들었다.

결론은 간단했다. open ai 모델의 언어 생성 능력만으론 사실성을 담보하기 어렵다. 그래서 검색과 출처를 모델의 문장 생성 과정에 끼워 넣는 Retrieval-Augmented Generation(RAG)로 방향을 틀었다. 답변의 근거를 찾아서 답변의 신뢰도를 높이는 것이 목표였다.

RAG를 붙이면서 설계를 이렇게 잡았다. 우선 등장인물 자료를 잘게 나눠 임베딩하고, ChromaDB 같은 벡터 스토어에 메타데이터와 함께 넣었다. 파편화로 맥락이 끊기는 걸 줄이려고 청크 크기와 오버랩을 여러 조합으로 테스트했다. UI는 Streamlit로 빠르게 구현했다.
