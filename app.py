r"""
5단계: Streamlit 웹 화면 (챗봇 UI)

지금까지 만든 RAG 파이프라인(rag.py)에 웹 화면만 붙인다.
- 사용자가 입력창에 질문 -> RagPipeline.ask() -> 답변 + 근거 문서 표시
- 대화 내용은 화면에 계속 쌓인다 (채팅처럼)

실행:
  streamlit run app.py

전제: 먼저 `python step3_index.py` 로 벡터 DB(.\chroma_db)가 만들어져 있어야 한다.
"""

import streamlit as st

from rag import RagPipeline

st.set_page_config(page_title="건설 문서 RAG 챗봇", page_icon="🏗️")


# ── 파이프라인은 한 번만 로딩 (임베딩 모델 로딩이 느리므로 캐시) ──────────
@st.cache_resource
def load_pipeline() -> RagPipeline:
    return RagPipeline()


pipeline = load_pipeline()


# ── 사이드바: 현재 어떤 구성으로 도는지 보여준다 ─────────────────────────
with st.sidebar:
    st.header("⚙️ 현재 구성")
    st.write(f"**LLM**  ·  `{type(pipeline.llm).__name__}`")
    st.write(f"모델: `{pipeline.llm.model}`")
    st.write(f"**임베딩**  ·  `{pipeline.store.embedder.model_name}`")
    st.write(f"**색인된 chunk**: {pipeline.store.collection.count()}개")
    st.write(f"**검색 개수(k)**: {pipeline.k}")
    st.caption(
        "LLM 을 바꾸려면 `.env` 의 `LLM_PROVIDER` 를 수정하고 앱을 재시작하세요 "
        "(ollama ↔ claude)."
    )
    if st.button("대화 비우기"):
        st.session_state.messages = []
        st.rerun()


# ── 메인 ────────────────────────────────────────────────────────────────
st.title("🏗️ 건설 문서 RAG 챗봇")
st.caption("`data/` 폴더의 시방서·계약서·공사보고서 내용을 근거로만 답합니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict]) -> None:
    """답변 아래에 접히는 형태로 근거 문서를 보여준다."""
    with st.expander(f"📎 근거 문서 {len(sources)}건 보기"):
        for i, h in enumerate(sources, start=1):
            st.markdown(
                f"**[자료 {i}] `{h['source']}` · 조각 #{h['chunk_index']}** "
                f"— 유사도 {h['similarity']:.3f}"
            )
            st.text(h["text"])
            st.divider()


# 지난 대화 다시 그리기
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# 입력창
if question := st.chat_input("건설 문서에 대해 질문하세요 (예: 지체상금은 하루에 얼마인가?)"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("문서 검색 + 답변 생성 중..."):
            result = pipeline.ask(question)
        st.markdown(result.answer)
        render_sources(result.sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )
