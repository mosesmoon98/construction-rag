"""
4단계: RAG 파이프라인 (Retrieval-Augmented Generation)

[흐름]
  질문
   -> 1) 벡터 검색: 질문과 의미가 가까운 chunk k개를 벡터 DB 에서 꺼낸다  (3단계 재사용)
   -> 2) 프롬프트 조립: 그 chunk 들을 '참고 자료'로 넣고 질문을 붙인다
   -> 3) LLM 호출: "이 자료만 근거로 답하라"고 지시하며 답변을 받는다      (1단계의 llm.py 재사용)
   -> 답변 + 사용한 자료(출처)

[왜 이렇게 하나]
  - LLM 은 학습하지 않은 '우리 회사 문서'의 구체적 수치를 모른다. 그래서 답할 때
    필요한 근거를 그때그때 찾아서(retrieval) 프롬프트에 넣어준다(augmented).
  - "자료에 없으면 없다고 하라"고 못박아 환각(없는 사실을 지어내는 것)을 줄인다.
  - 어떤 자료를 근거로 썼는지 함께 돌려줘서 사람이 검증할 수 있게 한다.
"""

from dataclasses import dataclass

from llm import get_llm
from vector_store import VectorStore

RAG_SYSTEM_PROMPT = """당신은 건설사업관리(CM/PM) 회사의 문서 도우미입니다.
아래에 주어지는 '참고 자료'만 근거로 사용자 질문에 답하세요.

규칙:
- 참고 자료에 있는 내용만으로 답합니다. 자료에 근거가 없으면
  "제공된 문서에서 관련 내용을 찾을 수 없습니다."라고만 답하세요.
- 추측하거나 일반 상식으로 빈칸을 채우지 마세요.
- 숫자·기준값·조항번호는 자료의 표현을 그대로 옮기세요.
- 답변 맨 끝에 근거로 사용한 자료 번호를 [자료 1], [자료 3] 형식으로 표기하세요.
- 한국어로, 군더더기 없이 답하세요."""


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[dict]      # vector_store.search() 가 돌려준 hit 목록 (출처 표시용)


def _format_context(hits: list[dict]) -> str:
    """검색된 chunk 들을 '[자료 N] (출처: ...) 본문' 형태의 한 덩어리 텍스트로."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[자료 {i}] (출처: {h['source']}, 조각 #{h['chunk_index']})\n{h['text']}"
        )
    return "\n\n".join(blocks)


class RagPipeline:
    def __init__(self, store: VectorStore | None = None, k: int = 4) -> None:
        self.store = store or VectorStore()   # 임베딩 모델 + ChromaDB 준비
        self.llm = get_llm()                  # .env 에 따라 Ollama 또는 Claude
        self.k = k                            # 질문당 가져올 참고 자료 개수

    def ask(self, question: str) -> RagAnswer:
        # 1) 검색
        hits = self.store.search(question, k=self.k)

        # 2) 프롬프트 조립
        user_prompt = (
            f"참고 자료:\n{_format_context(hits)}\n\n"
            f"---\n"
            f"질문: {question}\n\n"
            f"위 참고 자료만 근거로 답하세요."
        )

        # 3) LLM 호출
        answer = self.llm.chat(system=RAG_SYSTEM_PROMPT, prompt=user_prompt)

        return RagAnswer(question=question, answer=answer, sources=hits)
