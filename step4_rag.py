r"""
4단계 실행 스크립트: RAG 챗봇

전제: 먼저 step3_index.py 를 한 번 실행해 벡터 DB(.\chroma_db)가 있어야 한다.

실행:
  python step4_rag.py "지체상금은 하루에 얼마인가?"     # 질문 한 번
  python step4_rag.py                                   # 대화형 (여러 번 질문, 빈 줄이면 종료)
"""

import sys

from rag import RagAnswer, RagPipeline


def print_answer(result: RagAnswer) -> None:
    print("\n" + "=" * 70)
    print(f"질문: {result.question}")
    print("=" * 70)
    print(result.answer)
    print("\n--- 사용한 참고 자료 ---")
    for i, h in enumerate(result.sources, start=1):
        preview = h["text"].replace("\n", " ")[:60]
        print(
            f"[자료 {i}] {h['source']} #{h['chunk_index']}  "
            f"(유사도 {h['similarity']:.3f})"
        )
        print(f"        {preview}...")


def main() -> None:
    print("[준비] 임베딩 모델 + 벡터 DB + LLM 로딩 중...")
    pipeline = RagPipeline()

    # 인자로 질문을 주면 한 번만 답하고 종료
    if len(sys.argv) >= 2:
        question = " ".join(sys.argv[1:])
        print_answer(pipeline.ask(question))
        return

    # 인자가 없으면 대화형
    print("\n건설 문서 RAG 챗봇 — 질문을 입력하세요. (빈 줄 입력 시 종료)\n")
    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        print_answer(pipeline.ask(question))
        print()


if __name__ == "__main__":
    main()
