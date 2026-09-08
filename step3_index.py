r"""
3단계 실행 스크립트

1) data\ 의 모든 문서를 chunk 로 나눈다        (2단계 재사용)
2) 각 chunk 를 임베딩해서 ChromaDB 에 저장한다  (embedder + vector_store)
3) 예시 질문 몇 개로 검색해, 어떤 chunk 가 뽑히는지 확인한다

실행:
  python step3_index.py

실행이 끝나면 ./chroma_db 폴더가 생기고, 다음 단계(RAG)에서 이 DB 를 그대로 쓴다.
"""

from pathlib import Path

from chunker import chunk_directory
from vector_store import VectorStore

DATA_DIR = Path("data")

# 검색이 잘 되는지 확인할 예시 질문들 (일부러 문서와 표현을 다르게 씀)
SAMPLE_QUESTIONS = [
    "공사가 늦어지면 하루에 얼마를 물어내야 하나?",
    "기둥에 쓰는 콘크리트 강도 기준은?",
    "이번 주에 안전 점검에서 지적된 사항은?",
]


def main() -> None:
    # 1) chunk 만들기
    chunks = chunk_directory(DATA_DIR)
    print(f"[1] data\\ 에서 chunk {len(chunks)}개 생성\n")

    # 2) 임베딩 + 저장 (매번 처음부터 다시 만든다)
    store = VectorStore()
    store.reset()
    store.build_index(chunks)
    print()

    # 3) 예시 검색
    print("=" * 70)
    print("예시 질문으로 검색 테스트 (top-4)")
    print("=" * 70)
    for q in SAMPLE_QUESTIONS:
        print(f"\n❓ {q}")
        hits = store.search(q, k=4)
        for rank, h in enumerate(hits, start=1):
            preview = h["text"].replace("\n", " ")[:70]
            print(
                f"  {rank}. [{h['source']} #{h['chunk_index']}] "
                f"유사도 {h['similarity']:.3f}"
            )
            print(f"     {preview}...")


if __name__ == "__main__":
    main()
