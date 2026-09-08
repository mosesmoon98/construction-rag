r"""
2단계 실행 스크립트

data\ 폴더의 모든 .txt 문서를 chunk 로 나누고,
  - 문서별로 몇 조각이 나왔는지 / 조각 길이 통계
  - 실제 조각 2개의 내용 (겹치는 부분을 눈으로 확인)
를 출력한다.

실행:
  python step2_chunk.py
"""

from pathlib import Path

from chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_directory, chunk_document

DATA_DIR = Path("data")


def main() -> None:
    print(f"[설정] CHUNK_SIZE = {CHUNK_SIZE}자,  CHUNK_OVERLAP = {CHUNK_OVERLAP}자")
    print("=" * 60)

    for path in sorted(DATA_DIR.glob("*.txt")):
        chunks = chunk_document(path)
        sizes = [len(c.text) for c in chunks]
        original_len = len(path.read_text(encoding="utf-8"))
        print(f"■ {path.name}")
        print(f"   원문 {original_len:,}자  ->  chunk {len(chunks)}개")
        print(
            f"   조각 길이: 최소 {min(sizes)}자 / "
            f"평균 {sum(sizes) // len(sizes)}자 / 최대 {max(sizes)}자"
        )
        print()

    all_chunks = chunk_directory(DATA_DIR)
    print("=" * 60)
    print(f"=> data\\ 전체에서 총 {len(all_chunks)}개 chunk 생성")
    print("=" * 60)

    # 계약서의 처음 두 조각을 실제로 출력한다.
    # 두 번째 조각의 맨 앞이 첫 번째 조각의 꼬리와 겹치는 것을 확인할 수 있다.
    print("\n[예시] sample_03_contract.txt 의 첫 두 조각\n")
    sample_chunks = chunk_document(DATA_DIR / "sample_03_contract.txt")
    for c in sample_chunks[:2]:
        print("-" * 60)
        print(f"[{c.source}  #{c.chunk_index}]  ({len(c.text)}자)")
        print("-" * 60)
        print(c.text)
        print()


if __name__ == "__main__":
    main()
