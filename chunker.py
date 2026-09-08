"""
2단계: 문서를 작은 조각(chunk)으로 나누는 기능

[왜 나누나?]
  1) 임베딩 모델·LLM 에 한 번에 넣을 수 있는 글자 수에는 한계가 있다.
  2) 나중에 "질문과 관련된 부분만" 골라서 LLM 에게 주려면(=RAG 의 R, 검색),
     문서가 검색 가능한 작은 단위로 쪼개져 있어야 한다.
  3) 문서 전체를 매번 LLM 에 넣으면 느리고, (유료 API 라면) 비싸다.

[방법: 줄 단위 패킹 + 줄 단위 overlap]
  - 건설 문서(시방서 조항, 계약 조문, 보고서 항목)는 대부분 '한 줄 = 하나의
    완결된 항목' 이다. 그래서 줄을 기본 단위로 삼는다.
  - 줄들을 목표 크기(CHUNK_SIZE) 이하가 되도록 순서대로 묶는다.
  - 새 조각을 시작할 때, 직전 조각의 '마지막 몇 줄'(합쳐서 CHUNK_OVERLAP 이하)을
    앞에 다시 넣어 문맥이 이어지게 한다.
  - 이렇게 하면 조각의 시작과 끝이 항상 '줄 경계' 라서, 단어·숫자 중간에서
    잘리지 않는다. (첫 버전은 글자 단위로 잘라서 "27,720,00 / 0,000원" 처럼
    깨졌고, 그게 검색 품질을 떨어뜨렸다.)
"""

from dataclasses import dataclass
from pathlib import Path

# ── 조절 가능한 설정 ─────────────────────────────────────────────
CHUNK_SIZE = 400        # 한 조각의 목표 최대 길이 (글자 수)
CHUNK_OVERLAP = 80      # 이웃한 조각끼리 겹치는 길이 (글자 수, CHUNK_SIZE 의 20%)
# ────────────────────────────────────────────────────────────────

# CHUNK_SIZE 보다 긴 '한 줄'을 쪼갤 때만 쓰는 하위 경계들 (드문 경우).
_FALLBACK_SEPARATORS = [". ", "。", " ", ""]


def _split_oversized_line(line: str) -> list[str]:
    """CHUNK_SIZE 보다 긴 한 줄을 문장 -> 공백 -> 글자 순으로 쪼갠다."""
    parts = [line]
    for sep in _FALLBACK_SEPARATORS:
        if all(len(p) <= CHUNK_SIZE for p in parts):
            break
        nxt: list[str] = []
        for p in parts:
            if len(p) <= CHUNK_SIZE:
                nxt.append(p)
            elif sep == "":
                nxt.extend(p[i:i + CHUNK_SIZE] for i in range(0, len(p), CHUNK_SIZE))
            else:
                nxt.extend(p.split(sep))
        parts = nxt
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Chunk:
    """조각 하나. 텍스트뿐 아니라 '어디서 왔는지'(출처)도 함께 들고 다닌다.
    이 출처 정보는 4단계에서 답변의 근거를 표시할 때 쓴다."""

    text: str
    source: str          # 원본 파일명 (예: sample_03_contract.txt)
    chunk_index: int     # 그 문서 안에서 몇 번째 조각인지 (0부터)


def chunk_text(text: str) -> list[str]:
    """문자열 하나를 조각(문자열) 리스트로 나눈다."""
    text = text.replace("\r\n", "\n")

    # 1) 빈 줄을 뺀 '줄' 목록을 만든다. 너무 긴 줄은 미리 잘게 쪼갠다.
    units: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if len(line) <= CHUNK_SIZE:
            units.append(line)
        else:
            units.extend(_split_oversized_line(line))

    # 2) 줄들을 CHUNK_SIZE 이하로 묶고, 조각 사이에 줄 단위 overlap 을 준다.
    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        would_be = "\n".join([*current, unit])
        if current and len(would_be) > CHUNK_SIZE:
            chunks.append("\n".join(current))
            # 직전 조각의 마지막 줄들을 CHUNK_OVERLAP 이하까지 이월
            carry: list[str] = []
            for u in reversed(current):
                if len("\n".join([u, *carry])) > CHUNK_OVERLAP:
                    break
                carry.insert(0, u)
            current = carry
        current.append(unit)

    if current:
        chunks.append("\n".join(current))

    return [c.strip() for c in chunks if c.strip()]


def chunk_document(path: Path) -> list[Chunk]:
    """.txt 파일 하나를 Chunk 리스트로 (출처·번호 포함)."""
    text = path.read_text(encoding="utf-8")
    return [
        Chunk(text=piece, source=path.name, chunk_index=i)
        for i, piece in enumerate(chunk_text(text))
    ]


def chunk_directory(folder: Path) -> list[Chunk]:
    """폴더 안 모든 .txt 를 모아 하나의 Chunk 리스트로."""
    all_chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.txt")):
        all_chunks.extend(chunk_document(path))
    return all_chunks
