"""
2단계: 문서를 작은 조각(chunk)으로 나누는 기능

[왜 나누나?]
  1) 임베딩 모델·LLM에 한 번에 넣을 수 있는 글자 수에는 한계가 있다.
  2) 나중에 "질문과 관련된 부분만" 골라서 LLM에게 주려면(=RAG의 R, 검색),
     문서가 검색 가능한 작은 단위로 쪼개져 있어야 한다.
  3) 문서 전체를 매번 LLM에 넣으면 느리고, (유료 API라면) 비싸다.

[방법: 재귀적 문자 분할 (recursive character splitting)]
  - 먼저 '큰 경계'(빈 줄)로 자른다.
  - 조각이 여전히 목표 크기보다 크면, 더 작은 경계(줄바꿈 -> 문장 -> 공백 -> 글자)로
    한 단계씩 내려가며 다시 자른다.
  - 그래서 목표 크기를 넘지 않으면서도 문단·문장 같은 '의미 단위'를 최대한 보존한다.
  - 이웃한 조각끼리 끝-앞을 조금 겹치게(overlap) 해서, 경계에서 잘린 내용이
    최소 한 조각에는 온전히 들어가도록 한다.
  (LangChain 의 RecursiveCharacterTextSplitter 와 같은 아이디어를, 외부 라이브러리 없이
   직접 구현한 것.)
"""

from dataclasses import dataclass
from pathlib import Path

# ── 조절 가능한 설정 ─────────────────────────────────────────────
CHUNK_SIZE = 400        # 한 조각의 목표 최대 길이 (글자 수)
CHUNK_OVERLAP = 80      # 이웃한 조각끼리 겹치는 길이 (글자 수, CHUNK_SIZE 의 20%)

# 자를 때 위에서부터 순서대로 시도하는 경계들.
#   "\n\n" 문단  ->  "\n" 줄  ->  ". "/"。" 문장  ->  " " 단어  ->  "" 글자(최후)
SEPARATORS = ["\n\n", "\n", ". ", "。 ", "。", " ", ""]
# ────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """조각 하나. 텍스트뿐 아니라 '어디서 왔는지'(출처)도 함께 들고 다닌다.
    이 출처 정보는 4단계에서 답변의 근거를 표시할 때 쓴다."""

    text: str
    source: str          # 원본 파일명 (예: sample_03_contract.txt)
    chunk_index: int     # 그 문서 안에서 몇 번째 조각인지 (0부터)


def _recurse(text: str, separators: list[str]) -> list[str]:
    """separators 를 한 단계씩 내려가며 text 를 CHUNK_SIZE 이하 조각들로 쪼갠다."""
    # 이미 충분히 작으면 그대로 둔다.
    if len(text) <= CHUNK_SIZE:
        return [text]

    # 이 텍스트 안에 실제로 존재하는 첫 번째 경계를 고른다.
    sep = ""
    for s in separators:
        if s == "" or s in text:
            sep = s
            break

    # 더 쪼갤 경계가 없으면(=""), 어쩔 수 없이 글자 수로 강제 분할.
    if sep == "":
        return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    # 고른 경계로 1차 분할. 그래도 큰 조각은 '다음 단계 경계들'로 재귀.
    next_separators = separators[separators.index(sep) + 1:]
    pieces: list[str] = []
    for part in text.split(sep):
        if len(part) <= CHUNK_SIZE:
            pieces.append(part)
        else:
            pieces.extend(_recurse(part, next_separators))

    # 잘게 나뉜 조각들을 CHUNK_SIZE 이하로 다시 '뭉치고', 사이에 overlap 을 준다.
    return _merge(pieces, sep)


def _merge(pieces: list[str], sep: str) -> list[str]:
    """작은 조각들을 CHUNK_SIZE 이하로 이어붙이되, 새 조각을 시작할 때
    직전 조각의 마지막 CHUNK_OVERLAP 글자를 앞에 붙여 문맥이 이어지게 한다."""
    chunks: list[str] = []
    current = ""

    for part in pieces:
        candidate = part if current == "" else current + sep + part
        if len(candidate) <= CHUNK_SIZE or current == "":
            current = candidate
        else:
            chunks.append(current)
            overlap_tail = current[-CHUNK_OVERLAP:]        # 직전 조각의 꼬리
            current = overlap_tail + sep + part            # 다음 조각의 머리에 이어붙임

    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str) -> list[str]:
    """문자열 하나를 조각 리스트로."""
    text = text.replace("\r\n", "\n")                      # 윈도우 줄바꿈 정리
    return [c.strip() for c in _recurse(text, SEPARATORS) if c.strip()]


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
