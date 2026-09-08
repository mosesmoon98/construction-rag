r"""
1단계: 건설 문서 한 개를 읽어서 LLM으로 요약하기

이 스크립트가 하는 일 (딱 3가지):
  1) 텍스트 파일 하나를 읽는다
  2) 그 내용을 LLM에게 보내면서 "건설 문서 분석가"처럼 행동하라고 지시한다
  3) LLM이 만들어 준 요약을 화면에 출력한다

어떤 LLM을 쓸지는 llm.py 의 get_llm() 이 .env 를 보고 골라준다.
  - 기본값: Ollama (무료, 로컬)          -> .env 의 LLM_PROVIDER=ollama
  - 비교용: Claude API (유료)            -> .env 의 LLM_PROVIDER=claude

실행 방법 (PowerShell):
  python step1_summarize.py data\sample_01_spec_concrete.txt
  python step1_summarize.py data\sample_02_weekly_report.txt
  python step1_summarize.py data\sample_03_contract.txt
"""

import sys
from pathlib import Path

from llm import get_llm  # <- LLM 교체 계층. 여기선 provider가 뭔지 몰라도 된다.

# LLM에게 부여하는 역할과 출력 형식 지침 (= system prompt).
# "무엇을 하는 사람인지 + 어떤 형식으로 답할지"를 미리 못박아 두는 부분이다.
SYSTEM_PROMPT = """당신은 건설사업관리(CM/PM) 회사에서 일하는 문서 분석 전문가입니다.
시방서, 공사보고서, 계약서 같은 건설 문서를 읽고 핵심을 빠르게 정리합니다.
요약은 반드시 아래 형식으로 작성하세요:

1. 문서 종류: (예: 철근콘크리트 시방서 / 주간 공사보고서 / 공사도급계약서)
2. 한 줄 요약:
3. 핵심 내용 (불릿 5~8개):
4. 숫자·기준값 (문서에 나온 수치를 목록으로):
5. 주의·리스크 사항:

문서에 없는 내용을 지어내지 말고, 불확실하면 '문서에 명시되지 않음'이라고 쓰세요."""


def read_document(path: Path) -> str:
    """텍스트 파일을 읽어서 하나의 문자열로 돌려준다."""
    if not path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)
    # 한글이 깨지지 않도록 encoding="utf-8" 을 반드시 지정한다.
    # (윈도우 파이썬은 기본 인코딩이 cp949라서 지정하지 않으면 한글이 깨질 수 있다.)
    return path.read_text(encoding="utf-8")


def build_prompt(document_text: str) -> str:
    """LLM에게 보낼 사용자 메시지를 만든다 (지침 + 실제 문서)."""
    return (
        "다음 건설 문서를 위 형식에 맞춰 한국어로 요약해 주세요.\n\n"
        "---\n"
        f"{document_text}\n"
        "---"
    )


def main() -> None:
    # 명령줄에서 파일 경로를 받는다. 안 주면 샘플 1번을 기본으로 사용.
    if len(sys.argv) >= 2:
        target = Path(sys.argv[1])
    else:
        target = Path("data/sample_01_spec_concrete.txt")
        print(f"[안내] 파일 경로를 주지 않아 기본 파일을 사용합니다: {target}\n")

    document_text = read_document(target)

    llm = get_llm()  # .env 를 보고 Ollama 또는 Claude 객체를 만들어 준다

    print(f"[정보] 파일     : {target.name}")
    print(f"[정보] 글자 수  : {len(document_text):,}자")
    print(f"[정보] 사용 LLM : {type(llm).__name__}  (모델: {llm.model})\n")
    print("=" * 60)
    print("LLM에게 요약을 요청하는 중...")
    print("  (로컬 Ollama는 PC 사양에 따라 10초~1분 이상 걸릴 수 있습니다)")
    print("=" * 60 + "\n")

    summary = llm.chat(system=SYSTEM_PROMPT, prompt=build_prompt(document_text))
    print(summary)


if __name__ == "__main__":
    main()
