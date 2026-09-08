"""
LLM 교체 계층 (provider abstraction)

목적: "어떤 LLM을 쓰는지"를 이 파일 한 곳에서만 정한다.
     step1_summarize.py 같은 나머지 코드는 provider가 Ollama든 Claude든
     신경 쓰지 않고 llm.chat(system=..., prompt=...) 만 호출한다.

provider 선택은 .env 의 LLM_PROVIDER 값으로 한다 (ollama | claude). 기본값은 ollama.

    from llm import get_llm
    llm = get_llm()
    answer = llm.chat(system="너는 ...", prompt="...")
"""

import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv


class LLMClient(ABC):
    """모든 provider가 따라야 하는 공통 형태(인터페이스).

    이 클래스를 직접 쓰지는 않는다. 아래 OllamaClient / ClaudeClient 가
    이 형태(= chat 메서드 하나)를 똑같이 구현하기 때문에, 바깥 코드는
    둘 중 무엇을 받든 동일하게 사용할 수 있다.
    """

    model: str  # provider가 실제 사용하는 모델 이름 (출력/로그용)

    @abstractmethod
    def chat(self, system: str, prompt: str) -> str:
        """system 지침과 사용자 prompt를 받아 답변 문자열을 돌려준다."""
        raise NotImplementedError


class OllamaClient(LLMClient):
    """로컬 PC에서 도는 Ollama 서버를 사용한다. 완전 무료, 인터넷 불필요."""

    def __init__(self, model: str) -> None:
        # 지연 import: Claude만 쓰는 사람은 ollama 패키지가 없어도 되도록
        # 파일 맨 위가 아니라 여기서 import 한다.
        import ollama

        self._ollama = ollama
        self.model = model

    def chat(self, system: str, prompt: str) -> str:
        try:
            response = self._ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                # temperature 낮게 = 매번 비슷하고 사실 위주의 답 (요약에 적합)
                options={"temperature": 0.2},
            )
        except Exception as e:
            raise RuntimeError(
                "Ollama 호출 실패. 아래를 확인하세요:\n"
                "  1) Ollama가 실행 중인가? (작업표시줄 아이콘 확인, 없으면 터미널에서 `ollama serve`)\n"
                f"  2) 모델을 받아뒀는가?  ->  `ollama pull {self.model}`\n"
                f"[원본 오류] {e}"
            ) from e

        return response["message"]["content"]


class ClaudeClient(LLMClient):
    """Anthropic Claude API를 사용한다. 유료. 나중에 답변 품질 비교용."""

    def __init__(self, model: str) -> None:
        import anthropic  # 지연 import (Ollama만 쓰면 이 패키지 없어도 됨)

        self._client = anthropic.Anthropic()  # .env 의 ANTHROPIC_API_KEY 자동 사용
        self.model = model

    def chat(self, system: str, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n".join(b.text for b in response.content if b.type == "text")


def get_llm() -> LLMClient:
    """.env 설정을 읽어 알맞은 provider 객체를 만들어 돌려준다."""
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "ollama":
        return OllamaClient(model=os.getenv("OLLAMA_MODEL", "llama3.2"))
    if provider == "claude":
        return ClaudeClient(model=os.getenv("CLAUDE_MODEL", "claude-opus-5"))

    raise ValueError(
        f"알 수 없는 LLM_PROVIDER: {provider!r}  (ollama 또는 claude 만 가능)"
    )
