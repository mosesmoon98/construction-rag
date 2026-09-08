"""
3단계 (1/2): 임베딩 = 문장을 '의미를 담은 숫자 벡터'로 바꾸기

[임베딩이란?]
  "지체상금" 과 "지연 배상금" 은 글자는 다르지만 뜻이 비슷하다.
  임베딩 모델은 문장을 예: 384개의 숫자로 이뤄진 벡터로 바꾸는데,
  뜻이 비슷한 문장일수록 벡터가 서로 가깝게(방향이 비슷하게) 나온다.
  -> 그래서 "벡터 거리"로 의미 유사도를 계산할 수 있고,
     이것이 RAG 검색(키워드가 안 겹쳐도 관련 내용을 찾는 것)의 원리다.

[모델 선택: intfloat/multilingual-e5-small]
  - 다국어(한국어 포함) 지원, 384차원, 약 470MB 로 노트북 CPU 에서도 돌아감.
  - 입력 최대 512 토큰 -> 우리 chunk(400자 ≈ 400~600토큰)를 자르지 않고 인코딩.
  - e5 계열은 학습 방식 때문에 접두어가 필요하다:
      * 문서(저장할 내용) 앞에는  "passage: "
      * 질문(검색어)      앞에는  "query: "
    이 접두어를 붙여야 검색 성능이 제대로 나온다. (아래에서 자동 처리)
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384  # 이 모델이 내놓는 벡터의 길이


class Embedder:
    """sentence-transformers 모델을 감싼 얇은 래퍼.
    문서용/질문용 접두어를 자동으로 붙여 준다."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # 첫 실행 때 모델을 인터넷에서 내려받는다(약 470MB, 1회). 이후엔 캐시 사용.
        print(f"[임베딩] 모델 로딩 중: {model_name} (처음이면 다운로드로 몇 분 소요)")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """저장할 문서 조각들 -> 벡터 리스트. 'passage: ' 접두어를 붙인다."""
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self.model.encode(
            prefixed,
            normalize_embeddings=True,   # 길이를 1로 정규화 -> 코사인 유사도 계산에 적합
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """검색 질문 하나 -> 벡터. 'query: ' 접두어를 붙인다."""
        vector = self.model.encode(
            f"query: {text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()
