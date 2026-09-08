"""
3단계 (2/2): 벡터 DB = 임베딩을 저장하고, 질문 벡터와 가까운 것들을 빠르게 찾기

[벡터 DB 가 왜 필요한가?]
  chunk 가 19개면 그냥 반복문으로 거리 계산해도 된다. 하지만 실제 문서가
  수천~수만 chunk 가 되면 매번 전부 계산하기엔 느리다. 벡터 DB 는 근사 최근접
  이웃(ANN) 색인으로 "가까운 벡터 top-k" 를 빠르게 돌려준다.

[ChromaDB 를 고른 이유]
  - 로컬에서 파일로 동작(별도 서버 설치 불필요), 무료, 파이썬 한 줄로 시작.
  - 벡터와 함께 메타데이터(출처 파일명 등)를 저장/필터링할 수 있다.
  - 프로토타입~중간 규모에 적합. (더 큰 규모면 FAISS/pgvector/전용 DB 로 교체)

저장 위치: ./chroma_db  (폴더로 생성됨, .gitignore 에 이미 제외되어 있음)
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from chunker import Chunk
from embedder import Embedder

DB_PATH = "chroma_db"
COLLECTION_NAME = "construction_docs"


class VectorStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        # PersistentClient: 프로그램을 꺼도 ./chroma_db 폴더에 내용이 남는다.
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            # 거리 척도를 코사인으로. (정규화된 임베딩 + 코사인 = 의미 유사도 표준 조합)
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """컬렉션을 비우고 새로 만든다. 인덱스를 처음부터 다시 만들 때 사용."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # 아직 없으면 무시
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, chunks: list[Chunk]) -> None:
        """chunk 리스트를 임베딩해서 컬렉션에 저장(upsert)한다."""
        if not chunks:
            print("[벡터DB] 저장할 chunk 가 없습니다.")
            return

        texts = [c.text for c in chunks]
        # id 를 '파일명::번호' 로 고정 -> 다시 인덱싱해도 중복 안 되고 갱신됨.
        ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
        metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]

        print(f"[벡터DB] {len(chunks)}개 chunk 임베딩 중...")
        embeddings = self.embedder.embed_documents(texts)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"[벡터DB] 저장 완료. 현재 컬렉션 총 {self.collection.count()}개")

    def search(self, question: str, k: int = 4) -> list[dict]:
        """질문과 의미가 가까운 chunk top-k 를 돌려준다."""
        query_vec = self.embedder.embed_query(question)
        result = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
        )
        # chroma 결과는 리스트의 리스트 형태(질문 여러 개도 가능). 질문 1개이므로 [0] 만 사용.
        hits = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "chunk_index": meta["chunk_index"],
                    "distance": dist,          # 0에 가까울수록 유사 (코사인 거리)
                    "similarity": 1 - dist,    # 1에 가까울수록 유사 (보기 편하라고 변환)
                }
            )
        return hits
