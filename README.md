# 건설 문서 RAG 챗봇

건설 관련 문서(시방서, 공사보고서, 계약서)를 넣으면 **요약**하거나 **질문에 답변**하는
RAG(Retrieval-Augmented Generation, 검색 증강 생성) 챗봇.
문서에 근거가 없으면 지어내지 않고 "찾을 수 없습니다"라고 답하며, 답변마다 근거 문서를 함께 보여준다.

## 기술 스택 (완전 무료 · 로컬 실행)

| 역할 | 도구 | 비고 |
|---|---|---|
| LLM (답변 생성) | **Ollama** + `llama3.2` | 로컬 실행, 무료. `.env` 한 줄로 Claude API 로 교체 가능 |
| 임베딩 (문장 → 벡터) | **sentence-transformers** `multilingual-e5-small` | 로컬, 무료. 다국어(한국어) 384차원 |
| 벡터 검색 | **ChromaDB** | 로컬 파일 DB, 무료 |
| 웹 화면 | **Streamlit** | 채팅 UI + 근거 문서 표시 |

LLM 부분은 `llm.py`의 **교체 계층(어댑터)**으로 분리돼 있어, `.env`의 `LLM_PROVIDER`를
`ollama` ↔ `claude`로 바꾸기만 하면 코드 수정 없이 답변 품질을 비교할 수 있다.

## 단계별 진행 상황

| 단계 | 내용 | 파일 | 상태 |
|---|---|---|---|
| 1 | 문서 1개를 읽어 LLM으로 요약 | `step1_summarize.py` | ✅ |
| 2 | 문서를 작은 조각(chunk)으로 분할 | `step2_chunk.py`, `chunker.py` | ✅ |
| 3 | 조각을 임베딩해 벡터 DB에 저장 + 검색 | `step3_index.py`, `embedder.py`, `vector_store.py` | ✅ |
| 4 | 질문 → 검색 → LLM 답변 (RAG 완성) | `step4_rag.py`, `rag.py` | ✅ |
| 5 | Streamlit 웹 화면 | `app.py` | ✅ |

## 준비 (최초 1회)

### 1. Ollama 설치 + 모델 받기
```powershell
winget install --id Ollama.Ollama -e
# 새 터미널을 연 뒤:
ollama pull llama3.2
ollama run llama3.2 "한 문장으로 자기소개 해줘"   # 잘 나오면 /bye 로 종료
```

### 2. 파이썬 가상환경 + 패키지
```powershell
cd "$env:USERPROFILE\Desktop\construction-rag"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
`Activate.ps1`이 차단되면 한 번만: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 3. 설정 파일
```powershell
copy .env.example .env
```
기본값(Ollama)이면 그대로 둬도 됨. Claude와 비교하려면 `.env`에서 `LLM_PROVIDER=claude`로
바꾸고 `ANTHROPIC_API_KEY`를 채운 뒤 `pip install anthropic`.

## 실행

```powershell
# 1단계: 문서 하나 통째로 요약
python step1_summarize.py data\sample_01_spec_concrete.txt

# 2단계: 문서를 chunk 로 분할 (통계 + 예시 조각 출력)
python step2_chunk.py

# 3단계: 전체 문서 임베딩 → ChromaDB 저장 → 예시 질문으로 검색 테스트
python step3_index.py

# 4단계: RAG 챗봇 (터미널). 질문 1개 또는 대화형
python step4_rag.py "지체상금은 하루에 얼마인가?"
python step4_rag.py

# 5단계: RAG 챗봇 (웹). 브라우저에서 http://localhost:8501 자동 열림
#   ※ 먼저 step3_index.py 를 한 번 실행해 벡터 DB 가 있어야 함
streamlit run app.py
```

## 파일 구조

```
construction-rag/
├─ llm.py              LLM 교체 계층 (Ollama / Claude 공통 인터페이스)
├─ chunker.py          문서를 chunk 로 분할 (줄 단위 경계 + overlap)
├─ embedder.py         문장 → 벡터 (multilingual-e5-small 래퍼)
├─ vector_store.py     ChromaDB 래퍼 (build_index / search)
├─ rag.py              RAG 파이프라인 (검색 → 근거 주입 → LLM 답변)
├─ step1_summarize.py  1단계 실행 스크립트
├─ step2_chunk.py      2단계 실행 스크립트
├─ step3_index.py      3단계 실행 스크립트
├─ step4_rag.py        4단계 실행 스크립트 (터미널 챗봇)
├─ app.py              5단계 Streamlit 웹 챗봇
├─ requirements.txt    파이썬 패키지 목록
├─ .env.example        설정 양식 (복사해서 .env 로 사용)
├─ .gitignore
├─ chroma_db/          (자동 생성) 벡터 DB 저장 폴더 — git 제외
└─ data/               테스트용 가상 건설 문서 3개
   ├─ sample_01_spec_concrete.txt   철근콘크리트공사 시방서
   ├─ sample_02_weekly_report.txt   주간 공사보고서
   └─ sample_03_contract.txt        공사도급계약서 발췌
```

## 설계 메모 (왜 이렇게 만들었나)

- **LLM 어댑터 분리** — 민감한 건설 문서를 기본은 로컬(Ollama)로 처리하고, 필요 시에만
  외부 API로 전환. 벤더 종속 회피. `rag.py`는 provider가 무엇인지 모른다.
- **chunk 크기 400자 / overlap 80자** — 임베딩 모델 입력 한도(512토큰) 안에 들어오도록,
  그리고 조항·수치가 경계에서 잘려도 최소 한 조각엔 온전히 담기도록.
- **줄 단위 분할** — 첫 버전은 글자 단위로 잘라 숫자("27,720,00 / 0,000원")가 깨졌고
  검색 품질이 나빴음. 줄 경계 분할로 교체해 해결.
- **grounding 프롬프트** — "참고 자료에 없으면 없다고만 답하라 / 추측 금지"를 시스템
  프롬프트에 명시. 검색이 빗나가도 환각을 만들지 않게 함.
- **답변에 출처 동봉** — 각 답변에 근거 chunk(파일명·조각번호·유사도)를 함께 표시해
  사람이 원문을 검증할 수 있게 함.
- **레이어 분리** — chunker / embedder / vector_store / llm / rag 를 독립 모듈로.
  검색 엔진이나 LLM 을 교체해도 나머지는 그대로.
- **알려진 한계 / 개선 방향** — ① e5-small 은 유사도 점수 분리력이 약함 → 더 큰 임베딩
  모델(bge-m3), 의미검색+BM25 하이브리드, reranker. ② llama3.2 3B 는 출처 표기 지시를
  항상 지키진 않음 → 더 큰 모델 또는 Claude. ③ 현재 `.txt`만 지원 → `pypdf`로 PDF 파싱 추가.
```
