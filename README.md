# 건설 문서 RAG 챗봇

건설 관련 문서(시방서, 공사보고서, 계약서)를 넣으면 **요약**하거나 **질문에 답변**하는
RAG(Retrieval-Augmented Generation, 검색 증강 생성) 챗봇.

## 기술 스택 (완전 무료 · 로컬 실행)

| 역할 | 도구 | 비고 |
|---|---|---|
| LLM (답변 생성) | **Ollama** + `llama3.2` | 로컬 실행, 무료. `llm.py`에서 Claude API로 교체 가능 |
| 임베딩 (문장 → 벡터) | **sentence-transformers** | 3단계부터. 로컬, 무료 |
| 벡터 검색 | **ChromaDB** | 3단계부터. 로컬 파일 DB |
| 웹 화면 | **Streamlit** | 5단계 |

LLM 부분은 `llm.py`의 **교체 계층**으로 분리돼 있어, `.env`의 `LLM_PROVIDER`를
`ollama` ↔ `claude`로 바꾸기만 하면 답변 품질을 비교할 수 있다.

## 단계별 진행 상황

| 단계 | 내용 | 파일 | 상태 |
|---|---|---|---|
| 1 | 문서 1개를 읽어 LLM으로 요약 | `step1_summarize.py` | ✅ |
| 2 | 문서를 작은 조각(chunk)으로 분할 | `step2_chunk.py` | ⬜ |
| 3 | 조각을 임베딩해 벡터 DB에 저장 | `step3_index.py` | ⬜ |
| 4 | 질문 → 검색 → LLM 답변 (RAG 완성) | `step4_rag.py` | ⬜ |
| 5 | Streamlit 웹 화면 | `app.py` | ⬜ |

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

## 실행 (1단계)

```powershell
python step1_summarize.py data\sample_01_spec_concrete.txt
python step1_summarize.py data\sample_02_weekly_report.txt
python step1_summarize.py data\sample_03_contract.txt
```

## 파일 구조

```
construction-rag/
├─ llm.py              LLM 교체 계층 (Ollama / Claude 공통 인터페이스)
├─ step1_summarize.py  1단계: 문서 요약
├─ requirements.txt    파이썬 패키지 목록
├─ .env.example        설정 양식 (복사해서 .env 로 사용)
├─ .gitignore
└─ data/               테스트용 가상 건설 문서 3개
   ├─ sample_01_spec_concrete.txt   철근콘크리트공사 시방서
   ├─ sample_02_weekly_report.txt   주간 공사보고서
   └─ sample_03_contract.txt        공사도급계약서 발췌
```
