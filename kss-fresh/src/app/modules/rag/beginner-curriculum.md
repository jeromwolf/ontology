# RAG 초급 커리큘럼 (Step 1: Naive RAG 완벽 마스터)

## 🎯 학습 목표
RAG의 탄생 배경과 핵심 원리를 깊이 있게 이해하고, Naive RAG 시스템을 처음부터 끝까지 직접 구현할 수 있는 능력을 갖춥니다. 실제 프로덕션에서 발생하는 문제들을 경험하고 해결 방법을 체득합니다.

## 📚 총 학습 시간: 10시간

## 🏆 차별화된 교육 철학
- **"이론보다 실전"**: 모든 개념을 코드로 직접 구현
- **"실패에서 배우기"**: 일부러 문제 상황을 만들고 해결
- **"현업 그대로"**: 실제 기업에서 사용하는 도구와 방법론
- **"측정 가능한 성과"**: 명확한 평가 지표와 개선 방법

## 📋 커리큘럼 구성

### Module 1: RAG의 탄생 - LLM의 본질적 한계 극복하기 (2.5시간)

#### 1.1 LLM의 치명적 약점 직접 체험하기 (1시간)
**실습 중심 학습**
```python
# 실습 1: Hallucination 실제로 경험하기
class HallucinationDemo:
    def __init__(self):
        self.llm = OpenAI()
    
    def demonstrate_hallucination(self):
        # 존재하지 않는 정보 질문
        questions = [
            "2024년 노벨물리학상 수상자의 연구 내용을 설명해주세요",
            "김철수 교수의 'Quantum RAG Theory' 논문을 요약해주세요",
            "서울대학교 AI연구소의 2025년 연구 계획은?",
        ]
        
        for q in questions:
            response = self.llm.complete(q)
            print(f"질문: {q}")
            print(f"LLM 답변: {response}")
            print(f"사실 여부: ❌ (모두 가상의 정보)")
            print("-" * 50)
```

**핵심 인사이트**
- LLM은 '그럴듯한 거짓말'을 매우 잘함
- 학습 데이터에 없는 정보는 창작함
- 확률적 생성이라는 본질적 한계

#### 1.2 실시간 정보의 부재 (30분)
```python
# 실습 2: 최신 정보 한계 테스트
def test_knowledge_cutoff():
    current_events = [
        "오늘의 코스피 지수",
        "현재 비트코인 가격",
        "어제 발표된 애플 신제품",
        "이번주 날씨 예보"
    ]
    
    for event in current_events:
        print(f"❌ LLM은 '{event}'를 모릅니다")
        print(f"✅ RAG는 실시간 데이터를 검색해서 답할 수 있습니다")
```

#### 1.3 기업 내부 지식의 활용 불가 (1시간)
**실제 비즈니스 시나리오**
```python
# 기업이 직면하는 실제 문제
enterprise_challenges = {
    "법무팀": "우리 회사 계약서 템플릿에서 Force Majeure 조항은?",
    "인사팀": "연차 사용 규정 중 이월 가능 일수는?",
    "개발팀": "우리 API의 rate limiting 정책은?",
    "영업팀": "작년 3분기 매출 데이터와 전년 대비 성장률은?"
}

# LLM만으로는 절대 답할 수 없는 질문들
# RAG로 해결 가능한 실제 use case
```

### Module 2: Naive RAG 아키텍처 완벽 이해 (3시간)

#### 2.1 Naive RAG의 3단계 프로세스 (1시간)
```python
class NaiveRAGArchitecture:
    """
    Indexing → Retrieval → Generation
    각 단계를 명확히 분리하여 이해
    """
    
    def __init__(self):
        self.vector_store = ChromaDB()
        self.embedder = OpenAIEmbeddings()
        self.llm = ChatOpenAI()
    
    def indexing_phase(self, documents):
        """1단계: 인덱싱 - 문서를 검색 가능한 형태로 변환"""
        chunks = []
        for doc in documents:
            # 청킹: 문서를 적절한 크기로 분할
            doc_chunks = self.chunk_document(doc)
            
            # 임베딩: 각 청크를 벡터로 변환
            for chunk in doc_chunks:
                embedding = self.embedder.embed(chunk.text)
                chunks.append({
                    'id': chunk.id,
                    'text': chunk.text,
                    'embedding': embedding,
                    'metadata': chunk.metadata
                })
        
        # 벡터 저장소에 저장
        self.vector_store.add(chunks)
        return len(chunks)
    
    def retrieval_phase(self, query, top_k=5):
        """2단계: 검색 - 질문과 관련된 정보 찾기"""
        # 질문을 벡터로 변환
        query_embedding = self.embedder.embed(query)
        
        # 유사도 검색
        results = self.vector_store.search(
            query_embedding, 
            top_k=top_k
        )
        
        # 검색 결과 정리
        retrieved_chunks = [
            {
                'text': r.text,
                'score': r.score,
                'metadata': r.metadata
            }
            for r in results
        ]
        
        return retrieved_chunks
    
    def generation_phase(self, query, retrieved_chunks):
        """3단계: 생성 - 검색된 정보를 바탕으로 답변 생성"""
        # 프롬프트 구성
        context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
        
        prompt = f"""다음 정보를 참고하여 질문에 답하세요.
        
참고 정보:
{context}

질문: {query}

답변:"""
        
        # LLM으로 답변 생성
        response = self.llm.generate(prompt)
        
        return response
```

#### 2.2 각 단계별 심화 이해 (1시간 30분)
**Indexing 단계 상세 분석**
```python
def deep_dive_indexing():
    """인덱싱의 모든 것"""
    
    # 1. 문서 로더의 다양성
    loaders = {
        'pdf': PyPDFLoader,
        'docx': UnstructuredWordDocumentLoader,
        'html': BSHTMLLoader,
        'csv': CSVLoader,
        'json': JSONLoader
    }
    
    # 2. 청킹 전략의 중요성
    chunking_strategies = {
        'fixed_size': {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'pros': '구현 간단, 일정한 크기',
            'cons': '문맥 단절 가능성'
        },
        'sentence_based': {
            'sentences_per_chunk': 5,
            'pros': '문장 단위 보존',
            'cons': '크기 불균일'
        },
        'semantic': {
            'method': 'embedding_similarity',
            'pros': '의미적 일관성',
            'cons': '계산 비용 높음'
        }
    }
    
    # 3. 메타데이터의 활용
    metadata_examples = {
        'source': 'document.pdf',
        'page': 42,
        'section': 'Chapter 3.2',
        'date_created': '2024-01-15',
        'author': 'John Doe',
        'department': 'Engineering'
    }
```

**Retrieval 단계 상세 분석**
```python
def deep_dive_retrieval():
    """검색의 핵심 이해"""
    
    # 1. 벡터 유사도 메트릭
    similarity_metrics = {
        'cosine': {
            'formula': 'dot(A, B) / (norm(A) * norm(B))',
            'range': '[-1, 1]',
            'use_case': '가장 일반적, 방향성 중요'
        },
        'euclidean': {
            'formula': 'sqrt(sum((A - B)^2))',
            'range': '[0, ∞)',
            'use_case': '실제 거리가 중요한 경우'
        },
        'dot_product': {
            'formula': 'sum(A * B)',
            'range': '(-∞, ∞)',
            'use_case': '정규화된 벡터에서 빠른 계산'
        }
    }
    
    # 2. Top-K 선택의 trade-off
    topk_analysis = {
        'k=1': '정확도 높음, recall 낮음',
        'k=5': '균형잡힌 선택',
        'k=10': 'recall 높음, noise 증가',
        'k=20': '너무 많은 정보, 성능 저하'
    }
```

#### 2.3 첫 번째 Naive RAG 구현 (30분)
```python
# 실제 동작하는 Naive RAG 시스템
class MyFirstRAG:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_document(self, text):
        """문서 추가 및 임베딩"""
        # 간단한 청킹 (문장 단위)
        sentences = text.split('. ')
        
        for sentence in sentences:
            if len(sentence) > 10:  # 너무 짧은 문장 제외
                embedding = self.model.encode(sentence)
                self.documents.append(sentence)
                self.embeddings.append(embedding)
    
    def search(self, query, top_k=3):
        """검색"""
        query_embedding = self.model.encode(query)
        
        # 코사인 유사도 계산
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embedding.reshape(1, -1)
            )[0][0]
            similarities.append((i, sim))
        
        # Top-K 선택
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        return [self.documents[i] for i, _ in top_results]
    
    def answer(self, query):
        """RAG 답변 생성"""
        # 관련 문서 검색
        relevant_docs = self.search(query)
        
        # 컨텍스트 구성
        context = "\n".join(relevant_docs)
        
        # 프롬프트 생성
        prompt = f"""
Based on the following information, answer the question.

Information:
{context}

Question: {query}
Answer:"""
        
        # 여기서는 간단히 컨텍스트 반환 (실제로는 LLM 호출)
        return f"Based on the documents: {context[:200]}..."

# 사용 예시
rag = MyFirstRAG()
rag.add_document("Python은 1991년에 귀도 반 로섬이 개발한 프로그래밍 언어입니다.")
rag.add_document("Python은 간결하고 읽기 쉬운 문법으로 유명합니다.")
rag.add_document("기계학습과 데이터 분석에 Python이 널리 사용됩니다.")

answer = rag.answer("Python은 언제 만들어졌나요?")
print(answer)
```

### Module 3: Naive RAG의 실전 구현과 한계 체험 (3시간)

#### 3.1 실전 프로젝트 1: 회사 규정 Q&A 봇 (1시간)
```python
class CompanyPolicyRAG:
    """실제 회사에서 사용할 수 있는 규정 검색 시스템"""
    
    def __init__(self):
        self.vector_store = Chroma(
            collection_name="company_policies",
            embedding_function=OpenAIEmbeddings()
        )
        self.llm = ChatOpenAI(temperature=0)  # 정확성 중요
    
    def load_policies(self, policy_dir):
        """회사 규정 문서 로드"""
        policies = {
            "휴가규정.pdf": "연차, 병가, 경조사 휴가 규정",
            "보안정책.docx": "정보보안, 출입통제 규정",
            "인사규정.pdf": "채용, 평가, 승진 규정",
            "복리후생.docx": "복지제도, 지원금 규정"
        }
        
        for filename, description in policies.items():
            # 문서 로드 및 청킹
            loader = self.get_loader(filename)
            documents = loader.load()
            
            # 메타데이터 추가
            for doc in documents:
                doc.metadata.update({
                    'source': filename,
                    'category': description,
                    'last_updated': '2024-01-01'
                })
            
            # 벡터 스토어에 저장
            self.vector_store.add_documents(documents)
    
    def ask_policy(self, question):
        """규정 관련 질문에 답변"""
        # 1. 관련 규정 검색
        relevant_docs = self.vector_store.similarity_search(
            question, 
            k=5,
            filter=None  # 특정 카테고리로 필터링 가능
        )
        
        # 2. 출처 정보 포함한 답변 생성
        context = self.format_context_with_sources(relevant_docs)
        
        prompt = f"""당신은 회사 규정 전문가입니다.
다음 규정 문서를 참고하여 직원의 질문에 정확하게 답변하세요.

규정 내용:
{context}

질문: {question}

답변 시 주의사항:
1. 규정에 명시된 내용만 답변
2. 추측이나 일반론 금지
3. 출처 문서명 명시
4. 규정에 없는 경우 "규정에 명시되지 않음" 표시

답변:"""
        
        response = self.llm.invoke(prompt)
        return response.content
```

#### 3.2 실전 프로젝트 2: 기술 문서 검색 시스템 (1시간)
```python
class TechnicalDocRAG:
    """개발팀을 위한 기술 문서 RAG"""
    
    def __init__(self):
        self.setup_specialized_components()
    
    def setup_specialized_components(self):
        # 코드 특화 청킹
        self.code_splitter = RecursiveCharacterTextSplitter(
            separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        # 기술 문서 특화 임베딩
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def process_technical_docs(self):
        """다양한 기술 문서 처리"""
        doc_types = {
            'api_docs': self.process_api_documentation,
            'code_comments': self.process_code_files,
            'readme': self.process_markdown,
            'architecture': self.process_diagrams
        }
        
        for doc_type, processor in doc_types.items():
            documents = processor()
            self.index_documents(documents, doc_type)
    
    def intelligent_search(self, query):
        """스마트 검색"""
        # 쿼리 타입 분석
        query_type = self.analyze_query_type(query)
        
        if query_type == 'code_search':
            # 코드 검색에 특화된 처리
            return self.search_code_snippets(query)
        elif query_type == 'api_search':
            # API 검색에 특화된 처리
            return self.search_api_endpoints(query)
        else:
            # 일반 문서 검색
            return self.search_general_docs(query)
```

#### 3.3 Naive RAG의 한계 직접 경험하기 (1시간)
```python
class NaiveRAGLimitations:
    """Naive RAG의 한계를 직접 체험하는 실습"""
    
    def demonstrate_limitation_1_retrieval_quality(self):
        """한계 1: 검색 품질 문제"""
        # 문제 상황: 비슷한 단어는 많지만 실제 답은 없는 경우
        documents = [
            "Python은 프로그래밍 언어입니다.",
            "Java도 프로그래밍 언어입니다.",
            "프로그래밍 언어는 컴퓨터와 대화하는 도구입니다.",
            "많은 프로그래밍 언어가 존재합니다."
        ]
        
        query = "Python의 가비지 컬렉션은 어떻게 동작하나요?"
        
        # 결과: 관련 없는 문서들이 검색됨
        print("❌ 문제: 키워드는 매칭되지만 실제 답은 없음")
    
    def demonstrate_limitation_2_context_window(self):
        """한계 2: 컨텍스트 윈도우 제한"""
        # 너무 많은 문서를 검색하면?
        large_context = "문서 내용" * 1000  # 매우 긴 컨텍스트
        
        print("❌ 문제: LLM의 토큰 제한 초과")
        print("❌ 문제: 중요한 정보가 뒤쪽에 있으면 무시됨")
    
    def demonstrate_limitation_3_redundancy(self):
        """한계 3: 중복 정보 처리"""
        # 같은 내용이 여러 문서에 반복
        redundant_docs = [
            "회사 창립일은 2010년 1월 1일입니다.",
            "우리 회사는 2010년 1월 1일에 창립했습니다.",
            "2010년 1월 1일, 회사가 설립되었습니다."
        ]
        
        print("❌ 문제: 동일한 정보가 반복되어 토큰 낭비")
    
    def demonstrate_limitation_4_conflicting_info(self):
        """한계 4: 상충하는 정보 처리"""
        conflicting_docs = [
            "제품 가격은 10만원입니다. (2023년 자료)",
            "제품 가격은 12만원입니다. (2024년 자료)",
            "할인 중! 제품 가격 8만원 (프로모션)"
        ]
        
        print("❌ 문제: 어떤 정보가 정확한지 판단 불가")
        print("❌ 문제: 시간적 맥락 고려 불가")
```

### Module 4: 실전 평가와 개선 (1.5시간)

#### 4.1 성능 측정하기 (45분)
```python
class RAGEvaluator:
    """RAG 시스템 성능 평가"""
    
    def __init__(self, rag_system):
        self.rag = rag_system
        self.metrics = {}
    
    def evaluate_retrieval_quality(self, test_set):
        """검색 품질 평가"""
        metrics = {
            'precision_at_k': [],
            'recall_at_k': [],
            'mrr': []  # Mean Reciprocal Rank
        }
        
        for query, relevant_docs in test_set:
            retrieved = self.rag.search(query, top_k=5)
            
            # Precision@K 계산
            relevant_retrieved = len(set(retrieved) & set(relevant_docs))
            precision = relevant_retrieved / len(retrieved)
            metrics['precision_at_k'].append(precision)
            
            # Recall@K 계산
            recall = relevant_retrieved / len(relevant_docs)
            metrics['recall_at_k'].append(recall)
            
            # MRR 계산
            for i, doc in enumerate(retrieved):
                if doc in relevant_docs:
                    metrics['mrr'].append(1 / (i + 1))
                    break
        
        return {
            'avg_precision': np.mean(metrics['precision_at_k']),
            'avg_recall': np.mean(metrics['recall_at_k']),
            'mrr': np.mean(metrics['mrr'])
        }
    
    def evaluate_answer_quality(self, test_qa_pairs):
        """답변 품질 평가"""
        results = []
        
        for question, expected_answer in test_qa_pairs:
            generated_answer = self.rag.answer(question)
            
            # 다양한 평가 기준
            evaluation = {
                'relevance': self.check_relevance(generated_answer, question),
                'accuracy': self.check_accuracy(generated_answer, expected_answer),
                'completeness': self.check_completeness(generated_answer, expected_answer),
                'hallucination': self.detect_hallucination(generated_answer)
            }
            
            results.append(evaluation)
        
        return self.aggregate_results(results)
```

#### 4.2 개선 아이디어 도출 (45분)
```python
class ImprovementIdeas:
    """Naive RAG 개선 방향 탐색"""
    
    def brainstorm_improvements(self):
        improvements = {
            'pre_retrieval': [
                '더 나은 청킹 전략',
                '메타데이터 활용',
                '쿼리 확장',
                '문서 품질 필터링'
            ],
            'retrieval': [
                '하이브리드 검색 (벡터 + 키워드)',
                '의미적 유사도 개선',
                '다단계 검색',
                'MMR (최대 한계 관련성) 알고리즘'
            ],
            'post_retrieval': [
                '재순위화 (Reranking)',
                '중복 제거',
                '요약 및 압축',
                '신뢰도 점수 추가'
            ],
            'generation': [
                '더 나은 프롬프트 엔지니어링',
                '출처 명시',
                '답변 검증',
                '반복적 개선'
            ]
        }
        
        return improvements
```

## 🛠️ 실습 프로젝트

### 메인 프로젝트: "나만의 첫 RAG 시스템"
- 주제: 관심 분야의 문서 10개로 Q&A 시스템 구축
- 요구사항:
  - 최소 3가지 파일 형식 지원 (PDF, TXT, DOCX)
  - 50개 이상의 청크 생성
  - 검색 정확도 70% 이상 달성
  - 소스 추적 기능 구현
  - 성능 보고서 작성

### 도전 과제
1. **다국어 RAG**: 한국어/영어 동시 지원
2. **실시간 업데이트**: 새 문서 추가 시 즉시 반영
3. **시각화**: 검색 결과와 임베딩 공간 시각화

## 📊 평가 기준

### 필수 달성 목표
- [ ] LLM의 3가지 핵심 한계를 코드로 증명
- [ ] Naive RAG 전체 파이프라인 구현
- [ ] 최소 2개의 실전 프로젝트 완성
- [ ] 4가지 Naive RAG 한계 경험 및 문서화
- [ ] 검색 정확도 70% 이상 달성

### 추가 점수 항목
- [ ] 독창적인 RAG 활용 사례 제안
- [ ] 성능 개선 아이디어 3개 이상 구현
- [ ] 오픈소스 프로젝트로 공개

## 🎯 학습 성과
이 과정을 완료하면:
- Naive RAG의 모든 구성 요소를 직접 구현 가능
- 실제 비즈니스 문제에 RAG 적용 능력
- RAG 시스템의 성능 측정 및 평가 능력
- Advanced RAG로 나아갈 준비 완료

## 📚 필수 참고 자료
- [RAG 원논문 (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [LangChain RAG 튜토리얼](https://python.langchain.com/docs/use_cases/question_answering)
- [Chroma DB 공식 문서](https://docs.trychroma.com/)
- [OpenAI Embeddings 가이드](https://platform.openai.com/docs/guides/embeddings)

## ⏭️ 다음 단계: Advanced RAG
- Pre-retrieval 최적화 기법
- Post-retrieval 처리 전략
- 하이브리드 검색 시스템
- 성능 최적화 고급 기법

## 💡 핵심 메시지
"Naive RAG는 시작일 뿐입니다. 하지만 이 기초가 탄탄해야 고급 기법도 제대로 구현할 수 있습니다. 모든 한계를 직접 경험하고, 왜 Advanced RAG가 필요한지 체감하세요."