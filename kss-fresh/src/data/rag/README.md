# RAG Module Curriculum Structure

이 디렉토리는 RAG (Retrieval-Augmented Generation) 모듈의 체계적인 커리큘럼 데이터를 포함합니다.

## 📁 파일 구조

```
src/data/rag/
├── beginnerCurriculum.ts    # 초급 과정 (4개 섹션)
├── intermediateCurriculum.ts # 중급 과정 (5개 섹션)
├── advancedCurriculum.ts    # 고급 과정 (6개 섹션)
├── supplementaryCurriculum.ts # 보충 자료 (5개 주제)
└── index.ts                 # 통합 export 파일
```

## 🎯 각 과정 개요

### 초급 과정 (Beginner)
- **목표**: RAG의 기본 개념과 필요성 이해
- **기간**: 약 10시간
- **주요 내용**:
  1. RAG 기초 개념
  2. 문서 처리 기초
  3. 청킹 전략 이해
  4. 첫 RAG 시스템 만들기

### 중급 과정 (Intermediate)
- **목표**: 핵심 컴포넌트 마스터
- **기간**: 약 15시간
- **주요 내용**:
  1. 임베딩 심화 학습
  2. 벡터 데이터베이스 마스터
  3. 하이브리드 검색 구현
  4. RAG 프롬프트 엔지니어링
  5. 성능 최적화

### 고급 과정 (Advanced)
- **목표**: 프로덕션 레벨 구현
- **기간**: 약 20시간
- **주요 내용**:
  1. GraphRAG 아키텍처
  2. Multi-hop Reasoning
  3. 고급 Reranking 전략
  4. RAG 평가와 모니터링
  5. 프로덕션 배포와 스케일링
  6. 최신 연구 동향

### 보충 자료 (Supplementary)
- 한국어 특화 RAG
- Multimodal RAG
- 도메인 특화 RAG (의료, 법률, 금융)
- RAG 보안과 프라이버시
- RAG 대안 기술

## 🔗 연동 방법

### 1. 데이터 가져오기
```typescript
import { 
  beginnerCurriculum, 
  intermediateCurriculum,
  advancedCurriculum,
  supplementaryCurriculum 
} from '@/data/rag'
```

### 2. 커리큘럼 아이템 구조
```typescript
interface CurriculumItem {
  id: string
  title: string
  description: string
  topics: string[]
  resources: {
    type: 'chapter' | 'simulator' | 'external'
    title: string
    url?: string
    duration?: string
  }[]
  quiz?: {
    question: string
    options: string[]
    answer: number
  }[]
}
```

### 3. 페이지 연동
- `/app/modules/rag/page.tsx`에서 모든 커리큘럼 데이터 표시
- 각 과정별 진행률 추적
- 체크리스트를 통한 학습 완료 확인

## 📚 추가 리소스

`supplementaryResources` 객체에는 다음이 포함됩니다:
- **books**: 추천 도서 목록
- **papers**: 핵심 논문 링크
- **tools**: 필수 개발 도구 (LangChain, LlamaIndex, Haystack)
- **communities**: 관련 커뮤니티 정보

## 🎓 학습 경로 추천

1. **입문자**: 초급 과정 → 중급 과정 1-3장
2. **경험자**: 중급 과정 → 고급 과정
3. **전문가**: 고급 과정 + 최신 논문 연구

각 과정은 이전 단계의 지식을 기반으로 구성되어 있으므로, 순차적인 학습을 권장합니다.