# 온톨로지 마스터클래스 콘텐츠 마이그레이션 계획

## 현재 콘텐츠 구조 분석

### 기존 콘텐츠 (index.html 기반)
- 16개 챕터로 구성된 단일 HTML 파일
- 5개 파트로 그룹화:
  1. 온톨로지 기초 (챕터 1-3)
  2. 시맨틱 웹과 온톨로지 (챕터 4-6)
  3. 온톨로지 언어와 도구 (챕터 7-9)
  4. 온톨로지 설계와 구축 (챕터 10-12)
  5. 온톨로지 활용과 응용 (챕터 13-16)

### 추가 콘텐츠 (chapters/ 디렉토리)
- 각 챕터별 독립적인 HTML 파일
- 그래프 시각화 (graphs.js)
- 스타일링 (style.css)

## 마이그레이션 전략

### 1단계: 콘텐츠 추출 및 변환
```javascript
// 콘텐츠 추출 스크립트
const extractContent = async () => {
  // 1. HTML 파일 파싱
  // 2. 각 챕터별로 콘텐츠 분리
  // 3. Markdown 형식으로 변환
  // 4. 메타데이터 추출 (제목, 순서, 카테고리 등)
}
```

### 2단계: 데이터베이스 구조 매핑

#### MongoDB - 콘텐츠 저장
```javascript
{
  chapter_id: "ch01_what_is_ontology",
  module_id: "ontology_basics",
  title: "온톨로지란 무엇인가?",
  order: 1,
  content: {
    markdown: "# 온톨로지란 무엇인가?\n...",
    html: "<h1>온톨로지란 무엇인가?</h1>...",
    components: [
      {
        type: "text",
        content: "온톨로지(Ontology)는..."
      },
      {
        type: "definition",
        term: "온톨로지",
        definition: "특정 도메인의 개념과 관계를 형식적으로 표현한 것"
      }
    ]
  },
  learning_objectives: [
    "온톨로지의 정의 이해",
    "온톨로지의 구성 요소 파악",
    "온톨로지의 필요성 인식"
  ]
}
```

#### Neo4j - 지식 그래프 구조
```cypher
// 모듈 생성
CREATE (m1:Module {
  id: 'ontology_basics',
  title: '온톨로지 기초',
  order: 1
})

// 챕터 생성
CREATE (c1:Chapter {
  id: 'ch01_what_is_ontology',
  title: '온톨로지란 무엇인가?',
  order: 1
})

// 개념 노드 생성
CREATE (concept1:Concept {
  id: 'ontology',
  name: '온톨로지',
  definition: '특정 도메인의 개념과 관계를 형식적으로 표현한 것'
})

// 관계 설정
CREATE (c1)-[:BELONGS_TO]->(m1)
CREATE (c1)-[:COVERS]->(concept1)
```

### 3단계: 대화형 요소 추가

#### 시뮬레이터 구현
1. **온톨로지 빌더**: 드래그 앤 드롭으로 개념과 관계 생성
2. **RDF 트리플 생성기**: 주어-술어-목적어 관계 실습
3. **SPARQL 쿼리 플레이그라운드**: 실시간 쿼리 실행 및 결과 확인
4. **OWL 추론 엔진**: 논리적 추론 과정 시각화

#### 학습 활동
- 각 챕터별 퀴즈
- 실습 과제
- 프로젝트 기반 평가

### 4단계: 콘텐츠 향상

#### 추가할 기능
1. **3D 지식 그래프 시각화**: Three.js 활용
2. **실시간 협업**: 여러 사용자가 함께 온톨로지 구축
3. **AI 기반 피드백**: 설계 패턴 제안 및 오류 검출
4. **도메인별 템플릿**: 의료, 금융, 교육 등 분야별 온톨로지 템플릿

## 마이그레이션 도구

### 콘텐츠 변환 스크립트
```typescript
// packages/migration/src/content-converter.ts
import { JSDOM } from 'jsdom'
import { marked } from 'marked'
import { collections } from '@cognosphere/database'

export async function migrateChapter(chapterPath: string) {
  // HTML 파일 읽기
  const dom = new JSDOM(await readFile(chapterPath))
  const document = dom.window.document
  
  // 콘텐츠 추출
  const title = document.querySelector('h1')?.textContent
  const sections = document.querySelectorAll('section')
  
  // MongoDB에 저장
  const contentCollection = await collections.content()
  await contentCollection.insertOne({
    // ... 변환된 콘텐츠
  })
  
  // Neo4j에 메타데이터 저장
  // ... 그래프 관계 생성
}
```

### 데이터 시딩 스크립트
```typescript
// packages/database/src/seed.ts
import { prisma } from './prisma'
import { runQuery } from './neo4j'
import { collections } from './mongodb'

async function seed() {
  // 1. PostgreSQL: 샘플 사용자 및 진도 데이터
  // 2. Neo4j: 전체 커리큘럼 구조
  // 3. MongoDB: 변환된 콘텐츠
  // 4. Redis: 초기 캐시 데이터
}
```

## 타임라인

1. **1주차**: 콘텐츠 추출 및 변환 도구 개발
2. **2주차**: 데이터베이스 마이그레이션 실행
3. **3주차**: 대화형 시뮬레이터 개발
4. **4주차**: 테스트 및 최적화

## 성공 지표

- [ ] 모든 16개 챕터 콘텐츠 마이그레이션 완료
- [ ] 지식 그래프에 100개 이상의 개념 노드 생성
- [ ] 각 챕터별 최소 1개의 대화형 시뮬레이션 구현
- [ ] 기존 사용자 경험 개선 (페이지 로드 시간 50% 단축)
- [ ] 학습 진도 추적 및 분석 기능 구현