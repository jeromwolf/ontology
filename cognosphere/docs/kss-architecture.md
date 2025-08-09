# KSS Architecture - 하이브리드 접근법

## 🎯 목표
최소한의 구조로 시작하여 점진적으로 확장 가능한 아키텍처 구축

## 📁 프로젝트 구조

```
cognosphere/
├── apps/
│   ├── kss-web/              # KSS 메인 웹 애플리케이션
│   │   └── ontology-mvp/     # 온톨로지 시뮬레이터 MVP
│   └── api/                  # API 게이트웨이 (향후)
│
├── packages/
│   ├── shared/               # 공통 유틸리티, 타입, 상수
│   ├── ui/                   # 공통 UI 컴포넌트
│   └── simulators/           # 시뮬레이터 코어 (향후)
│
├── services/                 # 마이크로서비스 (향후)
│   ├── ontology/            # 온톨로지 서비스
│   ├── content/             # 콘텐츠 관리
│   └── user/                # 사용자 관리
│
└── docs/
    ├── architecture/         # 아키텍처 문서
    ├── api/                 # API 문서
    └── guides/              # 개발 가이드
```

## 🚀 Phase 1: MVP (현재)

### 1. 단일 앱으로 시작
- `apps/kss-web/ontology-mvp`에 모든 기능 집중
- Next.js App Router 사용
- 서버리스 함수로 백엔드 처리

### 2. 최소 공통 코드
```typescript
// packages/shared/types/ontology.ts
export interface Concept {
  id: string;
  label: string;
  description?: string;
}

export interface Triple {
  subject: Concept;
  predicate: string;
  object: Concept;
}
```

### 3. 간단한 API 규약
```typescript
// API Routes (Next.js)
POST   /api/ontology/triple     # Triple 생성
GET    /api/ontology/graph      # 그래프 조회
POST   /api/sparql/query        # SPARQL 실행
```

## 🔄 Phase 2: 분리 (1개월 후)

### 1. 서비스 분리 준비
- 비즈니스 로직을 packages로 이동
- API 레이어 추가
- 데이터베이스 추상화

### 2. A2A 패턴 도입
```typescript
// packages/agents/base.ts
export abstract class Agent {
  abstract process(input: any): Promise<any>;
  abstract communicate(target: Agent, message: any): Promise<any>;
}
```

## 🌐 Phase 3: 확장 (2-3개월 후)

### 1. 마이크로서비스 전환
- Docker 컨테이너화
- 서비스 메시 구현
- 이벤트 기반 통신

### 2. 멀티 도메인 지원
- LLM 시뮬레이터 추가
- 플러그인 시스템 구현
- 마켓플레이스 준비

## 🛠 기술 결정

### 현재 (MVP)
- **Frontend**: Next.js 14, TypeScript, Tailwind
- **Backend**: Next.js API Routes
- **Database**: Supabase (PostgreSQL)
- **Deploy**: Vercel

### 향후 (확장)
- **Container**: Docker, Kubernetes
- **Message**: Redis, RabbitMQ
- **Graph DB**: Neo4j
- **Monitor**: Prometheus, Grafana

## 📝 개발 원칙

1. **YAGNI (You Aren't Gonna Need It)**
   - 필요할 때까지 구현하지 않음
   - 과도한 추상화 피하기

2. **DRY (Don't Repeat Yourself)**
   - 공통 코드는 packages로
   - 재사용 가능한 컴포넌트

3. **KISS (Keep It Simple, Stupid)**
   - 복잡한 것보다 간단한 해결책
   - 이해하기 쉬운 코드

## 🎬 즉시 실행 계획

```bash
# 1. 기존 프로젝트 정리
cd cognosphere
rm -rf apps/ontology-mvp  # 중복 제거

# 2. KSS 웹 앱 생성
cd apps
npx create-next-app@latest kss-web --typescript --tailwind --app

# 3. 온톨로지 콘텐츠 이동
cp -r ../../chapters kss-web/public/content

# 4. 공통 패키지 설정
cd ../packages
mkdir -p shared/src/types
mkdir -p ui/src/components
```

이 구조로 시작하면 MVP를 빠르게 개발하면서도 향후 확장이 용이합니다.