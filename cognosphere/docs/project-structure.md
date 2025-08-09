# Cognosphere 프로젝트 구조

## 디렉토리 구조

```
cognosphere/
├── apps/
│   ├── web/                    # Next.js 14 웹 애플리케이션
│   │   ├── src/
│   │   │   ├── app/           # App Router 페이지
│   │   │   ├── components/    # React 컴포넌트
│   │   │   ├── lib/          # 유틸리티 함수
│   │   │   └── styles/       # 글로벌 스타일
│   │   ├── public/           # 정적 파일
│   │   └── package.json
│   │
│   └── api/                    # API 서버 (향후 구현)
│       ├── src/
│       │   ├── routes/       # API 엔드포인트
│       │   ├── services/     # 비즈니스 로직
│       │   └── middleware/   # Express 미들웨어
│       └── package.json
│
├── packages/
│   ├── ui/                     # 공유 UI 컴포넌트 라이브러리
│   │   ├── src/
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   └── ...
│   │   └── package.json
│   │
│   ├── database/               # 데이터베이스 클라이언트 및 스키마
│   │   ├── src/
│   │   │   ├── prisma.ts    # PostgreSQL (Prisma)
│   │   │   ├── neo4j.ts     # Neo4j 드라이버
│   │   │   ├── mongodb.ts   # MongoDB 클라이언트
│   │   │   └── redis.ts     # Redis 클라이언트
│   │   ├── prisma/
│   │   │   └── schema.prisma
│   │   └── package.json
│   │
│   ├── shared/                 # 공유 유틸리티 및 타입
│   │   ├── src/
│   │   │   ├── types/       # TypeScript 타입 정의
│   │   │   ├── utils/       # 공유 유틸리티
│   │   │   └── constants/   # 상수
│   │   └── package.json
│   │
│   ├── simulators/             # 온톨로지 시뮬레이터
│   │   ├── src/
│   │   │   ├── ontology-builder/
│   │   │   ├── rdf-editor/
│   │   │   ├── sparql-playground/
│   │   │   └── owl-reasoner/
│   │   └── package.json
│   │
│   └── typescript-config/      # 공유 TypeScript 설정
│       ├── base.json
│       ├── nextjs.json
│       └── package.json
│
├── docs/                       # 프로젝트 문서
│   ├── database-schema.md     # 데이터베이스 설계
│   ├── migration-plan.md      # 마이그레이션 계획
│   ├── mcp-tools-usage.md     # MCP 도구 활용
│   └── project-structure.md   # 프로젝트 구조 (이 파일)
│
├── scripts/                    # 유틸리티 스크립트
│   ├── migrate-content.ts     # 콘텐츠 마이그레이션
│   ├── seed-database.ts       # 데이터베이스 시딩
│   └── setup.sh              # 초기 설정
│
├── .github/                    # GitHub Actions
│   └── workflows/
│       ├── ci.yml            # CI 파이프라인
│       └── deploy.yml        # 배포 파이프라인
│
├── package.json               # 루트 package.json
├── turbo.json                # Turborepo 설정
├── pnpm-workspace.yaml       # pnpm 워크스페이스 설정
├── .gitignore
├── .env.example              # 환경변수 예시
└── README.md                 # 프로젝트 README
```

## 주요 기술 스택

### Frontend
- **Next.js 14**: App Router, Server Components
- **React 18**: UI 라이브러리
- **TypeScript**: 타입 안정성
- **Tailwind CSS**: 유틸리티 기반 스타일링
- **Framer Motion**: 애니메이션
- **Three.js**: 3D 시각화
- **Zustand**: 상태 관리
- **React Query**: 서버 상태 관리

### Backend
- **Node.js**: 런타임
- **Express**: API 서버 (향후)
- **Prisma**: PostgreSQL ORM
- **Neo4j Driver**: 그래프 데이터베이스
- **MongoDB Driver**: 문서 데이터베이스
- **Redis**: 캐싱 및 세션

### Infrastructure
- **Turborepo**: Monorepo 관리
- **pnpm**: 패키지 매니저
- **Docker**: 컨테이너화
- **GitHub Actions**: CI/CD

## 개발 명령어

```bash
# 의존성 설치
pnpm install

# 개발 서버 실행
pnpm dev

# 빌드
pnpm build

# 테스트
pnpm test

# 린트
pnpm lint

# 데이터베이스 마이그레이션
pnpm db:migrate

# 데이터베이스 시딩
pnpm db:seed
```

## 환경 설정

1. `.env.example`을 `.env.local`로 복사
2. 데이터베이스 연결 정보 설정
3. 인증 관련 시크릿 키 생성

## 배포

- **Web App**: Vercel
- **PostgreSQL**: Supabase/Neon
- **Neo4j**: Neo4j Aura
- **MongoDB**: MongoDB Atlas
- **Redis**: Upstash

## 기여 가이드

1. Feature 브랜치 생성: `feature/기능명`
2. 커밋 메시지 규칙: `type(scope): message`
3. PR 템플릿 사용
4. 코드 리뷰 필수