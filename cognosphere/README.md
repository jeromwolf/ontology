# Cognosphere - Hybrid Monorepo Architecture

온톨로지 MVP에 집중하면서 미래 확장성을 고려한 하이브리드 모노레포 구조입니다.

## 🏗️ 프로젝트 구조

```
cognosphere/
├── apps/
│   └── ontology-mvp/          # 온톨로지 학습 플랫폼 MVP
├── packages/
│   └── shared/                # 최소한의 공통 유틸리티
├── services/                  # 향후 마이크로서비스 준비
├── docs/
│   └── architecture/          # 시스템 설계 문서
└── turbo.json                # 빌드 최적화 설정
```

## 🚀 빠른 시작

### 1. 의존성 설치
```bash
npm install
```

### 2. 온톨로지 MVP 개발 서버 실행
```bash
npm run ontology:dev
```

### 3. 프로덕션 빌드
```bash
npm run ontology:build
npm run ontology:start
```

## 📦 워크스페이스 구조

### apps/ontology-mvp
- Next.js 14 기반 온톨로지 학습 플랫폼
- TypeScript + Tailwind CSS
- 포트: 3001

### packages/shared
- 공통 타입 정의
- 유틸리티 함수
- 비즈니스 로직 (최소한만 유지)

## 🛠️ 개발 명령어

```bash
# 전체 개발 서버
npm run dev

# 온톨로지 MVP만 실행
npm run ontology:dev

# 빌드
npm run build

# 린트
npm run lint

# 클린
npm run clean
```

## 🎯 MVP 집중 전략

1. **단순성 우선**: 불필요한 추상화 최소화
2. **빠른 반복**: 온톨로지 MVP에 모든 리소스 집중
3. **점진적 확장**: services/ 폴더는 준비만 해두고 MVP 이후 구현
4. **명확한 경계**: 각 앱은 독립적으로 배포 가능

## 📝 향후 확장 계획

1. **Phase 1 (현재)**: 온톨로지 MVP 완성
2. **Phase 2**: 인증/결제 서비스 분리
3. **Phase 3**: AI 서비스 통합
4. **Phase 4**: 멀티 플랫폼 확장