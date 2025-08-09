# KSS 마이그레이션 계획

## 현재 상황
- `apps/ontology-mvp`: 기존 온톨로지 MVP (Next.js)
- `apps/web`: 기존 웹 프로젝트
- `kss-simulator`: 별도 디렉토리에 있는 KSS 프로젝트

## 통합 계획

### Option 1: ontology-mvp를 KSS로 리브랜딩 ✅ (추천)
```bash
# 1. ontology-mvp를 kss-web으로 이름 변경
cd cognosphere/apps
mv ontology-mvp kss-web

# 2. package.json 업데이트
# name: "@kss/web"

# 3. 기존 온톨로지 콘텐츠 통합
cp -r ../../chapters kss-web/public/content
```

### Option 2: 새로 시작
- 클린 슬레이트로 시작
- 기존 코드 선택적 이동

## 즉시 실행 명령어

```bash
# Option 1 실행
cd /Users/kelly/Desktop/Space/project/Ontology/cognosphere

# 1. 백업
cp -r apps/ontology-mvp apps/ontology-mvp.backup

# 2. 리네임
mv apps/ontology-mvp apps/kss-web

# 3. package.json 수정
cd apps/kss-web
# package.json의 name을 "@kss/web"으로 변경

# 4. 온톨로지 콘텐츠 통합
cp -r ../../../chapters public/content

# 5. 의존성 설치
pnpm install
```

## 다음 단계
1. KSS 홈페이지 디자인
2. 온톨로지 시뮬레이터 MVP 기능 구현
3. API 규약 정의