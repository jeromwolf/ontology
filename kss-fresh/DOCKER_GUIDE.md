# 🐳 KSS Docker 실행 가이드

## 📋 사전 준비

### 1. Docker 설치 확인
```bash
docker --version
docker-compose --version
```

### 2. 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하세요:

```bash
# .env 파일 생성
cp .env.example .env
```

필수 환경 변수:
```env
# Database (Neon PostgreSQL)
DATABASE_URL="postgresql://username:password@host/database?sslmode=require"

# NextAuth
NEXTAUTH_SECRET="your-secret-key-here"
NEXTAUTH_URL="http://localhost:3000"

# OpenAI (Optional)
OPENAI_API_KEY="sk-..."

# KIS API (Optional - for Stock Analysis)
KIS_APP_KEY="your-kis-app-key"
KIS_APP_SECRET="your-kis-app-secret"
```

---

## 🚀 빠른 시작

### Option 1: Docker Compose (권장)

```bash
# 1. 빌드 및 실행
docker-compose up -d

# 2. 로그 확인
docker-compose logs -f

# 3. 중지
docker-compose down

# 4. 완전 삭제 (볼륨 포함)
docker-compose down -v
```

### Option 2: Docker 단독 실행

```bash
# 1. 이미지 빌드
docker build -t kss-platform:latest .

# 2. 컨테이너 실행
docker run -d \
  --name kss-platform \
  -p 3000:3000 \
  --env-file .env \
  kss-platform:latest

# 3. 로그 확인
docker logs -f kss-platform

# 4. 중지 및 삭제
docker stop kss-platform
docker rm kss-platform
```

---

## 📊 접속 확인

브라우저에서 다음 URL로 접속:
- **메인**: http://localhost:3000
- **온톨로지**: http://localhost:3000/modules/ontology
- **주식분석**: http://localhost:3000/modules/stock-analysis

---

## 🔧 유용한 명령어

### 컨테이너 상태 확인
```bash
docker-compose ps
```

### 실시간 로그 보기
```bash
docker-compose logs -f kss-app
```

### 컨테이너 내부 접속
```bash
docker-compose exec kss-app sh
```

### 이미지 크기 확인
```bash
docker images | grep kss
```

### 빌드 캐시 없이 재빌드
```bash
docker-compose build --no-cache
docker-compose up -d
```

---

## 🐛 트러블슈팅

### 1. 포트 충돌
```bash
# 포트 3000이 이미 사용중인 경우
# docker-compose.yml에서 포트 변경:
ports:
  - "3001:3000"  # 3001로 변경
```

### 2. 데이터베이스 연결 실패
```bash
# .env 파일의 DATABASE_URL 확인
# Prisma 클라이언트 재생성
docker-compose exec kss-app npx prisma generate
```

### 3. 빌드 오류
```bash
# 로컬에서 먼저 빌드 테스트
npm run build

# node_modules 삭제 후 재시도
rm -rf node_modules .next
npm install
docker-compose build --no-cache
```

### 4. 메모리 부족
Docker Desktop 설정에서 메모리 할당 증가:
- Settings → Resources → Memory → 4GB 이상

---

## 📈 프로덕션 배포

### Google Cloud Run 배포
```bash
# 1. Google Cloud 인증
gcloud auth login

# 2. Container Registry에 푸시
docker tag kss-platform:latest gcr.io/[PROJECT-ID]/kss-platform:latest
docker push gcr.io/[PROJECT-ID]/kss-platform:latest

# 3. Cloud Run 배포
gcloud run deploy kss-platform \
  --image gcr.io/[PROJECT-ID]/kss-platform:latest \
  --platform managed \
  --region asia-northeast3 \
  --allow-unauthenticated
```

### Docker Hub 배포
```bash
# 1. Docker Hub 로그인
docker login

# 2. 태그 지정 및 푸시
docker tag kss-platform:latest username/kss-platform:latest
docker push username/kss-platform:latest
```

---

## 🎯 최적화 팁

### 1. 멀티 스테이지 빌드
현재 Dockerfile은 3단계 빌드로 최적화되어 있습니다:
- **deps**: 의존성 설치
- **builder**: 앱 빌드
- **runner**: 실행 (최종 이미지 크기 최소화)

### 2. 빌드 캐시 활용
```bash
# BuildKit 활성화 (더 빠른 빌드)
export DOCKER_BUILDKIT=1
docker build -t kss-platform:latest .
```

### 3. 이미지 크기 최소화
- Alpine Linux 사용 (node:20-alpine)
- 프로덕션 의존성만 포함
- .dockerignore로 불필요한 파일 제외

---

## 📚 추가 자료

- [Next.js Docker 공식 가이드](https://nextjs.org/docs/deployment#docker-image)
- [Docker Compose 문서](https://docs.docker.com/compose/)
- [Google Cloud Run 가이드](https://cloud.google.com/run/docs)

---

## 🆘 문제 해결

문제가 발생하면:
1. 로그 확인: `docker-compose logs -f`
2. 헬스체크 확인: `docker-compose ps`
3. 컨테이너 재시작: `docker-compose restart`
4. 완전 재빌드: `docker-compose down && docker-compose build --no-cache && docker-compose up -d`

---

**생성일**: 2025-10-07
**버전**: 1.0.0
**프로젝트**: KSS (Knowledge Space Simulator)
