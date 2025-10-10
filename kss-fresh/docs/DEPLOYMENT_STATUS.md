# 🌐 KSS 배포 현황 및 설정

**작성일**: 2025-10-10
**배포 URL**: https://kss.ai.kr/
**상태**: ✅ 정상 운영 중

---

## 🎯 배포 정보

### 운영 환경
- **도메인**: https://kss.ai.kr/
- **IP 주소**: 34.160.100.11
- **플랫폼**: Google Cloud (Cloud Run / App Engine)
- **서버**: Google Frontend
- **프레임워크**: Next.js (확인됨)
- **프로토콜**: HTTP/2
- **상태**: ✅ 200 OK

### DNS 설정
```
Name:    kss.ai.kr
Address: 34.160.100.11 (Google Cloud Korea Region)
```

### HTTP 헤더
```
Server: Google Frontend
X-Powered-By: Next.js
Content-Security-Policy: default-src 'self' 'unsafe-inline' 'unsafe-eval' ...
Cache-Control: private, no-cache, no-store, max-age=0, must-revalidate
```

---

## 🚀 배포 방식

### 추정 배포 플랫폼

#### Option 1: Google Cloud Run (가능성 높음)
- **장점**:
  - Serverless, Auto-scaling
  - Docker 컨테이너 기반
  - 비용 효율적 (사용량 기반 과금)
- **특징**:
  - HTTP/2 지원 ✅
  - Google Frontend ✅
  - 컨테이너 배포 ✅

#### Option 2: Google App Engine
- **장점**:
  - 완전 관리형 서비스
  - 자동 스케일링
- **특징**:
  - Next.js 직접 지원

---

## 📦 현재 배포 파일

### 1. Dockerfile
**위치**: `/kss-fresh/Dockerfile`

```dockerfile
# 3단계 멀티스테이지 빌드
FROM node:20-alpine AS deps
# ... 의존성 설치

FROM node:20-alpine AS builder
# ... 빌드

FROM node:20-alpine AS runner
# ... 프로덕션 실행
CMD ["node", "server.js"]
```

**특징**:
- ✅ Alpine Linux 기반 (경량화)
- ✅ 멀티스테이지 빌드 (최적화)
- ✅ Health check 포함
- ✅ Non-root 사용자 보안

### 2. 배포 가이드
**위치**: `/kss-fresh/DOCKER_GUIDE.md`

**Google Cloud Run 배포 명령어**:
```bash
# 1. 인증
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

---

## 🔄 배포 프로세스

### 현재 배포 방식 (추정)
```
┌─────────────┐
│  Git Push   │
│   (main)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Manual    │  ← 현재 방식 (추정)
│   Deploy    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Google      │
│ Cloud Run   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ https://    │
│ kss.ai.kr   │
└─────────────┘
```

### 권장 배포 프로세스 (자동화)
```
┌─────────────┐
│  Git Push   │
│   (main)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   GitHub    │  ← 자동화 추가 가능
│   Actions   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Docker    │
│   Build     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Cloud     │
│   Run       │
│   Deploy    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ https://    │
│ kss.ai.kr   │
└─────────────┘
```

---

## 🔧 배포 자동화 설정 (선택사항)

### GitHub Actions + Cloud Build

**파일**: `.github/workflows/deploy.yml`

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches:
      - main

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  SERVICE_NAME: kss-platform
  REGION: asia-northeast3

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          project_id: ${{ secrets.GCP_PROJECT_ID }}

      - name: Configure Docker for GCR
        run: gcloud auth configure-docker

      - name: Build Docker image
        run: |
          docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:${{ github.sha }} .
          docker tag gcr.io/$PROJECT_ID/$SERVICE_NAME:${{ github.sha }} \
            gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

      - name: Push to Container Registry
        run: |
          docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:${{ github.sha }}
          docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy $SERVICE_NAME \
            --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
            --platform managed \
            --region $REGION \
            --allow-unauthenticated \
            --set-env-vars DATABASE_URL=${{ secrets.DATABASE_URL }} \
            --set-env-vars NEXTAUTH_SECRET=${{ secrets.NEXTAUTH_SECRET }} \
            --set-env-vars NEXTAUTH_URL=https://kss.ai.kr
```

**필요한 GitHub Secrets**:
- `GCP_PROJECT_ID`: Google Cloud 프로젝트 ID
- `GCP_SA_KEY`: Service Account 키 (JSON)
- `DATABASE_URL`: PostgreSQL 연결 문자열
- `NEXTAUTH_SECRET`: NextAuth 시크릿 키

---

## 📊 모니터링

### Cloud Run 대시보드
- **URL**: https://console.cloud.google.com/run
- **확인 사항**:
  - 요청 수
  - 응답 시간
  - 에러율
  - 메모리 사용량
  - CPU 사용률

### 로그 확인
```bash
# Cloud Run 로그 보기
gcloud run services logs read kss-platform \
  --region asia-northeast3 \
  --limit 50
```

### 메트릭 확인
```bash
# 서비스 상태 확인
gcloud run services describe kss-platform \
  --region asia-northeast3
```

---

## 🔐 환경 변수 설정

### Cloud Run 환경 변수 업데이트
```bash
gcloud run services update kss-platform \
  --region asia-northeast3 \
  --set-env-vars DATABASE_URL="postgresql://..." \
  --set-env-vars NEXTAUTH_SECRET="your-secret" \
  --set-env-vars OPENAI_API_KEY="sk-..."
```

### Secret Manager 사용 (권장)
```bash
# Secret 생성
echo -n "your-secret-value" | \
  gcloud secrets create DATABASE_URL --data-file=-

# Cloud Run에서 Secret 사용
gcloud run services update kss-platform \
  --region asia-northeast3 \
  --set-secrets DATABASE_URL=DATABASE_URL:latest
```

---

## 🚦 배포 체크리스트

### 배포 전 확인사항
- [ ] 로컬에서 `npm run build` 성공
- [ ] Docker 이미지 빌드 성공
- [ ] 환경 변수 설정 완료
- [ ] 데이터베이스 마이그레이션 완료
- [ ] Health check 엔드포인트 작동 (`/api/health`)

### 배포 후 확인사항
- [ ] https://kss.ai.kr/ 접속 확인
- [ ] 주요 페이지 렌더링 확인
  - [ ] 홈페이지
  - [ ] /modules/ontology
  - [ ] /modules/stock-analysis
  - [ ] /modules/llm
- [ ] API 엔드포인트 작동 확인
- [ ] 데이터베이스 연결 확인
- [ ] 로그에 에러 없는지 확인

---

## 🔍 트러블슈팅

### 배포 실패 시
```bash
# 최근 배포 로그 확인
gcloud run services logs read kss-platform --limit 100

# 서비스 상세 정보
gcloud run services describe kss-platform --region asia-northeast3

# 이전 버전으로 롤백
gcloud run services update kss-platform \
  --region asia-northeast3 \
  --image gcr.io/[PROJECT-ID]/kss-platform:[PREVIOUS_TAG]
```

### 사이트 접속 안 될 때
1. **DNS 확인**: `nslookup kss.ai.kr`
2. **Cloud Run 상태 확인**: GCP Console
3. **로그 확인**: Cloud Logging
4. **Health check**: `curl https://kss.ai.kr/api/health`

### 성능 이슈
```bash
# 인스턴스 수 증가
gcloud run services update kss-platform \
  --region asia-northeast3 \
  --min-instances 1 \
  --max-instances 10

# 메모리 증가
gcloud run services update kss-platform \
  --region asia-northeast3 \
  --memory 2Gi

# CPU 증가
gcloud run services update kss-platform \
  --region asia-northeast3 \
  --cpu 2
```

---

## 📈 비용 최적화

### Cloud Run 요금
- **요청 기반**: $0.40 per million requests
- **CPU 시간**: $0.00002400 per vCPU-second
- **메모리**: $0.00000250 per GiB-second

### 최적화 팁
1. **최소 인스턴스 0으로 설정** (트래픽 없을 때 비용 절감)
2. **적절한 메모리/CPU 할당** (과도한 리소스 방지)
3. **CDN 사용** (정적 파일 캐싱)
4. **이미지 최적화** (Docker 이미지 크기 최소화)

---

## 🎯 다음 단계

### 즉시 가능한 개선
1. **자동 배포 설정** (GitHub Actions)
2. **모니터링 강화** (Uptime checks, Alerts)
3. **CDN 연동** (Cloud CDN)
4. **백업 전략** (데이터베이스, 이미지)

### 중장기 개선
1. **멀티 리전 배포** (고가용성)
2. **Load Balancing** (트래픽 분산)
3. **Auto-scaling 최적화** (비용 vs 성능)
4. **보안 강화** (WAF, DDoS 방어)

---

## 📞 문의 및 지원

- **GCP 지원**: https://console.cloud.google.com/support
- **Cloud Run 문서**: https://cloud.google.com/run/docs
- **프로젝트 이슈**: https://github.com/jeromwolf/ontology/issues

---

**Last Updated**: 2025-10-10
**Status**: ✅ Production (정상 운영)
**Next Review**: 배포 자동화 설정 시

---

## 🎉 요약

✅ **현재 상태**: https://kss.ai.kr/ 정상 운영 중
✅ **플랫폼**: Google Cloud (Cloud Run 추정)
✅ **배포 방식**: 수동 배포 (추정)
✅ **성능**: HTTP/2, CDN 준비
✅ **보안**: HTTPS, CSP 헤더 적용

**다음 작업**: 배포 자동화 설정 권장 (선택사항)
