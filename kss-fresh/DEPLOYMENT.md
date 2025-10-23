# KSS Platform - Deployment Guide

## 🚀 Quick Deploy to Cloud Run

### Prerequisites

1. **Google Cloud SDK 설치**
```bash
# macOS
brew install google-cloud-sdk

# 또는 공식 설치 프로그램
# https://cloud.google.com/sdk/docs/install
```

2. **Docker 설치**
```bash
# Docker Desktop 설치
# https://www.docker.com/products/docker-desktop
```

3. **Google Cloud 인증**
```bash
gcloud auth login
gcloud auth configure-docker
```

### 🎯 간단 배포 (원클릭)

```bash
./deploy.sh
```

### 📝 커스텀 설정으로 배포

```bash
# 프로젝트 ID, 리전, 서비스명 지정
./deploy.sh --project YOUR_PROJECT_ID --region asia-northeast3 --service kss-platform
```

### 🌐 커스텀 도메인 설정 (kss.ai.kr)

#### 1. Cloud Run에 도메인 매핑

```bash
gcloud run domain-mappings create \
  --service=kss-platform \
  --domain=kss.ai.kr \
  --region=asia-northeast3
```

#### 2. DNS 레코드 설정

위 명령을 실행하면 DNS 레코드 추가 안내가 나옵니다:

```
Please add the following DNS records to your domain:

Type: A
Name: kss.ai.kr
Value: 216.239.32.21

Type: A
Name: kss.ai.kr
Value: 216.239.34.21

Type: A
Name: kss.ai.kr
Value: 216.239.36.21

Type: A
Name: kss.ai.kr
Value: 216.239.38.21

Type: AAAA
Name: kss.ai.kr
Value: 2001:4860:4802:32::15

Type: AAAA
Name: kss.ai.kr
Value: 2001:4860:4802:34::15

Type: AAAA
Name: kss.ai.kr
Value: 2001:4860:4802:36::15

Type: AAAA
Name: kss.ai.kr
Value: 2001:4860:4802:38::15
```

도메인 등록 업체(가비아, Route53 등)에서 이 레코드들을 추가하세요.

#### 3. SSL 인증서 자동 발급

Cloud Run이 자동으로 Let's Encrypt SSL 인증서를 발급합니다 (약 15분 소요).

### 📊 배포 후 확인

#### 서비스 URL 확인
```bash
gcloud run services describe kss-platform --region=asia-northeast3 --format='value(status.url)'
```

#### 로그 확인
```bash
gcloud run services logs read kss-platform --region=asia-northeast3 --limit=50
```

#### 실시간 로그 스트리밍
```bash
gcloud run services logs tail kss-platform --region=asia-northeast3
```

### 🔄 업데이트 배포

코드 수정 후:

```bash
# 1. Git 커밋
git add .
git commit -m "feat: Update AI Infrastructure module"
git push

# 2. 재배포
./deploy.sh
```

### 🎛️ 환경 변수 설정

#### Cloud Run에서 환경 변수 추가

```bash
gcloud run services update kss-platform \
  --region=asia-northeast3 \
  --set-env-vars="KEY1=value1,KEY2=value2"
```

#### 시크릿 환경 변수 (민감 정보)

```bash
# 1. Secret Manager에 시크릿 생성
echo -n "your-secret-value" | gcloud secrets create SECRET_NAME --data-file=-

# 2. Cloud Run에 연결
gcloud run services update kss-platform \
  --region=asia-northeast3 \
  --update-secrets=SECRET_NAME=SECRET_NAME:latest
```

### 📈 성능 튜닝

#### 리소스 조정

```bash
gcloud run services update kss-platform \
  --region=asia-northeast3 \
  --memory=4Gi \
  --cpu=4 \
  --max-instances=20 \
  --min-instances=1
```

#### 동시성 설정

```bash
gcloud run services update kss-platform \
  --region=asia-northeast3 \
  --concurrency=100
```

### 🛠️ 트러블슈팅

#### 빌드 실패 시
```bash
# Docker 빌드 캐시 제거
docker builder prune -a

# 재빌드
docker build --no-cache -t gcr.io/PROJECT_ID/kss-platform .
```

#### 메모리 부족 에러
```bash
# 메모리 증가
gcloud run services update kss-platform \
  --region=asia-northeast3 \
  --memory=4Gi
```

#### 타임아웃 에러
```bash
# 타임아웃 연장 (최대 3600초)
gcloud run services update kss-platform \
  --region=asia-northeast3 \
  --timeout=600
```

### 💰 비용 최적화

#### 최소 인스턴스 0으로 설정 (트래픽 없을 때 비용 0)
```bash
gcloud run services update kss-platform \
  --region=asia-northeast3 \
  --min-instances=0
```

#### 요청 기반 스케일링
```bash
gcloud run services update kss-platform \
  --region=asia-northeast3 \
  --cpu-throttling \
  --concurrency=80
```

### 📚 추가 리소스

- [Cloud Run 공식 문서](https://cloud.google.com/run/docs)
- [Next.js 배포 가이드](https://nextjs.org/docs/deployment)
- [Dockerfile 최적화](https://docs.docker.com/develop/dev-best-practices/)

### 🆘 Support

문제가 발생하면:
1. Cloud Run 로그 확인: `gcloud run services logs read kss-platform`
2. Docker 로컬 테스트: `docker build . && docker run -p 3000:3000 IMAGE_ID`
3. GitHub Issues: https://github.com/jeromwolf/ontology/issues

---

**마지막 업데이트**: 2025-10-22
**버전**: 1.0.0
