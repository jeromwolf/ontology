#!/bin/bash

# KSS Platform - Cloud Run Deployment Script
# 사용법: ./deploy.sh [--project PROJECT_ID] [--region REGION] [--service SERVICE_NAME]

set -e

# 기본 설정
PROJECT_ID="${PROJECT_ID:-kss-platform}"
REGION="${REGION:-asia-northeast3}"
SERVICE_NAME="${SERVICE_NAME:-kss-platform}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# 색상 코드
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

log_success() {
    echo -e "${GREEN}✓ ${NC}$1"
}

log_warning() {
    echo -e "${YELLOW}⚠ ${NC}$1"
}

log_error() {
    echo -e "${RED}✗ ${NC}$1"
}

# 파라미터 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --service)
            SERVICE_NAME="$2"
            IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
            shift 2
            ;;
        *)
            log_error "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

log_info "Starting deployment to Cloud Run..."
echo "  Project ID: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Service: ${SERVICE_NAME}"
echo "  Image: ${IMAGE_NAME}"
echo ""

# Step 1: Git 상태 확인
log_info "Checking git status..."
if [[ -n $(git status -s) ]]; then
    log_warning "You have uncommitted changes. Continue? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log_error "Deployment cancelled"
        exit 1
    fi
fi

# Step 2: Google Cloud 프로젝트 설정
log_info "Setting Google Cloud project..."
gcloud config set project ${PROJECT_ID}
log_success "Project set to ${PROJECT_ID}"

# Step 3: Docker 이미지 빌드
log_info "Building Docker image..."
docker build -t ${IMAGE_NAME}:latest -t ${IMAGE_NAME}:$(git rev-parse --short HEAD) .
log_success "Docker image built successfully"

# Step 4: Container Registry에 푸시
log_info "Pushing image to Google Container Registry..."
docker push ${IMAGE_NAME}:latest
docker push ${IMAGE_NAME}:$(git rev-parse --short HEAD)
log_success "Image pushed to GCR"

# Step 5: Cloud Run에 배포
log_info "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars "NODE_ENV=production,NEXT_TELEMETRY_DISABLED=1"

log_success "Deployment completed!"

# Step 6: 서비스 URL 출력
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')
echo ""
log_success "Service is now available at:"
echo -e "  ${GREEN}${SERVICE_URL}${NC}"
echo ""

# Step 7: 커스텀 도메인 확인
log_info "Checking custom domain mapping..."
DOMAIN_MAPPINGS=$(gcloud run domain-mappings list --region=${REGION} --format='value(metadata.name)' 2>/dev/null || echo "")

if [[ -n "$DOMAIN_MAPPINGS" ]]; then
    log_success "Custom domains:"
    echo "$DOMAIN_MAPPINGS" | while read -r domain; do
        echo -e "  ${GREEN}https://${domain}${NC}"
    done
else
    log_info "No custom domains configured"
    log_info "To add kss.ai.kr, run:"
    echo -e "  ${BLUE}gcloud run domain-mappings create --service=${SERVICE_NAME} --domain=kss.ai.kr --region=${REGION}${NC}"
fi

echo ""
log_success "Deployment script completed successfully! 🚀"
