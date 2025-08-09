# Services Directory

이 디렉토리는 향후 마이크로서비스 아키텍처로 전환 시 사용될 예정입니다.

## 계획된 서비스

### 1. auth-service
- 사용자 인증/인가
- JWT 토큰 관리
- OAuth 통합

### 2. payment-service
- 결제 처리
- 구독 관리
- 인보이스 생성

### 3. content-service
- 코스 콘텐츠 관리
- 버전 관리
- CDN 통합

### 4. analytics-service
- 학습 분석
- 진도 추적
- 리포트 생성

### 5. notification-service
- 이메일 알림
- 푸시 알림
- 인앱 알림

## MVP 단계에서는
현재는 모든 기능이 `apps/ontology-mvp` 내에 모놀리식으로 구현됩니다.
서비스 분리는 사용자 증가와 기능 복잡도에 따라 점진적으로 진행됩니다.