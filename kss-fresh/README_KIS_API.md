# KIS API 토큰 관리 시스템

한국투자증권(KIS) OpenAPI를 사용하여 실제 주식 데이터를 가져오는 시스템입니다. 하루에 한 번만 토큰을 요청하여 저장하고 관리하는 효율적인 구조로 설계되었습니다.

## 🎯 주요 특징

### 1. 자동 토큰 관리
- **하루 1회 제한**: 토큰 생성 후 24시간 동안 재사용
- **자동 갱신**: 토큰 만료 5분 전 자동 갱신
- **안전한 저장**: localStorage를 활용한 클라이언트 사이드 저장
- **유효성 검증**: 토큰 상태 실시간 모니터링

### 2. 실시간 데이터 지원
- **주식 시세**: 현재가, 등락률, 거래량 등
- **차트 데이터**: 일봉, 주봉, 월봉 캔들 데이터
- **다중 종목**: 여러 종목 동시 조회 (Rate Limiting 적용)
- **인기 종목**: 삼성전자, SK하이닉스 등 주요 종목 프리셋

### 3. 개발자 친화적 UI
- **토큰 상태 모니터링**: 실시간 상태 확인
- **원클릭 갱신**: 토큰 수동 갱신 기능
- **연결 테스트**: API 연결 상태 확인
- **설정 가이드**: 환경변수 설정 도움말

## 📁 파일 구조

```
src/
├── lib/
│   ├── auth/
│   │   └── kis-token-manager.ts     # 토큰 관리 핵심 로직
│   └── services/
│       └── kis-api-service.ts       # API 호출 서비스
├── components/charts/ProChart/
│   └── KISTokenStatus.tsx           # 토큰 상태 UI 컴포넌트
└── app/modules/stock-analysis/tools/
    ├── kis-manager/page.tsx         # 관리자 페이지
    └── pro-trading-chart/           # 실제 차트 구현
```

## 🚀 설정 방법

### 1. KIS API 키 발급
1. [KIS OpenAPI 포털](https://apiportal.koreainvestment.com) 방문
2. 회원가입 및 앱 등록
3. App Key와 App Secret 발급

### 2. 환경변수 설정
`.env.local` 파일에 다음 내용 추가:

```bash
# KIS API 설정
NEXT_PUBLIC_KIS_APP_KEY=your_app_key_here
NEXT_PUBLIC_KIS_APP_SECRET=your_app_secret_here
```

### 3. 서버 재시작
```bash
npm run dev
```

## 💻 사용 방법

### 기본 사용법
```typescript
import { kisApiService } from '@/lib/services/kis-api-service';

// 단일 종목 조회
const quote = await kisApiService.getStockQuote('005930'); // 삼성전자

// 차트 데이터 조회
const history = await kisApiService.getStockHistory('005930', 'D', 100);

// 여러 종목 조회
const quotes = await kisApiService.getMultipleQuotes(['005930', '000660']);

// 인기 종목 조회
const popular = await kisApiService.getPopularStocks();
```

### 토큰 관리
```typescript
import { kisTokenManager } from '@/lib/auth/kis-token-manager';

// 유효한 토큰 자동 획득
const token = await kisTokenManager.getValidToken();

// 토큰 상태 확인
const status = kisTokenManager.getTokenStatus();

// 강제 갱신
await kisTokenManager.forceRefreshToken();

// 토큰 삭제
kisTokenManager.clearToken();
```

## 📊 UI 컴포넌트

### KISTokenStatus 컴포넌트
```jsx
import KISTokenStatus from '@/components/charts/ProChart/KISTokenStatus';

function MyComponent() {
  return (
    <div>
      <KISTokenStatus />
    </div>
  );
}
```

**제공 기능:**
- 토큰 유효성 실시간 표시
- 만료 시간 카운트다운
- 원클릭 토큰 갱신
- API 연결 상태 확인
- 토큰 삭제 기능

## 🔧 관리자 도구

### 접근 경로
```
http://localhost:3000/modules/stock-analysis/tools/kis-manager
```

**제공 기능:**
1. **토큰 상태**: 실시간 모니터링 대시보드
2. **설정**: 환경변수 확인 및 가이드
3. **테스트**: API 연결 테스트 도구 (개발 중)
4. **문서**: 공식 문서 링크 및 내부 구현 가이드

## ⚡ 성능 최적화

### Rate Limiting
- **동시 요청 제한**: 5개 종목씩 배치 처리
- **요청 간격**: 200ms 지연으로 API 제한 준수
- **에러 핸들링**: 실패한 요청 자동 스킵

### 메모리 관리
- **토큰 캐싱**: localStorage 활용
- **자동 정리**: 만료된 토큰 자동 삭제
- **싱글톤 패턴**: 메모리 효율적인 인스턴스 관리

## 🛡️ 보안 고려사항

### 환경변수 보안
- `.env.local` 파일은 Git에 포함되지 않음
- 프로덕션 환경에서는 안전한 키 관리 시스템 사용 권장
- API 키 노출 방지를 위한 서버사이드 구현 고려

### 토큰 보안
- 클라이언트 사이드 저장 (localStorage)
- HTTPS 필수 (프로덕션 환경)
- 토큰 유효기간 준수 (24시간)

## 🔗 관련 링크

- [KIS OpenAPI 포털](https://apiportal.koreainvestment.com)
- [국내주식 API 문서](https://apiportal.koreainvestment.com/apiservice/apiservice-domestic-stock)
- [OAuth2 인증 가이드](https://apiportal.koreainvestment.com/apiservice/oauth2)

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. KIS API 사용은 한국투자증권의 이용약관을 따릅니다.