# 한국투자증권 API 설정 가이드

## 1. API 키 발급

### 1.1 한국투자증권 개발자 포털 가입
1. [한국투자증권 OpenAPI 포털](https://apiportal.koreainvestment.com/) 접속
2. 회원가입 진행
3. 본인인증 완료

### 1.2 앱 생성 및 API 키 발급
1. 마이페이지 → 앱 관리
2. "앱 추가하기" 클릭
3. 앱 정보 입력:
   - 앱 이름: KSS Stock Analysis
   - 앱 설명: KSS 플랫폼 주식 분석 모듈
   - 서비스 URL: http://localhost:3000 (개발용)
4. 생성 후 APP KEY와 APP SECRET 확인

### 1.3 계좌 연결
1. 모의투자 계좌 신청 (실제 거래 없이 시세만 조회할 경우 필수 아님)
2. 계좌번호 확인

## 2. 환경변수 설정

### 2.1 .env.local 파일 생성
```bash
# .env.local 파일에 추가
KIS_APP_KEY="발급받은_APP_KEY"
KIS_APP_SECRET="발급받은_APP_SECRET"
KIS_BASE_URL="https://openapi.koreainvestment.com:9443"
KIS_ACCOUNT_NO="계좌번호" # 선택사항
```

### 2.2 환경변수 확인
```bash
# 환경변수가 제대로 설정되었는지 확인
echo $KIS_APP_KEY
```

## 3. API 사용 제한 및 주의사항

### 3.1 API 호출 제한
- **초당 호출 제한**: 20회/초
- **일일 호출 제한**: 100,000회/일
- **동시 접속 제한**: 5개 세션

### 3.2 주의사항
- 개발/테스트 시에는 모의투자 서버 사용 권장
- API 키는 절대 GitHub 등에 노출하지 않도록 주의
- 프로덕션 환경에서는 별도의 API 키 사용

## 4. 구현된 기능

### 4.1 실시간 주가 조회
```typescript
// 현재가 조회
const stockData = await stockDataService.getStockData('005930')
console.log(stockData.currentPrice) // 현재 주가
```

### 4.2 차트 데이터 조회
```typescript
// 일봉 데이터 조회 (30일)
const chartData = await stockDataService.getChartData('005930', 30)
```

### 4.3 재무 정보 조회
```typescript
// 재무 데이터 조회
const financialData = await stockDataService.getFinancialData('005930')
console.log(financialData.eps) // 주당순이익
```

## 5. 테스트 방법

### 5.1 개발 서버 실행
```bash
npm run dev
```

### 5.2 주식 분석 모듈 접속
1. http://localhost:3000/modules/stock-analysis 접속
2. "재무 계산기" 시뮬레이터 클릭
3. "실시간 데이터" 체크박스 활성화
4. 종목 선택 후 데이터 확인

### 5.3 API 연결 확인
- 실시간 데이터가 로드되면 성공
- 에러 메시지가 표시되면 API 키 확인 필요

## 6. 트러블슈팅

### 6.1 API 키 관련 오류
```
Error: Token request failed: Unauthorized
```
**해결방법**: API 키와 SECRET이 올바른지 확인

### 6.2 CORS 오류
```
Access to fetch at 'https://openapi.koreainvestment.com:9443' from origin 'http://localhost:3000' has been blocked by CORS policy
```
**해결방법**: 서버 사이드에서 API 호출하도록 구현 (현재 구현된 방식)

### 6.3 호출 제한 초과
```
Error: Rate limit exceeded
```
**해결방법**: 
- API 호출 간격 조절 (50ms 이상)
- 캐싱 활용 (현재 5분 캐시 구현됨)

## 7. 향후 개선사항

1. **WebSocket 연결**: 실시간 호가 및 체결 정보
2. **주문 API 연동**: 모의투자 주문 기능
3. **업종별 분석**: 섹터별 시세 정보
4. **뉴스 연동**: 종목별 뉴스와 공시 정보
5. **알림 기능**: 가격 변동 알림

## 8. 참고 자료

- [한국투자증권 OpenAPI 문서](https://apiportal.koreainvestment.com/apiservice/oauth2)
- [API 레퍼런스](https://apiportal.koreainvestment.com/apiservice/reference)
- [샘플 코드](https://github.com/koreainvestment/open-api-sample)