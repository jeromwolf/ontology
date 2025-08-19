# 주식 시세 API 설정 가이드

## 🎯 추천 순서 (무료 → 유료)

### 1단계: Yahoo Finance (무료, 즉시 사용)
- **장점**: API 키 불필요, 즉시 사용 가능
- **단점**: 한국 주식은 20분 지연
- **용도**: 개발/테스트, 일봉 차트

```bash
# .env.local
# Yahoo Finance는 API 키 불필요
```

### 2단계: 한국투자증권 OpenAPI (무료, 신청 필요)
- **장점**: 실시간 시세, 공식 API
- **단점**: 신청 절차 필요 (1-2일)
- **신청**: https://apiportal.koreainvestment.com

```bash
# .env.local
KI_APP_KEY=your_app_key
KI_APP_SECRET=your_app_secret
KI_ACCOUNT_NO=your_account_number
```

### 3단계: Alpha Vantage (무료 제한)
- **장점**: 기술적 지표 제공
- **제한**: 분당 5회, 일 500회
- **신청**: https://www.alphavantage.co/support/#api-key

```bash
# .env.local
ALPHA_VANTAGE_API_KEY=your_api_key
```

## 📝 한국투자증권 API 신청 방법

1. **회원가입**
   - https://www.koreainvestment.com 회원가입
   - 계좌 개설 (위탁계좌 필요)

2. **API 신청**
   - https://apiportal.koreainvestment.com 접속
   - "API 신청" → "REST API"
   - 용도: "개인 투자 정보 조회"

3. **앱 생성**
   - 마이페이지 → 앱 관리
   - 신규 앱 생성
   - App Key, App Secret 발급

4. **모의투자 신청** (선택)
   - 실제 계좌 없이 테스트 가능
   - 모의투자 → 모의계좌 개설

## 🔧 API 통합 예시

```typescript
// /src/app/api/stock/realtime-price/route.ts
import YahooFinanceAPI from '@/lib/stock-api/yahoo-finance';
import KoreaInvestmentAPI from '@/lib/stock-api/korea-investment';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const stockCode = searchParams.get('code');
  
  // 1. 먼저 Yahoo Finance 시도 (즉시 사용 가능)
  const yahooApi = new YahooFinanceAPI();
  const yahooData = await yahooApi.getQuote(stockCode!, 'KOSPI');
  
  if (yahooData) {
    return NextResponse.json({
      source: 'yahoo',
      delayed: true,
      data: yahooData
    });
  }
  
  // 2. 한국투자증권 API (실시간)
  if (process.env.KI_APP_KEY) {
    const kiApi = new KoreaInvestmentAPI({
      appKey: process.env.KI_APP_KEY,
      appSecret: process.env.KI_APP_SECRET!,
      accountNo: process.env.KI_ACCOUNT_NO!,
      isPaper: true
    });
    
    const kiData = await kiApi.getCurrentPrice(stockCode!);
    return NextResponse.json({
      source: 'korea-investment',
      delayed: false,
      data: kiData
    });
  }
  
  // 3. 폴백: DB 데이터
  return NextResponse.json({
    source: 'database',
    delayed: false,
    data: await getFromDatabase(stockCode!)
  });
}
```

## 🚀 빠른 시작 (Yahoo Finance)

1. API 파일은 이미 생성됨
2. 바로 사용 가능:

```typescript
// 페이지에서 사용
const response = await fetch('/api/stock/realtime-price?code=005930');
const data = await response.json();
```

## ⚠️ 주의사항

1. **CORS 이슈**
   - 클라이언트에서 직접 호출 불가
   - Next.js API Routes 사용 필수

2. **Rate Limit**
   - Yahoo: 제한 없음 (과도한 사용 자제)
   - 한국투자: 초당 20회, 분당 1000회
   - Alpha Vantage: 분당 5회

3. **상업적 사용**
   - Yahoo Finance: 상업적 사용 시 라이선스 필요
   - 한국투자증권: 개인 용도만 가능
   - 상업 서비스는 유료 API 필요

## 📞 고급 옵션 (유료)

1. **Xing API** (이베스트투자증권)
   - 월 33,000원~
   - 실시간 시세, 차트, 뉴스

2. **KRX 정보데이터시스템**
   - 기업 문의
   - 가장 정확한 공식 데이터

3. **Bloomberg Terminal**
   - 월 $2,000+
   - 전문 트레이더용