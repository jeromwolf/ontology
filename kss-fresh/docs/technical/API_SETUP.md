# 📊 주식 뉴스 분석 API 설정 가이드

## 필요한 API 목록 및 가입 방법

### 1. NewsAPI.org (필수)
- **가입**: https://newsapi.org/register
- **무료 플랜**: 1000 요청/일
- **용도**: 뉴스 수집
```bash
NEWS_API_KEY=your_key_here
```

### 2. Alpha Vantage (추천)
- **가입**: https://www.alphavantage.co/support/#api-key
- **무료 플랜**: 500 요청/일
- **용도**: 감성 분석, 주가 데이터
```bash
ALPHA_VANTAGE_KEY=your_key_here
```

### 3. OpenAI API (추천)
- **가입**: https://platform.openai.com/signup
- **용도**: 뉴스 영향도 분석, 온톨로지 생성
- **비용**: 사용량 기반 과금
```bash
OPENAI_API_KEY=your_key_here
```

### 4. 한국 증권 API (선택)

#### 한국투자증권 OpenAPI
- **가입**: https://apiportal.koreainvestment.com/
- **계좌 필요**: 모의투자 계좌 무료
- **용도**: 한국 주식 실시간 시세
```bash
KIS_APP_KEY=your_app_key
KIS_APP_SECRET=your_app_secret
```

#### 네이버 증권 (비공식)
- **API 키 불필요**
- **제한**: Rate limit 있음
- **용도**: 간단한 시세 조회

### 5. 추가 옵션

#### Polygon.io
- **가입**: https://polygon.io/
- **무료 플랜**: 제한적
- **용도**: 미국 주식 실시간 데이터
```bash
POLYGON_API_KEY=your_key
```

#### IEX Cloud
- **가입**: https://iexcloud.io/
- **무료 플랜**: 50,000 크레딧/월
- **용도**: 재무제표, 기업 정보
```bash
IEX_CLOUD_KEY=your_key
```

## 환경 변수 설정

1. `.env.local` 파일 생성:
```bash
cp .env.local.example .env.local
```

2. API 키 입력:
```env
# 필수
NEWS_API_KEY=your_newsapi_key
OPENAI_API_KEY=your_openai_key

# 추천
ALPHA_VANTAGE_KEY=your_alphavantage_key

# 선택
KIS_APP_KEY=your_korea_investment_key
KIS_APP_SECRET=your_korea_investment_secret

# 데이터베이스 (PostgreSQL 또는 SQLite)
DATABASE_URL="postgresql://user:password@localhost:5432/kss_news"
# 또는 개발용 SQLite
# DATABASE_URL="file:./dev.db"

# 크론잡 보안 (Vercel 배포 시)
CRON_SECRET=your_cron_secret_key
ADMIN_SECRET=your_admin_secret_key
```

## API 사용 예제

### 뉴스 분석 요청
```javascript
const response = await fetch('/api/news-analysis', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    company: '삼성전자',
    ticker: '005930'
  })
})

const data = await response.json()
// {
//   company: '삼성전자',
//   newsCount: 42,
//   sentiment: 0.65,
//   ontologyAnalysis: {
//     impact: { direct: 45, indirect: 20, sector: 35 },
//     relatedCompanies: ['SK하이닉스', 'LG전자'],
//     keywords: ['반도체', 'AI', '실적']
//   },
//   recommendation: '매수'
// }
```

## 주의사항

1. **API 키 보안**: 절대 GitHub에 커밋하지 마세요
2. **Rate Limit**: 각 API의 요청 제한을 확인하세요
3. **비용 관리**: OpenAI 등 유료 API는 사용량을 모니터링하세요
4. **캐싱**: 동일한 요청은 캐싱하여 API 호출을 줄이세요

## 무료 대안

API 비용이 부담되시면:
1. **뉴스**: RSS 피드 직접 파싱
2. **감성분석**: Hugging Face 무료 모델 사용
3. **주가**: Yahoo Finance 스크래핑 (제한적)
4. **온톨로지**: 자체 구축한 지식 그래프 사용