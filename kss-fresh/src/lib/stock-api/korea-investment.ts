// 한국투자증권 OpenAPI 연동
// https://apiportal.koreainvestment.com

interface KoreaInvestmentConfig {
  appKey: string;
  appSecret: string;
  accountNo?: string;
  isPaper?: boolean; // 모의투자 여부
}

class KoreaInvestmentAPI {
  private baseUrl: string;
  private config: KoreaInvestmentConfig;
  private accessToken: string | null = null;
  private tokenExpiry: number | null = null;
  private static tokenCache: { token: string; expiry: number } | null = null;

  constructor(config: KoreaInvestmentConfig) {
    this.config = config;
    this.baseUrl = config.isPaper 
      ? 'https://openapivts.koreainvestment.com:29443' // 모의투자
      : 'https://openapi.koreainvestment.com:9443';    // 실전투자
  }

  // 접근 토큰 발급 (24시간 캐싱)
  async getAccessToken(): Promise<string> {
    try {
      // 캐시된 토큰 확인
      const now = Date.now();
      
      // 메모리 캐시 확인
      if (KoreaInvestmentAPI.tokenCache && KoreaInvestmentAPI.tokenCache.expiry > now) {
        const remainingHours = Math.floor((KoreaInvestmentAPI.tokenCache.expiry - now) / (1000 * 60 * 60));
        console.log(`Using cached token from memory (${remainingHours}h remaining)`);
        this.accessToken = KoreaInvestmentAPI.tokenCache.token;
        this.tokenExpiry = KoreaInvestmentAPI.tokenCache.expiry;
        return this.accessToken;
      }

      // 파일 시스템에서 토큰 읽기 시도
      try {
        const fs = require('fs').promises;
        const path = require('path');
        const tokenPath = path.join(process.cwd(), '.ki-token-cache.json');
        
        const tokenData = await fs.readFile(tokenPath, 'utf-8');
        const cached = JSON.parse(tokenData);
        
        if (cached.expiry > now && cached.appKey === this.config.appKey) {
          const remainingHours = Math.floor((cached.expiry - now) / (1000 * 60 * 60));
          console.log(`Using cached token from file (${remainingHours}h remaining)`);
          this.accessToken = cached.token;
          this.tokenExpiry = cached.expiry;
          KoreaInvestmentAPI.tokenCache = { token: cached.token, expiry: cached.expiry };
          return this.accessToken;
        }
      } catch (err) {
        // 파일이 없거나 읽기 실패 - 정상적인 상황
      }

      // 새 토큰 발급
      console.log('Requesting new access token from Korea Investment API');
      const response = await fetch(`${this.baseUrl}/oauth2/tokenP`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          grant_type: 'client_credentials',
          appkey: this.config.appKey,
          appsecret: this.config.appSecret,
        }),
      });

      if (!response.ok) {
        throw new Error(`Token request failed: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.access_token) {
        this.accessToken = data.access_token;
        // 토큰 만료 시간 설정 (24시간 - 1시간 여유)
        this.tokenExpiry = now + (23 * 60 * 60 * 1000);
        
        // 메모리 캐시 저장
        KoreaInvestmentAPI.tokenCache = { token: this.accessToken, expiry: this.tokenExpiry };
        
        // 파일에 저장 (서버 재시작 시에도 유지)
        try {
          const fs = require('fs').promises;
          const path = require('path');
          const tokenPath = path.join(process.cwd(), '.ki-token-cache.json');
          
          await fs.writeFile(tokenPath, JSON.stringify({
            token: this.accessToken,
            expiry: this.tokenExpiry,
            appKey: this.config.appKey,
            createdAt: new Date().toISOString()
          }), 'utf-8');
          
          console.log('Token cached to file');
        } catch (err) {
          console.error('Failed to cache token to file:', err);
        }
        
        return this.accessToken;
      } else {
        throw new Error('No access token in response');
      }
    } catch (error) {
      console.error('Failed to get access token:', error);
      throw error;
    }
  }

  // 현재가 조회
  async getCurrentPrice(stockCode: string): Promise<any> {
    if (!this.accessToken) {
      await this.getAccessToken();
    }

    const response = await fetch(
      `${this.baseUrl}/uapi/domestic-stock/v1/quotations/inquire-price?` +
      `FID_COND_MRKT_DIV_CODE=J&FID_INPUT_ISCD=${stockCode}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json; charset=utf-8',
          'authorization': `Bearer ${this.accessToken}`,
          'appkey': this.config.appKey,
          'appsecret': this.config.appSecret,
          'tr_id': 'FHKST01010100', // 주식현재가 시세
        },
      }
    );

    if (!response.ok) {
      console.error('Korea Investment API error:', response.status);
      return null;
    }

    const data = await response.json();
    
    // API 응답 형식에 맞게 변환
    if (data.rt_cd === '0' && data.output) {
      const output = data.output;
      return {
        symbol: stockCode,
        regularMarketPrice: parseFloat(output.stck_prpr), // 현재가
        regularMarketChange: parseFloat(output.prdy_vrss), // 전일대비
        regularMarketChangePercent: parseFloat(output.prdy_ctrt), // 전일대비율
        regularMarketVolume: parseInt(output.acml_vol), // 누적거래량
        marketCap: parseFloat(output.stck_prpr) * parseFloat(output.lstn_stcn), // 시가총액
        regularMarketTime: new Date().toISOString()
      };
    }
    
    return null;
  }

  // 일봉 데이터 조회
  async getDailyPrice(stockCode: string, startDate: string, endDate: string): Promise<any> {
    if (!this.accessToken) {
      await this.getAccessToken();
    }

    const response = await fetch(
      `${this.baseUrl}/uapi/domestic-stock/v1/quotations/inquire-daily-price?` +
      `FID_COND_MRKT_DIV_CODE=J&FID_INPUT_ISCD=${stockCode}&` +
      `FID_PERIOD_DIV_CODE=D&FID_ORG_ADJ_PRC=1&` +
      `FID_INPUT_DATE_1=${startDate}&FID_INPUT_DATE_2=${endDate}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'authorization': `Bearer ${this.accessToken}`,
          'appkey': this.config.appKey,
          'appsecret': this.config.appSecret,
          'tr_id': 'FHKST01010400', // 국내주식 일별 시세
        },
      }
    );

    return await response.json();
  }

  // 업종 지수 조회
  async getSectorIndex(): Promise<any> {
    if (!this.accessToken) {
      await this.getAccessToken();
    }

    const response = await fetch(
      `${this.baseUrl}/uapi/domestic-stock/v1/quotations/inquire-index-price?` +
      `FID_COND_MRKT_DIV_CODE=U&FID_INPUT_ISCD=0000`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'authorization': `Bearer ${this.accessToken}`,
          'appkey': this.config.appKey,
          'appsecret': this.config.appSecret,
          'tr_id': 'FHPUP02100000', // 업종지수 시세
        },
      }
    );

    return await response.json();
  }
}

export default KoreaInvestmentAPI;

// 사용 예시
/*
const api = new KoreaInvestmentAPI({
  appKey: process.env.KI_APP_KEY!,
  appSecret: process.env.KI_APP_SECRET!,
  accountNo: process.env.KI_ACCOUNT_NO!,
  isPaper: true, // 모의투자로 시작
});

// 삼성전자 현재가 조회
const price = await api.getCurrentPrice('005930');
console.log(price);
*/