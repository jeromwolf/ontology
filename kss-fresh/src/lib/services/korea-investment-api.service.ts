/**
 * 한국투자증권 OpenAPI 서비스
 * @see https://apiportal.koreainvestment.com/
 */

interface KISConfig {
  appKey: string;
  appSecret: string;
  baseUrl: string;
  accountNo: string;
}

interface KISTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

interface KISStockPrice {
  stck_prpr: string;          // 주식 현재가
  prdy_vrss: string;          // 전일 대비
  prdy_ctrt: string;          // 전일 대비율
  stck_oprc: string;          // 시가
  stck_hgpr: string;          // 고가
  stck_lwpr: string;          // 저가
  acml_vol: string;           // 누적 거래량
  acml_tr_pbmn: string;       // 누적 거래대금
}

interface KISOrderbookItem {
  askp: string;               // 매도호가
  askp_rsqn: string;          // 매도호가 잔량
  bidp: string;               // 매수호가
  bidp_rsqn: string;          // 매수호가 잔량
}

export class KoreaInvestmentAPI {
  private config: KISConfig;
  private accessToken: string | null = null;
  private tokenExpiry: Date | null = null;

  constructor() {
    this.config = {
      appKey: process.env.KIS_APP_KEY || '',
      appSecret: process.env.KIS_APP_SECRET || '',
      baseUrl: process.env.KIS_BASE_URL || 'https://openapi.koreainvestment.com:9443',
      accountNo: process.env.KIS_ACCOUNT_NO || ''
    };
  }

  /**
   * 액세스 토큰 발급
   */
  private async getAccessToken(): Promise<string> {
    // 토큰이 있고 유효하면 재사용
    if (this.accessToken && this.tokenExpiry && this.tokenExpiry > new Date()) {
      return this.accessToken;
    }

    const url = `${this.config.baseUrl}/oauth2/tokenP`;
    const body = {
      grant_type: 'client_credentials',
      appkey: this.config.appKey,
      appsecret: this.config.appSecret
    };

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        throw new Error(`Token request failed: ${response.statusText}`);
      }

      const data: KISTokenResponse = await response.json();
      this.accessToken = data.access_token;
      // 토큰 만료 시간 설정 (여유를 두고 5분 전에 갱신)
      this.tokenExpiry = new Date(Date.now() + (data.expires_in - 300) * 1000);
      
      return this.accessToken;
    } catch (error) {
      console.error('Failed to get access token:', error);
      throw error;
    }
  }

  /**
   * 주식 현재가 조회
   */
  async getCurrentPrice(stockCode: string): Promise<KISStockPrice> {
    const token = await this.getAccessToken();
    const url = `${this.config.baseUrl}/uapi/domestic-stock/v1/quotations/inquire-price`;
    
    const headers = {
      'Content-Type': 'application/json',
      'authorization': `Bearer ${token}`,
      'appkey': this.config.appKey,
      'appsecret': this.config.appSecret,
      'tr_id': 'FHKST01010100' // 주식 현재가 조회
    };

    const params = new URLSearchParams({
      fid_cond_mrkt_div_code: 'J', // 주식
      fid_input_iscd: stockCode
    });

    try {
      const response = await fetch(`${url}?${params}`, {
        method: 'GET',
        headers
      });

      if (!response.ok) {
        throw new Error(`Price request failed: ${response.statusText}`);
      }

      const data = await response.json();
      return data.output;
    } catch (error) {
      console.error('Failed to get current price:', error);
      throw error;
    }
  }

  /**
   * 주식 호가 조회
   */
  async getOrderbook(stockCode: string): Promise<KISOrderbookItem[]> {
    const token = await this.getAccessToken();
    const url = `${this.config.baseUrl}/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn`;
    
    const headers = {
      'Content-Type': 'application/json',
      'authorization': `Bearer ${token}`,
      'appkey': this.config.appKey,
      'appsecret': this.config.appSecret,
      'tr_id': 'FHKST01010200' // 주식 호가 조회
    };

    const params = new URLSearchParams({
      fid_cond_mrkt_div_code: 'J',
      fid_input_iscd: stockCode
    });

    try {
      const response = await fetch(`${url}?${params}`, {
        method: 'GET',
        headers
      });

      if (!response.ok) {
        throw new Error(`Orderbook request failed: ${response.statusText}`);
      }

      const data = await response.json();
      // 호가 데이터는 output1에 배열로 제공됨
      return data.output1;
    } catch (error) {
      console.error('Failed to get orderbook:', error);
      throw error;
    }
  }

  /**
   * 주식 일봉 조회 (차트 데이터)
   */
  async getDailyPrices(stockCode: string, startDate: string, endDate: string) {
    const token = await this.getAccessToken();
    const url = `${this.config.baseUrl}/uapi/domestic-stock/v1/quotations/inquire-daily-price`;
    
    const headers = {
      'Content-Type': 'application/json',
      'authorization': `Bearer ${token}`,
      'appkey': this.config.appKey,
      'appsecret': this.config.appSecret,
      'tr_id': 'FHKST01010400' // 주식 일봉 조회
    };

    const params = new URLSearchParams({
      fid_cond_mrkt_div_code: 'J',
      fid_input_iscd: stockCode,
      fid_org_adj_prc: '1', // 수정주가 적용
      fid_period_div_code: 'D' // 일봉
    });

    try {
      const response = await fetch(`${url}?${params}`, {
        method: 'GET',
        headers
      });

      if (!response.ok) {
        throw new Error(`Daily prices request failed: ${response.statusText}`);
      }

      const data = await response.json();
      // 일봉 데이터 가공
      return data.output.map((item: any) => ({
        date: item.stck_bsop_date,
        open: parseFloat(item.stck_oprc),
        high: parseFloat(item.stck_hgpr),
        low: parseFloat(item.stck_lwpr),
        close: parseFloat(item.stck_clpr),
        volume: parseInt(item.acml_vol),
        value: parseFloat(item.acml_tr_pbmn)
      }));
    } catch (error) {
      console.error('Failed to get daily prices:', error);
      throw error;
    }
  }

  /**
   * 주식 재무정보 조회
   */
  async getFinancialInfo(stockCode: string) {
    const token = await this.getAccessToken();
    const url = `${this.config.baseUrl}/uapi/domestic-stock/v1/finance/financial-ratio`;
    
    const headers = {
      'Content-Type': 'application/json',
      'authorization': `Bearer ${token}`,
      'appkey': this.config.appKey,
      'appsecret': this.config.appSecret,
      'tr_id': 'FHKST66430300' // 재무비율 조회
    };

    const params = new URLSearchParams({
      fid_cond_mrkt_div_code: 'J',
      fid_input_iscd: stockCode
    });

    try {
      const response = await fetch(`${url}?${params}`, {
        method: 'GET',
        headers
      });

      if (!response.ok) {
        throw new Error(`Financial info request failed: ${response.statusText}`);
      }

      const data = await response.json();
      return data.output;
    } catch (error) {
      console.error('Failed to get financial info:', error);
      throw error;
    }
  }

  /**
   * 여러 종목의 현재가 일괄 조회
   */
  async getMultipleStockPrices(stockCodes: string[]): Promise<Map<string, KISStockPrice>> {
    const priceMap = new Map<string, KISStockPrice>();
    
    // API 호출 제한을 고려하여 순차적으로 처리
    for (const code of stockCodes) {
      try {
        const price = await this.getCurrentPrice(code);
        priceMap.set(code, price);
        // API 호출 간격 조절 (초당 20회 제한 고려)
        await new Promise(resolve => setTimeout(resolve, 50));
      } catch (error) {
        console.error(`Failed to get price for ${code}:`, error);
      }
    }
    
    return priceMap;
  }

  /**
   * 업종별 시세 조회
   */
  async getSectorPrices(sectorCode: string) {
    const token = await this.getAccessToken();
    const url = `${this.config.baseUrl}/uapi/domestic-stock/v1/quotations/inquire-sector-price`;
    
    const headers = {
      'Content-Type': 'application/json',
      'authorization': `Bearer ${token}`,
      'appkey': this.config.appKey,
      'appsecret': this.config.appSecret,
      'tr_id': 'FHKUP03500100' // 업종별 시세 조회
    };

    const params = new URLSearchParams({
      fid_cond_mrkt_div_code: 'U',
      fid_input_iscd: sectorCode
    });

    try {
      const response = await fetch(`${url}?${params}`, {
        method: 'GET',
        headers
      });

      if (!response.ok) {
        throw new Error(`Sector prices request failed: ${response.statusText}`);
      }

      const data = await response.json();
      return data.output;
    } catch (error) {
      console.error('Failed to get sector prices:', error);
      throw error;
    }
  }
}

// 싱글톤 인스턴스 export
export const koreaInvestmentAPI = new KoreaInvestmentAPI();