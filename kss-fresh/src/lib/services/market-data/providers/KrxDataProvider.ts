// KRX (Korea Exchange) Data Provider
// Real-time and historical data from Korean stock market

import { MarketData, OHLCV, OrderBook, PriceData } from '../types';

export interface KrxApiConfig {
  apiKey: string;
  secretKey: string;
  baseUrl: string;
  timeout: number;
}

export interface KrxRealTimeData {
  isuSrtCd: string;      // 종목 단축 코드
  trdPrc: number;        // 현재가
  cmpprevddPrc: number;  // 전일 대비
  opnprc: number;        // 시가
  hgprc: number;         // 고가
  lwprc: number;         // 저가
  accTrdvol: number;     // 누적 거래량
  accTrdval: number;     // 누적 거래 대금
  trdTm: string;         // 체결 시간
}

export interface KrxCompanyInfo {
  isuSrtCd: string;      // 종목 코드
  isuAbbrv: string;      // 종목명
  mktTpNm: string;       // 시장 구분
  sectTpNm: string;      // 업종명
  mktcap: number;        // 시가총액
  sttlAmt: number;       // 상장 주식수
  parval: number;        // 액면가
  eps: number;           // 주당순이익
  per: number;           // 주가수익비율
  pbr: number;           // 주가순자산비율
  bps: number;           // 주당순자산
}

export class KrxDataProvider {
  private config: KrxApiConfig;
  private accessToken: string | null = null;
  private tokenExpiresAt: Date | null = null;
  
  constructor(config: KrxApiConfig) {
    this.config = config;
  }

  // Authenticate with KRX API
  async authenticate(): Promise<string> {
    if (this.accessToken && this.tokenExpiresAt && new Date() < this.tokenExpiresAt) {
      return this.accessToken;
    }

    try {
      const response = await fetch(`${this.config.baseUrl}/auth/token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': this.config.apiKey
        },
        body: JSON.stringify({
          grant_type: 'client_credentials',
          client_id: this.config.apiKey,
          client_secret: this.config.secretKey,
          scope: 'market_data'
        })
      });

      if (!response.ok) {
        throw new Error(`KRX authentication failed: ${response.statusText}`);
      }

      const data = await response.json();
      this.accessToken = data.access_token;
      this.tokenExpiresAt = new Date(Date.now() + (data.expires_in * 1000));
      
      return this.accessToken;
    } catch (error) {
      console.error('KRX authentication error:', error);
      throw error;
    }
  }

  // Get real-time market data
  async getRealTimeData(ticker: string): Promise<MarketData> {
    const token = await this.authenticate();
    
    try {
      const response = await fetch(
        `${this.config.baseUrl}/market/real-time/${ticker}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          signal: AbortSignal.timeout(this.config.timeout)
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch KRX data: ${response.statusText}`);
      }

      const krxData: KrxRealTimeData = await response.json();
      return this.transformKrxToMarketData(krxData);
    } catch (error) {
      console.error(`Error fetching KRX data for ${ticker}:`, error);
      throw error;
    }
  }

  // Get historical OHLCV data
  async getHistoricalData(
    ticker: string,
    startDate: Date,
    endDate: Date,
    interval: '1D' | '1W' | '1M' = '1D'
  ): Promise<OHLCV[]> {
    const token = await this.authenticate();
    
    try {
      const params = new URLSearchParams({
        isuSrtCd: ticker,
        strtDd: this.formatDate(startDate),
        endDd: this.formatDate(endDate),
        interval: interval
      });

      const response = await fetch(
        `${this.config.baseUrl}/market/historical?${params}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          signal: AbortSignal.timeout(this.config.timeout)
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch KRX historical data: ${response.statusText}`);
      }

      const data = await response.json();
      return data.map((item: any) => this.transformKrxToOHLCV(item));
    } catch (error) {
      console.error(`Error fetching KRX historical data for ${ticker}:`, error);
      throw error;
    }
  }

  // Get order book data
  async getOrderBook(ticker: string): Promise<OrderBook> {
    const token = await this.authenticate();
    
    try {
      const response = await fetch(
        `${this.config.baseUrl}/market/orderbook/${ticker}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          signal: AbortSignal.timeout(this.config.timeout)
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch KRX order book: ${response.statusText}`);
      }

      const data = await response.json();
      return this.transformKrxToOrderBook(data, ticker);
    } catch (error) {
      console.error(`Error fetching KRX order book for ${ticker}:`, error);
      throw error;
    }
  }

  // Get company information
  async getCompanyInfo(ticker: string): Promise<KrxCompanyInfo> {
    const token = await this.authenticate();
    
    try {
      const response = await fetch(
        `${this.config.baseUrl}/company/info/${ticker}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          signal: AbortSignal.timeout(this.config.timeout)
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch KRX company info: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`Error fetching KRX company info for ${ticker}:`, error);
      throw error;
    }
  }

  // Get top gainers/losers
  async getTopMovers(type: 'gainers' | 'losers' | 'volume', limit = 20): Promise<MarketData[]> {
    const token = await this.authenticate();
    
    try {
      const response = await fetch(
        `${this.config.baseUrl}/market/movers/${type}?limit=${limit}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          signal: AbortSignal.timeout(this.config.timeout)
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch KRX top movers: ${response.statusText}`);
      }

      const data = await response.json();
      return data.map((item: KrxRealTimeData) => this.transformKrxToMarketData(item));
    } catch (error) {
      console.error(`Error fetching KRX top movers:`, error);
      throw error;
    }
  }

  // Transform KRX data to standard MarketData format
  private transformKrxToMarketData(krxData: KrxRealTimeData): MarketData {
    return {
      ticker: krxData.isuSrtCd,
      exchange: 'KRX',
      price: krxData.trdPrc,
      currency: 'KRW',
      timestamp: new Date(),
      volume: krxData.accTrdvol,
      change: krxData.cmpprevddPrc,
      changePercent: (krxData.cmpprevddPrc / (krxData.trdPrc - krxData.cmpprevddPrc)) * 100,
      high: krxData.hgprc,
      low: krxData.lwprc,
      open: krxData.opnprc,
      previousClose: krxData.trdPrc - krxData.cmpprevddPrc
    };
  }

  // Transform KRX data to OHLCV format
  private transformKrxToOHLCV(krxData: any): OHLCV {
    return {
      timestamp: new Date(krxData.trdDd),
      open: krxData.opnprc,
      high: krxData.hgprc,
      low: krxData.lwprc,
      close: krxData.clsprc,
      volume: krxData.accTrdvol
    };
  }

  // Transform KRX data to OrderBook format
  private transformKrxToOrderBook(krxData: any, ticker: string): OrderBook {
    const bids = krxData.bidList?.map((bid: any) => ({
      price: bid.bidPrc,
      volume: bid.bidQty,
      orders: bid.bidCnt || 1
    })) || [];

    const asks = krxData.askList?.map((ask: any) => ({
      price: ask.askPrc,
      volume: ask.askQty,
      orders: ask.askCnt || 1
    })) || [];

    const bestBid = bids[0]?.price || 0;
    const bestAsk = asks[0]?.price || 0;

    return {
      ticker,
      timestamp: new Date(),
      bids,
      asks,
      spread: bestAsk - bestBid,
      midPrice: (bestBid + bestAsk) / 2
    };
  }

  // Format date for KRX API
  private formatDate(date: Date): string {
    return date.toISOString().slice(0, 10).replace(/-/g, '');
  }

  // Check if market is open
  isMarketOpen(): boolean {
    const now = new Date();
    const koreaTime = new Date(now.toLocaleString("en-US", {timeZone: "Asia/Seoul"}));
    const hour = koreaTime.getHours();
    const minute = koreaTime.getMinutes();
    const day = koreaTime.getDay();
    
    // Monday to Friday, 9:00 AM to 3:30 PM KST
    return day >= 1 && day <= 5 && 
           ((hour === 9 && minute >= 0) || (hour > 9 && hour < 15) || (hour === 15 && minute <= 30));
  }

  // Get supported tickers
  async getSupportedTickers(): Promise<string[]> {
    const token = await this.authenticate();
    
    try {
      const response = await fetch(
        `${this.config.baseUrl}/market/tickers`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch KRX tickers: ${response.statusText}`);
      }

      const data = await response.json();
      return data.map((item: any) => item.isuSrtCd);
    } catch (error) {
      console.error('Error fetching KRX tickers:', error);
      throw error;
    }
  }
}

// Export default instance
export const krxProvider = new KrxDataProvider({
  apiKey: process.env.NEXT_PUBLIC_KRX_API_KEY || '',
  secretKey: process.env.KRX_SECRET_KEY || '',
  baseUrl: process.env.NEXT_PUBLIC_KRX_API_URL || 'https://api.krx.co.kr/v1',
  timeout: 10000
});