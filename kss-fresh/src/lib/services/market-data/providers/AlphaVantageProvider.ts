// Alpha Vantage Data Provider
// Real-time and historical data for US markets (NYSE, NASDAQ)

import { MarketData, OHLCV, PriceData } from '../types';

export interface AlphaVantageConfig {
  apiKey: string;
  baseUrl: string;
  timeout: number;
  requestsPerMinute: number;
}

export interface AlphaVantageQuote {
  '01. symbol': string;
  '02. open': string;
  '03. high': string;
  '04. low': string;
  '05. price': string;
  '06. volume': string;
  '07. latest trading day': string;
  '08. previous close': string;
  '09. change': string;
  '10. change percent': string;
}

export interface AlphaVantageTimeSeriesDaily {
  'Meta Data': {
    '1. Information': string;
    '2. Symbol': string;
    '3. Last Refreshed': string;
    '4. Output Size': string;
    '5. Time Zone': string;
  };
  'Time Series (Daily)': {
    [date: string]: {
      '1. open': string;
      '2. high': string;
      '3. low': string;
      '4. close': string;
      '5. volume': string;
    };
  };
}

export interface AlphaVantageCompanyOverview {
  Symbol: string;
  AssetType: string;
  Name: string;
  Description: string;
  CIK: string;
  Exchange: string;
  Currency: string;
  Country: string;
  Sector: string;
  Industry: string;
  Address: string;
  MarketCapitalization: string;
  EBITDA: string;
  PERatio: string;
  PEGRatio: string;
  BookValue: string;
  DividendPerShare: string;
  DividendYield: string;
  EPS: string;
  RevenuePerShareTTM: string;
  ProfitMargin: string;
  OperatingMarginTTM: string;
  ReturnOnAssetsTTM: string;
  ReturnOnEquityTTM: string;
  RevenueTTM: string;
  GrossProfitTTM: string;
  DilutedEPSTTM: string;
  QuarterlyEarningsGrowthYOY: string;
  QuarterlyRevenueGrowthYOY: string;
  AnalystTargetPrice: string;
  TrailingPE: string;
  ForwardPE: string;
  PriceToSalesRatioTTM: string;
  PriceToBookRatio: string;
  EVToRevenue: string;
  EVToEBITDA: string;
  Beta: string;
  '52WeekHigh': string;
  '52WeekLow': string;
  '50DayMovingAverage': string;
  '200DayMovingAverage': string;
  SharesOutstanding: string;
  DividendDate: string;
  ExDividendDate: string;
}

export class AlphaVantageProvider {
  private config: AlphaVantageConfig;
  private requestQueue: Array<() => Promise<any>> = [];
  private lastRequestTime = 0;
  private isProcessingQueue = false;
  
  constructor(config: AlphaVantageConfig) {
    this.config = config;
  }

  // Rate limiting queue to respect API limits
  private async queueRequest<T>(requestFn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.requestQueue.push(async () => {
        try {
          const result = await requestFn();
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
      
      if (!this.isProcessingQueue) {
        this.processQueue();
      }
    });
  }

  private async processQueue(): Promise<void> {
    if (this.isProcessingQueue || this.requestQueue.length === 0) {
      return;
    }

    this.isProcessingQueue = true;

    while (this.requestQueue.length > 0) {
      const now = Date.now();
      const timeSinceLastRequest = now - this.lastRequestTime;
      const minInterval = (60 / this.config.requestsPerMinute) * 1000;

      if (timeSinceLastRequest < minInterval) {
        await new Promise(resolve => setTimeout(resolve, minInterval - timeSinceLastRequest));
      }

      const request = this.requestQueue.shift();
      if (request) {
        await request();
        this.lastRequestTime = Date.now();
      }
    }

    this.isProcessingQueue = false;
  }

  // Get real-time quote
  async getRealTimeData(ticker: string): Promise<MarketData> {
    return this.queueRequest(async () => {
      try {
        const url = `${this.config.baseUrl}/query?function=GLOBAL_QUOTE&symbol=${ticker}&apikey=${this.config.apiKey}`;
        const response = await fetch(url, {
          signal: AbortSignal.timeout(this.config.timeout)
        });

        if (!response.ok) {
          throw new Error(`Alpha Vantage API error: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data['Error Message']) {
          throw new Error(`Alpha Vantage error: ${data['Error Message']}`);
        }

        if (data['Note']) {
          throw new Error(`Alpha Vantage rate limit: ${data['Note']}`);
        }

        const quote = data['Global Quote'] as AlphaVantageQuote;
        return this.transformQuoteToMarketData(quote);
      } catch (error) {
        console.error(`Error fetching Alpha Vantage data for ${ticker}:`, error);
        throw error;
      }
    });
  }

  // Get historical daily data
  async getHistoricalData(
    ticker: string,
    outputSize: 'compact' | 'full' = 'compact'
  ): Promise<OHLCV[]> {
    return this.queueRequest(async () => {
      try {
        const url = `${this.config.baseUrl}/query?function=TIME_SERIES_DAILY&symbol=${ticker}&outputsize=${outputSize}&apikey=${this.config.apiKey}`;
        const response = await fetch(url, {
          signal: AbortSignal.timeout(this.config.timeout)
        });

        if (!response.ok) {
          throw new Error(`Alpha Vantage API error: ${response.statusText}`);
        }

        const data: AlphaVantageTimeSeriesDaily = await response.json();
        
        if (data['Error Message']) {
          throw new Error(`Alpha Vantage error: ${data['Error Message']}`);
        }

        if (data['Note']) {
          throw new Error(`Alpha Vantage rate limit: ${data['Note']}`);
        }

        const timeSeries = data['Time Series (Daily)'];
        return Object.entries(timeSeries)
          .map(([date, values]) => this.transformTimeSeriestoOHLCV(date, values))
          .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
      } catch (error) {
        console.error(`Error fetching Alpha Vantage historical data for ${ticker}:`, error);
        throw error;
      }
    });
  }

  // Get intraday data
  async getIntradayData(
    ticker: string,
    interval: '1min' | '5min' | '15min' | '30min' | '60min' = '5min',
    outputSize: 'compact' | 'full' = 'compact'
  ): Promise<OHLCV[]> {
    return this.queueRequest(async () => {
      try {
        const url = `${this.config.baseUrl}/query?function=TIME_SERIES_INTRADAY&symbol=${ticker}&interval=${interval}&outputsize=${outputSize}&apikey=${this.config.apiKey}`;
        const response = await fetch(url, {
          signal: AbortSignal.timeout(this.config.timeout)
        });

        if (!response.ok) {
          throw new Error(`Alpha Vantage API error: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data['Error Message']) {
          throw new Error(`Alpha Vantage error: ${data['Error Message']}`);
        }

        if (data['Note']) {
          throw new Error(`Alpha Vantage rate limit: ${data['Note']}`);
        }

        const timeSeries = data[`Time Series (${interval})`];
        return Object.entries(timeSeries)
          .map(([timestamp, values]) => this.transformTimeSeriestoOHLCV(timestamp, values as any))
          .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
      } catch (error) {
        console.error(`Error fetching Alpha Vantage intraday data for ${ticker}:`, error);
        throw error;
      }
    });
  }

  // Get company overview
  async getCompanyOverview(ticker: string): Promise<AlphaVantageCompanyOverview> {
    return this.queueRequest(async () => {
      try {
        const url = `${this.config.baseUrl}/query?function=OVERVIEW&symbol=${ticker}&apikey=${this.config.apiKey}`;
        const response = await fetch(url, {
          signal: AbortSignal.timeout(this.config.timeout)
        });

        if (!response.ok) {
          throw new Error(`Alpha Vantage API error: ${response.statusText}`);
        }

        const data: AlphaVantageCompanyOverview = await response.json();
        
        if (data['Error Message']) {
          throw new Error(`Alpha Vantage error: ${data['Error Message']}`);
        }

        if (data['Note']) {
          throw new Error(`Alpha Vantage rate limit: ${data['Note']}`);
        }

        return data;
      } catch (error) {
        console.error(`Error fetching Alpha Vantage company overview for ${ticker}:`, error);
        throw error;
      }
    });
  }

  // Get top gainers and losers
  async getTopGainersLosers(): Promise<{
    top_gainers: MarketData[];
    top_losers: MarketData[];
    most_actively_traded: MarketData[];
  }> {
    return this.queueRequest(async () => {
      try {
        const url = `${this.config.baseUrl}/query?function=TOP_GAINERS_LOSERS&apikey=${this.config.apiKey}`;
        const response = await fetch(url, {
          signal: AbortSignal.timeout(this.config.timeout)
        });

        if (!response.ok) {
          throw new Error(`Alpha Vantage API error: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data['Error Message']) {
          throw new Error(`Alpha Vantage error: ${data['Error Message']}`);
        }

        if (data['Note']) {
          throw new Error(`Alpha Vantage rate limit: ${data['Note']}`);
        }

        return {
          top_gainers: data.top_gainers?.map((item: any) => this.transformMoverToMarketData(item)) || [],
          top_losers: data.top_losers?.map((item: any) => this.transformMoverToMarketData(item)) || [],
          most_actively_traded: data.most_actively_traded?.map((item: any) => this.transformMoverToMarketData(item)) || []
        };
      } catch (error) {
        console.error('Error fetching Alpha Vantage top gainers/losers:', error);
        throw error;
      }
    });
  }

  // Get technical indicators
  async getTechnicalIndicator(
    ticker: string,
    indicator: string,
    interval: string,
    timePeriod?: number,
    seriesType?: string
  ): Promise<any> {
    return this.queueRequest(async () => {
      try {
        let url = `${this.config.baseUrl}/query?function=${indicator}&symbol=${ticker}&interval=${interval}&apikey=${this.config.apiKey}`;
        
        if (timePeriod) {
          url += `&time_period=${timePeriod}`;
        }
        
        if (seriesType) {
          url += `&series_type=${seriesType}`;
        }

        const response = await fetch(url, {
          signal: AbortSignal.timeout(this.config.timeout)
        });

        if (!response.ok) {
          throw new Error(`Alpha Vantage API error: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data['Error Message']) {
          throw new Error(`Alpha Vantage error: ${data['Error Message']}`);
        }

        if (data['Note']) {
          throw new Error(`Alpha Vantage rate limit: ${data['Note']}`);
        }

        return data;
      } catch (error) {
        console.error(`Error fetching Alpha Vantage technical indicator ${indicator} for ${ticker}:`, error);
        throw error;
      }
    });
  }

  // Transform Alpha Vantage quote to MarketData
  private transformQuoteToMarketData(quote: AlphaVantageQuote): MarketData {
    const price = parseFloat(quote['05. price']);
    const previousClose = parseFloat(quote['08. previous close']);
    const change = parseFloat(quote['09. change']);
    const changePercent = parseFloat(quote['10. change percent'].replace('%', ''));

    return {
      ticker: quote['01. symbol'],
      exchange: this.getExchangeFromTicker(quote['01. symbol']),
      price,
      currency: 'USD',
      timestamp: new Date(quote['07. latest trading day']),
      volume: parseInt(quote['06. volume']),
      change,
      changePercent,
      high: parseFloat(quote['03. high']),
      low: parseFloat(quote['04. low']),
      open: parseFloat(quote['02. open']),
      previousClose
    };
  }

  // Transform time series data to OHLCV
  private transformTimeSeriestoOHLCV(timestamp: string, values: any): OHLCV {
    return {
      timestamp: new Date(timestamp),
      open: parseFloat(values['1. open']),
      high: parseFloat(values['2. high']),
      low: parseFloat(values['3. low']),
      close: parseFloat(values['4. close']),
      volume: parseInt(values['5. volume'])
    };
  }

  // Transform mover data to MarketData
  private transformMoverToMarketData(mover: any): MarketData {
    const price = parseFloat(mover.price);
    const changeAmount = parseFloat(mover.change_amount);
    const changePercent = parseFloat(mover.change_percentage.replace('%', ''));

    return {
      ticker: mover.ticker,
      exchange: this.getExchangeFromTicker(mover.ticker),
      price,
      currency: 'USD',
      timestamp: new Date(),
      volume: parseInt(mover.volume),
      change: changeAmount,
      changePercent,
      previousClose: price - changeAmount
    };
  }

  // Determine exchange from ticker symbol
  private getExchangeFromTicker(ticker: string): 'NYSE' | 'NASDAQ' {
    // This is a simplified logic - in production, you'd have a comprehensive mapping
    const nasdaqTickers = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE'];
    return nasdaqTickers.includes(ticker) ? 'NASDAQ' : 'NYSE';
  }

  // Check if US market is open
  isMarketOpen(): boolean {
    const now = new Date();
    const usTime = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
    const hour = usTime.getHours();
    const minute = usTime.getMinutes();
    const day = usTime.getDay();
    
    // Monday to Friday, 9:30 AM to 4:00 PM EST
    return day >= 1 && day <= 5 && 
           ((hour === 9 && minute >= 30) || (hour > 9 && hour < 16));
  }

  // Get current queue status
  getQueueStatus(): { queueLength: number; isProcessing: boolean } {
    return {
      queueLength: this.requestQueue.length,
      isProcessing: this.isProcessingQueue
    };
  }
}

// Export default instance
export const alphaVantageProvider = new AlphaVantageProvider({
  apiKey: process.env.NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY || '',
  baseUrl: 'https://www.alphavantage.co',
  timeout: 15000,
  requestsPerMinute: 5 // Free tier limit
});