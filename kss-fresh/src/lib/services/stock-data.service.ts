/**
 * 주식 데이터 통합 서비스
 * 한국투자증권 API와 기타 데이터 소스를 통합 관리
 */

import { koreaInvestmentAPI } from './korea-investment-api.service';

export interface StockData {
  code: string;
  name: string;
  currentPrice: number;
  changeAmount: number;
  changeRate: number;
  volume: number;
  marketCap?: number;
  per?: number;
  pbr?: number;
  roe?: number;
  eps?: number;
}

export interface ChartData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface FinancialData {
  revenue: number;
  operatingIncome: number;
  netIncome: number;
  eps: number;
  bps: number;
  roe: number;
  roa: number;
  debtRatio: number;
}

// 주요 한국 주식 종목 코드
export const KOREAN_STOCKS = {
  '삼성전자': '005930',
  'SK하이닉스': '000660',
  'LG에너지솔루션': '373220',
  'NAVER': '035420',
  '카카오': '035720',
  '현대차': '005380',
  '기아': '000270',
  'SK이노베이션': '096770',
  'LG화학': '051910',
  '삼성바이오로직스': '207940',
  '셀트리온': '068270',
  'KB금융': '105560',
  '신한지주': '055550',
  '하나금융지주': '086790',
  'POSCO홀딩스': '005490'
};

export class StockDataService {
  private cache = new Map<string, { data: any; timestamp: number }>();
  private cacheTimeout = 5 * 60 * 1000; // 5분

  /**
   * 캐시된 데이터 가져오기
   */
  private getCachedData(key: string): any | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }
    return null;
  }

  /**
   * 데이터 캐싱
   */
  private setCachedData(key: string, data: any): void {
    this.cache.set(key, { data, timestamp: Date.now() });
  }

  /**
   * 실시간 주식 데이터 조회
   */
  async getStockData(stockCode: string): Promise<StockData> {
    const cacheKey = `stock_${stockCode}`;
    const cached = this.getCachedData(cacheKey);
    if (cached) return cached;

    try {
      const priceData = await koreaInvestmentAPI.getCurrentPrice(stockCode);
      
      const stockData: StockData = {
        code: stockCode,
        name: this.getStockName(stockCode),
        currentPrice: parseFloat(priceData.stck_prpr),
        changeAmount: parseFloat(priceData.prdy_vrss),
        changeRate: parseFloat(priceData.prdy_ctrt),
        volume: parseInt(priceData.acml_vol)
      };

      this.setCachedData(cacheKey, stockData);
      return stockData;
    } catch (error) {
      console.error('Failed to get stock data:', error);
      // 에러 시 목업 데이터 반환
      return this.getMockStockData(stockCode);
    }
  }

  /**
   * 여러 종목 데이터 일괄 조회
   */
  async getMultipleStocks(stockCodes: string[]): Promise<StockData[]> {
    const results: StockData[] = [];
    
    for (const code of stockCodes) {
      const data = await this.getStockData(code);
      results.push(data);
    }
    
    return results;
  }

  /**
   * 차트 데이터 조회
   */
  async getChartData(stockCode: string, days: number = 30): Promise<ChartData[]> {
    const cacheKey = `chart_${stockCode}_${days}`;
    const cached = this.getCachedData(cacheKey);
    if (cached) return cached;

    try {
      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - days);
      
      const chartData = await koreaInvestmentAPI.getDailyPrices(
        stockCode,
        this.formatDate(startDate),
        this.formatDate(endDate)
      );

      this.setCachedData(cacheKey, chartData);
      return chartData;
    } catch (error) {
      console.error('Failed to get chart data:', error);
      return this.getMockChartData(days);
    }
  }

  /**
   * 재무 데이터 조회
   */
  async getFinancialData(stockCode: string): Promise<FinancialData> {
    const cacheKey = `financial_${stockCode}`;
    const cached = this.getCachedData(cacheKey);
    if (cached) return cached;

    try {
      const financialInfo = await koreaInvestmentAPI.getFinancialInfo(stockCode);
      
      // API 응답을 FinancialData 형식으로 변환
      const financialData: FinancialData = {
        revenue: parseFloat(financialInfo.sale_account || '0'),
        operatingIncome: parseFloat(financialInfo.sale_cost || '0'),
        netIncome: parseFloat(financialInfo.current_net_income || '0'),
        eps: parseFloat(financialInfo.eps || '0'),
        bps: parseFloat(financialInfo.bps || '0'),
        roe: parseFloat(financialInfo.roe || '0'),
        roa: parseFloat(financialInfo.roa || '0'),
        debtRatio: parseFloat(financialInfo.debt_ratio || '0')
      };

      this.setCachedData(cacheKey, financialData);
      return financialData;
    } catch (error) {
      console.error('Failed to get financial data:', error);
      return this.getMockFinancialData();
    }
  }

  /**
   * 포트폴리오용 자산 데이터 조회
   */
  async getPortfolioAssets(): Promise<any[]> {
    const topStocks = Object.entries(KOREAN_STOCKS).slice(0, 10);
    const assets = [];

    for (const [name, code] of topStocks) {
      try {
        const stockData = await this.getStockData(code);
        const chartData = await this.getChartData(code, 252); // 1년 데이터
        
        // 수익률과 변동성 계산
        const returns = this.calculateReturns(chartData);
        const volatility = this.calculateVolatility(returns);
        const annualReturn = this.calculateAnnualReturn(returns);

        assets.push({
          name,
          ticker: code,
          expectedReturn: annualReturn,
          risk: volatility,
          currentPrice: stockData.currentPrice
        });
      } catch (error) {
        console.error(`Failed to get asset data for ${name}:`, error);
      }
    }

    return assets;
  }

  /**
   * 종목 코드로 종목명 조회
   */
  private getStockName(stockCode: string): string {
    const entry = Object.entries(KOREAN_STOCKS).find(([_, code]) => code === stockCode);
    return entry ? entry[0] : stockCode;
  }

  /**
   * 날짜 포맷팅 (YYYYMMDD)
   */
  private formatDate(date: Date): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}${month}${day}`;
  }

  /**
   * 수익률 계산
   */
  private calculateReturns(chartData: ChartData[]): number[] {
    const returns: number[] = [];
    for (let i = 1; i < chartData.length; i++) {
      const returnRate = (chartData[i].close - chartData[i - 1].close) / chartData[i - 1].close;
      returns.push(returnRate);
    }
    return returns;
  }

  /**
   * 변동성 계산 (표준편차)
   */
  private calculateVolatility(returns: number[]): number {
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    return Math.sqrt(variance) * Math.sqrt(252); // 연간 변동성
  }

  /**
   * 연간 수익률 계산
   */
  private calculateAnnualReturn(returns: number[]): number {
    const totalReturn = returns.reduce((acc, r) => acc * (1 + r), 1) - 1;
    return totalReturn * (252 / returns.length);
  }

  /**
   * 목업 데이터 생성 (API 실패 시 폴백)
   */
  private getMockStockData(stockCode: string): StockData {
    const basePrice = 50000 + Math.random() * 100000;
    const changeRate = (Math.random() - 0.5) * 10;
    
    return {
      code: stockCode,
      name: this.getStockName(stockCode),
      currentPrice: basePrice,
      changeAmount: basePrice * changeRate / 100,
      changeRate: changeRate,
      volume: Math.floor(Math.random() * 10000000)
    };
  }

  private getMockChartData(days: number): ChartData[] {
    const data: ChartData[] = [];
    let basePrice = 50000 + Math.random() * 50000;
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      const change = (Math.random() - 0.5) * 0.05;
      basePrice = basePrice * (1 + change);
      
      const high = basePrice * (1 + Math.random() * 0.02);
      const low = basePrice * (1 - Math.random() * 0.02);
      const open = low + Math.random() * (high - low);
      const close = low + Math.random() * (high - low);
      
      data.push({
        date: date.toISOString().split('T')[0],
        open,
        high,
        low,
        close,
        volume: Math.floor(Math.random() * 10000000)
      });
    }
    
    return data;
  }

  private getMockFinancialData(): FinancialData {
    return {
      revenue: Math.floor(Math.random() * 100000) * 100000000,
      operatingIncome: Math.floor(Math.random() * 20000) * 100000000,
      netIncome: Math.floor(Math.random() * 15000) * 100000000,
      eps: Math.floor(Math.random() * 10000),
      bps: Math.floor(Math.random() * 100000),
      roe: Math.random() * 30,
      roa: Math.random() * 15,
      debtRatio: Math.random() * 200
    };
  }
}

// 싱글톤 인스턴스 export
export const stockDataService = new StockDataService();