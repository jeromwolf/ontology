/**
 * 미국 주식 데이터 서비스
 * Twelve Data / Alpha Vantage API를 통해 실시간 미국 주식 데이터 제공
 */

export interface USStockQuote {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  previousClose: number;
  timestamp: string;
}

export interface USStockChartData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export class USStockService {
  private static instance: USStockService;
  
  private constructor() {}
  
  public static getInstance(): USStockService {
    if (!USStockService.instance) {
      USStockService.instance = new USStockService();
    }
    return USStockService.instance;
  }
  
  /**
   * 미국 주식 차트 데이터 가져오기
   */
  async getChartData(symbol: string, interval: string = '5min'): Promise<USStockChartData[]> {
    try {
      const response = await fetch(`/api/stock/us?symbol=${symbol}&interval=${interval}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch data: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      return result.data || [];
    } catch (error) {
      console.error('Failed to fetch US stock data:', error);
      // 오류 시 빈 배열 반환
      return [];
    }
  }
  
  /**
   * 미국 주식 실시간 견적 가져오기
   */
  async getQuote(symbol: string): Promise<USStockQuote | null> {
    try {
      const chartData = await this.getChartData(symbol, '1min');
      
      if (chartData.length === 0) {
        return null;
      }
      
      const latest = chartData[chartData.length - 1];
      const previous = chartData[chartData.length - 2] || latest;
      
      const price = latest.close;
      const previousClose = previous.close;
      const change = price - previousClose;
      const changePercent = (change / previousClose) * 100;
      
      // 일일 최고/최저 계산
      const dayData = chartData.slice(-78); // 약 6.5시간 (미국 장중)
      const high = Math.max(...dayData.map(d => d.high));
      const low = Math.min(...dayData.map(d => d.low));
      const volume = dayData.reduce((sum, d) => sum + d.volume, 0);
      
      return {
        symbol,
        price,
        change,
        changePercent,
        volume,
        high,
        low,
        open: dayData[0]?.open || price,
        previousClose,
        timestamp: latest.time,
      };
    } catch (error) {
      console.error('Failed to get quote:', error);
      return null;
    }
  }
  
  /**
   * 여러 미국 주식 견적 가져오기
   */
  async getMultipleQuotes(symbols: string[]): Promise<Record<string, USStockQuote>> {
    const quotes: Record<string, USStockQuote> = {};
    
    // 병렬로 요청
    const promises = symbols.map(async (symbol) => {
      const quote = await this.getQuote(symbol);
      if (quote) {
        quotes[symbol] = quote;
      }
    });
    
    await Promise.all(promises);
    return quotes;
  }
  
  /**
   * 미국 시장 개장 여부 확인
   */
  isMarketOpen(): boolean {
    const now = new Date();
    const easternTime = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
    const day = easternTime.getDay();
    const hours = easternTime.getHours();
    const minutes = easternTime.getMinutes();
    
    // 주말 제외
    if (day === 0 || day === 6) return false;
    
    // 개장 시간: 9:30 AM - 4:00 PM ET
    const marketOpen = hours * 60 + minutes >= 9 * 60 + 30;
    const marketClose = hours * 60 + minutes < 16 * 60;
    
    return marketOpen && marketClose;
  }
  
  /**
   * 프리마켓/애프터마켓 여부 확인
   */
  getMarketSession(): 'pre-market' | 'regular' | 'after-hours' | 'closed' {
    const now = new Date();
    const easternTime = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
    const day = easternTime.getDay();
    const hours = easternTime.getHours();
    const minutes = easternTime.getMinutes();
    const totalMinutes = hours * 60 + minutes;
    
    // 주말
    if (day === 0 || day === 6) return 'closed';
    
    // 프리마켓: 4:00 AM - 9:30 AM ET
    if (totalMinutes >= 4 * 60 && totalMinutes < 9 * 60 + 30) {
      return 'pre-market';
    }
    
    // 정규장: 9:30 AM - 4:00 PM ET
    if (totalMinutes >= 9 * 60 + 30 && totalMinutes < 16 * 60) {
      return 'regular';
    }
    
    // 애프터마켓: 4:00 PM - 8:00 PM ET
    if (totalMinutes >= 16 * 60 && totalMinutes < 20 * 60) {
      return 'after-hours';
    }
    
    return 'closed';
  }
}

export const usStockService = USStockService.getInstance();