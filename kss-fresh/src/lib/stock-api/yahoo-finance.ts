// Yahoo Finance API 연동 (무료, 지연 시세)
// 한국 주식은 .KS (KOSPI) 또는 .KQ (KOSDAQ) 접미사 사용

interface YahooQuote {
  symbol: string;
  longName: string;
  regularMarketPrice: number;
  regularMarketChange: number;
  regularMarketChangePercent: number;
  regularMarketVolume: number;
  marketCap: number;
  regularMarketTime: string;
}

class YahooFinanceAPI {
  private baseUrl = 'https://query1.finance.yahoo.com/v8/finance';

  // 종목 코드 변환 (한국 -> Yahoo)
  private convertToYahooSymbol(koreanCode: string, market: 'KOSPI' | 'KOSDAQ' = 'KOSPI'): string {
    // 지수인 경우
    if (koreanCode === 'KOSPI') return '^KS11';
    if (koreanCode === 'KOSDAQ') return '^KQ11';
    
    // 일반 종목
    const suffix = market === 'KOSPI' ? '.KS' : '.KQ';
    return `${koreanCode}${suffix}`;
  }

  // 현재가 조회
  async getQuote(stockCode: string, market: 'KOSPI' | 'KOSDAQ' = 'KOSPI'): Promise<YahooQuote | null> {
    const symbol = this.convertToYahooSymbol(stockCode, market);
    
    try {
      const response = await fetch(
        `${this.baseUrl}/quote?symbols=${symbol}`,
        {
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
          }
        }
      );
      
      if (!response.ok) {
        console.error(`Yahoo Finance API returned ${response.status}`);
        return null;
      }
      
      const data = await response.json();
      const quote = data.quoteResponse?.result?.[0];
      
      if (!quote) return null;

      return {
        symbol: quote.symbol,
        longName: quote.longName || quote.shortName || quote.symbol,
        regularMarketPrice: quote.regularMarketPrice || 0,
        regularMarketChange: quote.regularMarketChange || 0,
        regularMarketChangePercent: quote.regularMarketChangePercent || 0,
        regularMarketVolume: quote.regularMarketVolume || 0,
        marketCap: quote.marketCap || 0,
        regularMarketTime: quote.regularMarketTime ? new Date(quote.regularMarketTime * 1000).toISOString() : new Date().toISOString(),
      };
    } catch (error) {
      console.error('Yahoo Finance API error:', error);
      return null;
    }
  }

  // 다중 종목 조회
  async getMultipleQuotes(stockCodes: Array<{code: string, market: 'KOSPI' | 'KOSDAQ'}>): Promise<YahooQuote[]> {
    const symbols = stockCodes
      .map(stock => this.convertToYahooSymbol(stock.code, stock.market))
      .join(',');
    
    try {
      const response = await fetch(
        `${this.baseUrl}/quote?symbols=${symbols}`,
        {
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
          }
        }
      );
      
      if (!response.ok) {
        console.error(`Yahoo Finance API returned ${response.status}`);
        return [];
      }
      
      const data = await response.json();
      const quotes = data.quoteResponse?.result || [];
      
      return quotes.map((quote: any) => ({
        symbol: quote.symbol,
        longName: quote.longName || quote.shortName || quote.symbol,
        regularMarketPrice: quote.regularMarketPrice || 0,
        regularMarketChange: quote.regularMarketChange || 0,
        regularMarketChangePercent: quote.regularMarketChangePercent || 0,
        regularMarketVolume: quote.regularMarketVolume || 0,
        marketCap: quote.marketCap || 0,
        regularMarketTime: quote.regularMarketTime ? new Date(quote.regularMarketTime * 1000).toISOString() : new Date().toISOString(),
      }));
    } catch (error) {
      console.error('Yahoo Finance API error:', error);
      return [];
    }
  }

  // 과거 데이터 조회 (차트용)
  async getHistoricalData(
    stockCode: string, 
    market: 'KOSPI' | 'KOSDAQ' = 'KOSPI',
    period: '1d' | '5d' | '1mo' | '3mo' | '6mo' | '1y' = '1mo'
  ): Promise<any> {
    const symbol = this.convertToYahooSymbol(stockCode, market);
    
    try {
      const response = await fetch(
        `${this.baseUrl}/chart/${symbol}?interval=1d&range=${period}`
      );
      
      const data = await response.json();
      return data.chart.result[0];
    } catch (error) {
      console.error('Yahoo Finance API error:', error);
      return null;
    }
  }
}

export default YahooFinanceAPI;

// 사용 예시
/*
const yahooApi = new YahooFinanceAPI();

// 삼성전자 현재가
const samsung = await yahooApi.getQuote('005930', 'KOSPI');
console.log(samsung);

// 여러 종목 한번에
const stocks = await yahooApi.getMultipleQuotes([
  { code: '005930', market: 'KOSPI' },  // 삼성전자
  { code: '000660', market: 'KOSPI' },  // SK하이닉스
  { code: '035720', market: 'KOSPI' },  // 카카오
]);
*/