// Naver Finance 비공식 API (무료, 실시간)
// 주의: 비공식 API이므로 변경될 수 있음

interface NaverStockPrice {
  code: string;           // 종목코드
  name: string;           // 종목명
  now: number;           // 현재가
  diff: number;          // 전일대비
  rate: number;          // 등락률
  quant: number;         // 거래량
  amount: number;        // 거래대금
  high: number;          // 고가
  low: number;           // 저가
  open: number;          // 시가
  yesterday: number;     // 전일종가
  marketSum: number;     // 시가총액 (억원)
}

interface NaverMarketIndex {
  name: string;
  value: number;
  change: number;
  changePercent: number;
}

class NaverFinanceAPI {
  // CORS 문제로 서버사이드에서만 사용 가능
  
  // 개별 종목 시세
  async getStockPrice(stockCode: string): Promise<NaverStockPrice | null> {
    try {
      const response = await fetch(
        `https://api.finance.naver.com/service/itemSummary.naver?itemcode=${stockCode}`,
        {
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
          }
        }
      );
      
      const data = await response.json();
      
      return {
        code: data.itemcode,
        name: data.itemname,
        now: data.now,
        diff: data.diff,
        rate: data.rate,
        quant: data.quant,
        amount: data.amount,
        high: data.high,
        low: data.low,
        open: data.open,
        yesterday: data.yesterday,
        marketSum: data.marketSum,
      };
    } catch (error) {
      console.error('Naver Finance API error:', error);
      return null;
    }
  }

  // 상승률 상위 종목
  async getTopGainers(market: 'KOSPI' | 'KOSDAQ' = 'KOSPI'): Promise<any[]> {
    const marketCode = market === 'KOSPI' ? 'KOSPI' : 'KOSDAQ';
    
    try {
      const response = await fetch(
        `https://api.finance.naver.com/siseJson.naver?symbol=${marketCode}&requestType=1&count=10&startTime=&endTime=&timeframe=day`,
        {
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
          }
        }
      );
      
      const text = await response.text();
      // Naver returns JSONP format, need to parse
      const jsonStr = text.replace(/^[^(]*\(/, '').replace(/\);?\s*$/, '');
      const data = JSON.parse(jsonStr);
      
      return data;
    } catch (error) {
      console.error('Naver Finance API error:', error);
      return [];
    }
  }

  // 시장 지수 (KOSPI, KOSDAQ)
  async getMarketIndex(indexCode: string = 'KOSPI'): Promise<NaverMarketIndex | null> {
    try {
      const response = await fetch(
        `https://api.finance.naver.com/service/itemSummary.naver?itemcode=${indexCode}`,
        {
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
          }
        }
      );
      
      const data = await response.json();
      
      return {
        name: indexCode,
        value: data.now,
        change: data.diff,
        changePercent: data.rate,
      };
    } catch (error) {
      console.error('Naver Finance API error:', error);
      return null;
    }
  }

  // 업종별 시세
  async getSectorData(): Promise<any[]> {
    try {
      const response = await fetch(
        'https://finance.naver.com/api/sise/upjongList.naver',
        {
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
          }
        }
      );
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Naver Finance API error:', error);
      return [];
    }
  }
}

export default NaverFinanceAPI;

// 사용 예시 (서버 사이드에서만)
/*
const naverApi = new NaverFinanceAPI();

// 삼성전자 시세
const samsung = await naverApi.getStockPrice('005930');
console.log(samsung);

// KOSPI 지수
const kospiIndex = await naverApi.getMarketIndex('KOSPI');
console.log(kospiIndex);
*/