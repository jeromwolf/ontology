import { NextResponse } from 'next/server';
import YahooFinanceAPI from '@/lib/stock-api/yahoo-finance';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const type = searchParams.get('type'); // indices, stocks
    
    const yahooApi = new YahooFinanceAPI();
    
    if (type === 'indices') {
      // 주요 지수
      const indices = await yahooApi.getMultipleQuotes([
        { code: 'KOSPI', market: 'KOSPI' },    // ^KS11
        { code: 'KOSDAQ', market: 'KOSDAQ' },  // ^KQ11
      ]);
      
      // 미국 지수는 직접 심볼 사용
      const usIndices = await fetch(
        'https://query1.finance.yahoo.com/v8/finance/quote?symbols=^DJI,^IXIC,^GSPC'
      ).then(res => res.json());
      
      return NextResponse.json({
        korean: indices,
        us: usIndices.quoteResponse.result
      });
    }
    
    if (type === 'top-stocks') {
      // 주요 종목들
      const stocks = await yahooApi.getMultipleQuotes([
        { code: '005930', market: 'KOSPI' },  // 삼성전자
        { code: '000660', market: 'KOSPI' },  // SK하이닉스
        { code: '035420', market: 'KOSPI' },  // NAVER
        { code: '035720', market: 'KOSPI' },  // 카카오
        { code: '207940', market: 'KOSPI' },  // 삼성바이오로직스
        { code: '068270', market: 'KOSPI' },  // 셀트리온
        { code: '005380', market: 'KOSPI' },  // 현대차
        { code: '051910', market: 'KOSPI' },  // LG화학
        { code: '006400', market: 'KOSPI' },  // 삼성SDI
        { code: '373220', market: 'KOSPI' },  // LG에너지솔루션
      ]);
      
      // 상승/하락 정렬
      const gainers = stocks
        .filter(s => s.regularMarketChangePercent > 0)
        .sort((a, b) => b.regularMarketChangePercent - a.regularMarketChangePercent)
        .slice(0, 5);
        
      const losers = stocks
        .filter(s => s.regularMarketChangePercent < 0)
        .sort((a, b) => a.regularMarketChangePercent - b.regularMarketChangePercent)
        .slice(0, 5);
        
      const mostActive = stocks
        .sort((a, b) => b.regularMarketVolume - a.regularMarketVolume)
        .slice(0, 5);
      
      return NextResponse.json({
        gainers,
        losers,
        mostActive,
        allStocks: stocks
      });
    }
    
    // 개별 종목
    const code = searchParams.get('code');
    const market = searchParams.get('market') as 'KOSPI' | 'KOSDAQ' || 'KOSPI';
    
    if (code) {
      const data = await yahooApi.getQuote(code, market);
      return NextResponse.json(data);
    }
    
    return NextResponse.json({ error: 'Invalid request' }, { status: 400 });
    
  } catch (error) {
    console.error('Yahoo Finance API error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch stock data' },
      { status: 500 }
    );
  }
}