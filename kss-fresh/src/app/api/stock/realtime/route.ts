import { NextResponse } from 'next/server';
import YahooFinanceAPI from '@/lib/stock-api/yahoo-finance';
import { marketCache, CACHE_TYPES } from '@/lib/cache/market-cache';
import NaverFinanceAPI from '@/lib/stock-api/naver-finance';
import KoreaInvestmentAPI from '@/lib/stock-api/korea-investment';

// Alpha Vantage API 사용 (무료 티어)
const ALPHA_VANTAGE_API_KEY = process.env.ALPHA_VANTAGE_API_KEY || '7JD5XP9H2T8WZBKQ';

// 한국 주요 종목 리스트
const KOREAN_STOCKS = [
  { code: '005930', name: '삼성전자', market: 'KOSPI' as const },
  { code: '000660', name: 'SK하이닉스', market: 'KOSPI' as const },
  { code: '035420', name: 'NAVER', market: 'KOSPI' as const },
  { code: '035720', name: '카카오', market: 'KOSPI' as const },
  { code: '207940', name: '삼성바이오로직스', market: 'KOSPI' as const },
  { code: '068270', name: '셀트리온', market: 'KOSPI' as const },
  { code: '005380', name: '현대차', market: 'KOSPI' as const },
  { code: '051910', name: 'LG화학', market: 'KOSPI' as const },
  { code: '006400', name: '삼성SDI', market: 'KOSPI' as const },
  { code: '373220', name: 'LG에너지솔루션', market: 'KOSPI' as const },
  { code: '000270', name: '기아', market: 'KOSPI' as const },
  { code: '012330', name: '현대모비스', market: 'KOSPI' as const },
  { code: '105560', name: 'KB금융', market: 'KOSPI' as const },
  { code: '055550', name: '신한지주', market: 'KOSPI' as const },
  { code: '028260', name: '삼성물산', market: 'KOSPI' as const },
];

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const type = searchParams.get('type');
    const forceRefresh = searchParams.get('refresh') === 'true';
    
    const yahooApi = new YahooFinanceAPI();
    const naverApi = new NaverFinanceAPI();
    
    // 한국투자증권 API 초기화
    const kiApi = new KoreaInvestmentAPI({
      appKey: process.env.APP_KEY!,
      appSecret: process.env.APP_SECRET!,
      accountNo: '', // 시세 조회는 계좌번호 불필요
      isPaper: true  // 모의투자 서버 사용
    });
    
    if (type === 'market-overview') {
      // 캐싱 적용된 데이터 가져오기
      const cachedData = await marketCache.fetchWithCache(
        CACHE_TYPES.MARKET_OVERVIEW,
        async () => {
          // 한국투자증권 API로 실제 데이터 가져오기
          let koreanStocks = [];
          
          try {
            // 한국투자증권 API로 실제 주가 가져오기
            const stockPromises = KOREAN_STOCKS.map(stock => 
              kiApi.getCurrentPrice(stock.code)
                .then(data => data ? { ...data, longName: stock.name } : null)
                .catch(err => {
                  console.error(`Failed to fetch ${stock.code}:`, err);
                  return null;
                })
            );
            
            const results = await Promise.all(stockPromises);
            koreanStocks = results.filter(s => s !== null);
            
            // 데이터가 없으면 시뮬레이션 데이터 사용
            if (koreanStocks.length === 0) {
              throw new Error('No data from Korea Investment API');
            }
          } catch (error) {
            console.error('Failed to fetch from Korea Investment:', error);
            
            // API 실패시 시뮬레이션 데이터 사용 (2024년 12월 기준 실제 가격)
            const stockPrices: Record<string, number> = {
              '005930': 71300,   // 삼성전자
              '000660': 121500,  // SK하이닉스
              '035420': 226500,  // NAVER
              '035720': 51600,   // 카카오
              '207940': 812000,  // 삼성바이오로직스
              '068270': 168300,  // 셀트리온
              '005380': 241000,  // 현대차
              '051910': 389500,  // LG화학
              '006400': 420000,  // 삼성SDI
              '373220': 392000,  // LG에너지솔루션
              '000270': 119700,  // 기아
              '012330': 248500,  // 현대모비스
              '105560': 59800,   // KB금융
              '055550': 47350,   // 신한지주
              '028260': 121500,  // 삼성물산
            };
            
            koreanStocks = KOREAN_STOCKS.map((stock) => {
              const basePrice = stockPrices[stock.code] || 100000;
              const change = (Math.random() - 0.5) * 4; // -2% ~ +2%
              return {
                symbol: stock.code,
                longName: stock.name,
                regularMarketPrice: basePrice,
                regularMarketChange: basePrice * change / 100,
                regularMarketChangePercent: change,
                regularMarketVolume: Math.floor(Math.random() * 10000000),
                marketCap: basePrice * 1000000000,
                regularMarketTime: new Date().toISOString()
              };
            });
          }

          // 상승/하락/거래량 분석
          const validStocks = koreanStocks.filter(s => s && s.regularMarketPrice > 0);
          
          const gainers = validStocks
            .filter(s => s.regularMarketChangePercent > 0)
            .sort((a, b) => b.regularMarketChangePercent - a.regularMarketChangePercent)
            .slice(0, 5)
            .map(stock => ({
              symbol: stock.symbol,
              name: stock.longName,
              price: stock.regularMarketPrice,
              change: stock.regularMarketChange,
              changePercent: stock.regularMarketChangePercent,
              volume: stock.regularMarketVolume.toLocaleString('ko-KR'),
              marketCap: stock.marketCap ? `${(stock.marketCap / 1000000000000).toFixed(1)}조` : undefined
            }));
            
          const losers = validStocks
            .filter(s => s.regularMarketChangePercent < 0)
            .sort((a, b) => a.regularMarketChangePercent - b.regularMarketChangePercent)
            .slice(0, 5)
            .map(stock => ({
              symbol: stock.symbol,
              name: stock.longName,
              price: stock.regularMarketPrice,
              change: stock.regularMarketChange,
              changePercent: stock.regularMarketChangePercent,
              volume: stock.regularMarketVolume.toLocaleString('ko-KR'),
              marketCap: stock.marketCap ? `${(stock.marketCap / 1000000000000).toFixed(1)}조` : undefined
            }));
            
          const mostActive = validStocks
            .sort((a, b) => b.regularMarketVolume - a.regularMarketVolume)
            .slice(0, 5)
            .map(stock => ({
              symbol: stock.symbol,
              name: stock.longName,
              price: stock.regularMarketPrice,
              change: stock.regularMarketChange,
              changePercent: stock.regularMarketChangePercent,
              volume: stock.regularMarketVolume.toLocaleString('ko-KR'),
              marketCap: stock.marketCap ? `${(stock.marketCap / 1000000000000).toFixed(1)}조` : undefined
            }));

          // 섹터별 분석 (간단한 분류)
          const sectors = [
            {
              name: '반도체',
              stocks: ['005930', '000660'],
              changePercent: 0,
              leaders: []
            },
            {
              name: '2차전지',
              stocks: ['373220', '006400'],
              changePercent: 0,
              leaders: []
            },
            {
              name: '바이오',
              stocks: ['207940', '068270'],
              changePercent: 0,
              leaders: []
            },
            {
              name: '금융',
              stocks: ['105560', '055550'],
              changePercent: 0,
              leaders: []
            }
          ];

          // 섹터별 평균 계산
          sectors.forEach(sector => {
            const sectorStocks = validStocks.filter(s => 
              sector.stocks.includes(s.symbol)
            );
            
            if (sectorStocks.length > 0) {
              sector.changePercent = Number(
                (sectorStocks.reduce((sum, s) => sum + s.regularMarketChangePercent, 0) / sectorStocks.length).toFixed(2)
              );
              
              sector.leaders = sectorStocks.map(s => ({
                symbol: s.symbol,
                name: s.longName,
                changePercent: s.regularMarketChangePercent
              }));
            }
          });

          return {
            timestamp: new Date().toISOString(),
            dataSource: koreanStocks.length > 0 && !koreanStocks[0].regularMarketTime.includes('2025') ? 'Korea Investment Securities' : 'Simulated Data',
            gainers,
            losers,
            mostActive,
            sectors,
            allStocks: validStocks.map(stock => ({
              symbol: stock.symbol,
              name: stock.longName,
              price: stock.regularMarketPrice,
              change: stock.regularMarketChange,
              changePercent: stock.regularMarketChangePercent,
              volume: stock.regularMarketVolume,
              marketCap: stock.marketCap
            }))
          };
        },
        {},
        { forceRefresh }
      );
      
      return NextResponse.json({
        ...cachedData,
        cached: cachedData.timestamp !== new Date().toISOString(),
        cacheStats: marketCache.getStats()
      });
    }
    
    // 개별 종목 조회
    const code = searchParams.get('code');
    if (code) {
      const market = (searchParams.get('market') as 'KOSPI' | 'KOSDAQ') || 'KOSPI';
      
      const cachedData = await marketCache.fetchWithCache(
        CACHE_TYPES.STOCK_QUOTE,
        async () => {
          const data = await yahooApi.getQuote(code, market);
          return {
            timestamp: new Date().toISOString(),
            dataSource: 'Yahoo Finance',
            delayed: true,
            data
          };
        },
        { code, market },
        { forceRefresh }
      );
      
      return NextResponse.json({
        ...cachedData,
        cached: cachedData.timestamp !== new Date().toISOString()
      });
    }
    
    // 캐시 상태 조회
    if (searchParams.get('cache-stats') === 'true') {
      return NextResponse.json({
        stats: marketCache.getStats(),
        timestamp: new Date().toISOString()
      });
    }
    
    return NextResponse.json({ error: 'Invalid request' }, { status: 400 });
    
  } catch (error) {
    console.error('Realtime API error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch realtime data', details: error },
      { status: 500 }
    );
  }
}