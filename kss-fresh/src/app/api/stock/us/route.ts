import { NextResponse } from 'next/server';

const TWELVE_DATA_API_KEY = process.env.TWELVE_DATA_API_KEY;
const ALPHA_VANTAGE_API_KEY = process.env.ALPHA_VANTAGE_API_KEY;

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const symbol = searchParams.get('symbol');
  const interval = searchParams.get('interval') || '5min';
  const includeDaily = searchParams.get('includeDaily') === 'true';
  
  if (!symbol) {
    return NextResponse.json({ error: 'Symbol is required' }, { status: 400 });
  }

  try {
    // Twelve Data API 우선 사용
    if (TWELVE_DATA_API_KEY) {
      console.log('Using Twelve Data API for:', symbol);
      const response = await fetch(
        `https://api.twelvedata.com/time_series?symbol=${symbol}&interval=${interval}&apikey=${TWELVE_DATA_API_KEY}&outputsize=100&timezone=America/New_York`
      );
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.status === 'error') {
          console.error('Twelve Data error:', data.message);
        } else if (data.values && data.values.length > 0) {
          // Twelve Data 형식을 우리 차트 형식으로 변환
          const chartData = data.values.map((item: any) => ({
            time: item.datetime,
            open: parseFloat(item.open),
            high: parseFloat(item.high),
            low: parseFloat(item.low),
            close: parseFloat(item.close),
            volume: parseInt(item.volume),
          })).reverse(); // 시간 순서대로 정렬
          
          // 전일 종가 가져오기
          let previousClose = null;
          if (includeDaily) {
            try {
              const dailyResponse = await fetch(
                `https://api.twelvedata.com/time_series?symbol=${symbol}&interval=1day&outputsize=2&apikey=${TWELVE_DATA_API_KEY}`
              );
              if (dailyResponse.ok) {
                const dailyData = await dailyResponse.json();
                if (dailyData.values && dailyData.values.length >= 2) {
                  previousClose = parseFloat(dailyData.values[1].close);
                }
              }
            } catch (error) {
              console.error('Failed to fetch daily data:', error);
            }
          }
          
          return NextResponse.json({
            symbol: data.meta.symbol,
            data: chartData,
            previousClose,
            meta: {
              currency: data.meta.currency,
              exchange: data.meta.exchange,
              type: data.meta.type,
            }
          });
        }
      }
    }
    
    // 백업: Alpha Vantage API 사용
    if (ALPHA_VANTAGE_API_KEY) {
      const alphaInterval = interval === '5min' ? '5min' : '60min';
      const response = await fetch(
        `https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=${symbol}&interval=${alphaInterval}&apikey=${ALPHA_VANTAGE_API_KEY}&outputsize=full`
      );
      
      if (response.ok) {
        const data = await response.json();
        
        if (data['Error Message']) {
          console.error('Alpha Vantage error:', data['Error Message']);
        } else if (data[`Time Series (${alphaInterval})`]) {
          const timeSeries = data[`Time Series (${alphaInterval})`];
          const chartData = Object.entries(timeSeries).map(([time, values]: [string, any]) => ({
            time,
            open: parseFloat(values['1. open']),
            high: parseFloat(values['2. high']),
            low: parseFloat(values['3. low']),
            close: parseFloat(values['4. close']),
            volume: parseInt(values['5. volume']),
          })).reverse();
          
          return NextResponse.json({
            symbol: symbol,
            data: chartData.slice(0, 100), // 최근 100개만
            meta: {
              currency: 'USD',
              exchange: 'NASDAQ',
              type: 'stock',
            }
          });
        }
      }
    }
    
    // API 키가 없거나 실패한 경우 데모 데이터 반환
    console.log('미국 주식 API 키가 없거나 실패하여 데모 데이터를 반환합니다.');
    return NextResponse.json({
      symbol: symbol,
      data: generateDemoData(symbol),
      meta: {
        currency: 'USD',
        exchange: 'DEMO',
        type: 'stock',
        demo: true,
      }
    });
    
  } catch (error) {
    console.error('US Stock API Error:', error);
    return NextResponse.json({ 
      error: '데이터를 가져오는 중 오류가 발생했습니다.',
      details: error instanceof Error ? error.message : 'Unknown error' 
    }, { status: 500 });
  }
}

// 데모 데이터 생성
function generateDemoData(symbol: string) {
  const data = [];
  const now = new Date();
  
  // 기본 가격 설정 (2024년 8월 기준 실제 가격 근사치)
  const basePrices: Record<string, number> = {
    'TSLA': 212,    // 실제 약 $212
    'AAPL': 226,    // 실제 약 $226
    'MSFT': 410,    // 실제 약 $410
    'GOOGL': 163,   // 실제 약 $163
    'AMZN': 178,    // 실제 약 $178
    'NVDA': 125,    // 실제 약 $125 (액면분할 후)
    'META': 513,    // 실제 약 $513
    'NFLX': 612,    // 실제 약 $612
  };
  
  let basePrice = basePrices[symbol] || 100;
  
  for (let i = 100; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 5 * 60 * 1000);
    const volatility = 0.02;
    const change = (Math.random() - 0.5) * basePrice * volatility;
    const close = Math.max(1, basePrice + change);
    const open = basePrice;
    const high = Math.max(open, close) * (1 + Math.random() * 0.01);
    const low = Math.min(open, close) * (1 - Math.random() * 0.01);
    
    data.push({
      time: time.toISOString(),
      open: Math.round(open * 100) / 100,
      high: Math.round(high * 100) / 100,
      low: Math.round(low * 100) / 100,
      close: Math.round(close * 100) / 100,
      volume: Math.floor(Math.random() * 10000000) + 1000000,
    });
    
    basePrice = close;
  }
  
  return data;
}