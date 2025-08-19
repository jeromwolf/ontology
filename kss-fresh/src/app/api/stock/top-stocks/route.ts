import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const type = searchParams.get('type'); // gainers, losers, active
    const limit = parseInt(searchParams.get('limit') || '5');

    // 종목 마스터 데이터 조회
    const stocks = await prisma.stock_Symbol.findMany({
      where: {
        market: {
          in: ['KOSPI', 'KOSDAQ']
        }
      },
      include: {
        quotes: {
          orderBy: {
            date: 'desc'
          },
          take: 1
        }
      }
    });

    // 시뮬레이션 데이터 (DB에 데이터가 없을 경우)
    if (stocks.length === 0) {
      const sampleStocks = {
        gainers: [
          { symbol: '005930', name: '삼성전자', price: 68500, change: 2100, changePercent: 3.16, volume: '15,234,567' },
          { symbol: '000660', name: 'SK하이닉스', price: 115000, change: 3500, changePercent: 3.14, volume: '8,901,234' },
          { symbol: '035420', name: 'NAVER', price: 215000, change: 6000, changePercent: 2.87, volume: '1,234,567' },
          { symbol: '035720', name: '카카오', price: 45200, change: 1200, changePercent: 2.73, volume: '3,456,789' },
          { symbol: '207940', name: '삼성바이오로직스', price: 785000, change: 20000, changePercent: 2.61, volume: '234,567' },
        ],
        losers: [
          { symbol: '005380', name: '현대차', price: 185000, change: -5500, changePercent: -2.89, volume: '2,345,678' },
          { symbol: '051910', name: 'LG화학', price: 412000, change: -11000, changePercent: -2.60, volume: '567,890' },
          { symbol: '006400', name: '삼성SDI', price: 398000, change: -9500, changePercent: -2.33, volume: '345,678' },
          { symbol: '028260', name: '삼성물산', price: 102000, change: -2300, changePercent: -2.21, volume: '890,123' },
          { symbol: '105560', name: 'KB금융', price: 52100, change: -1100, changePercent: -2.07, volume: '1,234,567' },
        ],
        active: [
          { symbol: '005930', name: '삼성전자', price: 68500, change: 2100, changePercent: 3.16, volume: '15,234,567', marketCap: '408.5조' },
          { symbol: '373220', name: 'LG에너지솔루션', price: 425000, change: -5000, changePercent: -1.16, volume: '12,345,678', marketCap: '99.8조' },
          { symbol: '000270', name: '기아', price: 82500, change: 1500, changePercent: 1.85, volume: '9,876,543', marketCap: '33.5조' },
          { symbol: '068270', name: '셀트리온', price: 172000, change: 3000, changePercent: 1.77, volume: '8,765,432', marketCap: '23.7조' },
          { symbol: '012330', name: '현대모비스', price: 215000, change: -3000, changePercent: -1.38, volume: '7,654,321', marketCap: '20.3조' },
        ]
      };

      // 종목 마스터 데이터 생성
      const allStocks = [...sampleStocks.gainers, ...sampleStocks.losers, ...sampleStocks.active];
      const uniqueStocks = Array.from(new Map(allStocks.map(s => [s.symbol, s])).values());

      for (const stock of uniqueStocks) {
        await prisma.stock_Symbol.upsert({
          where: { symbol: stock.symbol },
          update: {},
          create: {
            symbol: stock.symbol,
            nameKr: stock.name,
            market: stock.symbol.startsWith('0') ? 'KOSPI' : 'KOSDAQ',
            sector: '기술', // 임시 섹터
          }
        });
      }

      return NextResponse.json(sampleStocks[type as keyof typeof sampleStocks] || []);
    }

    // 실제 데이터 정렬 및 필터링
    let sortedStocks = stocks
      .filter(stock => stock.quotes.length > 0)
      .map(stock => ({
        symbol: stock.symbol,
        name: stock.nameKr,
        price: stock.quotes[0].close,
        change: stock.quotes[0].change,
        changePercent: stock.quotes[0].changePercent,
        volume: stock.quotes[0].volume.toString(),
        marketCap: stock.marketCap ? `${(Number(stock.marketCap) / 1000000000000).toFixed(1)}조` : undefined
      }));

    switch (type) {
      case 'gainers':
        sortedStocks = sortedStocks
          .filter(s => s.changePercent > 0)
          .sort((a, b) => b.changePercent - a.changePercent);
        break;
      case 'losers':
        sortedStocks = sortedStocks
          .filter(s => s.changePercent < 0)
          .sort((a, b) => a.changePercent - b.changePercent);
        break;
      case 'active':
        sortedStocks = sortedStocks
          .sort((a, b) => parseInt(b.volume) - parseInt(a.volume));
        break;
    }

    return NextResponse.json(sortedStocks.slice(0, limit));
  } catch (error) {
    console.error('Failed to fetch top stocks:', error);
    return NextResponse.json(
      { error: 'Failed to fetch top stocks' },
      { status: 500 }
    );
  } finally {
    await prisma.$disconnect();
  }
}