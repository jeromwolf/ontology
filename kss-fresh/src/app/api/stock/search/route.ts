import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// 한국 주요 종목 데이터 (초기 데이터)
const KOREAN_STOCKS = [
  { code: '005930', name: '삼성전자', market: 'KOSPI', sector: '전기전자' },
  { code: '000660', name: 'SK하이닉스', market: 'KOSPI', sector: '전기전자' },
  { code: '035420', name: 'NAVER', market: 'KOSPI', sector: 'IT' },
  { code: '035720', name: '카카오', market: 'KOSPI', sector: 'IT' },
  { code: '207940', name: '삼성바이오로직스', market: 'KOSPI', sector: '바이오' },
  { code: '068270', name: '셀트리온', market: 'KOSPI', sector: '바이오' },
  { code: '005380', name: '현대차', market: 'KOSPI', sector: '자동차' },
  { code: '051910', name: 'LG화학', market: 'KOSPI', sector: '화학' },
  { code: '006400', name: '삼성SDI', market: 'KOSPI', sector: '전기전자' },
  { code: '373220', name: 'LG에너지솔루션', market: 'KOSPI', sector: '전기전자' },
  { code: '000270', name: '기아', market: 'KOSPI', sector: '자동차' },
  { code: '012330', name: '현대모비스', market: 'KOSPI', sector: '자동차' },
  { code: '105560', name: 'KB금융', market: 'KOSPI', sector: '금융' },
  { code: '055550', name: '신한지주', market: 'KOSPI', sector: '금융' },
  { code: '028260', name: '삼성물산', market: 'KOSPI', sector: '유통' },
  { code: '003670', name: '포스코퓨처엠', market: 'KOSPI', sector: '화학' },
  { code: '066570', name: 'LG전자', market: 'KOSPI', sector: '전기전자' },
  { code: '034730', name: 'SK이노베이션', market: 'KOSPI', sector: '화학' },
  { code: '015760', name: '한국전력', market: 'KOSPI', sector: '전기가스' },
  { code: '032830', name: '삼성생명', market: 'KOSPI', sector: '보험' },
  { code: '003550', name: 'LG', market: 'KOSPI', sector: '지주회사' },
  { code: '034220', name: 'LG디스플레이', market: 'KOSPI', sector: '전기전자' },
  { code: '010130', name: '고려아연', market: 'KOSPI', sector: '비철금속' },
  { code: '009150', name: '삼성전기', market: 'KOSPI', sector: '전기전자' },
  { code: '086790', name: '하나금융지주', market: 'KOSPI', sector: '금융' },
  { code: '017670', name: 'SK텔레콤', market: 'KOSPI', sector: '통신' },
  { code: '316140', name: '우리금융지주', market: 'KOSPI', sector: '금융' },
  { code: '030200', name: 'KT', market: 'KOSPI', sector: '통신' },
  { code: '352820', name: '하이브', market: 'KOSPI', sector: '엔터테인먼트' },
  { code: '005490', name: 'POSCO홀딩스', market: 'KOSPI', sector: '철강' },
];

// GET: 종목 검색
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const query = searchParams.get('q')?.toLowerCase() || '';
    const limit = parseInt(searchParams.get('limit') || '10');
    
    if (!query || query.length < 1) {
      return NextResponse.json({ results: [] });
    }

    // 먼저 DB에서 검색
    let results = await prisma.stock_Symbol.findMany({
      where: {
        OR: [
          { code: { contains: query, mode: 'insensitive' } },
          { name: { contains: query, mode: 'insensitive' } },
        ]
      },
      take: limit,
      orderBy: [
        { marketCap: 'desc' },
        { name: 'asc' }
      ]
    });

    // DB에 데이터가 없으면 초기 데이터로 검색
    if (results.length === 0) {
      results = KOREAN_STOCKS
        .filter(stock => 
          stock.code.toLowerCase().includes(query) ||
          stock.name.toLowerCase().includes(query)
        )
        .slice(0, limit)
        .map(stock => ({
          id: stock.code,
          code: stock.code,
          name: stock.name,
          market: stock.market,
          sector: stock.sector,
          currentPrice: 0,
          marketCap: 0,
          isActive: true,
          createdAt: new Date(),
          updatedAt: new Date()
        }));
    }

    // 결과 포맷팅
    const formattedResults = results.map(stock => ({
      code: stock.code,
      name: stock.name,
      market: stock.market,
      sector: stock.sector,
      display: `${stock.code} - ${stock.name}`
    }));

    return NextResponse.json({ 
      results: formattedResults,
      total: formattedResults.length 
    });
  } catch (error) {
    console.error('Stock search error:', error);
    return NextResponse.json(
      { error: 'Failed to search stocks' },
      { status: 500 }
    );
  }
}

// POST: 종목 데이터 초기화 (관리자용)
export async function POST(request: NextRequest) {
  try {
    // 초기 데이터 DB에 저장
    const existingCount = await prisma.stock_Symbol.count();
    
    if (existingCount === 0) {
      // 초기 데이터가 없을 때만 삽입
      const symbols = KOREAN_STOCKS.map(stock => ({
        code: stock.code,
        name: stock.name,
        market: stock.market,
        sector: stock.sector,
        currentPrice: 0,
        marketCap: 0,
        isActive: true
      }));

      await prisma.stock_Symbol.createMany({
        data: symbols,
        skipDuplicates: true
      });

      return NextResponse.json({ 
        message: 'Stock symbols initialized',
        count: symbols.length 
      });
    }

    return NextResponse.json({ 
      message: 'Stock symbols already exist',
      count: existingCount 
    });
  } catch (error) {
    console.error('Stock initialization error:', error);
    return NextResponse.json(
      { error: 'Failed to initialize stocks' },
      { status: 500 }
    );
  }
}