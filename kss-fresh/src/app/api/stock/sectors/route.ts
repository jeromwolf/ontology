import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET() {
  try {
    // 섹터별 종목 데이터 조회
    const sectors = await prisma.stock_Symbol.groupBy({
      by: ['sector'],
      _count: {
        sector: true
      }
    });

    // 시뮬레이션 데이터 (DB에 데이터가 없을 경우)
    if (sectors.length === 0) {
      const sampleSectors = [
        { 
          name: '반도체', 
          changePercent: 2.85,
          leaders: [
            { symbol: '005930', name: '삼성전자', changePercent: 3.16 },
            { symbol: '000660', name: 'SK하이닉스', changePercent: 3.14 }
          ]
        },
        {
          name: '2차전지',
          changePercent: -1.23,
          leaders: [
            { symbol: '373220', name: 'LG에너지솔루션', changePercent: -1.16 },
            { symbol: '006400', name: '삼성SDI', changePercent: -2.33 }
          ]
        },
        {
          name: '바이오',
          changePercent: 1.45,
          leaders: [
            { symbol: '207940', name: '삼성바이오로직스', changePercent: 2.61 },
            { symbol: '068270', name: '셀트리온', changePercent: 1.77 }
          ]
        },
        {
          name: '금융',
          changePercent: -0.89,
          leaders: [
            { symbol: '105560', name: 'KB금융', changePercent: -2.07 },
            { symbol: '055550', name: '신한지주', changePercent: -0.52 }
          ]
        }
      ];

      return NextResponse.json(sampleSectors);
    }

    // 실제 섹터별 데이터 처리
    const sectorData = await Promise.all(
      sectors.map(async (sector) => {
        const stocks = await prisma.stock_Symbol.findMany({
          where: { sector: sector.sector },
          include: {
            quotes: {
              orderBy: { date: 'desc' },
              take: 1
            }
          },
          take: 2 // 섹터별 상위 2개 종목
        });

        const validStocks = stocks.filter(s => s.quotes.length > 0);
        const avgChange = validStocks.reduce((sum, s) => sum + s.quotes[0].changePercent, 0) / validStocks.length;

        return {
          name: sector.sector,
          changePercent: Number(avgChange.toFixed(2)),
          leaders: validStocks.map(s => ({
            symbol: s.symbol,
            name: s.nameKr,
            changePercent: s.quotes[0].changePercent
          }))
        };
      })
    );

    return NextResponse.json(sectorData);
  } catch (error) {
    console.error('Failed to fetch sectors:', error);
    return NextResponse.json(
      { error: 'Failed to fetch sectors' },
      { status: 500 }
    );
  } finally {
    await prisma.$disconnect();
  }
}