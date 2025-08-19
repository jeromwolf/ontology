import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET() {
  try {
    // 시장 지수 조회
    const indices = await prisma.stock_MarketIndex.findMany({
      orderBy: {
        symbol: 'asc'
      }
    });

    // 시뮬레이션 데이터 (DB에 데이터가 없을 경우)
    if (indices.length === 0) {
      const defaultIndices = [
        { symbol: 'KOSPI', name: '코스피', value: 2501.23, change: 15.67, changePercent: 0.63 },
        { symbol: 'KOSDAQ', name: '코스닥', value: 698.45, change: -8.21, changePercent: -1.16 },
        { symbol: 'KOSPI200', name: '코스피200', value: 321.15, change: 2.34, changePercent: 0.73 },
        { symbol: 'DJI', name: '다우존스', value: 34721.12, change: 182.01, changePercent: 0.53 },
        { symbol: 'NASDAQ', name: '나스닥', value: 13711.00, change: -123.45, changePercent: -0.89 },
        { symbol: 'S&P500', name: 'S&P 500', value: 4488.28, change: 23.17, changePercent: 0.52 },
      ];

      // 시뮬레이션 데이터를 DB에 저장
      for (const index of defaultIndices) {
        await prisma.stock_MarketIndex.upsert({
          where: { symbol: index.symbol },
          update: {
            value: index.value,
            change: index.change,
            changePercent: index.changePercent,
          },
          create: index,
        });
      }

      return NextResponse.json(defaultIndices);
    }

    return NextResponse.json(indices);
  } catch (error) {
    console.error('Failed to fetch market indices:', error);
    return NextResponse.json(
      { error: 'Failed to fetch market indices' },
      { status: 500 }
    );
  } finally {
    await prisma.$disconnect();
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { symbol, value, change, changePercent } = body;

    const updatedIndex = await prisma.stock_MarketIndex.update({
      where: { symbol },
      data: {
        value,
        change,
        changePercent,
      },
    });

    return NextResponse.json(updatedIndex);
  } catch (error) {
    console.error('Failed to update market index:', error);
    return NextResponse.json(
      { error: 'Failed to update market index' },
      { status: 500 }
    );
  } finally {
    await prisma.$disconnect();
  }
}