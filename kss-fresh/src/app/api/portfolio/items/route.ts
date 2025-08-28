import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { getServerSession } from 'next-auth';

const prisma = new PrismaClient();

// POST: 포트폴리오에 종목 추가
export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession();
    
    if (!session?.user?.email) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const user = await prisma.user.findUnique({
      where: { email: session.user.email }
    });

    if (!user) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    const body = await request.json();
    const { 
      portfolioId, 
      symbolId, 
      quantity, 
      purchasePrice,
      purchaseDate,
      notes 
    } = body;

    // 포트폴리오 권한 확인
    const portfolio = await prisma.stock_Portfolio.findFirst({
      where: {
        id: portfolioId,
        userId: user.id
      }
    });

    if (!portfolio) {
      return NextResponse.json({ error: 'Portfolio not found' }, { status: 404 });
    }

    // 종목 정보 확인
    const symbol = await prisma.stock_Symbol.findUnique({
      where: { id: symbolId }
    });

    if (!symbol) {
      return NextResponse.json({ error: 'Symbol not found' }, { status: 404 });
    }

    // 기존 아이템 확인 (같은 종목이 있으면 수량 추가)
    const existingItem = await prisma.stock_PortfolioItem.findFirst({
      where: {
        portfolioId,
        stockId: symbolId
      }
    });

    let item;
    if (existingItem) {
      // 평균 단가 계산
      const totalQuantity = existingItem.quantity + quantity;
      const totalCost = (existingItem.quantity * existingItem.avgPrice) + (quantity * purchasePrice);
      const avgPrice = totalCost / totalQuantity;

      item = await prisma.stock_PortfolioItem.update({
        where: { id: existingItem.id },
        data: {
          quantity: totalQuantity,
          avgPrice: avgPrice,
          currentPrice: purchasePrice // 최신 가격으로 업데이트
        }
      });
    } else {
      // 새 아이템 생성
      item = await prisma.stock_PortfolioItem.create({
        data: {
          portfolioId,
          stockId: symbolId,
          quantity,
          avgPrice: purchasePrice,
          currentPrice: purchasePrice
        }
      });
    }

    // 거래 내역 기록
    await prisma.stock_Transaction.create({
      data: {
        portfolioId,
        symbol: symbolId,
        type: 'buy',
        quantity,
        price: purchasePrice
      }
    });

    // 포트폴리오 총 가치 업데이트
    const updatedItems = await prisma.stock_PortfolioItem.findMany({
      where: { portfolioId }
    });

    const totalValue = updatedItems.reduce((sum, item) => {
      return sum + (item.quantity * item.currentPrice);
    }, portfolio.cash);

    // 현재 스키마에는 totalValue 필드가 없으므로 업데이트 생략
    // totalValue 계산만 수행: ${totalValue}

    return NextResponse.json(item);
  } catch (error) {
    console.error('Portfolio item POST error:', error);
    return NextResponse.json(
      { error: 'Failed to add portfolio item' },
      { status: 500 }
    );
  }
}

// PUT: 포트폴리오 아이템 수정
export async function PUT(request: NextRequest) {
  try {
    const session = await getServerSession();
    
    if (!session?.user?.email) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await request.json();
    const { id, quantity, currentPrice, notes } = body;

    if (!id) {
      return NextResponse.json({ error: 'Item ID is required' }, { status: 400 });
    }

    // 권한 확인
    const item = await prisma.stock_PortfolioItem.findUnique({
      where: { id },
      include: {
        portfolio: true
      }
    });

    if (!item) {
      return NextResponse.json({ error: 'Item not found' }, { status: 404 });
    }

    const user = await prisma.user.findUnique({
      where: { email: session.user.email }
    });

    if (item.portfolio.userId !== user?.id) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const updated = await prisma.stock_PortfolioItem.update({
      where: { id },
      data: {
        quantity,
        currentPrice
      }
    });

    // 포트폴리오 총 가치 업데이트
    const updatedItems = await prisma.stock_PortfolioItem.findMany({
      where: { portfolioId: item.portfolioId }
    });

    const totalValue = updatedItems.reduce((sum, item) => {
      return sum + (item.quantity * item.currentPrice);
    }, item.portfolio.cash);

    // 포트폴리오 총 가치 업데이트는 스키마에 totalValue 필드가 없으므로 생략

    return NextResponse.json(updated);
  } catch (error) {
    console.error('Portfolio item PUT error:', error);
    return NextResponse.json(
      { error: 'Failed to update portfolio item' },
      { status: 500 }
    );
  }
}

// DELETE: 포트폴리오에서 종목 제거
export async function DELETE(request: NextRequest) {
  try {
    const session = await getServerSession();
    
    if (!session?.user?.email) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id) {
      return NextResponse.json({ error: 'Item ID is required' }, { status: 400 });
    }

    // 권한 확인
    const item = await prisma.stock_PortfolioItem.findUnique({
      where: { id },
      include: {
        portfolio: true
      }
    });

    if (!item) {
      return NextResponse.json({ error: 'Item not found' }, { status: 404 });
    }

    const user = await prisma.user.findUnique({
      where: { email: session.user.email }
    });

    if (item.portfolio.userId !== user?.id) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // 매도 거래 기록
    await prisma.stock_Transaction.create({
      data: {
        portfolioId: item.portfolioId,
        symbol: item.stockId,
        type: 'sell',
        quantity: item.quantity,
        price: item.currentPrice
      }
    });

    // 아이템 삭제
    await prisma.stock_PortfolioItem.delete({
      where: { id }
    });

    // 포트폴리오 총 가치 업데이트
    const updatedItems = await prisma.stock_PortfolioItem.findMany({
      where: { portfolioId: item.portfolioId }
    });

    const totalValue = updatedItems.reduce((sum, item) => {
      return sum + (item.quantity * item.currentPrice);
    }, item.portfolio.cash);

    // 포트폴리오 총 가치 업데이트는 스키마에 totalValue 필드가 없으므로 생략

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Portfolio item DELETE error:', error);
    return NextResponse.json(
      { error: 'Failed to delete portfolio item' },
      { status: 500 }
    );
  }
}