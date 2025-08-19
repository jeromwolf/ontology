import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { getServerSession } from 'next-auth';

const prisma = new PrismaClient();

// GET: 사용자의 포트폴리오 목록 조회
export async function GET(request: NextRequest) {
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

    const portfolios = await prisma.stock_Portfolio.findMany({
      where: { userId: user.id },
      include: {
        items: {
          include: {
            symbol: true
          }
        }
      },
      orderBy: { createdAt: 'desc' }
    });

    // 포트폴리오 통계 계산
    const portfoliosWithStats = portfolios.map(portfolio => {
      const totalValue = portfolio.items.reduce((sum, item) => {
        return sum + (item.quantity * item.currentPrice);
      }, 0);

      const totalCost = portfolio.items.reduce((sum, item) => {
        return sum + (item.quantity * item.purchasePrice);
      }, 0);

      const totalReturn = totalValue - totalCost;
      const returnPercent = totalCost > 0 ? (totalReturn / totalCost) * 100 : 0;

      return {
        ...portfolio,
        totalValue,
        totalCost,
        totalReturn,
        returnPercent
      };
    });

    return NextResponse.json(portfoliosWithStats);
  } catch (error) {
    console.error('Portfolio GET error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch portfolios' },
      { status: 500 }
    );
  }
}

// POST: 새 포트폴리오 생성
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
    const { name, description, initialCash = 0 } = body;

    if (!name) {
      return NextResponse.json({ error: 'Portfolio name is required' }, { status: 400 });
    }

    const portfolio = await prisma.stock_Portfolio.create({
      data: {
        userId: user.id,
        name,
        description,
        totalValue: initialCash,
        cash: initialCash,
        isActive: true
      }
    });

    return NextResponse.json(portfolio);
  } catch (error) {
    console.error('Portfolio POST error:', error);
    return NextResponse.json(
      { error: 'Failed to create portfolio' },
      { status: 500 }
    );
  }
}

// PUT: 포트폴리오 업데이트
export async function PUT(request: NextRequest) {
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
    const { id, name, description, isActive } = body;

    if (!id) {
      return NextResponse.json({ error: 'Portfolio ID is required' }, { status: 400 });
    }

    // 권한 확인
    const portfolio = await prisma.stock_Portfolio.findFirst({
      where: {
        id,
        userId: user.id
      }
    });

    if (!portfolio) {
      return NextResponse.json({ error: 'Portfolio not found' }, { status: 404 });
    }

    const updated = await prisma.stock_Portfolio.update({
      where: { id },
      data: {
        name,
        description,
        isActive
      }
    });

    return NextResponse.json(updated);
  } catch (error) {
    console.error('Portfolio PUT error:', error);
    return NextResponse.json(
      { error: 'Failed to update portfolio' },
      { status: 500 }
    );
  }
}

// DELETE: 포트폴리오 삭제
export async function DELETE(request: NextRequest) {
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

    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id) {
      return NextResponse.json({ error: 'Portfolio ID is required' }, { status: 400 });
    }

    // 권한 확인
    const portfolio = await prisma.stock_Portfolio.findFirst({
      where: {
        id,
        userId: user.id
      }
    });

    if (!portfolio) {
      return NextResponse.json({ error: 'Portfolio not found' }, { status: 404 });
    }

    // 관련 데이터 삭제 (Cascade)
    await prisma.$transaction([
      prisma.stock_Transaction.deleteMany({
        where: { portfolioId: id }
      }),
      prisma.stock_PortfolioItem.deleteMany({
        where: { portfolioId: id }
      }),
      prisma.stock_Portfolio.delete({
        where: { id }
      })
    ]);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Portfolio DELETE error:', error);
    return NextResponse.json(
      { error: 'Failed to delete portfolio' },
      { status: 500 }
    );
  }
}