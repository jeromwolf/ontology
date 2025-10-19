/**
 * ArXiv Monitor - Papers API
 * GET /api/arxiv-monitor/papers?status=...&limit=...
 */

import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const status = searchParams.get('status') || undefined
    const limit = parseInt(searchParams.get('limit') || '50')
    const offset = parseInt(searchParams.get('offset') || '0')

    // 필터 조건
    const where = status ? { status: status as any } : {}

    // 논문 목록 조회
    const papers = await prisma.arXiv_Paper.findMany({
      where,
      orderBy: { createdAt: 'desc' },
      take: limit,
      skip: offset,
      include: {
        processingLogs: {
          orderBy: { createdAt: 'desc' },
          take: 3,
        },
      },
    })

    // 전체 개수
    const total = await prisma.arXiv_Paper.count({ where })

    await prisma.$disconnect()

    return NextResponse.json({
      success: true,
      data: {
        papers,
        pagination: {
          total,
          limit,
          offset,
          hasMore: offset + limit < total,
        },
      },
    })
  } catch (error) {
    await prisma.$disconnect()
    console.error('Error in papers API:', error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
