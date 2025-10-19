/**
 * ArXiv Monitor - Status API
 * GET /api/arxiv-monitor/status
 */

import { NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function GET() {
  try {
    // 전체 통계
    const totalPapers = await prisma.arXiv_Paper.count()

    // 상태별 카운트
    const statusCounts = await prisma.arXiv_Paper.groupBy({
      by: ['status'],
      _count: true,
    })

    const statusMap: Record<string, number> = {}
    statusCounts.forEach((item) => {
      statusMap[item.status] = item._count
    })

    // 최근 논문 (5개)
    const recentPapers = await prisma.arXiv_Paper.findMany({
      orderBy: { createdAt: 'desc' },
      take: 5,
      select: {
        id: true,
        arxivId: true,
        title: true,
        status: true,
        createdAt: true,
        publishedDate: true,
      },
    })

    // 최근 처리 로그 (10개)
    const recentLogs = await prisma.arXiv_ProcessingLog.findMany({
      orderBy: { createdAt: 'desc' },
      take: 10,
      include: {
        paper: {
          select: {
            arxivId: true,
            title: true,
          },
        },
      },
    })

    // 비용 추정 (간단히 요약된 논문 수 기준)
    const summarizedCount = statusMap.SUMMARIZED || 0
    const mdxGeneratedCount = statusMap.MDX_GENERATED || 0
    const publishedCount = statusMap.PUBLISHED || 0
    const totalSummarized = summarizedCount + mdxGeneratedCount + publishedCount
    const estimatedCost = totalSummarized * 0.015 // $0.015 per paper

    await prisma.$disconnect()

    return NextResponse.json({
      success: true,
      data: {
        summary: {
          total: totalPapers,
          crawled: statusMap.CRAWLED || 0,
          summarized: statusMap.SUMMARIZED || 0,
          mdxGenerated: statusMap.MDX_GENERATED || 0,
          published: statusMap.PUBLISHED || 0,
          failed: statusMap.FAILED || 0,
          estimatedCost: estimatedCost.toFixed(3),
        },
        recentPapers,
        recentLogs,
      },
    })
  } catch (error) {
    await prisma.$disconnect()
    console.error('Error in status API:', error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
