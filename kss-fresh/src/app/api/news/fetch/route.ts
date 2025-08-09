import { NextRequest, NextResponse } from 'next/server'
import { newsManager } from '../../news-analysis/news-manager'

// GET: 뉴스 조회 (캐시 우선)
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const query = searchParams.get('q') || '삼성전자'
    const forceRefresh = searchParams.get('refresh') === 'true'
    const priority = searchParams.get('priority') as 'high' | 'medium' | 'low' || 'medium'

    // 시간대별 차등 업데이트 적용하여 뉴스 가져오기
    const result = await newsManager.getNews(query, {
      forceRefresh,
      priority
    })

    // 현재 업데이트 주기 정보 추가
    const stats = await newsManager.getStats()
    
    return NextResponse.json({
      success: true,
      query,
      updateFrequency: stats.currentUpdateFreq,
      nextUpdateIn: `${stats.currentUpdateFreq}분`,
      data: result,
      timestamp: new Date().toISOString()
    })
  } catch (error) {
    console.error('뉴스 조회 오류:', error)
    return NextResponse.json(
      { 
        success: false,
        error: '뉴스를 가져오는 중 오류가 발생했습니다.' 
      },
      { status: 500 }
    )
  }
}

// POST: 뉴스 분석 요청
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { query, companies, analyze = true } = body

    // 여러 회사 동시 분석
    if (companies && Array.isArray(companies)) {
      const results = await Promise.all(
        companies.map(company => 
          newsManager.getNews(company, { priority: 'high' })
        )
      )
      
      return NextResponse.json({
        success: true,
        companies,
        results,
        timestamp: new Date().toISOString()
      })
    }

    // 단일 쿼리 분석
    const result = await newsManager.getNews(query, {
      priority: 'high'
    })

    return NextResponse.json({
      success: true,
      query,
      data: result,
      timestamp: new Date().toISOString()
    })
  } catch (error) {
    console.error('뉴스 분석 오류:', error)
    return NextResponse.json(
      { 
        success: false,
        error: '뉴스 분석 중 오류가 발생했습니다.' 
      },
      { status: 500 }
    )
  }
}