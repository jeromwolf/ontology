import { NextResponse } from 'next/server'
import { newsManager } from '../../news-analysis/news-manager'

// 백그라운드 업데이트를 위한 크론잡 엔드포인트
// Vercel Cron 또는 외부 크론 서비스에서 호출

export async function GET(request: Request) {
  try {
    // 보안: 크론 시크릿 확인 (Vercel Cron 사용 시)
    const authHeader = request.headers.get('authorization')
    if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const now = new Date()
    const hour = now.getHours()
    const day = now.getDay()
    
    // 주말은 스킵 (또는 최소 업데이트)
    if (day === 0 || day === 6) {
      console.log('📅 주말 - 최소 업데이트 모드')
      // 주요 키워드만 업데이트
      const majorKeywords = ['삼성전자', 'SK하이닉스', '반도체']
      for (const keyword of majorKeywords) {
        await newsManager.getNews(keyword, { priority: 'low' })
      }
      
      return NextResponse.json({
        success: true,
        mode: 'weekend',
        updated: majorKeywords.length
      })
    }
    
    // 시간대별 업데이트 대상 결정
    let updateTargets: string[] = []
    let priority: 'high' | 'medium' | 'low' = 'medium'
    
    if (hour >= 8 && hour < 9) {
      // 장 시작 전: 주요 종목 집중 업데이트
      updateTargets = [
        '삼성전자', 'SK하이닉스', 'LG에너지솔루션',
        'NAVER', '카카오', '현대차', '기아',
        'POSCO홀딩스', 'LG화학', 'SK이노베이션'
      ]
      priority = 'high'
    } else if (hour >= 9 && hour < 16) {
      // 장중: 전체 종목 순환 업데이트
      updateTargets = [
        '삼성전자', 'SK하이닉스', 'LG에너지솔루션',
        'NAVER', '카카오', '현대차', '기아',
        'POSCO홀딩스', 'LG화학', 'SK이노베이션',
        '삼성바이오로직스', '셀트리온', '삼성SDI',
        'LG전자', 'SK텔레콤', 'KB금융', '신한금융',
        '하이브', 'CJ ENM', '넷마블'
      ]
      priority = 'high'
    } else if (hour >= 15 && hour < 16) {
      // 장 마감: 주요 종목 + 이슈 종목
      updateTargets = [
        '삼성전자', 'SK하이닉스', 'LG에너지솔루션',
        'AI 반도체', '전기차', '2차전지'
      ]
      priority = 'high'
    } else {
      // 장외시간: 섹터별 키워드 위주
      updateTargets = [
        '반도체', 'IT', '바이오', '2차전지',
        '금융', '자동차', '엔터테인먼트'
      ]
      priority = 'low'
    }
    
    // 순차적 업데이트 (API 제한 회피)
    const results = []
    for (const target of updateTargets) {
      try {
        await newsManager.getNews(target, { priority })
        results.push({ target, status: 'updated' })
        
        // API rate limit 회피를 위한 딜레이
        await new Promise(resolve => setTimeout(resolve, 1000))
      } catch (error) {
        console.error(`업데이트 실패: ${target}`, error)
        results.push({ target, status: 'failed' })
      }
    }
    
    // 캐시 정리
    await newsManager.cleanupCache()
    
    // 통계 업데이트
    const stats = await newsManager.getStats()
    
    return NextResponse.json({
      success: true,
      timestamp: now.toISOString(),
      marketHour: hour,
      priority,
      updated: results.filter(r => r.status === 'updated').length,
      failed: results.filter(r => r.status === 'failed').length,
      stats: {
        totalCached: stats.totalCached,
        apiCallsToday: stats.recentAPICalls,
        estimatedCost: stats.estimatedMonthlyCost
      }
    })
  } catch (error) {
    console.error('크론잡 오류:', error)
    return NextResponse.json(
      { 
        success: false,
        error: '크론잡 실행 중 오류가 발생했습니다.' 
      },
      { status: 500 }
    )
  }
}

// POST: 수동 업데이트 트리거
export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { targets = [], priority = 'medium' } = body
    
    // 관리자 권한 확인
    const authHeader = request.headers.get('authorization')
    if (authHeader !== `Bearer ${process.env.ADMIN_SECRET}`) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }
    
    const results = []
    for (const target of targets) {
      try {
        await newsManager.getNews(target, { priority, forceRefresh: true })
        results.push({ target, status: 'updated' })
      } catch (error) {
        results.push({ target, status: 'failed' })
      }
    }
    
    return NextResponse.json({
      success: true,
      manual: true,
      updated: results.filter(r => r.status === 'updated').length,
      failed: results.filter(r => r.status === 'failed').length,
      results
    })
  } catch (error) {
    return NextResponse.json(
      { success: false, error: '수동 업데이트 실패' },
      { status: 500 }
    )
  }
}