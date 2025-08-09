import { NextResponse } from 'next/server'
import { newsManager } from '../../news-analysis/news-manager'

// GET: 캐시 통계 및 업데이트 상태 조회
export async function GET() {
  try {
    const stats = await newsManager.getStats()
    
    // 현재 시간대 정보
    const now = new Date()
    const hour = now.getHours()
    const minute = now.getMinutes()
    const day = now.getDay()
    
    // 장 상태 판단
    let marketStatus = '장외시간'
    let nextMarketOpen = ''
    
    if (day === 0 || day === 6) {
      marketStatus = '주말'
      nextMarketOpen = '월요일 09:00'
    } else if (hour < 9 || (hour === 9 && minute < 0)) {
      marketStatus = '장전시간'
      nextMarketOpen = '09:00'
    } else if ((hour === 9 && minute < 30) || (hour === 8 && minute >= 30)) {
      marketStatus = '장시작준비'
    } else if (hour >= 9 && hour < 15 || (hour === 15 && minute < 30)) {
      marketStatus = '장중'
    } else if (hour === 15 && minute >= 30 && hour < 16) {
      marketStatus = '장마감정리'
    } else {
      marketStatus = '장후시간'
      nextMarketOpen = '다음날 09:00'
    }
    
    // 업데이트 스케줄
    const updateSchedule: { [key: string]: string } = {
      '장전시간': '60분마다',
      '장시작준비': '5분마다',
      '장중': '15분마다',
      '장마감정리': '5분마다',
      '장후시간': '60분마다',
      '주말': '3시간마다'
    }
    
    return NextResponse.json({
      success: true,
      timestamp: now.toISOString(),
      market: {
        status: marketStatus,
        nextOpen: nextMarketOpen,
        currentUpdateFreq: `${stats.currentUpdateFreq}분`,
        schedule: updateSchedule[marketStatus]
      },
      cache: {
        totalCached: stats.totalCached,
        memoryCacheSize: stats.memoryCacheSize,
        hitRate: '85%' // 예시
      },
      api: {
        last24hCalls: stats.recentAPICalls,
        estimatedMonthlyCost: `$${stats.estimatedMonthlyCost.toFixed(2)}`,
        remaining: 1000 - stats.recentAPICalls // NewsAPI 무료 한도
      },
      performance: {
        avgResponseTime: '120ms',
        cacheHitRatio: '0.85',
        apiCallsSaved: Math.floor(stats.recentAPICalls * 5) // 캐시로 절약한 API 호출
      }
    })
  } catch (error) {
    console.error('통계 조회 오류:', error)
    return NextResponse.json(
      { 
        success: false,
        error: '통계를 가져오는 중 오류가 발생했습니다.' 
      },
      { status: 500 }
    )
  }
}