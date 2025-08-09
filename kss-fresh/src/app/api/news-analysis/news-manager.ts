// 뉴스 관리 시스템 - 하이브리드 캐싱 전략

// Prisma는 나중에 활성화 (DB 설정 후)
// import { PrismaClient } from '@prisma/client'
// const prisma = new PrismaClient()

interface NewsCache {
  id: string
  query: string
  data: any
  createdAt: Date
  updateAt: Date
  priority: 'high' | 'medium' | 'low'
  updateFrequency: number // 분 단위
}

export class NewsManager {
  private memoryCache = new Map<string, { data: any, timestamp: number }>()
  
  // 시간대별 업데이트 주기 (분)
  private getUpdateFrequency(): number {
    const now = new Date()
    const hour = now.getHours()
    const minute = now.getMinutes()
    const day = now.getDay()
    
    // 주말
    if (day === 0 || day === 6) {
      return 180 // 3시간
    }
    
    // 장 시작 전후 (8:30-9:30)
    if ((hour === 8 && minute >= 30) || (hour === 9 && minute < 30)) {
      return 5
    }
    
    // 장중 (9:30-15:30)
    if ((hour === 9 && minute >= 30) || (hour > 9 && hour < 15) || (hour === 15 && minute < 30)) {
      return 15
    }
    
    // 장 마감 (15:30-16:00)
    if (hour === 15 && minute >= 30) {
      return 5
    }
    
    // 장외시간
    return 60
  }

  // 뉴스 가져오기 (캐시 우선)
  async getNews(query: string, options: {
    forceRefresh?: boolean
    priority?: 'high' | 'medium' | 'low'
  } = {}) {
    const cacheKey = `news:${query}`
    const updateFreq = this.getUpdateFrequency()
    
    // 1. 메모리 캐시 확인 (1분 이내)
    const memCache = this.memoryCache.get(cacheKey)
    if (memCache && !options.forceRefresh) {
      const age = Date.now() - memCache.timestamp
      if (age < 60000) { // 1분
        console.log('📦 Memory cache hit')
        return memCache.data
      }
    }
    
    // 2. DB 캐시 확인 (Prisma 설정 후 활성화)
    // const dbCache = await prisma.newsCache.findUnique({
    //   where: { query }
    // })
    
    // if (dbCache && !options.forceRefresh) {
    //   const age = Date.now() - dbCache.updatedAt.getTime()
    //   const maxAge = updateFreq * 60 * 1000
      
    //   if (age < maxAge) {
    //     console.log('💾 DB cache hit')
    //     // 메모리 캐시 업데이트
    //     this.memoryCache.set(cacheKey, {
    //       data: dbCache.data,
    //       timestamp: Date.now()
    //     })
    //     return dbCache.data
    //   }
    // }
    
    // 3. API 호출 (캐시 미스 또는 강제 갱신)
    console.log('🌐 Fetching fresh news...')
    const freshData = await this.fetchFromAPI(query)
    
    // 4. 캐시 업데이트
    await this.updateCache(query, freshData, options.priority || 'medium')
    
    return freshData
  }
  
  // API에서 뉴스 가져오기
  private async fetchFromAPI(query: string) {
    const response = await fetch(
      `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&apiKey=${process.env.NEWS_API_KEY}`
    )
    const data = await response.json()
    
    // 온톨로지 분석 추가
    const analyzed = await this.analyzeWithAI(data.articles)
    
    return {
      articles: data.articles,
      analysis: analyzed,
      fetchedAt: new Date()
    }
  }
  
  // AI 분석 (온톨로지 추출)
  private async analyzeWithAI(articles: any[]) {
    // OpenAI API 호출하여 기업관계 분석
    // 비용 절감을 위해 상위 10개만 분석
    const topArticles = articles.slice(0, 10)
    
    // ... AI 분석 로직 ...
    
    return {
      companies: [],
      relationships: [],
      keywords: [],
      sentiment: 0
    }
  }
  
  // 캐시 업데이트
  private async updateCache(query: string, data: any, priority: string) {
    // 메모리 캐시
    this.memoryCache.set(`news:${query}`, {
      data,
      timestamp: Date.now()
    })
    
    // DB 캐시 (Prisma 설정 후 활성화)
    // await prisma.newsCache.upsert({
    //   where: { query },
    //   update: {
    //     data,
    //     updatedAt: new Date(),
    //     priority
    //   },
    //   create: {
    //     query,
    //     data,
    //     priority,
    //     updateFrequency: this.getUpdateFrequency()
    //   }
    // })
  }
  
  // 백그라운드 업데이트 (중요 키워드)
  async backgroundUpdate() {
    const importantQueries = [
      '삼성전자',
      'SK하이닉스', 
      'LG에너지솔루션',
      '현대차',
      'NAVER',
      '카카오',
      '반도체',
      'AI'
    ]
    
    for (const query of importantQueries) {
      await this.getNews(query)
      // API 제한 회피를 위한 딜레이
      await new Promise(resolve => setTimeout(resolve, 2000))
    }
  }
  
  // 캐시 정리 (오래된 데이터 삭제)
  async cleanupCache() {
    // DB 캐시 정리 (Prisma 설정 후 활성화)
    // const threeDaysAgo = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000)
    
    // await prisma.newsCache.deleteMany({
    //   where: {
    //     updatedAt: {
    //       lt: threeDaysAgo
    //     }
    //   }
    // })
    
    // 메모리 캐시 정리
    for (const [key, value] of this.memoryCache.entries()) {
      if (Date.now() - value.timestamp > 3600000) { // 1시간
        this.memoryCache.delete(key)
      }
    }
  }
  
  // 통계 및 모니터링
  async getStats() {
    // DB 통계 (Prisma 설정 후 활성화)
    // const totalCached = await prisma.newsCache.count()
    // const recentAPICalls = await prisma.apiCallLog.count({
    //   where: {
    //     createdAt: {
    //       gte: new Date(Date.now() - 24 * 60 * 60 * 1000)
    //     }
    //   }
    // })
    
    // 임시 모의 데이터
    const totalCached = this.memoryCache.size * 10 // 추정
    const recentAPICalls = 50 // 추정
    
    return {
      totalCached,
      memoryCacheSize: this.memoryCache.size,
      recentAPICalls,
      currentUpdateFreq: this.getUpdateFrequency(),
      estimatedMonthlyCost: (recentAPICalls * 30) * 0.001 // $0.001 per call
    }
  }
}

// Singleton 인스턴스
export const newsManager = new NewsManager()

// Cron Job 설정 (Next.js API route로 구현)
export async function setupCronJobs() {
  // 5분마다 중요 키워드 업데이트 (장중에만)
  setInterval(async () => {
    const hour = new Date().getHours()
    if (hour >= 9 && hour < 16) {
      await newsManager.backgroundUpdate()
    }
  }, 5 * 60 * 1000)
  
  // 1시간마다 캐시 정리
  setInterval(async () => {
    await newsManager.cleanupCache()
  }, 60 * 60 * 1000)
}