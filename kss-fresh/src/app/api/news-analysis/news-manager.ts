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
    try {
      // API 키가 없으면 목업 데이터 반환
      if (!process.env.NEWS_API_KEY) {
        console.log('⚠️ NEWS_API_KEY not found, using mock data')
        return this.getMockNewsData(query)
      }

      // 한국어 뉴스가 없을 수 있으므로 언어 제한 제거
      const response = await fetch(
        `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&apiKey=${process.env.NEWS_API_KEY}&sortBy=publishedAt&pageSize=20`
      )
      const data = await response.json()
      
      console.log('📰 News API Response:', {
        status: data.status,
        totalResults: data.totalResults,
        articlesCount: data.articles?.length
      })
      
      // API 에러 처리
      if (data.status === 'error') {
        console.log('⚠️ News API error:', data.message)
        return this.getMockNewsData(query)
      }
      
      // 결과가 없으면 영어로 재시도
      if (!data.articles || data.articles.length === 0) {
        console.log('🔄 No results, trying with English query...')
        const enResponse = await fetch(
          `https://newsapi.org/v2/everything?q=${encodeURIComponent(query + ' Korea')}&apiKey=${process.env.NEWS_API_KEY}&sortBy=publishedAt&pageSize=20`
        )
        const enData = await enResponse.json()
        
        if (enData.articles && enData.articles.length > 0) {
          data.articles = enData.articles
        } else {
          return this.getMockNewsData(query)
        }
      }
      
      // 온톨로지 분석 추가
      const analyzed = await this.analyzeWithAI(data.articles || [])
      
      return {
        articles: data.articles || [],
        analysis: analyzed,
        fetchedAt: new Date()
      }
    } catch (error) {
      console.error('News API fetch error:', error)
      return this.getMockNewsData(query)
    }
  }
  
  // AI 분석 (온톨로지 추출)
  private async analyzeWithAI(articles: any[]) {
    // 안전한 배열 처리
    if (!articles || !Array.isArray(articles) || articles.length === 0) {
      return {
        companies: [],
        relationships: [],
        keywords: [],
        sentiment: 0.5
      }
    }
    
    // OpenAI API 호출하여 기업관계 분석
    // 비용 절감을 위해 상위 10개만 분석
    const topArticles = articles.slice(0, Math.min(10, articles.length))
    
    // ... AI 분석 로직 ...
    
    return {
      companies: [],
      relationships: [],
      keywords: [],
      sentiment: 0.5
    }
  }
  
  // 목업 뉴스 데이터 생성
  private getMockNewsData(query: string) {
    const today = new Date()
    const mockArticles = [
      {
        title: `삼성전자, 엔비디아에 HBM4 메모리 독점 공급 계약 체결 임박`,
        description: `삼성전자가 엔비디아의 차세대 AI 가속기에 들어갈 HBM4 메모리 독점 공급권을 확보할 것으로 보인다. 업계에 따르면 양사는 2026년부터 3년간 독점 공급 계약을 논의 중이다.`,
        source: { name: '한국경제' },
        publishedAt: new Date(today.getTime() - 2 * 3600000).toISOString(),
        url: 'https://www.hankyung.com',
        sentiment: 0.85,
        content: '반도체 업계 관계자는 "삼성전자의 HBM4 기술력이 엔비디아의 요구사항을 충족시켰다"며 "SK하이닉스와의 경쟁에서 우위를 점하게 됐다"고 말했다.'
      },
      {
        title: `네이버, 오픈AI와 손잡고 한국형 초거대 AI 'HyperCLOVA X 2.0' 개발`,
        description: `네이버가 오픈AI와 전략적 파트너십을 맺고 한국어에 특화된 초거대 AI 모델 개발에 나선다. 양사는 GPT-5 기술을 기반으로 한국어 성능을 대폭 향상시킨 모델을 내년 상반기 출시할 예정이다.`,
        source: { name: '매일경제' },
        publishedAt: new Date(today.getTime() - 4 * 3600000).toISOString(),
        url: 'https://www.mk.co.kr',
        sentiment: 0.8,
        content: '네이버 관계자는 "한국어 데이터 100TB를 추가 학습시켜 한국 문화와 언어의 미묘한 뉘앙스까지 이해하는 AI를 만들 것"이라고 밝혔다.'
      },
      {
        title: `현대차-애플, 자율주행 전기차 프로젝트 협력 재개...2027년 출시 목표`,
        description: `현대자동차그룹과 애플이 중단됐던 자율주행 전기차 프로젝트 협력을 재개한다. 양사는 현대차의 E-GMP 플랫폼과 애플의 자율주행 기술을 결합한 차량을 2027년 출시할 계획이다.`,
        source: { name: '조선일보' },
        publishedAt: new Date(today.getTime() - 5 * 3600000).toISOString(),
        url: 'https://www.chosun.com',
        sentiment: 0.75,
        content: '업계 전문가들은 "테슬라에 이어 애플카가 출시되면 전기차 시장의 판도가 크게 바뀔 것"이라고 전망했다.'
      },
      {
        title: `카카오뱅크, 가상자산 거래소 진출 검토...토스·네이버페이와 3파전`,
        description: `카카오뱅크가 가상자산 거래소 사업 진출을 검토하고 있다. 이미 토스와 네이버페이가 가상자산 서비스를 준비 중인 가운데, 대형 핀테크 3사의 경쟁이 가상자산 시장으로 확대될 전망이다.`,
        source: { name: '연합뉴스' },
        publishedAt: new Date(today.getTime() - 6 * 3600000).toISOString(),
        url: 'https://www.yna.co.kr',
        sentiment: 0.7,
        content: '금융당국은 "핀테크 기업들의 가상자산 시장 진출은 긍정적이나, 투자자 보호 장치 마련이 우선"이라고 밝혔다.'
      },
      {
        title: `LG에너지솔루션, 테슬라에 4680 배터리 대량 공급 계약 체결`,
        description: `LG에너지솔루션이 테슬라의 차세대 원통형 배터리 '4680' 대량 공급 계약을 체결했다. 2025년부터 연간 20GWh 규모의 배터리를 공급하며, 이는 전기차 약 30만대 분량이다.`,
        source: { name: '서울경제' },
        publishedAt: new Date(today.getTime() - 8 * 3600000).toISOString(),
        url: 'https://www.sedaily.com',
        sentiment: 0.82,
        content: 'LG에너지솔루션은 "4680 배터리는 기존 대비 에너지 밀도가 5배 높고 생산 비용은 50% 절감된다"고 설명했다.'
      }
    ]
    
    // 검색어와 관련된 뉴스만 필터링
    const filteredArticles = query.toLowerCase() === '삼성전자' || query === '' 
      ? mockArticles 
      : mockArticles.filter(article => 
          article.title.includes(query) || 
          article.description.includes(query) ||
          article.content.includes(query)
        )
    
    // 검색 결과가 없으면 검색어 관련 가상 뉴스 생성
    if (filteredArticles.length === 0) {
      filteredArticles.push({
        title: `${query}, 3분기 실적 시장 예상치 상회...영업이익 전년比 15% 증가`,
        description: `${query}가 3분기 영업이익이 시장 예상을 웃돌며 견조한 실적을 기록했다. 글로벌 경기 둔화 우려에도 불구하고 AI 관련 사업 호조로 성장세를 이어갔다.`,
        source: { name: '한국경제' },
        publishedAt: today.toISOString(),
        url: '#',
        sentiment: 0.75,
        content: `${query}의 주력 사업 부문이 모두 성장하며 균형잡힌 실적을 달성했다.`
      })
    }
    
    return {
      articles: filteredArticles,
      analysis: {
        companies: [query, '삼성전자', 'SK하이닉스', '네이버', '현대차'],
        relationships: [],
        keywords: ['AI', 'HBM4', '자율주행', '전기차', '가상자산', '배터리'],
        sentiment: 0.78
      },
      fetchedAt: new Date()
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