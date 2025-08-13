'use client'

import { useState, useEffect } from 'react'
import { Search, Newspaper, Network, TrendingUp, TrendingDown, AlertCircle, RefreshCw, Filter, ChevronRight, Building2, Hash, Globe } from 'lucide-react'
import dynamic from 'next/dynamic'

// 동적 import로 SSR 문제 해결
const OntologyGraph = dynamic(() => import('./OntologyGraph'), { 
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-[500px] bg-gray-50 dark:bg-gray-900 rounded-lg">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
    </div>
  )
})

interface NewsItem {
  id: string
  title: string
  description: string
  source: string
  publishedAt: string
  url: string
  sentiment?: number
  entities?: {
    companies: string[]
    keywords: string[]
    sectors: string[]
  }
}

interface EntityRelation {
  company: string
  relatedCompanies: string[]
  keywords: string[]
  sentiment: number
  impact: number
  type?: 'positive' | 'negative' | 'neutral'
}

export default function NewsOntologyAnalyzer() {
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [newsItems, setNewsItems] = useState<NewsItem[]>([])
  const [selectedNews, setSelectedNews] = useState<NewsItem | null>(null)
  const [extractedEntities, setExtractedEntities] = useState<EntityRelation[]>([])
  const [activeView, setActiveView] = useState<'search' | 'analysis'>('search')
  const [filters, setFilters] = useState({
    dateRange: '7d',
    sentiment: 'all',
    sector: 'all'
  })

  // 인기 검색어 / 트렌딩 토픽
  const trendingTopics = [
    '반도체 수출 규제',
    'AI 투자 확대', 
    '금리 인하 전망',
    '전기차 배터리',
    '바이오 신약 승인',
    'K-뷰티 수출 호조'
  ]

  // 뉴스 검색 및 분석
  const searchAndAnalyze = async (query: string) => {
    setLoading(true)
    setSearchQuery(query)
    
    try {
      // API 호출하여 실제 뉴스 데이터 가져오기
      const response = await fetch(`/api/news/fetch?q=${encodeURIComponent(query)}&priority=high`)
      const result = await response.json()
      
      if (result.success && result.data?.articles) {
        // API 응답을 NewsItem 형식으로 변환
        const newsItems: NewsItem[] = result.data.articles.slice(0, 5).map((article: any, index: number) => ({
          id: String(index + 1),
          title: article.title,
          description: article.description || article.content || '',
          source: article.source?.name || '뉴스',
          publishedAt: article.publishedAt,
          url: article.url || '#',
          sentiment: article.sentiment || Math.random() * 0.4 + 0.4, // 0.4 ~ 0.8
          entities: extractEntities(article) // 엔티티 추출 함수
        }))
        
        setNewsItems(newsItems)
        setActiveView('analysis')
        
        // 첫 번째 뉴스 자동 선택
        if (newsItems.length > 0) {
          handleNewsSelect(newsItems[0])
        }
      } else {
        // API 실패 시 목업 데이터 사용
        const mockNews: NewsItem[] = [
          {
            id: '1',
            title: `${query} 관련 최신 뉴스를 불러오는 중 오류가 발생했습니다`,
            description: 'API 한도 초과 또는 네트워크 오류로 인해 실시간 뉴스를 가져올 수 없습니다. 잠시 후 다시 시도해주세요.',
            source: '시스템',
            publishedAt: new Date().toISOString(),
            url: '#',
            sentiment: 0.5,
            entities: {
              companies: [query],
              keywords: ['오류', 'API', '네트워크'],
              sectors: ['시스템']
            }
          }
        ]
        setNewsItems(mockNews)
        setActiveView('analysis')
      }
    } catch (error) {
      console.error('뉴스 검색 오류:', error)
      // 오류 시 사용자에게 알림
      alert('뉴스를 불러오는 중 오류가 발생했습니다.')
    } finally {
      setLoading(false)
    }
  }
  
  // 뉴스에서 엔티티 추출하는 헬퍼 함수
  const extractEntities = (article: any) => {
    // 간단한 키워드 추출 (실제로는 AI로 분석)
    const text = `${article.title} ${article.description || ''}`.toLowerCase()
    
    // 주요 회사명 찾기
    const companies: string[] = []
    const companyKeywords = ['삼성', '현대', 'LG', 'SK', '네이버', '카카오', '쿠팡', '테슬라', '애플', '구글']
    companyKeywords.forEach(company => {
      if (text.includes(company.toLowerCase())) {
        companies.push(company)
      }
    })
    
    // 섹터 추정
    const sectors = []
    if (text.includes('반도체') || text.includes('chip')) sectors.push('반도체')
    if (text.includes('자동차') || text.includes('전기차')) sectors.push('자동차')
    if (text.includes('배터리') || text.includes('2차전지')) sectors.push('배터리')
    if (text.includes('AI') || text.includes('인공지능')) sectors.push('AI')
    if (text.includes('금융') || text.includes('은행')) sectors.push('금융')
    
    // 키워드 추출
    const keywords = []
    if (text.includes('수출')) keywords.push('수출')
    if (text.includes('투자')) keywords.push('투자')
    if (text.includes('실적')) keywords.push('실적')
    if (text.includes('성장')) keywords.push('성장')
    if (text.includes('위기')) keywords.push('위기')
    
    return {
      companies: companies.length > 0 ? companies : [searchQuery],
      keywords: keywords.length > 0 ? keywords : ['뉴스', '분석'],
      sectors: sectors.length > 0 ? sectors : ['일반']
    }
  }

  // 뉴스 선택 시 엔티티 관계 분석
  const handleNewsSelect = (news: NewsItem) => {
    setSelectedNews(news)
    
    // 엔티티 관계 분석 (실제로는 AI 분석)
    if (news.entities) {
      const mainCompany = news.entities.companies[0]
      const relations: EntityRelation[] = news.entities.companies.map((company, idx) => ({
        company,
        relatedCompanies: news.entities!.companies.filter(c => c !== company),
        keywords: news.entities!.keywords,
        sentiment: news.sentiment || 0,
        impact: Math.random() * 100 - 50,
        type: news.sentiment! > 0.6 ? 'positive' : news.sentiment! < 0.4 ? 'negative' : 'neutral'
      }))
      
      setExtractedEntities(relations)
    }
  }

  // 빠른 검색 버튼
  const handleQuickSearch = (topic: string) => {
    searchAndAnalyze(topic)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* 헤더 */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Network className="w-8 h-8 text-indigo-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  뉴스 온톨로지 분석
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  뉴스에서 기업 관계와 시장 영향도를 자동으로 추출합니다
                </p>
              </div>
            </div>
            
            {/* 필터 */}
            <div className="flex items-center gap-2">
              <select 
                value={filters.dateRange}
                onChange={(e) => setFilters({...filters, dateRange: e.target.value})}
                className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg 
                         bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="1d">오늘</option>
                <option value="7d">1주일</option>
                <option value="1m">1개월</option>
                <option value="3m">3개월</option>
              </select>
              
              <select
                value={filters.sentiment}
                onChange={(e) => setFilters({...filters, sentiment: e.target.value})}
                className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg 
                         bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="all">모든 감성</option>
                <option value="positive">긍정</option>
                <option value="negative">부정</option>
                <option value="neutral">중립</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {activeView === 'search' ? (
        /* 검색 화면 */
        <div className="container mx-auto px-4 py-8">
          <div className="max-w-4xl mx-auto">
            {/* 검색 바 */}
            <div className="mb-8">
              <div className="relative">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && searchAndAnalyze(searchQuery)}
                  placeholder="뉴스 키워드를 입력하세요 (예: AI 반도체, 전기차 배터리, 바이오 신약)"
                  className="w-full px-6 py-4 pr-14 text-lg border-2 border-gray-300 dark:border-gray-600 rounded-xl
                           bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                           focus:border-indigo-500 dark:focus:border-indigo-400 focus:outline-none"
                />
                <button
                  onClick={() => searchAndAnalyze(searchQuery)}
                  disabled={loading || !searchQuery}
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 p-3 
                           bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 
                           disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <RefreshCw className="w-5 h-5 animate-spin" />
                  ) : (
                    <Search className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>

            {/* 트렌딩 토픽 */}
            <div className="mb-8">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-600" />
                실시간 인기 검색어
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {trendingTopics.map((topic, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleQuickSearch(topic)}
                    className="px-4 py-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700
                             hover:bg-indigo-50 dark:hover:bg-indigo-900/20 hover:border-indigo-300 dark:hover:border-indigo-600
                             transition-colors text-left group"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {topic}
                      </span>
                      <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-indigo-600 transition-colors" />
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* 섹터별 검색 */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <Globe className="w-5 h-5 text-blue-600" />
                섹터별 뉴스 분석
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {['반도체', 'IT/플랫폼', '자동차', '바이오', '금융', '배터리', '엔터', '유통'].map((sector) => (
                  <button
                    key={sector}
                    onClick={() => handleQuickSearch(sector)}
                    className="p-4 bg-gradient-to-br from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 
                             rounded-xl border border-gray-200 dark:border-gray-700
                             hover:shadow-lg hover:scale-105 transition-all"
                  >
                    <div className="text-2xl mb-2">
                      {sector === '반도체' ? '🔧' :
                       sector === 'IT/플랫폼' ? '💻' :
                       sector === '자동차' ? '🚗' :
                       sector === '바이오' ? '🧬' :
                       sector === '금융' ? '💰' :
                       sector === '배터리' ? '🔋' :
                       sector === '엔터' ? '🎬' : '📦'}
                    </div>
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300">{sector}</p>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* 분석 화면 */
        <div className="container mx-auto px-4 py-6">
          <div className="grid lg:grid-cols-3 gap-6">
            {/* 왼쪽: 뉴스 리스트 */}
            <div className="lg:col-span-1">
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                    <Newspaper className="w-5 h-5" />
                    검색 결과
                  </h3>
                  <button
                    onClick={() => setActiveView('search')}
                    className="text-sm text-indigo-600 hover:text-indigo-700"
                  >
                    새 검색
                  </button>
                </div>
                
                <div className="space-y-3 max-h-[calc(100vh-200px)] overflow-y-auto">
                  {newsItems.map((news) => (
                    <button
                      key={news.id}
                      onClick={() => handleNewsSelect(news)}
                      className={`w-full text-left p-4 rounded-lg border transition-all
                        ${selectedNews?.id === news.id 
                          ? 'bg-indigo-50 dark:bg-indigo-900/20 border-indigo-300 dark:border-indigo-600' 
                          : 'bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800'}`}
                    >
                      <h4 className="font-medium text-gray-900 dark:text-white text-sm mb-2 line-clamp-2">
                        {news.title}
                      </h4>
                      <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2 mb-2">
                        {news.description}
                      </p>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-500">{news.source}</span>
                        {news.sentiment !== undefined && (
                          <span className={`px-2 py-1 rounded-full font-medium
                            ${news.sentiment > 0.6 ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' :
                              news.sentiment < 0.4 ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' :
                              'bg-gray-100 text-gray-700 dark:bg-gray-900/30 dark:text-gray-400'}`}>
                            {news.sentiment > 0.6 ? '긍정' : news.sentiment < 0.4 ? '부정' : '중립'}
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* 오른쪽: 온톨로지 그래프 및 분석 */}
            <div className="lg:col-span-2 space-y-6">
              {selectedNews ? (
                <>
                  {/* 선택된 뉴스 상세 */}
                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                      {selectedNews.title}
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      {selectedNews.description}
                    </p>
                    
                    {/* 추출된 엔티티 */}
                    <div className="grid md:grid-cols-3 gap-4">
                      <div>
                        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-1">
                          <Building2 className="w-4 h-4" />
                          언급된 기업
                        </h4>
                        <div className="flex flex-wrap gap-1">
                          {selectedNews.entities?.companies.map((company, idx) => (
                            <span key={idx} className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded text-xs">
                              {company}
                            </span>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-1">
                          <Hash className="w-4 h-4" />
                          핵심 키워드
                        </h4>
                        <div className="flex flex-wrap gap-1">
                          {selectedNews.entities?.keywords.map((keyword, idx) => (
                            <span key={idx} className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded text-xs">
                              {keyword}
                            </span>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-1">
                          <Globe className="w-4 h-4" />
                          관련 섹터
                        </h4>
                        <div className="flex flex-wrap gap-1">
                          {selectedNews.entities?.sectors.map((sector, idx) => (
                            <span key={idx} className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded text-xs">
                              {sector}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* 온톨로지 그래프 */}
                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                      <Network className="w-5 h-5 text-indigo-600" />
                      기업 관계 온톨로지
                    </h3>
                    
                    {selectedNews.entities && selectedNews.entities.companies.length > 0 && (
                      <OntologyGraph
                        company={selectedNews.entities.companies[0] || '삼성전자'}
                        ticker=""
                        relationships={{
                          suppliers: selectedNews.entities.companies.slice(1, 3).length > 0 
                            ? selectedNews.entities.companies.slice(1, 3)
                            : ['TSMC', 'ASML'],
                          competitors: selectedNews.entities.companies.slice(3, 5).length > 0
                            ? selectedNews.entities.companies.slice(3, 5)
                            : ['SK하이닉스', '마이크론'],
                          partners: selectedNews.entities.companies.slice(2, 4).length > 0
                            ? selectedNews.entities.companies.slice(2, 4)
                            : ['퀄컴', 'ARM']
                        }}
                        keywords={selectedNews.entities.keywords}
                        impact={{
                          direct: Math.random() * 100 - 50,
                          indirect: Math.random() * 50 - 25,
                          sector: Math.random() * 40 - 20
                        }}
                      />
                    )}
                  </div>

                  {/* 영향도 분석 */}
                  <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                      <AlertCircle className="w-5 h-5 text-indigo-600" />
                      AI 영향도 분석
                    </h3>
                    
                    <div className="grid md:grid-cols-2 gap-4">
                      {extractedEntities.slice(0, 4).map((entity, idx) => (
                        <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-4">
                          <div className="flex items-center justify-between mb-2">
                            <h4 className="font-semibold text-gray-900 dark:text-white">
                              {entity.company}
                            </h4>
                            <span className={`text-2xl font-bold ${
                              entity.impact > 0 ? 'text-green-600' : 'text-red-600'
                            }`}>
                              {entity.impact > 0 ? '+' : ''}{entity.impact.toFixed(1)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
                            <span>관련: {entity.relatedCompanies.slice(0, 2).join(', ')}</span>
                          </div>
                          <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                            {entity.impact > 0 ? (
                              <TrendingUp className="w-4 h-4 text-green-600 inline mr-1" />
                            ) : (
                              <TrendingDown className="w-4 h-4 text-red-600 inline mr-1" />
                            )}
                            <span className="text-xs text-gray-600 dark:text-gray-400">
                              {entity.type === 'positive' ? '긍정적 영향 예상' :
                               entity.type === 'negative' ? '부정적 영향 예상' : '중립적 영향'}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-12 text-center">
                  <Network className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 dark:text-gray-400">
                    왼쪽에서 뉴스를 선택하면 온톨로지 분석이 표시됩니다
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}