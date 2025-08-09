'use client'

import { useState, useEffect } from 'react'
import { TrendingUp, TrendingDown, AlertCircle, Newspaper, Building2, Network, BarChart3, Brain, RefreshCw } from 'lucide-react'
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

// 뉴스 영향도 온톨로지 타입
interface NewsOntology {
  company: string
  ticker: string
  relatedCompanies: string[]
  industry: string
  keywords: string[]
  sentiment: number
  impact: {
    direct: number
    indirect: number
    sector: number
  }
  relationships: {
    suppliers: string[]
    competitors: string[]
    partners: string[]
  }
}

interface NewsItem {
  title: string
  description: string
  source: string
  publishedAt: string
  url: string
  sentiment?: number
}

interface StockInfo {
  ticker: string
  name: string
  price: number
  change: number
  changePercent: number
}

export default function NewsImpactAnalyzer() {
  const [selectedCompany, setSelectedCompany] = useState('삼성전자')
  const [selectedTicker, setSelectedTicker] = useState('005930')
  const [loading, setLoading] = useState(false)
  const [newsData, setNewsData] = useState<NewsItem[]>([])
  const [ontologyData, setOntologyData] = useState<NewsOntology | null>(null)
  const [stockInfo, setStockInfo] = useState<StockInfo | null>(null)
  const [activeTab, setActiveTab] = useState<'news' | 'ontology' | 'impact'>('news')

  // 주요 종목 리스트
  const majorStocks = [
    { name: '삼성전자', ticker: '005930' },
    { name: 'SK하이닉스', ticker: '000660' },
    { name: 'LG에너지솔루션', ticker: '373220' },
    { name: 'NAVER', ticker: '035420' },
    { name: '카카오', ticker: '035720' },
    { name: '현대차', ticker: '005380' },
    { name: '기아', ticker: '000270' },
    { name: 'POSCO홀딩스', ticker: '005490' },
  ]

  // 뉴스 분석 API 호출
  const analyzeNews = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/news-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          company: selectedCompany,
          ticker: selectedTicker
        })
      })

      if (response.ok) {
        const data = await response.json()
        setNewsData(data.news || [])
        setOntologyData(data.ontologyAnalysis)
        setStockInfo(data.currentPrice)
      }
    } catch (error) {
      console.error('뉴스 분석 실패:', error)
      // 모의 데이터로 테스트
      setMockData()
    } finally {
      setLoading(false)
    }
  }

  // 모의 데이터 설정 (API 연결 전 테스트용)
  const setMockData = () => {
    setNewsData([
      {
        title: "삼성전자, AI 반도체 대규모 투자 발표",
        description: "삼성전자가 차세대 AI 반도체 개발에 10조원 규모의 투자를 발표했다.",
        source: "한국경제",
        publishedAt: new Date().toISOString(),
        url: "#",
        sentiment: 0.8
      },
      {
        title: "반도체 시장 회복세 전망",
        description: "글로벌 반도체 시장이 2025년 하반기부터 본격적인 회복세에 들어설 전망이다.",
        source: "매일경제",
        publishedAt: new Date().toISOString(),
        url: "#",
        sentiment: 0.6
      },
      {
        title: "미중 무역 갈등 재점화 우려",
        description: "미중 무역 갈등이 다시 격화될 조짐을 보이며 반도체 업계에 불확실성이 커지고 있다.",
        source: "연합뉴스",
        publishedAt: new Date().toISOString(),
        url: "#",
        sentiment: -0.4
      }
    ])

    setOntologyData({
      company: selectedCompany,
      ticker: selectedTicker,
      relatedCompanies: ['SK하이닉스', 'TSMC', 'Intel'],
      industry: '반도체/전자',
      keywords: ['AI', '반도체', 'HBM', '파운드리', '투자'],
      sentiment: 0.65,
      impact: {
        direct: 45,
        indirect: 25,
        sector: 35
      },
      relationships: {
        suppliers: ['ASML', '도쿄일렉트론', '램리서치'],
        competitors: ['TSMC', 'Intel', 'SK하이닉스'],
        partners: ['엔비디아', 'AMD', '구글']
      }
    })

    setStockInfo({
      ticker: selectedTicker,
      name: selectedCompany,
      price: 72500,
      change: 1200,
      changePercent: 1.68
    })
  }

  // 영향도 색상 결정
  const getImpactColor = (impact: number) => {
    if (impact > 50) return 'text-green-600 dark:text-green-400'
    if (impact > 20) return 'text-green-500 dark:text-green-500'
    if (impact > -20) return 'text-gray-600 dark:text-gray-400'
    if (impact > -50) return 'text-red-500 dark:text-red-500'
    return 'text-red-600 dark:text-red-400'
  }

  // 영향도 배경색 결정
  const getImpactBg = (impact: number) => {
    if (impact > 50) return 'bg-green-100 dark:bg-green-900/30'
    if (impact > 20) return 'bg-green-50 dark:bg-green-900/20'
    if (impact > -20) return 'bg-gray-50 dark:bg-gray-900/20'
    if (impact > -50) return 'bg-red-50 dark:bg-red-900/20'
    return 'bg-red-100 dark:bg-red-900/30'
  }

  // 감성 점수 표시
  const getSentimentDisplay = (sentiment: number) => {
    const percentage = Math.round(Math.abs(sentiment) * 100)
    if (sentiment > 0.5) return { text: '매우 긍정', color: 'text-green-600', icon: '😊' }
    if (sentiment > 0) return { text: '긍정', color: 'text-green-500', icon: '🙂' }
    if (sentiment > -0.5) return { text: '중립', color: 'text-gray-600', icon: '😐' }
    return { text: '부정', color: 'text-red-600', icon: '😟' }
  }

  useEffect(() => {
    // 컴포넌트 마운트 시 모의 데이터 로드
    setMockData()
  }, [])

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
          <Brain className="w-7 h-7 text-indigo-600" />
          뉴스 영향도 온톨로지 분석
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          AI 기반 뉴스 분석으로 기업간 관계와 시장 영향도를 시각화합니다
        </p>
      </div>

      {/* 종목 선택 */}
      <div className="mb-6 flex gap-4 items-end">
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            종목 선택
          </label>
          <select
            value={selectedTicker}
            onChange={(e) => {
              const stock = majorStocks.find(s => s.ticker === e.target.value)
              if (stock) {
                setSelectedTicker(stock.ticker)
                setSelectedCompany(stock.name)
              }
            }}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                     bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            {majorStocks.map(stock => (
              <option key={stock.ticker} value={stock.ticker}>
                {stock.name} ({stock.ticker})
              </option>
            ))}
          </select>
        </div>
        <button
          onClick={analyzeNews}
          disabled={loading}
          className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 
                   disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              분석 중...
            </>
          ) : (
            <>
              <Brain className="w-4 h-4" />
              뉴스 분석
            </>
          )}
        </button>
      </div>

      {/* 현재 주가 정보 */}
      {stockInfo && (
        <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white text-lg">
                {stockInfo.name}
              </h4>
              <p className="text-gray-600 dark:text-gray-400 text-sm">{stockInfo.ticker}</p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {stockInfo.price.toLocaleString()}원
              </p>
              <p className={`text-sm font-medium ${stockInfo.change > 0 ? 'text-red-600' : 'text-blue-600'}`}>
                {stockInfo.change > 0 ? '▲' : '▼'} {Math.abs(stockInfo.change).toLocaleString()}원 
                ({stockInfo.changePercent > 0 ? '+' : ''}{stockInfo.changePercent}%)
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 탭 네비게이션 */}
      <div className="flex gap-2 mb-6 border-b border-gray-200 dark:border-gray-700">
        <button
          onClick={() => setActiveTab('news')}
          className={`px-4 py-2 font-medium transition-colors relative
            ${activeTab === 'news' 
              ? 'text-indigo-600 dark:text-indigo-400' 
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'}`}
        >
          <span className="flex items-center gap-2">
            <Newspaper className="w-4 h-4" />
            최신 뉴스
          </span>
          {activeTab === 'news' && (
            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-600 dark:bg-indigo-400" />
          )}
        </button>
        <button
          onClick={() => setActiveTab('ontology')}
          className={`px-4 py-2 font-medium transition-colors relative
            ${activeTab === 'ontology' 
              ? 'text-indigo-600 dark:text-indigo-400' 
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'}`}
        >
          <span className="flex items-center gap-2">
            <Network className="w-4 h-4" />
            온톨로지 관계
          </span>
          {activeTab === 'ontology' && (
            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-600 dark:bg-indigo-400" />
          )}
        </button>
        <button
          onClick={() => setActiveTab('impact')}
          className={`px-4 py-2 font-medium transition-colors relative
            ${activeTab === 'impact' 
              ? 'text-indigo-600 dark:text-indigo-400' 
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'}`}
        >
          <span className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            영향도 분석
          </span>
          {activeTab === 'impact' && (
            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-600 dark:bg-indigo-400" />
          )}
        </button>
      </div>

      {/* 뉴스 탭 */}
      {activeTab === 'news' && (
        <div className="space-y-4">
          {newsData.map((news, idx) => {
            const sentiment = getSentimentDisplay(news.sentiment || 0)
            return (
              <div key={idx} className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-semibold text-gray-900 dark:text-white flex-1">
                    {news.title}
                  </h4>
                  <span className={`text-sm font-medium ${sentiment.color} flex items-center gap-1`}>
                    <span className="text-lg">{sentiment.icon}</span>
                    {sentiment.text}
                  </span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  {news.description}
                </p>
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-500">
                  <span>{news.source}</span>
                  <span>{new Date(news.publishedAt).toLocaleDateString()}</span>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* 온톨로지 관계 탭 */}
      {activeTab === 'ontology' && ontologyData && (
        <div className="space-y-6">
          {/* 관련 기업 네트워크 */}
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-blue-900 dark:text-blue-300 mb-3 flex items-center gap-2">
                <Building2 className="w-4 h-4" />
                공급업체
              </h5>
              <div className="space-y-2">
                {ontologyData.relationships.suppliers.map((company, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">{company}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-red-900 dark:text-red-300 mb-3 flex items-center gap-2">
                <Building2 className="w-4 h-4" />
                경쟁사
              </h5>
              <div className="space-y-2">
                {ontologyData.relationships.competitors.map((company, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-red-500 rounded-full" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">{company}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-green-900 dark:text-green-300 mb-3 flex items-center gap-2">
                <Building2 className="w-4 h-4" />
                파트너사
              </h5>
              <div className="space-y-2">
                {ontologyData.relationships.partners.map((company, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">{company}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 주요 키워드 */}
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h5 className="font-semibold text-purple-900 dark:text-purple-300 mb-3">
              주요 키워드
            </h5>
            <div className="flex flex-wrap gap-2">
              {ontologyData.keywords.map((keyword, idx) => (
                <span key={idx} className="px-3 py-1 bg-purple-100 dark:bg-purple-800 text-purple-700 dark:text-purple-300 rounded-full text-sm">
                  #{keyword}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 영향도 분석 탭 */}
      {activeTab === 'impact' && ontologyData && (
        <div className="space-y-6">
          {/* 영향도 시각화 */}
          <div className="grid md:grid-cols-3 gap-4">
            <div className={`rounded-lg p-6 ${getImpactBg(ontologyData.impact.direct)}`}>
              <div className="text-center">
                <div className={`text-4xl font-bold ${getImpactColor(ontologyData.impact.direct)}`}>
                  {ontologyData.impact.direct > 0 ? '+' : ''}{ontologyData.impact.direct}
                </div>
                <p className="text-gray-600 dark:text-gray-400 mt-2">직접 영향도</p>
                <div className="mt-3">
                  {ontologyData.impact.direct > 0 ? (
                    <TrendingUp className="w-8 h-8 mx-auto text-green-600 dark:text-green-400" />
                  ) : (
                    <TrendingDown className="w-8 h-8 mx-auto text-red-600 dark:text-red-400" />
                  )}
                </div>
              </div>
            </div>

            <div className={`rounded-lg p-6 ${getImpactBg(ontologyData.impact.indirect)}`}>
              <div className="text-center">
                <div className={`text-4xl font-bold ${getImpactColor(ontologyData.impact.indirect)}`}>
                  {ontologyData.impact.indirect > 0 ? '+' : ''}{ontologyData.impact.indirect}
                </div>
                <p className="text-gray-600 dark:text-gray-400 mt-2">간접 영향도</p>
                <div className="mt-3">
                  <Network className="w-8 h-8 mx-auto text-blue-600 dark:text-blue-400" />
                </div>
              </div>
            </div>

            <div className={`rounded-lg p-6 ${getImpactBg(ontologyData.impact.sector)}`}>
              <div className="text-center">
                <div className={`text-4xl font-bold ${getImpactColor(ontologyData.impact.sector)}`}>
                  {ontologyData.impact.sector > 0 ? '+' : ''}{ontologyData.impact.sector}
                </div>
                <p className="text-gray-600 dark:text-gray-400 mt-2">섹터 영향도</p>
                <div className="mt-3">
                  <BarChart3 className="w-8 h-8 mx-auto text-purple-600 dark:text-purple-400" />
                </div>
              </div>
            </div>
          </div>

          {/* 투자 추천 */}
          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
            <h5 className="font-bold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-indigo-600" />
              AI 투자 의견
            </h5>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-gray-700 dark:text-gray-300">종합 감성 점수</span>
                <span className="font-bold text-lg">
                  {(ontologyData.sentiment * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-700 dark:text-gray-300">투자 추천</span>
                <span className={`font-bold text-lg ${
                  ontologyData.impact.direct > 50 ? 'text-green-600' :
                  ontologyData.impact.direct > 20 ? 'text-green-500' :
                  ontologyData.impact.direct > -20 ? 'text-gray-600' :
                  ontologyData.impact.direct > -50 ? 'text-red-500' :
                  'text-red-600'
                }`}>
                  {ontologyData.impact.direct > 50 ? '강력 매수' :
                   ontologyData.impact.direct > 20 ? '매수' :
                   ontologyData.impact.direct > -20 ? '중립' :
                   ontologyData.impact.direct > -50 ? '매도' :
                   '강력 매도'}
                </span>
              </div>
              <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  * AI 분석 결과는 참고용이며, 실제 투자 결정은 전문가 상담 후 신중히 결정하세요.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}