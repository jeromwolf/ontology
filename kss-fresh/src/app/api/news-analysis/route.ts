import { NextRequest, NextResponse } from 'next/server'

// API 키 설정 (환경 변수로 관리)
const NEWS_API_KEY = process.env.NEWS_API_KEY
const OPENAI_API_KEY = process.env.OPENAI_API_KEY
const ALPHA_VANTAGE_KEY = process.env.ALPHA_VANTAGE_KEY

// 뉴스 온톨로지 구조
interface NewsOntology {
  company: string
  ticker: string
  relatedCompanies: string[]
  industry: string
  keywords: string[]
  sentiment: number
  impact: {
    direct: number      // 직접 영향도 (-100 ~ +100)
    indirect: number    // 간접 영향도
    sector: number      // 섹터 영향도
  }
  relationships: {
    suppliers: string[]
    competitors: string[]
    partners: string[]
  }
}

// 1. 뉴스 데이터 수집
async function fetchNews(company: string, ticker: string) {
  const newsUrl = `https://newsapi.org/v2/everything?q=${encodeURIComponent(company)}&apiKey=${NEWS_API_KEY}&language=ko&sortBy=publishedAt`
  
  const response = await fetch(newsUrl)
  const data = await response.json()
  
  return data.articles
}

// 2. 감성 분석 (Alpha Vantage)
async function analyzeSentiment(ticker: string) {
  const sentimentUrl = `https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=${ticker}&apikey=${ALPHA_VANTAGE_KEY}`
  
  const response = await fetch(sentimentUrl)
  const data = await response.json()
  
  return data.feed?.map((item: any) => ({
    title: item.title,
    sentiment: item.overall_sentiment_score,
    relevance: item.ticker_sentiment?.[0]?.relevance_score || 0
  }))
}

// 3. 온톨로지 기반 영향도 분석
async function analyzeImpact(news: any[], company: string): Promise<NewsOntology> {
  // OpenAI GPT를 사용한 영향도 분석
  const prompt = `
    다음 뉴스들이 ${company}에 미칠 영향을 분석해주세요:
    ${news.slice(0, 5).map(n => n.title).join('\n')}
    
    다음 형식으로 답변해주세요:
    - 직접 영향도: -100 ~ +100
    - 간접 영향도: -100 ~ +100
    - 관련 기업들
    - 주요 키워드
  `
  
  // GPT API 호출 (실제 구현 시)
  // const analysis = await callOpenAI(prompt)
  
  // 임시 모의 데이터
  return {
    company,
    ticker: '',
    relatedCompanies: ['삼성SDI', 'SK하이닉스', 'LG전자'],
    industry: '전자/반도체',
    keywords: ['반도체', 'AI', '수출', '실적'],
    sentiment: 0.65,
    impact: {
      direct: 45,
      indirect: 20,
      sector: 35
    },
    relationships: {
      suppliers: ['ASML', 'TSMC'],
      competitors: ['Intel', 'TSMC'],
      partners: ['Google', 'Microsoft']
    }
  }
}

// 4. 한국 주식 시세 조회
async function getKoreanStockPrice(ticker: string) {
  // 네이버 금융 API (비공식)
  const url = `https://api.finance.naver.com/service/itemSummary.nhn?itemcode=${ticker}`
  
  try {
    const response = await fetch(url)
    const data = await response.json()
    return data
  } catch (error) {
    console.error('주가 조회 실패:', error)
    return null
  }
}

// 5. 통합 분석 API 엔드포인트
export async function POST(request: NextRequest) {
  try {
    const { company, ticker } = await request.json()
    
    // 병렬로 데이터 수집
    const [news, sentiment, stockPrice] = await Promise.all([
      fetchNews(company, ticker),
      analyzeSentiment(ticker),
      getKoreanStockPrice(ticker)
    ])
    
    // 온톨로지 기반 영향도 분석
    const impact = await analyzeImpact(news, company)
    
    // 결과 통합
    const result = {
      timestamp: new Date().toISOString(),
      company,
      ticker,
      currentPrice: stockPrice,
      newsCount: news.length,
      news: news.slice(0, 10),
      sentiment,
      ontologyAnalysis: impact,
      recommendation: getRecommendation(impact.impact.direct)
    }
    
    return NextResponse.json(result)
  } catch (error) {
    console.error('뉴스 분석 오류:', error)
    return NextResponse.json(
      { error: '뉴스 분석 중 오류가 발생했습니다.' },
      { status: 500 }
    )
  }
}

// 투자 추천 생성
function getRecommendation(impactScore: number): string {
  if (impactScore > 50) return '강력 매수'
  if (impactScore > 20) return '매수'
  if (impactScore > -20) return '중립'
  if (impactScore > -50) return '매도'
  return '강력 매도'
}

export async function GET() {
  return NextResponse.json({
    message: '뉴스 분석 API',
    endpoints: {
      POST: '/api/news-analysis',
      body: {
        company: '회사명',
        ticker: '종목코드'
      }
    }
  })
}