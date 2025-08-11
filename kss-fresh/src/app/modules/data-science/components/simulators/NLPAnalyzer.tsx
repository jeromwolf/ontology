'use client'

import { useState, useEffect } from 'react'
import { MessageSquare, Tag, BarChart3, Zap, Globe, Smile, Hash, TrendingUp } from 'lucide-react'

interface AnalysisResult {
  tokens: string[]
  entities: { text: string; type: string; start: number; end: number }[]
  sentiment: { score: number; label: string }
  keywords: { word: string; score: number }[]
  wordFrequency: { [key: string]: number }
  summary?: string
  language?: string
  topics?: string[]
}

export default function NLPAnalyzer() {
  const [inputText, setInputText] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [activeTab, setActiveTab] = useState<'tokens' | 'entities' | 'sentiment' | 'keywords' | 'wordcloud'>('tokens')
  
  // 샘플 텍스트
  const sampleTexts = {
    korean: `인공지능은 현대 사회에서 가장 중요한 기술 중 하나입니다. 
머신러닝과 딥러닝의 발전으로 인해 우리의 일상생활이 크게 변화하고 있습니다. 
자율주행 자동차, 음성 인식, 이미지 분석 등 다양한 분야에서 AI가 활용되고 있습니다.`,
    english: `Artificial intelligence has become one of the most transformative technologies of our time. 
Machine learning algorithms are revolutionizing industries from healthcare to finance. 
Natural language processing enables computers to understand and generate human language with remarkable accuracy.`,
    review: `이 제품은 정말 훌륭합니다! 배송도 빠르고 품질도 최고예요. 
다만 가격이 조금 비싼 편이지만, 그만한 가치가 있다고 생각합니다. 
다음에도 꼭 재구매할 예정입니다. 강력 추천!`,
    news: `오늘 주식시장은 전반적으로 상승세를 보였습니다. 
코스피 지수는 2,500포인트를 돌파하며 연중 최고치를 경신했습니다. 
외국인 투자자들의 순매수가 이어지면서 시장에 긍정적인 영향을 미쳤습니다.`
  }
  
  // 간단한 토큰화 (실제로는 더 정교한 방법 필요)
  const tokenize = (text: string): string[] => {
    // 한글과 영어를 다르게 처리
    const koreanRegex = /[\uAC00-\uD7AF]+|[a-zA-Z]+|\d+|[^\s\w\uAC00-\uD7AF]+/g
    const tokens = text.match(koreanRegex) || []
    return tokens.filter(token => token.trim().length > 0)
  }
  
  // 개체명 인식 시뮬레이션
  const recognizeEntities = (text: string): AnalysisResult['entities'] => {
    const entities: AnalysisResult['entities'] = []
    
    // 간단한 규칙 기반 NER
    const patterns = [
      { regex: /인공지능|AI|머신러닝|딥러닝|자연어처리|NLP/gi, type: 'TECH' },
      { regex: /\d{4}년|\d{1,2}월|\d{1,2}일|오늘|내일|어제/g, type: 'DATE' },
      { regex: /서울|부산|대구|인천|광주|대전|울산/g, type: 'LOCATION' },
      { regex: /\d+원|\d+달러|\d+포인트/g, type: 'MONEY' },
      { regex: /삼성|LG|현대|SK|네이버|카카오/g, type: 'ORG' },
      { regex: /\d+%|\d+퍼센트/g, type: 'PERCENT' }
    ]
    
    patterns.forEach(({ regex, type }) => {
      let match
      while ((match = regex.exec(text)) !== null) {
        entities.push({
          text: match[0],
          type,
          start: match.index,
          end: match.index + match[0].length
        })
      }
    })
    
    return entities.sort((a, b) => a.start - b.start)
  }
  
  // 감정 분석 시뮬레이션
  const analyzeSentiment = (text: string): AnalysisResult['sentiment'] => {
    // 긍정/부정 단어 사전
    const positiveWords = ['좋다', '훌륭', '최고', '추천', '만족', '행복', '성공', '발전', '상승', 'good', 'great', 'excellent', 'amazing', 'wonderful']
    const negativeWords = ['나쁘다', '실망', '최악', '불만', '실패', '하락', '위험', '문제', 'bad', 'terrible', 'awful', 'disappointing']
    
    let positiveCount = 0
    let negativeCount = 0
    
    const lowerText = text.toLowerCase()
    
    positiveWords.forEach(word => {
      const regex = new RegExp(word, 'gi')
      const matches = lowerText.match(regex)
      if (matches) positiveCount += matches.length
    })
    
    negativeWords.forEach(word => {
      const regex = new RegExp(word, 'gi')
      const matches = lowerText.match(regex)
      if (matches) negativeCount += matches.length
    })
    
    const score = (positiveCount - negativeCount) / Math.max(positiveCount + negativeCount, 1)
    
    let label = '중립'
    if (score > 0.2) label = '긍정'
    else if (score > 0.5) label = '매우 긍정'
    else if (score < -0.2) label = '부정'
    else if (score < -0.5) label = '매우 부정'
    
    return {
      score: Math.max(-1, Math.min(1, score)),
      label
    }
  }
  
  // 키워드 추출 (TF-IDF 간단 구현)
  const extractKeywords = (tokens: string[]): AnalysisResult['keywords'] => {
    const stopWords = new Set(['은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로', '와', '과', 'the', 'is', 'at', 'which', 'on', 'a', 'an'])
    
    // 단어 빈도 계산
    const wordFreq: { [key: string]: number } = {}
    tokens.forEach(token => {
      const lower = token.toLowerCase()
      if (!stopWords.has(lower) && token.length > 1) {
        wordFreq[lower] = (wordFreq[lower] || 0) + 1
      }
    })
    
    // TF 스코어 계산 및 정렬
    const keywords = Object.entries(wordFreq)
      .map(([word, freq]) => ({
        word,
        score: freq / tokens.length
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
    
    return keywords
  }
  
  // 단어 빈도 계산
  const calculateWordFrequency = (tokens: string[]): { [key: string]: number } => {
    const freq: { [key: string]: number } = {}
    tokens.forEach(token => {
      const lower = token.toLowerCase()
      if (token.length > 1) {
        freq[lower] = (freq[lower] || 0) + 1
      }
    })
    return freq
  }
  
  // 언어 감지 (간단한 구현)
  const detectLanguage = (text: string): string => {
    const koreanChars = text.match(/[\uAC00-\uD7AF]/g)
    const englishChars = text.match(/[a-zA-Z]/g)
    
    const koreanRatio = (koreanChars?.length || 0) / text.length
    const englishRatio = (englishChars?.length || 0) / text.length
    
    if (koreanRatio > 0.3) return '한국어'
    if (englishRatio > 0.5) return '영어'
    return '기타'
  }
  
  // 토픽 모델링 시뮬레이션
  const extractTopics = (text: string, keywords: AnalysisResult['keywords']): string[] => {
    const topicMap: { [key: string]: string[] } = {
      '기술': ['AI', '인공지능', '머신러닝', '딥러닝', '기술', '알고리즘', 'technology', 'machine', 'learning'],
      '비즈니스': ['시장', '투자', '매출', '성장', '전략', '경영', 'business', 'market', 'investment'],
      '의견': ['좋다', '나쁘다', '추천', '만족', '실망', 'good', 'bad', 'recommend'],
      '데이터': ['분석', '통계', '데이터', '정보', '결과', 'analysis', 'data', 'statistics']
    }
    
    const topics: string[] = []
    const lowerText = text.toLowerCase()
    
    Object.entries(topicMap).forEach(([topic, words]) => {
      const matchCount = words.filter(word => lowerText.includes(word)).length
      if (matchCount >= 2) topics.push(topic)
    })
    
    return topics.length > 0 ? topics : ['일반']
  }
  
  // 텍스트 요약 (추출적 요약)
  const summarizeText = (text: string): string => {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10)
    if (sentences.length <= 2) return text
    
    // 각 문장의 중요도 계산 (단어 빈도 기반)
    const tokens = tokenize(text)
    const wordFreq = calculateWordFrequency(tokens)
    
    const sentenceScores = sentences.map(sentence => {
      const sentTokens = tokenize(sentence)
      const score = sentTokens.reduce((sum, token) => {
        return sum + (wordFreq[token.toLowerCase()] || 0)
      }, 0) / sentTokens.length
      
      return { sentence, score }
    })
    
    // 상위 2개 문장 선택
    const topSentences = sentenceScores
      .sort((a, b) => b.score - a.score)
      .slice(0, 2)
      .sort((a, b) => sentences.indexOf(a.sentence) - sentences.indexOf(b.sentence))
    
    return topSentences.map(s => s.sentence.trim()).join('. ') + '.'
  }
  
  // 분석 수행
  const analyzeText = () => {
    if (!inputText.trim()) return
    
    setIsAnalyzing(true)
    
    // 시뮬레이션을 위한 지연
    setTimeout(() => {
      const tokens = tokenize(inputText)
      const entities = recognizeEntities(inputText)
      const sentiment = analyzeSentiment(inputText)
      const keywords = extractKeywords(tokens)
      const wordFrequency = calculateWordFrequency(tokens)
      const language = detectLanguage(inputText)
      const topics = extractTopics(inputText, keywords)
      const summary = summarizeText(inputText)
      
      setAnalysisResult({
        tokens,
        entities,
        sentiment,
        keywords,
        wordFrequency,
        language,
        topics,
        summary
      })
      
      setIsAnalyzing(false)
    }, 1000)
  }
  
  // 워드 클라우드 그리기
  const drawWordCloud = (wordFreq: { [key: string]: number }) => {
    const canvas = document.getElementById('wordcloud-canvas') as HTMLCanvasElement
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    ctx.clearRect(0, 0, 600, 400)
    
    // 상위 30개 단어만 표시
    const words = Object.entries(wordFreq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 30)
    
    if (words.length === 0) return
    
    const maxFreq = words[0][1]
    
    // 랜덤 위치에 단어 배치
    words.forEach(([word, freq], index) => {
      const fontSize = Math.max(12, (freq / maxFreq) * 40)
      const x = Math.random() * (600 - word.length * fontSize / 2)
      const y = Math.random() * (400 - fontSize) + fontSize
      
      ctx.font = `${fontSize}px sans-serif`
      ctx.fillStyle = `hsl(${Math.random() * 360}, 70%, 50%)`
      ctx.fillText(word, x, y)
    })
  }
  
  useEffect(() => {
    if (analysisResult?.wordFrequency && activeTab === 'wordcloud') {
      drawWordCloud(analysisResult.wordFrequency)
    }
  }, [analysisResult, activeTab])
  
  return (
    <div className="w-full max-w-6xl mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6">자연어 처리 분석기</h2>
        
        <div className="grid lg:grid-cols-2 gap-6">
          {/* 입력 영역 */}
          <div>
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">텍스트 입력</label>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="분석할 텍스트를 입력하세요..."
                className="w-full h-48 px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 resize-none"
              />
              <div className="flex justify-between items-center mt-2">
                <span className="text-sm text-gray-500">
                  {inputText.length}자 | {tokenize(inputText).length}개 토큰
                </span>
                <button
                  onClick={analyzeText}
                  disabled={!inputText.trim() || isAnalyzing}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    isAnalyzing
                      ? 'bg-gray-400 text-white cursor-not-allowed'
                      : 'bg-blue-500 text-white hover:bg-blue-600'
                  }`}
                >
                  {isAnalyzing ? '분석 중...' : '분석 시작'}
                </button>
              </div>
            </div>
            
            {/* 샘플 텍스트 */}
            <div>
              <h3 className="text-sm font-medium mb-2">샘플 텍스트</h3>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(sampleTexts).map(([key, text]) => (
                  <button
                    key={key}
                    onClick={() => setInputText(text)}
                    className="px-3 py-2 text-sm bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                  >
                    {key === 'korean' ? '한국어 예시' :
                     key === 'english' ? '영어 예시' :
                     key === 'review' ? '리뷰 텍스트' : '뉴스 기사'}
                  </button>
                ))}
              </div>
            </div>
            
            {/* 기본 정보 */}
            {analysisResult && (
              <div className="mt-6 bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h3 className="font-semibold mb-3">기본 정보</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">언어:</span>
                    <span className="font-medium">{analysisResult.language}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">토큰 수:</span>
                    <span className="font-medium">{analysisResult.tokens.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">고유 단어:</span>
                    <span className="font-medium">{Object.keys(analysisResult.wordFrequency).length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">개체명:</span>
                    <span className="font-medium">{analysisResult.entities.length}개</span>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* 결과 영역 */}
          <div>
            {analysisResult ? (
              <>
                {/* 탭 네비게이션 */}
                <div className="flex gap-2 mb-4 flex-wrap">
                  {[
                    { id: 'tokens', icon: Hash, label: '토큰' },
                    { id: 'entities', icon: Tag, label: '개체명' },
                    { id: 'sentiment', icon: Smile, label: '감정' },
                    { id: 'keywords', icon: Zap, label: '키워드' },
                    { id: 'wordcloud', icon: BarChart3, label: '워드클라우드' }
                  ].map(({ id, icon: Icon, label }) => (
                    <button
                      key={id}
                      onClick={() => setActiveTab(id as any)}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg font-medium transition-colors ${
                        activeTab === id
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      {label}
                    </button>
                  ))}
                </div>
                
                {/* 탭 콘텐츠 */}
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 h-96 overflow-auto">
                  {activeTab === 'tokens' && (
                    <div>
                      <h3 className="font-semibold mb-3">토큰 분석</h3>
                      <div className="flex flex-wrap gap-2">
                        {analysisResult.tokens.map((token, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-white dark:bg-gray-600 rounded text-sm"
                          >
                            {token}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {activeTab === 'entities' && (
                    <div>
                      <h3 className="font-semibold mb-3">개체명 인식</h3>
                      <div className="space-y-2">
                        {analysisResult.entities.length > 0 ? (
                          analysisResult.entities.map((entity, index) => (
                            <div key={index} className="flex items-center gap-3">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                entity.type === 'TECH' ? 'bg-blue-100 text-blue-700' :
                                entity.type === 'DATE' ? 'bg-green-100 text-green-700' :
                                entity.type === 'LOCATION' ? 'bg-purple-100 text-purple-700' :
                                entity.type === 'MONEY' ? 'bg-yellow-100 text-yellow-700' :
                                entity.type === 'ORG' ? 'bg-red-100 text-red-700' :
                                'bg-gray-100 text-gray-700'
                              }`}>
                                {entity.type}
                              </span>
                              <span>{entity.text}</span>
                            </div>
                          ))
                        ) : (
                          <p className="text-gray-500">인식된 개체명이 없습니다.</p>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {activeTab === 'sentiment' && (
                    <div>
                      <h3 className="font-semibold mb-3">감정 분석</h3>
                      <div className="text-center py-8">
                        <div className={`inline-flex items-center justify-center w-32 h-32 rounded-full mb-4 ${
                          analysisResult.sentiment.score > 0.2 ? 'bg-green-100' :
                          analysisResult.sentiment.score < -0.2 ? 'bg-red-100' :
                          'bg-gray-100'
                        }`}>
                          <Smile className={`w-16 h-16 ${
                            analysisResult.sentiment.score > 0.2 ? 'text-green-600' :
                            analysisResult.sentiment.score < -0.2 ? 'text-red-600' :
                            'text-gray-600'
                          }`} />
                        </div>
                        <p className="text-2xl font-bold mb-2">{analysisResult.sentiment.label}</p>
                        <p className="text-gray-600 dark:text-gray-400">
                          감정 점수: {(analysisResult.sentiment.score * 100).toFixed(1)}%
                        </p>
                        
                        {/* 감정 점수 바 */}
                        <div className="mt-6 w-full max-w-md mx-auto">
                          <div className="h-4 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                            <div
                              className={`h-full transition-all duration-500 ${
                                analysisResult.sentiment.score > 0 ? 'bg-green-500' : 'bg-red-500'
                              }`}
                              style={{
                                width: `${Math.abs(analysisResult.sentiment.score) * 50 + 50}%`,
                                marginLeft: analysisResult.sentiment.score < 0 ? `${50 + analysisResult.sentiment.score * 50}%` : '50%'
                              }}
                            />
                          </div>
                          <div className="flex justify-between mt-2 text-xs text-gray-500">
                            <span>매우 부정</span>
                            <span>중립</span>
                            <span>매우 긍정</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {activeTab === 'keywords' && (
                    <div>
                      <h3 className="font-semibold mb-3">주요 키워드</h3>
                      <div className="space-y-3">
                        {analysisResult.keywords.map((keyword, index) => (
                          <div key={index} className="flex items-center gap-3">
                            <span className="font-medium">{index + 1}.</span>
                            <span className="flex-1">{keyword.word}</span>
                            <div className="w-32 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                              <div
                                className="h-full bg-blue-500 rounded-full"
                                style={{ width: `${keyword.score * 500}%` }}
                              />
                            </div>
                            <span className="text-sm text-gray-500">
                              {(keyword.score * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {activeTab === 'wordcloud' && (
                    <div>
                      <h3 className="font-semibold mb-3">워드 클라우드</h3>
                      <canvas
                        id="wordcloud-canvas"
                        width={600}
                        height={400}
                        className="w-full border border-gray-300 dark:border-gray-600 rounded"
                      />
                    </div>
                  )}
                </div>
                
                {/* 추가 분석 결과 */}
                <div className="mt-4 grid grid-cols-2 gap-4">
                  {analysisResult.topics && (
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                      <h4 className="font-semibold mb-2 text-sm">주제 분류</h4>
                      <div className="flex flex-wrap gap-2">
                        {analysisResult.topics.map((topic, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-blue-100 dark:bg-blue-800 text-blue-700 dark:text-blue-200 rounded text-xs"
                          >
                            {topic}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {analysisResult.summary && (
                    <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
                      <h4 className="font-semibold mb-2 text-sm">요약</h4>
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        {analysisResult.summary}
                      </p>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="flex items-center justify-center h-96 text-gray-500">
                텍스트를 입력하고 분석을 시작하세요
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}