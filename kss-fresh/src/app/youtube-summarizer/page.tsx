'use client'

import { useState } from 'react'
import Navigation from '@/components/Navigation'
import { 
  Play, Pause, Download, Copy, Share2, Clock,
  FileText, Brain, Sparkles, ArrowRight,
  Youtube, MessageSquare, BookOpen, TrendingUp,
  CheckCircle2, AlertCircle, Loader2, Eye,
  BarChart3, Hash, Users, Calendar
} from 'lucide-react'

interface VideoInfo {
  id: string
  title: string
  channel: string
  duration: string
  views: string
  publishedAt: string
  thumbnail: string
  description: string
}

interface SummarySection {
  title: string
  content: string
  timestamp?: string
  importance: 'high' | 'medium' | 'low'
}

interface DetailedSection {
  timestamp: string
  title: string
  summary: string
  keyPoints: string[]
  quotes?: string[]
}

interface AnalysisResult {
  videoInfo: VideoInfo
  summary: {
    overview: string
    keyPoints: SummarySection[]
    detailedSections: DetailedSection[]
    actionItems: string[]
    tags: string[]
  }
  transcript: {
    segments: Array<{
      start: number
      end: number
      text: string
    }>
    language: string
    confidence: number
  }
  analytics: {
    readingTime: number
    complexity: 'beginner' | 'intermediate' | 'advanced'
    topics: string[]
    sentiment: 'positive' | 'neutral' | 'negative'
  }
}

export default function YouTubeSummarizer() {
  const [url, setUrl] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [activeTab, setActiveTab] = useState<'summary' | 'detailed' | 'transcript' | 'analytics'>('summary')
  const [error, setError] = useState('')

  // YouTube URL 유효성 검사
  const isValidYouTubeUrl = (url: string) => {
    const patterns = [
      /(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)/,
      /(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)/
    ]
    return patterns.some(pattern => pattern.test(url))
  }

  // YouTube 비디오 ID 추출
  const extractVideoId = (url: string) => {
    const match = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)/)
    return match ? match[1] : null
  }

  // 시뮬레이션된 분석 수행
  const analyzeVideo = async () => {
    if (!url.trim()) {
      setError('YouTube URL을 입력해주세요.')
      return
    }

    if (!isValidYouTubeUrl(url)) {
      setError('올바른 YouTube URL을 입력해주세요.')
      return
    }

    setError('')
    setIsAnalyzing(true)

    // 시뮬레이션 지연
    await new Promise(resolve => setTimeout(resolve, 3000))

    const videoId = extractVideoId(url)
    
    // 시뮬레이션 데이터
    const mockResult: AnalysisResult = {
      videoInfo: {
        id: videoId || 'demo',
        title: 'ChatGPT와 GPT-4의 차이점과 활용법 완벽 가이드',
        channel: 'AI 교육 채널',
        duration: '15:42',
        views: '127K',
        publishedAt: '2024-01-15',
        thumbnail: `https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`,
        description: 'ChatGPT와 GPT-4의 주요 차이점을 알아보고, 실제 업무에서 어떻게 활용할 수 있는지 자세히 설명합니다.'
      },
      summary: {
        overview: 'ChatGPT와 GPT-4의 핵심 차이점을 다루며, 각각의 장단점과 실무 활용 사례를 제시합니다. 특히 멀티모달 기능, 추론 능력, 창의성 측면에서 GPT-4의 향상된 성능을 강조합니다.',
        keyPoints: [
          {
            title: 'GPT-4의 주요 개선사항',
            content: '멀티모달 지원, 향상된 추론 능력, 더 긴 컨텍스트 윈도우를 제공합니다.',
            timestamp: '2:15',
            importance: 'high'
          },
          {
            title: '실무 활용 사례',
            content: '코드 리뷰, 문서 작성, 창작 활동에서 GPT-4가 보여주는 차별화된 성능을 소개합니다.',
            timestamp: '7:30',
            importance: 'high'
          },
          {
            title: '비용 대비 효과',
            content: 'ChatGPT Plus와 GPT-4 API의 가격 차이와 각 상황별 최적의 선택 기준을 제시합니다.',
            timestamp: '12:00',
            importance: 'medium'
          }
        ],
        detailedSections: [
          {
            timestamp: '0:00-2:30',
            title: '인트로 및 배경 설명',
            summary: 'ChatGPT와 GPT-4의 등장 배경과 이번 영상에서 다룰 주요 내용을 소개합니다. OpenAI의 발전 과정과 두 모델의 출시 시기, 그리고 사용자들이 가장 궁금해하는 차이점들을 개괄적으로 설명합니다.',
            keyPoints: [
              'ChatGPT는 2022년 11월, GPT-4는 2023년 3월 출시',
              '두 모델 모두 대화형 AI이지만 성능과 기능에서 차이',
              '실무 활용도와 비용 효율성이 주요 선택 기준'
            ],
            quotes: [
              '오늘은 많은 분들이 궁금해하시는 ChatGPT와 GPT-4의 실질적인 차이점을 알아보겠습니다'
            ]
          },
          {
            timestamp: '2:30-6:15',
            title: 'GPT-4의 핵심 개선사항',
            summary: 'GPT-4가 ChatGPT(GPT-3.5) 대비 향상된 주요 기능들을 자세히 설명합니다. 멀티모달 기능, 더 긴 컨텍스트 처리 능력, 향상된 추론 능력을 실제 예시와 함께 보여줍니다.',
            keyPoints: [
              '텍스트와 이미지를 동시에 처리하는 멀티모달 기능',
              '32K 토큰까지 처리 가능한 확장된 컨텍스트 윈도우',
              '복잡한 논리적 추론과 문제 해결 능력 향상',
              '더 정확하고 일관성 있는 답변 생성'
            ],
            quotes: [
              'GPT-4의 가장 큰 변화는 바로 이미지를 이해할 수 있다는 점입니다',
              '32,000토큰이면 대략 25페이지 분량의 문서를 한 번에 처리할 수 있어요'
            ]
          },
          {
            timestamp: '6:15-10:45',
            title: '실무 활용 사례 비교',
            summary: '실제 업무 환경에서 ChatGPT와 GPT-4를 어떻게 활용할 수 있는지 구체적인 사례를 통해 비교합니다. 코딩, 문서 작성, 창작 활동, 데이터 분석 등 다양한 영역에서의 성능 차이를 보여줍니다.',
            keyPoints: [
              '코드 리뷰와 디버깅에서 GPT-4의 뛰어난 정확도',
              '긴 문서 요약과 분석에서의 성능 차이',
              '창의적 글쓰기와 아이디어 생성 능력 비교',
              '복잡한 엑셀 함수와 데이터 분석 지원'
            ],
            quotes: [
              'GPT-4는 제가 작성한 코드의 버그를 정확히 찾아내더라고요',
              '긴 보고서를 요약할 때 ChatGPT는 중간에 내용을 놓치는 경우가 있어요'
            ]
          },
          {
            timestamp: '10:45-13:30',
            title: '비용 분석 및 가성비 비교',
            summary: 'ChatGPT Plus($20/월)와 GPT-4 API 사용료를 비교하고, 사용 패턴에 따른 최적의 선택 방법을 안내합니다. 개인 사용자와 기업 사용자를 구분하여 권장사항을 제시합니다.',
            keyPoints: [
              'ChatGPT Plus: 월 $20 무제한 사용',
              'GPT-4 API: 토큰당 과금, 대량 사용시 비용 부담',
              '개인 사용자는 Plus 구독이 경제적',
              '기업은 API 연동을 통한 맞춤형 활용 권장'
            ],
            quotes: [
              '하루에 몇 시간씩 사용한다면 Plus가 훨씬 경제적이에요',
              'API는 정확히 사용한 만큼만 지불하니까 가끔 쓰는 분들에게 좋죠'
            ]
          },
          {
            timestamp: '13:30-15:42',
            title: '선택 가이드 및 마무리',
            summary: '개인의 사용 목적과 예산에 따른 최적의 선택 가이드라인을 제시하고, 향후 AI 기술 발전 방향에 대한 전망을 공유합니다. 구독자들의 질문에 대한 답변도 포함되어 있습니다.',
            keyPoints: [
              '학습 목적: ChatGPT로 시작 후 필요시 GPT-4 업그레이드',
              '업무 활용: 정확도가 중요하다면 GPT-4 권장',
              '창작 활동: 두 모델의 차이 체험 후 선택',
              '미래에는 더 강력하고 저렴한 모델 출시 예정'
            ],
            quotes: [
              '완벽한 정답은 없어요. 본인의 사용 패턴을 파악하는 게 가장 중요합니다',
              'AI는 계속 발전하고 있으니까 너무 고민하지 마시고 일단 시작해보세요'
            ]
          }
        ],
        actionItems: [
          'GPT-4의 멀티모달 기능을 활용한 이미지 분석 실습',
          '업무별 ChatGPT vs GPT-4 선택 가이드라인 수립',
          'API 비용 최적화 전략 검토'
        ],
        tags: ['ChatGPT', 'GPT-4', '인공지능', 'OpenAI', '실무활용', '가이드']
      },
      transcript: {
        segments: [
          { start: 0, end: 30, text: '안녕하세요! 오늘은 ChatGPT와 GPT-4의 차이점에 대해 자세히 알아보겠습니다.' },
          { start: 30, end: 60, text: '먼저 GPT-4의 가장 큰 특징인 멀티모달 기능부터 살펴보겠습니다.' },
          { start: 60, end: 90, text: 'GPT-4는 텍스트뿐만 아니라 이미지도 이해할 수 있어서...' }
        ],
        language: 'ko',
        confidence: 0.95
      },
      analytics: {
        readingTime: 8,
        complexity: 'intermediate',
        topics: ['AI/ML', '도구 활용', '생산성', '기술 가이드'],
        sentiment: 'positive'
      }
    }

    setResult(mockResult)
    setIsAnalyzing(false)
  }

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    // 간단한 성공 피드백 (실제 구현시 toast 사용)
    alert('클립보드에 복사되었습니다!')
  }

  const getImportanceColor = (importance: string) => {
    switch (importance) {
      case 'high': return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
      case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400'
      case 'low': return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-400'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-400'
    }
  }

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return <TrendingUp className="w-4 h-4 text-green-500" />
      case 'negative': return <TrendingUp className="w-4 h-4 text-red-500 rotate-180" />
      default: return <BarChart3 className="w-4 h-4 text-gray-500" />
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Navigation />
      
      <main className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 mx-auto bg-gradient-to-r from-red-600 to-orange-500 rounded-2xl flex items-center justify-center mb-4">
            <Youtube className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            YouTube Summarizer
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-6">
            AI가 YouTube 동영상을 분석하여 핵심 내용을 요약해드립니다
          </p>
        </div>

        {/* Input Section */}
        <div className="max-w-4xl mx-auto mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4">
              <Eye className="w-5 h-5 text-red-500" />
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                동영상 분석
              </h2>
            </div>
            
            <div className="flex gap-3">
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="YouTube URL을 입력하세요... (예: https://youtube.com/watch?v=...)"
                className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg 
                         bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                         focus:ring-2 focus:ring-red-500 focus:border-transparent"
                disabled={isAnalyzing}
              />
              <button
                onClick={analyzeVideo}
                disabled={isAnalyzing || !url.trim()}
                className="px-6 py-3 bg-gradient-to-r from-red-600 to-orange-500 text-white rounded-lg 
                         hover:from-red-700 hover:to-orange-600 transition-all disabled:opacity-50
                         flex items-center gap-2 font-semibold"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    분석 중...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    분석 시작
                  </>
                )}
              </button>
            </div>
            
            {error && (
              <div className="mt-3 flex items-center gap-2 text-red-600 dark:text-red-400">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{error}</span>
              </div>
            )}
            
            <div className="mt-4 text-xs text-gray-500 dark:text-gray-400">
              * 현재 시뮬레이션 모드로 작동합니다. 실제 YouTube API 연동 시 모든 공개 동영상을 분석할 수 있습니다.
            </div>
          </div>
        </div>

        {/* Loading State */}
        {isAnalyzing && (
          <div className="max-w-4xl mx-auto mb-8">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="text-center">
                <div className="w-12 h-12 mx-auto bg-gradient-to-r from-red-600 to-orange-500 rounded-full flex items-center justify-center mb-4">
                  <Loader2 className="w-6 h-6 text-white animate-spin" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  동영상 분석 중...
                </h3>
                <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                  <p>• 동영상 정보 수집</p>
                  <p>• 자막 및 음성 추출</p>
                  <p>• AI 기반 내용 분석</p>
                  <p>• 요약 및 핵심 포인트 생성</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="max-w-6xl mx-auto space-y-6">
            {/* Video Info */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
              <div className="flex flex-col md:flex-row">
                <div className="md:w-1/3">
                  <img
                    src={result.videoInfo.thumbnail}
                    alt={result.videoInfo.title}
                    className="w-full h-48 md:h-full object-cover"
                    onError={(e) => {
                      // 썸네일 로드 실패시 플레이스홀더
                      e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="225" viewBox="0 0 400 225"><rect width="400" height="225" fill="%23374151"/><text x="200" y="112" text-anchor="middle" fill="white" font-size="16">YouTube Video</text></svg>'
                    }}
                  />
                </div>
                <div className="md:w-2/3 p-6">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                    {result.videoInfo.title}
                  </h3>
                  <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400 mb-3">
                    <span className="flex items-center gap-1">
                      <Users className="w-4 h-4" />
                      {result.videoInfo.channel}
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      {result.videoInfo.duration}
                    </span>
                    <span className="flex items-center gap-1">
                      <Eye className="w-4 h-4" />
                      {result.videoInfo.views} views
                    </span>
                    <span className="flex items-center gap-1">
                      <Calendar className="w-4 h-4" />
                      {result.videoInfo.publishedAt}
                    </span>
                  </div>
                  <p className="text-gray-700 dark:text-gray-300 text-sm line-clamp-3">
                    {result.videoInfo.description}
                  </p>
                </div>
              </div>
            </div>

            {/* Analytics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-blue-500" />
                  <span className="text-sm font-medium text-gray-600 dark:text-gray-400">읽기 시간</span>
                </div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {result.analytics.readingTime}분
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="w-4 h-4 text-green-500" />
                  <span className="text-sm font-medium text-gray-600 dark:text-gray-400">난이도</span>
                </div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white capitalize">
                  {result.analytics.complexity}
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  {getSentimentIcon(result.analytics.sentiment)}
                  <span className="text-sm font-medium text-gray-600 dark:text-gray-400">감정</span>
                </div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white capitalize">
                  {result.analytics.sentiment}
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <Hash className="w-4 h-4 text-purple-500" />
                  <span className="text-sm font-medium text-gray-600 dark:text-gray-400">주제</span>
                </div>
                <p className="text-sm font-bold text-gray-900 dark:text-white">
                  {result.analytics.topics.length}개 토픽
                </p>
              </div>
            </div>

            {/* Tabs */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
              <div className="border-b border-gray-200 dark:border-gray-700">
                <nav className="flex">
                  <button
                    onClick={() => setActiveTab('summary')}
                    className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                      activeTab === 'summary'
                        ? 'border-red-500 text-red-600 dark:text-red-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      요약
                    </div>
                  </button>
                  <button
                    onClick={() => setActiveTab('detailed')}
                    className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                      activeTab === 'detailed'
                        ? 'border-red-500 text-red-600 dark:text-red-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <BookOpen className="w-4 h-4" />
                      상세 요약
                    </div>
                  </button>
                  <button
                    onClick={() => setActiveTab('transcript')}
                    className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                      activeTab === 'transcript'
                        ? 'border-red-500 text-red-600 dark:text-red-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <MessageSquare className="w-4 h-4" />
                      전체 스크립트
                    </div>
                  </button>
                  <button
                    onClick={() => setActiveTab('analytics')}
                    className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                      activeTab === 'analytics'
                        ? 'border-red-500 text-red-600 dark:text-red-400'
                        : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <BarChart3 className="w-4 h-4" />
                      상세 분석
                    </div>
                  </button>
                </nav>
              </div>

              <div className="p-6">
                {activeTab === 'summary' && (
                  <div className="space-y-6">
                    {/* Overview */}
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                        <Brain className="w-5 h-5 text-red-500" />
                        전체 요약
                      </h4>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                          {result.summary.overview}
                        </p>
                        <button
                          onClick={() => copyToClipboard(result.summary.overview)}
                          className="mt-3 text-sm text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 flex items-center gap-1"
                        >
                          <Copy className="w-3 h-3" />
                          복사
                        </button>
                      </div>
                    </div>

                    {/* Key Points */}
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                        <CheckCircle2 className="w-5 h-5 text-green-500" />
                        핵심 포인트
                      </h4>
                      <div className="space-y-3">
                        {result.summary.keyPoints.map((point, index) => (
                          <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                            <div className="flex items-start justify-between mb-2">
                              <h5 className="font-semibold text-gray-900 dark:text-white">
                                {point.title}
                              </h5>
                              <div className="flex items-center gap-2">
                                {point.timestamp && (
                                  <span className="text-xs text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/20 px-2 py-1 rounded">
                                    {point.timestamp}
                                  </span>
                                )}
                                <span className={`text-xs px-2 py-1 rounded ${getImportanceColor(point.importance)}`}>
                                  {point.importance}
                                </span>
                              </div>
                            </div>
                            <p className="text-gray-700 dark:text-gray-300 text-sm">
                              {point.content}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Action Items */}
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                        <ArrowRight className="w-5 h-5 text-blue-500" />
                        실행 항목
                      </h4>
                      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                        <ul className="space-y-2">
                          {result.summary.actionItems.map((item, index) => (
                            <li key={index} className="flex items-start gap-2">
                              <CheckCircle2 className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                              <span className="text-gray-700 dark:text-gray-300 text-sm">{item}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    {/* Tags */}
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                        <Hash className="w-5 h-5 text-purple-500" />
                        관련 태그
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {result.summary.tags.map((tag, index) => (
                          <span
                            key={index}
                            className="px-3 py-1 bg-purple-100 dark:bg-purple-900/20 text-purple-800 dark:text-purple-400 rounded-full text-sm"
                          >
                            #{tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === 'detailed' && (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between mb-6">
                      <h4 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                        <BookOpen className="w-5 h-5 text-blue-500" />
                        구간별 상세 요약
                      </h4>
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        {result.summary.detailedSections.length}개 구간
                      </span>
                    </div>
                    
                    <div className="space-y-6">
                      {result.summary.detailedSections.map((section, index) => (
                        <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
                          {/* Section Header */}
                          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-4 border-b border-gray-200 dark:border-gray-700">
                            <div className="flex items-center justify-between">
                              <h5 className="text-lg font-bold text-gray-900 dark:text-white">
                                {section.title}
                              </h5>
                              <span className="px-3 py-1 bg-blue-600 text-white text-sm rounded-full font-medium">
                                {section.timestamp}
                              </span>
                            </div>
                          </div>
                          
                          {/* Section Content */}
                          <div className="p-6 space-y-4">
                            {/* Summary */}
                            <div>
                              <h6 className="font-semibold text-gray-900 dark:text-white mb-2 text-sm uppercase tracking-wide">
                                📝 구간 요약
                              </h6>
                              <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                                {section.summary}
                              </p>
                            </div>
                            
                            {/* Key Points */}
                            <div>
                              <h6 className="font-semibold text-gray-900 dark:text-white mb-3 text-sm uppercase tracking-wide">
                                🎯 핵심 포인트
                              </h6>
                              <ul className="space-y-2">
                                {section.keyPoints.map((point, pointIndex) => (
                                  <li key={pointIndex} className="flex items-start gap-2">
                                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                                    <span className="text-gray-700 dark:text-gray-300 text-sm leading-relaxed">
                                      {point}
                                    </span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                            
                            {/* Quotes */}
                            {section.quotes && section.quotes.length > 0 && (
                              <div>
                                <h6 className="font-semibold text-gray-900 dark:text-white mb-3 text-sm uppercase tracking-wide">
                                  💬 주요 발언
                                </h6>
                                <div className="space-y-2">
                                  {section.quotes.map((quote, quoteIndex) => (
                                    <blockquote key={quoteIndex} className="border-l-4 border-blue-500 pl-4 italic">
                                      <p className="text-gray-600 dark:text-gray-400 text-sm">
                                        "{quote}"
                                      </p>
                                    </blockquote>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            {/* Copy Section Button */}
                            <div className="pt-2 border-t border-gray-100 dark:border-gray-800">
                              <button
                                onClick={() => {
                                  const sectionText = `
${section.title} (${section.timestamp})

${section.summary}

핵심 포인트:
${section.keyPoints.map(point => `• ${point}`).join('\n')}

${section.quotes ? `주요 발언:\n${section.quotes.map(quote => `"${quote}"`).join('\n')}` : ''}
                                  `.trim()
                                  copyToClipboard(sectionText)
                                }}
                                className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 flex items-center gap-1"
                              >
                                <Copy className="w-3 h-3" />
                                이 구간 복사
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    {/* Copy All Detailed Summary */}
                    <div className="text-center pt-4 border-t border-gray-200 dark:border-gray-700">
                      <button
                        onClick={() => {
                          const allSections = result.summary.detailedSections.map(section => `
${section.title} (${section.timestamp})

${section.summary}

핵심 포인트:
${section.keyPoints.map(point => `• ${point}`).join('\n')}

${section.quotes ? `주요 발언:\n${section.quotes.map(quote => `"${quote}"`).join('\n')}` : ''}
                          `).join('\n' + '='.repeat(50) + '\n')
                          copyToClipboard(allSections)
                        }}
                        className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2 mx-auto"
                      >
                        <Copy className="w-4 h-4" />
                        전체 상세요약 복사
                      </button>
                    </div>
                  </div>
                )}

                {activeTab === 'transcript' && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h4 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                        <MessageSquare className="w-5 h-5 text-green-500" />
                        전체 스크립트
                      </h4>
                      <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                        <span>언어: {result.transcript.language}</span>
                        <span>신뢰도: {Math.round(result.transcript.confidence * 100)}%</span>
                      </div>
                    </div>
                    
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                      {result.transcript.segments.map((segment, index) => (
                        <div key={index} className="flex gap-3 group">
                          <span className="text-sm text-red-600 dark:text-red-400 font-mono bg-red-100 dark:bg-red-900/20 px-2 py-1 rounded">
                            {formatDuration(segment.start)}
                          </span>
                          <p className="text-gray-700 dark:text-gray-300 text-sm leading-relaxed flex-1">
                            {segment.text}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {activeTab === 'analytics' && (
                  <div className="space-y-6">
                    <h4 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-purple-500" />
                      상세 분석
                    </h4>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h5 className="font-semibold text-gray-900 dark:text-white mb-3">주요 토픽</h5>
                        <div className="space-y-2">
                          {result.analytics.topics.map((topic, index) => (
                            <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                              <span className="text-gray-700 dark:text-gray-300">{topic}</span>
                              <span className="text-sm text-gray-500 dark:text-gray-400">
                                {Math.floor(Math.random() * 30 + 70)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <h5 className="font-semibold text-gray-900 dark:text-white mb-3">콘텐츠 특성</h5>
                        <div className="space-y-3">
                          <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                            <span className="text-gray-700 dark:text-gray-300">교육성</span>
                            <div className="flex items-center gap-2">
                              <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                                <div className="bg-green-500 h-2 rounded-full" style={{width: '85%'}}></div>
                              </div>
                              <span className="text-sm text-gray-500 dark:text-gray-400">85%</span>
                            </div>
                          </div>
                          
                          <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                            <span className="text-gray-700 dark:text-gray-300">실용성</span>
                            <div className="flex items-center gap-2">
                              <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                                <div className="bg-blue-500 h-2 rounded-full" style={{width: '92%'}}></div>
                              </div>
                              <span className="text-sm text-gray-500 dark:text-gray-400">92%</span>
                            </div>
                          </div>
                          
                          <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                            <span className="text-gray-700 dark:text-gray-300">참여도</span>
                            <div className="flex items-center gap-2">
                              <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                                <div className="bg-purple-500 h-2 rounded-full" style={{width: '78%'}}></div>
                              </div>
                              <span className="text-sm text-gray-500 dark:text-gray-400">78%</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-wrap gap-3 justify-center">
              <button
                onClick={() => copyToClipboard(JSON.stringify(result.summary, null, 2))}
                className="px-6 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <Copy className="w-4 h-4" />
                전체 요약 복사
              </button>
              
              <button
                onClick={() => {
                  const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `youtube-summary-${result.videoInfo.id}.json`
                  document.body.appendChild(a)
                  a.click()
                  document.body.removeChild(a)
                  URL.revokeObjectURL(url)
                }}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                JSON 다운로드
              </button>
              
              <button
                onClick={() => {
                  if (navigator.share) {
                    navigator.share({
                      title: result.videoInfo.title,
                      text: result.summary.overview,
                      url: window.location.href
                    })
                  }
                }}
                className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <Share2 className="w-4 h-4" />
                공유하기
              </button>
            </div>
          </div>
        )}

        {/* Features Info */}
        {!result && !isAnalyzing && (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 text-center">
                주요 기능
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="w-12 h-12 mx-auto bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl flex items-center justify-center mb-3">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">AI 요약</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    고급 AI가 동영상 내용을 분석하여 핵심 포인트를 추출합니다
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="w-12 h-12 mx-auto bg-gradient-to-r from-green-500 to-teal-500 rounded-xl flex items-center justify-center mb-3">
                    <MessageSquare className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">전체 스크립트</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    타임스탬프와 함께 전체 대화 내용을 텍스트로 제공합니다
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="w-12 h-12 mx-auto bg-gradient-to-r from-orange-500 to-red-500 rounded-xl flex items-center justify-center mb-3">
                    <BarChart3 className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">상세 분석</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    난이도, 감정, 주제 분석 등 다양한 메트릭을 제공합니다
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}