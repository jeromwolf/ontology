'use client'

import { useState } from 'react'
import { Eye, Database, TrendingUp, AlertCircle, Lock, Unlock } from 'lucide-react'

interface QueryResult {
  input: string
  output: string
  confidence: number
}

interface ExtractionProgress {
  queriesMade: number
  accuracyExtracted: number
  parametersRevealed: number
  timeElapsed: number
}

export default function ModelExtractionSimulator() {
  const [targetModel, setTargetModel] = useState<'image-classifier' | 'nlp-model' | 'recommendation'>('image-classifier')
  const [extractionMethod, setExtractionMethod] = useState<'active' | 'passive' | 'hybrid'>('active')
  const [queryBudget, setQueryBudget] = useState(1000)
  const [isExtracting, setIsExtracting] = useState(false)
  const [extractionProgress, setExtractionProgress] = useState<ExtractionProgress | null>(null)
  const [queryHistory, setQueryHistory] = useState<QueryResult[]>([])
  const [extractedModel, setExtractedModel] = useState<string | null>(null)

  const modelDetails = {
    'image-classifier': {
      name: 'Image Classifier',
      description: 'ResNet-50 기반 이미지 분류 모델',
      parameters: '25M',
      classes: 1000,
      api_cost: '$0.001/query'
    },
    'nlp-model': {
      name: 'Text Sentiment Analyzer',
      description: 'BERT 기반 감성 분석 모델',
      parameters: '110M',
      classes: 3,
      api_cost: '$0.002/query'
    },
    'recommendation': {
      name: 'Recommendation Engine',
      description: '협업 필터링 추천 시스템',
      parameters: '50M',
      classes: 'N/A',
      api_cost: '$0.005/query'
    }
  }

  const startExtraction = async () => {
    setIsExtracting(true)
    setExtractionProgress(null)
    setQueryHistory([])
    setExtractedModel(null)

    // 추출 과정 시뮬레이션
    const totalQueries = Math.min(queryBudget, 1000)
    const intervalTime = 100 // ms per query simulation

    for (let i = 0; i < totalQueries; i++) {
      await new Promise(resolve => setTimeout(resolve, intervalTime))
      
      // 쿼리 결과 시뮬레이션
      const sampleInputs = {
        'image-classifier': ['cat.jpg', 'dog.jpg', 'car.jpg', 'tree.jpg'],
        'nlp-model': ['좋은 영화였어요', '별로였음', '그냥 그랬어요', '최고!'],
        'recommendation': ['user123', 'user456', 'user789', 'user101']
      }

      const sampleOutputs = {
        'image-classifier': ['cat: 0.95', 'dog: 0.89', 'car: 0.92', 'tree: 0.78'],
        'nlp-model': ['positive: 0.87', 'negative: 0.91', 'neutral: 0.65', 'positive: 0.94'],
        'recommendation': ['item_A: 4.2', 'item_B: 3.8', 'item_C: 4.5', 'item_D: 3.1']
      }

      const input = sampleInputs[targetModel][i % sampleInputs[targetModel].length]
      const output = sampleOutputs[targetModel][i % sampleOutputs[targetModel].length]
      const confidence = 0.6 + Math.random() * 0.4

      const newQuery: QueryResult = {
        input,
        output,
        confidence
      }

      setQueryHistory(prev => [...prev.slice(-9), newQuery]) // Keep last 10 queries

      // 진행상황 업데이트
      const progress: ExtractionProgress = {
        queriesMade: i + 1,
        accuracyExtracted: Math.min(95, (i + 1) / totalQueries * 85 + Math.random() * 10),
        parametersRevealed: Math.min(100, (i + 1) / totalQueries * 90 + Math.random() * 10),
        timeElapsed: (i + 1) * intervalTime / 1000
      }

      setExtractionProgress(progress)
    }

    // 추출 완료
    const finalAccuracy = 75 + Math.random() * 20
    setExtractedModel(`추출된 모델 (정확도: ${finalAccuracy.toFixed(1)}%)`)
    setIsExtracting(false)
  }

  const stopExtraction = () => {
    setIsExtracting(false)
  }

  const resetSimulation = () => {
    setIsExtracting(false)
    setExtractionProgress(null)
    setQueryHistory([])
    setExtractedModel(null)
  }

  return (
    <div className="space-y-6">
      {/* 설정 패널 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">추출 설정</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              타겟 모델
            </label>
            <select
              value={targetModel}
              onChange={(e) => setTargetModel(e.target.value as any)}
              className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md text-gray-900 dark:text-white"
            >
              <option value="image-classifier">이미지 분류기</option>
              <option value="nlp-model">텍스트 감성 분석</option>
              <option value="recommendation">추천 시스템</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              추출 방법
            </label>
            <select
              value={extractionMethod}
              onChange={(e) => setExtractionMethod(e.target.value as any)}
              className="w-full px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md text-gray-900 dark:text-white"
            >
              <option value="active">능동적 추출</option>
              <option value="passive">수동적 추출</option>
              <option value="hybrid">하이브리드</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              쿼리 예산: {queryBudget}
            </label>
            <input
              type="range"
              min="100"
              max="5000"
              step="100"
              value={queryBudget}
              onChange={(e) => setQueryBudget(parseInt(e.target.value))}
              className="w-full"
              disabled={isExtracting}
            />
          </div>
        </div>
      </div>

      {/* 타겟 모델 정보 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              타겟 모델 정보
            </h3>
            <div className="space-y-2 text-sm">
              <p><span className="text-gray-600 dark:text-gray-400">모델:</span> {modelDetails[targetModel].name}</p>
              <p><span className="text-gray-600 dark:text-gray-400">설명:</span> {modelDetails[targetModel].description}</p>
              <p><span className="text-gray-600 dark:text-gray-400">파라미터:</span> {modelDetails[targetModel].parameters}</p>
              <p><span className="text-gray-600 dark:text-gray-400">클래스:</span> {modelDetails[targetModel].classes}</p>
              <p><span className="text-gray-600 dark:text-gray-400">API 비용:</span> {modelDetails[targetModel].api_cost}</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-red-600">
            <Lock className="w-5 h-5" />
            <span className="text-sm font-medium">보호됨</span>
          </div>
        </div>
      </div>

      {/* 제어 버튼 */}
      <div className="flex gap-4">
        <button
          onClick={startExtraction}
          disabled={isExtracting}
          className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Eye className="w-5 h-5" />
          {isExtracting ? '추출 중...' : '모델 추출 시작'}
        </button>
        
        {isExtracting && (
          <button
            onClick={stopExtraction}
            className="flex items-center gap-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            중지
          </button>
        )}
        
        <button
          onClick={resetSimulation}
          className="flex items-center gap-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          초기화
        </button>
      </div>

      {/* 진행 상황 */}
      {extractionProgress && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">추출 진행 상황</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {extractionProgress.queriesMade}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">쿼리 수행</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {extractionProgress.accuracyExtracted.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">정확도 추출</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                {extractionProgress.parametersRevealed.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">파라미터 추정</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                {extractionProgress.timeElapsed.toFixed(1)}s
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">경과 시간</div>
            </div>
          </div>

          {/* 진행 바 */}
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-700 dark:text-gray-300">전체 진행도</span>
                <span className="text-gray-700 dark:text-gray-300">
                  {((extractionProgress.queriesMade / queryBudget) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${(extractionProgress.queriesMade / queryBudget) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 쿼리 히스토리 */}
      {queryHistory.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">최근 쿼리</h3>
          
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {queryHistory.slice().reverse().map((query, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="flex-1">
                  <span className="text-sm font-mono text-gray-900 dark:text-white">
                    {query.input}
                  </span>
                  <span className="mx-2 text-gray-400">→</span>
                  <span className="text-sm font-mono text-blue-600 dark:text-blue-400">
                    {query.output}
                  </span>
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  {(query.confidence * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 추출 결과 */}
      {extractedModel && (
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 border border-red-200 dark:border-red-800">
          <div className="flex items-start gap-4">
            <AlertCircle className="w-6 h-6 text-red-600 mt-1" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-red-900 dark:text-red-100 mb-2">
                모델 추출 완료
              </h3>
              <p className="text-red-800 dark:text-red-200 mb-4">
                {extractedModel}로 타겟 모델의 복제본이 생성되었습니다.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">추출된 정보</h4>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 모델 아키텍처 추정</li>
                    <li>• 결정 경계 매핑</li>
                    <li>• 파라미터 가중치 근사</li>
                    <li>• 훈련 데이터 특성 추론</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">보안 영향</h4>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• 지적재산권 침해</li>
                    <li>• 비즈니스 모델 유출</li>
                    <li>• 경쟁 우위 상실</li>
                    <li>• 추가 공격 가능성</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 방어 방법 */}
      <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-green-900 dark:text-green-100 mb-3">
          모델 추출 방어 방법
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium text-green-800 dark:text-green-200 mb-2">예방 조치</h4>
            <ul className="text-sm text-green-700 dark:text-green-300 space-y-1">
              <li>• API 호출 제한 및 모니터링</li>
              <li>• 쿼리 패턴 이상 탐지</li>
              <li>• 출력 노이즈 추가</li>
              <li>• 차등 프라이버시 적용</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-green-800 dark:text-green-200 mb-2">탐지 방법</h4>
            <ul className="text-sm text-green-700 dark:text-green-300 space-y-1">
              <li>• 비정상적 사용 패턴 감지</li>
              <li>• 모델 워터마킹</li>
              <li>• 허니팟 쿼리 삽입</li>
              <li>• 사용자 행동 분석</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}