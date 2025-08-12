'use client'

import { useState, useEffect } from 'react'
import { 
  Brain, BarChart3, Activity, AlertCircle, Info, 
  Play, Settings, CheckCircle, X, Download,
  Shuffle, GitBranch, Box, Layers, TreePine, Zap
} from 'lucide-react'

interface ModelResult {
  name: string
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  trainTime: number
  confusionMatrix: number[][]
  icon: React.ReactNode
  color: string
  description?: string
}

interface Dataset {
  name: string
  features: string[]
  target: string
  samples: number
  classes: string[]
  description: string
}

const SAMPLE_DATASETS: Dataset[] = [
  {
    name: 'Iris 분류',
    features: ['꽃받침 길이', '꽃받침 너비', '꽃잎 길이', '꽃잎 너비'],
    target: '품종',
    samples: 150,
    classes: ['Setosa', 'Versicolor', 'Virginica'],
    description: '붓꽃 품종 분류 (3개 클래스)'
  },
  {
    name: '신용카드 사기 탐지',
    features: ['거래액', '시간', '위치', '가맹점 유형', '이전 거래 패턴'],
    target: '사기 여부',
    samples: 5000,
    classes: ['정상', '사기'],
    description: '이진 분류 문제'
  },
  {
    name: '고객 이탈 예측',
    features: ['가입 기간', '월 이용료', '서비스 이용량', '고객 문의 횟수', '결제 연체'],
    target: '이탈 여부',
    samples: 3000,
    classes: ['유지', '이탈'],
    description: '이진 분류 문제'
  }
]

export default function ClassificationModelComparator() {
  const [selectedDataset, setSelectedDataset] = useState<Dataset>(SAMPLE_DATASETS[0])
  const [isTraining, setIsTraining] = useState(false)
  const [modelResults, setModelResults] = useState<ModelResult[]>([])
  const [selectedModel, setSelectedModel] = useState<ModelResult | null>(null)
  const [showOneHotEncoding, setShowOneHotEncoding] = useState(false)
  const [encodingExample, setEncodingExample] = useState<any>(null)

  // 모델 정의
  const models = [
    { 
      name: 'Logistic Regression', 
      icon: <Activity className="w-5 h-5" />, 
      color: 'blue',
      description: '해석이 용이한 기준선 모델',
      strength: '해석력',
      performance: 'baseline'
    },
    { 
      name: 'Random Forest', 
      icon: <TreePine className="w-5 h-5" />, 
      color: 'green',
      description: '강력한 앙상블 모델',
      strength: '안정성',
      performance: 'strong'
    },
    { 
      name: 'XGBoost', 
      icon: <Zap className="w-5 h-5" />, 
      color: 'purple',
      description: '최고 성능의 부스팅 모델',
      strength: '정확도',
      performance: 'strong'
    },
    { 
      name: 'SVM', 
      icon: <GitBranch className="w-5 h-5" />, 
      color: 'orange',
      description: '비선형 분류에 강력',
      strength: '복잡한 경계',
      performance: 'strong'
    },
    { 
      name: 'k-NN', 
      icon: <Box className="w-5 h-5" />, 
      color: 'pink',
      description: '스케일링 후 비교용',
      strength: '단순성',
      performance: 'comparison'
    },
    { 
      name: 'Gradient Boost', 
      icon: <Layers className="w-5 h-5" />, 
      color: 'red',
      description: '순차적 앙상블 학습',
      strength: '정확도',
      performance: 'strong'
    }
  ]

  // 혼동행렬 생성 (시뮬레이션)
  const generateConfusionMatrix = (accuracy: number, numClasses: number): number[][] => {
    const matrix: number[][] = []
    const total = 100
    
    if (numClasses === 2) {
      // 이진 분류
      const tp = Math.floor(accuracy * total / 100)
      const tn = Math.floor(accuracy * total / 100 * 0.9)
      const fp = Math.floor((100 - accuracy) * total / 100 * 0.4)
      const fn = total - tp - tn - fp
      
      return [[tp, fp], [fn, tn]]
    } else {
      // 다중 분류
      for (let i = 0; i < numClasses; i++) {
        matrix[i] = []
        for (let j = 0; j < numClasses; j++) {
          if (i === j) {
            matrix[i][j] = Math.floor(accuracy * total / numClasses / 100)
          } else {
            matrix[i][j] = Math.floor((100 - accuracy) * total / numClasses / 100 / (numClasses - 1))
          }
        }
      }
    }
    
    return matrix
  }

  // 모델 학습 시뮬레이션
  const trainModels = async () => {
    setIsTraining(true)
    setModelResults([])
    
    for (let i = 0; i < models.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 800))
      
      const model = models[i]
      const baseAccuracy = 75 + Math.random() * 20
      const accuracy = Math.min(95, baseAccuracy + (i === 1 || i === 2 ? 5 : 0)) // RF, XGBoost 보너스
      
      const result: ModelResult = {
        name: model.name,
        accuracy: accuracy,
        precision: accuracy - Math.random() * 3,
        recall: accuracy - Math.random() * 5,
        f1Score: accuracy - Math.random() * 2,
        trainTime: 0.5 + Math.random() * 4,
        confusionMatrix: generateConfusionMatrix(accuracy, selectedDataset.classes.length),
        icon: model.icon,
        color: model.color,
        description: model.description
      }
      
      setModelResults(prev => [...prev, result])
    }
    
    setIsTraining(false)
  }

  // One-Hot Encoding 예제 생성
  const generateOneHotExample = () => {
    const categories = ['Red', 'Green', 'Blue']
    const original = ['Green', 'Red', 'Blue', 'Green']
    const encoded = original.map(color => 
      categories.map(cat => cat === color ? 1 : 0)
    )
    
    setEncodingExample({
      categories,
      original,
      encoded
    })
    setShowOneHotEncoding(true)
  }

  // CSV 다운로드
  const downloadResults = () => {
    let csv = 'Model,Accuracy,Precision,Recall,F1-Score,Training Time\n'
    modelResults.forEach(model => {
      csv += `${model.name},${model.accuracy.toFixed(2)},${model.precision.toFixed(2)},${model.recall.toFixed(2)},${model.f1Score.toFixed(2)},${model.trainTime.toFixed(2)}\n`
    })
    
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `classification_models_comparison_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-2">분류 모델 비교 실험실</h2>
        <p className="text-blue-100">
          6가지 주요 분류 알고리즘의 성능을 비교하고 혼동행렬을 분석합니다
        </p>
      </div>

      {/* 데이터셋 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4">데이터셋 선택</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {SAMPLE_DATASETS.map((dataset) => (
            <button
              key={dataset.name}
              onClick={() => setSelectedDataset(dataset)}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                selectedDataset.name === dataset.name
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
              }`}
            >
              <h4 className="font-medium mb-1">{dataset.name}</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                {dataset.description}
              </p>
              <div className="flex items-center gap-4 text-xs text-gray-500">
                <span>{dataset.samples} 샘플</span>
                <span>{dataset.classes.length} 클래스</span>
                <span>{dataset.features.length} 특성</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 전처리 옵션 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5" />
          데이터 전처리 파이프라인
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium mb-3">전처리 설정</h4>
            <div className="space-y-2">
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="rounded" />
                <span>결측치 처리 (평균값/최빈값 대체)</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="rounded" />
                <span>Min-Max 스케일링 (수치형만, 훈련 세트 기준)</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="rounded" />
                <span>범주형 자동 처리 (One-Hot + 결측치 대치)</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="rounded" />
                <span>Stratified Split (클래스 비율 유지)</span>
              </label>
            </div>
            <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-sm">
              <p className="text-yellow-700 dark:text-yellow-300">
                <strong>⚠️ 데이터 누수 방지</strong><br/>
                스케일링은 훈련 세트에만 fit되고, 테스트 세트는 transform만 적용
              </p>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium mb-3">One-Hot Encoding 예제</h4>
            <button
              onClick={generateOneHotExample}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
            >
              인코딩 예제 보기
            </button>
            
            {showOneHotEncoding && encodingExample && (
              <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg text-sm">
                <p className="font-medium mb-2">원본: {encodingExample.original.join(', ')}</p>
                <p className="font-medium mb-1">인코딩 결과:</p>
                <div className="font-mono text-xs">
                  {encodingExample.encoded.map((row: number[], idx: number) => (
                    <div key={idx}>
                      {encodingExample.original[idx]}: [{row.join(', ')}]
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 모델 학습 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Brain className="w-5 h-5" />
            모델 학습 및 비교
          </h3>
          {modelResults.length > 0 && (
            <button
              onClick={downloadResults}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              결과 다운로드
            </button>
          )}
        </div>

        <button
          onClick={trainModels}
          disabled={isTraining}
          className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2 mb-6"
        >
          {isTraining ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent" />
              모델 학습 중... ({modelResults.length}/6)
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              6개 모델 학습 시작
            </>
          )}
        </button>

        {/* 모델 결과 */}
        {modelResults.length > 0 && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {modelResults.map((model) => (
                <div
                  key={model.name}
                  onClick={() => setSelectedModel(model)}
                  className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                    selectedModel?.name === model.name
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <div className={`text-${model.color}-600`}>
                      {model.icon}
                    </div>
                    <h4 className="font-medium">{model.name}</h4>
                  </div>
                  <p className="text-xs text-gray-500 mb-3">
                    {modelResults.find(m => m.name === model.name)?.description || ''}
                  </p>
                  
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">정확도:</span>
                      <span className="font-medium">{model.accuracy.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">F1-Score:</span>
                      <span className="font-medium">{model.f1Score.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">학습 시간:</span>
                      <span className="font-medium">{model.trainTime.toFixed(1)}s</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 혼동행렬 */}
      {selectedModel && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">
            {selectedModel.name} - 혼동행렬 (Confusion Matrix)
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3">혼동행렬</h4>
              <div className="overflow-x-auto">
                <table className="w-full border border-gray-200 dark:border-gray-700">
                  <thead>
                    <tr className="bg-gray-50 dark:bg-gray-700">
                      <th className="p-2 border border-gray-200 dark:border-gray-600"></th>
                      {selectedDataset.classes.map((cls, idx) => (
                        <th key={idx} className="p-2 border border-gray-200 dark:border-gray-600 text-sm">
                          예측: {cls}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {selectedModel.confusionMatrix.map((row, i) => (
                      <tr key={i}>
                        <td className="p-2 border border-gray-200 dark:border-gray-600 font-medium text-sm">
                          실제: {selectedDataset.classes[i]}
                        </td>
                        {row.map((value, j) => (
                          <td 
                            key={j} 
                            className={`p-2 border border-gray-200 dark:border-gray-600 text-center ${
                              i === j ? 'bg-green-100 dark:bg-green-900/30 font-medium' : 'bg-red-50 dark:bg-red-900/20'
                            }`}
                          >
                            {value}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            
            <div>
              <h4 className="font-medium mb-3">성능 지표</h4>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">정확도 (Accuracy)</span>
                    <span className="text-sm font-medium">{selectedModel.accuracy.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${selectedModel.accuracy}%` }}
                    />
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">정밀도 (Precision)</span>
                    <span className="text-sm font-medium">{selectedModel.precision.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${selectedModel.precision}%` }}
                    />
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">재현율 (Recall)</span>
                    <span className="text-sm font-medium">{selectedModel.recall.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-orange-500 h-2 rounded-full"
                      style={{ width: `${selectedModel.recall}%` }}
                    />
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">F1-Score</span>
                    <span className="text-sm font-medium">{selectedModel.f1Score.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-purple-500 h-2 rounded-full"
                      style={{ width: `${selectedModel.f1Score}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    * 불균형 데이터셋에서 균형잡힌 지표
                  </p>
                </div>
              </div>
              
              {selectedDataset.classes.length === 2 && (
                <div className="mt-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg text-sm">
                  <p className="text-purple-700 dark:text-purple-300">
                    <strong>💡 F1-Score (양성=1 기준)</strong><br/>
                    불균형 이진 분류에서 정밀도와 재현율의 조화평균으로 균형잡힌 평가 제공
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* MLOps 파이프라인 */}
      <div className="bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <GitBranch className="w-5 h-5" />
          MLOps 파이프라인
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <div className="text-blue-600 mb-2">
              <Shuffle className="w-8 h-8" />
            </div>
            <h4 className="font-medium mb-1">1. 데이터 수집</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              실시간 데이터 파이프라인
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <div className="text-green-600 mb-2">
              <Settings className="w-8 h-8" />
            </div>
            <h4 className="font-medium mb-1">2. 전처리</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              자동화된 피처 엔지니어링
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <div className="text-purple-600 mb-2">
              <Brain className="w-8 h-8" />
            </div>
            <h4 className="font-medium mb-1">3. 모델 학습</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              자동 하이퍼파라미터 튜닝
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <div className="text-orange-600 mb-2">
              <Activity className="w-8 h-8" />
            </div>
            <h4 className="font-medium mb-1">4. 모니터링</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              성능 추적 및 드리프트 감지
            </p>
          </div>
        </div>
      </div>

      {/* 정보 패널 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 mt-0.5" />
          <div className="text-sm text-blue-700 dark:text-blue-300">
            <p className="font-medium mb-1">분류 모델 비교 실험실</p>
            <p>
              이 시뮬레이터는 6가지 주요 분류 알고리즘의 성능을 비교합니다.
              일반적으로 <strong>SVM, Gradient Boosting, XGBoost, Random Forest</strong>가 강력하며,
              로지스틱 회귀는 해석 용이한 기준선, k-NN은 스케일링 후 비교용으로 유용합니다.
            </p>
            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="font-medium mb-1">혼동행렬 해석:</p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>대각선: 올바른 예측 (True Positive/Negative)</li>
                  <li>비대각선: 잘못된 예측 (False Positive/Negative)</li>
                  <li>행: 실제 클래스, 열: 예측 클래스</li>
                </ul>
              </div>
              <div>
                <p className="font-medium mb-1">성능 지표:</p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>정확도: 전체 정답률</li>
                  <li>정밀도: 예측한 것 중 정답 비율</li>
                  <li>재현율: 실제 정답 중 예측 비율</li>
                  <li>F1-Score: 정밀도와 재현율의 조화 평균</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}