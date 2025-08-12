'use client'

import { useState, useRef } from 'react'
import { 
  Upload, Play, BarChart3, Brain, Sparkles, FileUp, 
  Settings, Info, TrendingUp, Search, Download,
  ChevronRight, AlertCircle, Check, X
} from 'lucide-react'

interface Dataset {
  name: string
  description: string
  rows: number
  columns: number
  target: string
  task: 'classification' | 'regression' | 'clustering' | 'timeseries'
}

interface ModelResult {
  name: string
  accuracy?: number
  rmse?: number
  r2?: number
  precision?: number
  recall?: number
  f1?: number
  trainTime: number
}

const SAMPLE_DATASETS: Dataset[] = [
  {
    name: 'Iris Dataset',
    description: '붓꽃 분류 (3종류)',
    rows: 150,
    columns: 4,
    target: 'species',
    task: 'classification'
  },
  {
    name: 'Boston Housing',
    description: '주택 가격 예측',
    rows: 506,
    columns: 13,
    target: 'price',
    task: 'regression'
  },
  {
    name: 'Customer Churn',
    description: '고객 이탈 예측',
    rows: 5000,
    columns: 20,
    target: 'churn',
    task: 'classification'
  },
  {
    name: 'Sales Forecast',
    description: '판매량 시계열 예측',
    rows: 365,
    columns: 5,
    target: 'sales',
    task: 'timeseries'
  }
]

export default function MLPlaygroundPyCaret() {
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentStep, setCurrentStep] = useState<'upload' | 'setup' | 'eda' | 'compare' | 'interpret'>('upload')
  const [modelResults, setModelResults] = useState<ModelResult[]>([])
  const [bestModel, setBestModel] = useState<ModelResult | null>(null)
  const [showEDA, setShowEDA] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setUploadedFile(file)
      setCurrentStep('setup')
    }
  }

  const handleDatasetSelect = (dataset: Dataset) => {
    setSelectedDataset(dataset)
    setCurrentStep('setup')
  }

  const runAutoML = async () => {
    setIsProcessing(true)
    setCurrentStep('compare')
    
    // 시뮬레이션: 여러 모델 학습 및 비교
    const models = [
      { name: 'Random Forest', accuracy: 0.95, precision: 0.94, recall: 0.96, f1: 0.95, trainTime: 2.3 },
      { name: 'XGBoost', accuracy: 0.97, precision: 0.96, recall: 0.97, f1: 0.96, trainTime: 3.1 },
      { name: 'LightGBM', accuracy: 0.96, precision: 0.95, recall: 0.97, f1: 0.96, trainTime: 1.8 },
      { name: 'SVM', accuracy: 0.92, precision: 0.91, recall: 0.93, f1: 0.92, trainTime: 4.5 },
      { name: 'Logistic Regression', accuracy: 0.88, precision: 0.87, recall: 0.89, f1: 0.88, trainTime: 0.5 },
      { name: 'Neural Network', accuracy: 0.94, precision: 0.93, recall: 0.95, f1: 0.94, trainTime: 5.2 }
    ]

    // 점진적으로 결과 추가
    for (let i = 0; i < models.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000))
      setModelResults(prev => [...prev, models[i]])
    }

    // 최고 성능 모델 선택
    const best = models.reduce((prev, current) => 
      (current.accuracy || 0) > (prev.accuracy || 0) ? current : prev
    )
    setBestModel(best)
    setIsProcessing(false)
  }

  const runEDA = () => {
    setCurrentStep('eda')
    setShowEDA(true)
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-2">ML Playground with PyCaret</h2>
        <p className="text-blue-100">
          코드 없이 머신러닝 모델을 학습하고 비교해보세요. PyCaret의 AutoML 기능을 시뮬레이션합니다.
        </p>
      </div>

      {/* 진행 단계 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-4">
        <div className="flex items-center justify-between">
          {['upload', 'setup', 'eda', 'compare', 'interpret'].map((step, idx) => (
            <div key={step} className="flex items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
                currentStep === step 
                  ? 'bg-blue-500 text-white' 
                  : idx < ['upload', 'setup', 'eda', 'compare', 'interpret'].indexOf(currentStep)
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-500'
              }`}>
                {idx < ['upload', 'setup', 'eda', 'compare', 'interpret'].indexOf(currentStep) 
                  ? <Check className="w-5 h-5" />
                  : idx + 1
                }
              </div>
              <div className="ml-2">
                <div className="text-sm font-medium">
                  {step === 'upload' && '데이터 업로드'}
                  {step === 'setup' && 'Setup 설정'}
                  {step === 'eda' && '탐색적 분석'}
                  {step === 'compare' && '모델 비교'}
                  {step === 'interpret' && '결과 해석'}
                </div>
              </div>
              {idx < 4 && <ChevronRight className="w-5 h-5 mx-4 text-gray-400" />}
            </div>
          ))}
        </div>
      </div>

      {/* Step 1: 데이터 업로드 */}
      {currentStep === 'upload' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 파일 업로드 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <FileUp className="w-5 h-5" />
              CSV 파일 업로드
            </h3>
            <div 
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
            >
              <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p className="text-gray-600 dark:text-gray-400 mb-2">
                클릭하여 파일 선택 또는 드래그 앤 드롭
              </p>
              <p className="text-sm text-gray-500">
                CSV, Excel 파일 지원 (최대 10MB)
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileUpload}
                className="hidden"
              />
            </div>
            {uploadedFile && (
              <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  업로드됨: {uploadedFile.name}
                </p>
              </div>
            )}
          </div>

          {/* 샘플 데이터셋 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              샘플 데이터셋
            </h3>
            <div className="space-y-3">
              {SAMPLE_DATASETS.map((dataset) => (
                <button
                  key={dataset.name}
                  onClick={() => handleDatasetSelect(dataset)}
                  className="w-full text-left p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-all"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium">{dataset.name}</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {dataset.description}
                      </p>
                      <div className="flex items-center gap-4 mt-1 text-xs text-gray-500">
                        <span>{dataset.rows} rows</span>
                        <span>{dataset.columns} columns</span>
                        <span className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">
                          {dataset.task}
                        </span>
                      </div>
                    </div>
                    <ChevronRight className="w-5 h-5 text-gray-400" />
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Step 2: Setup 설정 */}
      {currentStep === 'setup' && selectedDataset && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">PyCaret Setup 설정</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium mb-2">타겟 변수</label>
              <select className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700">
                <option>{selectedDataset.target}</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">학습/테스트 분할</label>
              <select className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700">
                <option>80/20</option>
                <option>70/30</option>
                <option>75/25</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">교차 검증 폴드</label>
              <input 
                type="number" 
                defaultValue={5}
                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">랜덤 시드</label>
              <input 
                type="number" 
                defaultValue={42}
                className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700"
              />
            </div>
          </div>

          <div className="mt-6 flex gap-4">
            <button
              onClick={runEDA}
              className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2"
            >
              <Search className="w-4 h-4" />
              탐색적 데이터 분석
            </button>
            <button
              onClick={runAutoML}
              className="px-6 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors flex items-center gap-2"
            >
              <Brain className="w-4 h-4" />
              AutoML 실행
            </button>
          </div>
        </div>
      )}

      {/* Step 3: EDA */}
      {currentStep === 'eda' && showEDA && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4">탐색적 데이터 분석 (EDA)</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* 데이터 정보 */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-medium mb-2">데이터 정보</h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">행 수:</span>
                  <span>{selectedDataset?.rows}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">열 수:</span>
                  <span>{selectedDataset?.columns}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">결측치:</span>
                  <span className="text-green-600">0</span>
                </div>
              </div>
            </div>

            {/* 타겟 분포 */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-medium mb-2">타겟 변수 분포</h4>
              <div className="h-32 bg-white dark:bg-gray-800 rounded flex items-center justify-center">
                <BarChart3 className="w-16 h-16 text-gray-300" />
              </div>
            </div>

            {/* 상관관계 */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-medium mb-2">상관관계 히트맵</h4>
              <div className="h-32 bg-white dark:bg-gray-800 rounded flex items-center justify-center">
                <div className="grid grid-cols-3 gap-1">
                  {[...Array(9)].map((_, i) => (
                    <div 
                      key={i} 
                      className="w-4 h-4 rounded"
                      style={{
                        backgroundColor: `rgba(59, 130, 246, ${Math.random()})`
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          <button
            onClick={runAutoML}
            className="mt-6 px-6 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors flex items-center gap-2"
          >
            <Brain className="w-4 h-4" />
            모델 학습 시작
          </button>
        </div>
      )}

      {/* Step 4: 모델 비교 */}
      {currentStep === 'compare' && (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Sparkles className="w-5 h-5" />
            모델 성능 비교
          </h3>

          {isProcessing && (
            <div className="mb-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent"></div>
                <span className="text-blue-700 dark:text-blue-300">
                  모델 학습 중... ({modelResults.length}/6)
                </span>
              </div>
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4">모델</th>
                  <th className="text-center py-3 px-4">정확도</th>
                  <th className="text-center py-3 px-4">정밀도</th>
                  <th className="text-center py-3 px-4">재현율</th>
                  <th className="text-center py-3 px-4">F1 Score</th>
                  <th className="text-center py-3 px-4">학습 시간</th>
                </tr>
              </thead>
              <tbody>
                {modelResults.map((model, idx) => (
                  <tr 
                    key={idx} 
                    className={`border-b border-gray-100 dark:border-gray-700 ${
                      bestModel?.name === model.name ? 'bg-green-50 dark:bg-green-900/20' : ''
                    }`}
                  >
                    <td className="py-3 px-4 font-medium">
                      {model.name}
                      {bestModel?.name === model.name && (
                        <span className="ml-2 text-xs bg-green-500 text-white px-2 py-0.5 rounded">
                          Best
                        </span>
                      )}
                    </td>
                    <td className="text-center py-3 px-4">
                      {((model.accuracy || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="text-center py-3 px-4">
                      {((model.precision || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="text-center py-3 px-4">
                      {((model.recall || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="text-center py-3 px-4">
                      {((model.f1 || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="text-center py-3 px-4">
                      {model.trainTime.toFixed(1)}s
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {bestModel && !isProcessing && (
            <div className="mt-6 flex gap-4">
              <button
                onClick={() => setCurrentStep('interpret')}
                className="px-6 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center gap-2"
              >
                <TrendingUp className="w-4 h-4" />
                모델 해석
              </button>
              <button className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2">
                <Download className="w-4 h-4" />
                모델 다운로드
              </button>
            </div>
          )}
        </div>
      )}

      {/* Step 5: 모델 해석 */}
      {currentStep === 'interpret' && bestModel && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 피처 중요도 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">피처 중요도</h3>
            <div className="space-y-3">
              {['Feature A', 'Feature B', 'Feature C', 'Feature D'].map((feature, idx) => (
                <div key={feature}>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">{feature}</span>
                    <span className="text-sm font-medium">{(100 - idx * 20)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${100 - idx * 20}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* SHAP 값 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">SHAP 값 시각화</h3>
            <div className="h-64 bg-gray-50 dark:bg-gray-700 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <Brain className="w-16 h-16 text-gray-300 mx-auto mb-2" />
                <p className="text-sm text-gray-500">SHAP 값 플롯</p>
              </div>
            </div>
          </div>

          {/* 혼동 행렬 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">혼동 행렬</h3>
            <div className="grid grid-cols-2 gap-2 max-w-xs mx-auto">
              <div className="bg-green-100 dark:bg-green-900/30 p-8 rounded text-center">
                <div className="text-2xl font-bold text-green-700 dark:text-green-300">85</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">True Positive</div>
              </div>
              <div className="bg-red-100 dark:bg-red-900/30 p-8 rounded text-center">
                <div className="text-2xl font-bold text-red-700 dark:text-red-300">5</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">False Positive</div>
              </div>
              <div className="bg-red-100 dark:bg-red-900/30 p-8 rounded text-center">
                <div className="text-2xl font-bold text-red-700 dark:text-red-300">3</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">False Negative</div>
              </div>
              <div className="bg-green-100 dark:bg-green-900/30 p-8 rounded text-center">
                <div className="text-2xl font-bold text-green-700 dark:text-green-300">92</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">True Negative</div>
              </div>
            </div>
          </div>

          {/* ROC 커브 */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold mb-4">ROC 커브</h3>
            <div className="h-64 bg-gray-50 dark:bg-gray-700 rounded-lg flex items-center justify-center">
              <svg viewBox="0 0 100 100" className="w-full h-full max-w-xs">
                <line x1="0" y1="100" x2="100" y2="0" stroke="gray" strokeDasharray="2,2" />
                <path 
                  d="M 0,100 Q 20,20 100,0" 
                  fill="none" 
                  stroke="blue" 
                  strokeWidth="2"
                />
                <text x="50" y="95" fontSize="8" textAnchor="middle">
                  AUC = 0.97
                </text>
              </svg>
            </div>
          </div>
        </div>
      )}

      {/* 정보 패널 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 mt-0.5" />
          <div className="text-sm text-blue-700 dark:text-blue-300">
            <p className="font-medium mb-1">PyCaret AutoML 시뮬레이션</p>
            <p>
              이 시뮬레이터는 PyCaret의 주요 기능을 체험할 수 있도록 설계되었습니다. 
              실제 PyCaret은 더 많은 모델과 고급 기능을 제공합니다.
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              <span className="px-2 py-1 bg-blue-100 dark:bg-blue-800 rounded text-xs">
                setup() - 데이터 준비
              </span>
              <span className="px-2 py-1 bg-blue-100 dark:bg-blue-800 rounded text-xs">
                compare_models() - 모델 비교
              </span>
              <span className="px-2 py-1 bg-blue-100 dark:bg-blue-800 rounded text-xs">
                interpret_model() - 모델 해석
              </span>
              <span className="px-2 py-1 bg-blue-100 dark:bg-blue-800 rounded text-xs">
                predict_model() - 예측
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}