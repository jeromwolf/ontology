'use client'

import { useState, useEffect } from 'react'
import { 
  Wine, TrendingUp, AlertCircle, Info, Download, 
  BarChart3, DollarSign, Grape, Calendar, MapPin,
  Star, Award, Filter, Brain, ChevronRight
} from 'lucide-react'

interface WineData {
  id: number
  name: string
  year: number
  region: string
  variety: string
  alcohol: number
  acidity: number
  sugar: number
  tannins: number
  rating: number
  reviews: number
  actualPrice: number
  predictedPrice?: number
}

// 실제 와인 데이터 (샘플)
const WINE_DATASET: WineData[] = [
  { id: 1, name: "Château Margaux", year: 2010, region: "Bordeaux", variety: "Cabernet Sauvignon", alcohol: 13.5, acidity: 3.4, sugar: 2.1, tannins: 3.8, rating: 95, reviews: 342, actualPrice: 850 },
  { id: 2, name: "Opus One", year: 2018, region: "Napa Valley", variety: "Cabernet Blend", alcohol: 14.5, acidity: 3.6, sugar: 1.8, tannins: 3.5, rating: 93, reviews: 256, actualPrice: 400 },
  { id: 3, name: "Penfolds Grange", year: 2016, region: "South Australia", variety: "Shiraz", alcohol: 14.8, acidity: 3.3, sugar: 2.3, tannins: 4.0, rating: 98, reviews: 189, actualPrice: 650 },
  { id: 4, name: "Domaine Romanée-Conti", year: 2015, region: "Burgundy", variety: "Pinot Noir", alcohol: 13.0, acidity: 3.7, sugar: 1.5, tannins: 2.8, rating: 99, reviews: 412, actualPrice: 15000 },
  { id: 5, name: "Sassicaia", year: 2019, region: "Tuscany", variety: "Cabernet Franc", alcohol: 13.8, acidity: 3.5, sugar: 2.0, tannins: 3.3, rating: 94, reviews: 298, actualPrice: 280 },
  { id: 6, name: "Château Le Pin", year: 2012, region: "Pomerol", variety: "Merlot", alcohol: 13.5, acidity: 3.4, sugar: 2.2, tannins: 3.0, rating: 96, reviews: 167, actualPrice: 3200 },
  { id: 7, name: "Vega Sicilia Único", year: 2011, region: "Ribera del Duero", variety: "Tempranillo", alcohol: 14.0, acidity: 3.3, sugar: 1.9, tannins: 3.7, rating: 95, reviews: 234, actualPrice: 420 },
  { id: 8, name: "Château d'Yquem", year: 2017, region: "Sauternes", variety: "Semillon", alcohol: 13.5, acidity: 4.2, sugar: 8.5, tannins: 0.5, rating: 97, reviews: 178, actualPrice: 380 },
]

// 지역별 가격 승수
const REGION_MULTIPLIERS: { [key: string]: number } = {
  "Bordeaux": 1.5,
  "Burgundy": 2.0,
  "Napa Valley": 1.3,
  "Tuscany": 1.2,
  "Pomerol": 1.8,
  "South Australia": 1.0,
  "Ribera del Duero": 1.1,
  "Sauternes": 1.4
}

export default function WinePricePredictor() {
  const [selectedWine, setSelectedWine] = useState<WineData | null>(null)
  const [customWine, setCustomWine] = useState({
    year: 2020,
    region: "Bordeaux",
    variety: "Cabernet Sauvignon",
    alcohol: 13.5,
    acidity: 3.5,
    sugar: 2.0,
    tannins: 3.5,
    rating: 90,
    reviews: 100
  })
  const [predictions, setPredictions] = useState<WineData[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const [modelAccuracy, setModelAccuracy] = useState(0)
  const [showCustomForm, setShowCustomForm] = useState(false)

  // 간단한 가격 예측 모델 (실제로는 ML 모델 사용)
  const predictPrice = (wine: Omit<WineData, 'id' | 'name' | 'actualPrice' | 'predictedPrice'>) => {
    // 기본 가격 계산
    let basePrice = 50

    // 연도 효과 (오래된 와인일수록 비쌈)
    const age = 2024 - wine.year
    basePrice += age * 5

    // 평점 효과 (지수적 증가)
    basePrice *= Math.pow(wine.rating / 85, 3)

    // 지역 효과
    const regionMultiplier = REGION_MULTIPLIERS[wine.region] || 1.0
    basePrice *= regionMultiplier

    // 알코올 도수 효과
    basePrice *= (wine.alcohol / 12)

    // 리뷰 수 효과 (인기도)
    basePrice *= (1 + wine.reviews / 1000)

    // 품종별 조정
    if (wine.variety === "Pinot Noir") basePrice *= 1.3
    if (wine.variety === "Cabernet Sauvignon") basePrice *= 1.2

    // 랜덤 변동성 추가 (±10%)
    const randomFactor = 0.9 + Math.random() * 0.2
    
    return Math.round(basePrice * randomFactor)
  }

  // 모델 학습 시뮬레이션
  const trainModel = async () => {
    setIsTraining(true)
    setPredictions([])
    
    // 학습 시뮬레이션
    for (let i = 0; i < WINE_DATASET.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 300))
      
      const wine = WINE_DATASET[i]
      const predicted = predictPrice(wine)
      
      setPredictions(prev => [...prev, {
        ...wine,
        predictedPrice: predicted
      }])
    }

    // 정확도 계산 (MAE 기반)
    const totalError = WINE_DATASET.reduce((sum, wine, idx) => {
      const predicted = predictPrice(wine)
      return sum + Math.abs(wine.actualPrice - predicted) / wine.actualPrice
    }, 0)
    
    const accuracy = Math.max(0, 100 - (totalError / WINE_DATASET.length * 100))
    setModelAccuracy(accuracy)
    setIsTraining(false)
  }

  // 커스텀 와인 가격 예측
  const predictCustomWine = () => {
    const predicted = predictPrice(customWine)
    alert(`예상 가격: $${predicted}`)
  }

  // CSV 다운로드 기능
  const downloadData = (format: 'csv' | 'json') => {
    if (format === 'csv') {
      // CSV 헤더
      let csv = 'ID,Name,Year,Region,Variety,Alcohol,Acidity,Sugar,Tannins,Rating,Reviews,Actual Price,Predicted Price\n'
      
      // 데이터 추가
      predictions.forEach(wine => {
        csv += `${wine.id},"${wine.name}",${wine.year},"${wine.region}","${wine.variety}",${wine.alcohol},${wine.acidity},${wine.sugar},${wine.tannins},${wine.rating},${wine.reviews},${wine.actualPrice},${wine.predictedPrice || ''}\n`
      })

      // 다운로드
      const blob = new Blob([csv], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `wine_price_predictions_${new Date().toISOString().split('T')[0]}.csv`
      a.click()
      URL.revokeObjectURL(url)
    } else {
      // JSON 다운로드
      const json = JSON.stringify(predictions, null, 2)
      const blob = new Blob([json], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `wine_price_predictions_${new Date().toISOString().split('T')[0]}.json`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl p-6">
        <div className="flex items-center gap-3 mb-2">
          <Wine className="w-8 h-8" />
          <h2 className="text-2xl font-bold">와인 가격 예측 AI</h2>
        </div>
        <p className="text-purple-100">
          와인의 특성을 분석하여 적정 가격을 예측하는 머신러닝 모델을 체험해보세요
        </p>
      </div>

      {/* 모델 학습 섹션 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5" />
          모델 학습
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              실제 와인 데이터를 사용하여 가격 예측 모델을 학습시킵니다.
              데이터셋에는 {WINE_DATASET.length}개의 고급 와인 정보가 포함되어 있습니다.
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <Grape className="w-4 h-4 text-purple-500" />
                <span>품종별 특성 분석</span>
              </div>
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4 text-purple-500" />
                <span>생산 지역 프리미엄 반영</span>
              </div>
              <div className="flex items-center gap-2">
                <Star className="w-4 h-4 text-purple-500" />
                <span>전문가 평점 가중치</span>
              </div>
              <div className="flex items-center gap-2">
                <Calendar className="w-4 h-4 text-purple-500" />
                <span>빈티지 연도 효과</span>
              </div>
            </div>
          </div>
          
          <div className="flex flex-col items-center justify-center">
            {modelAccuracy > 0 ? (
              <div className="text-center">
                <div className="text-4xl font-bold text-green-600 mb-2">
                  {modelAccuracy.toFixed(1)}%
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">모델 정확도</p>
              </div>
            ) : (
              <button
                onClick={trainModel}
                disabled={isTraining}
                className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 flex items-center gap-2"
              >
                {isTraining ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                    학습 중...
                  </>
                ) : (
                  <>
                    <Brain className="w-5 h-5" />
                    모델 학습 시작
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* 학습 결과 테이블 */}
        {predictions.length > 0 && (
          <>
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-medium text-gray-700 dark:text-gray-300">예측 결과</h4>
              <div className="flex gap-2">
                <button
                  onClick={() => downloadData('csv')}
                  className="px-3 py-1.5 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors flex items-center gap-1"
                >
                  <Download className="w-4 h-4" />
                  CSV 다운로드
                </button>
                <button
                  onClick={() => downloadData('json')}
                  className="px-3 py-1.5 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-1"
                >
                  <Download className="w-4 h-4" />
                  JSON 다운로드
                </button>
              </div>
            </div>
            <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4">와인명</th>
                  <th className="text-center py-3 px-4">연도</th>
                  <th className="text-center py-3 px-4">지역</th>
                  <th className="text-center py-3 px-4">평점</th>
                  <th className="text-center py-3 px-4">실제 가격</th>
                  <th className="text-center py-3 px-4">예측 가격</th>
                  <th className="text-center py-3 px-4">오차</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((wine) => {
                  const error = Math.abs((wine.actualPrice - wine.predictedPrice!) / wine.actualPrice * 100)
                  return (
                    <tr key={wine.id} className="border-b border-gray-100 dark:border-gray-700">
                      <td className="py-3 px-4 font-medium">{wine.name}</td>
                      <td className="text-center py-3 px-4">{wine.year}</td>
                      <td className="text-center py-3 px-4">{wine.region}</td>
                      <td className="text-center py-3 px-4">
                        <span className="inline-flex items-center gap-1">
                          <Star className="w-3 h-3 text-yellow-500" />
                          {wine.rating}
                        </span>
                      </td>
                      <td className="text-center py-3 px-4 font-medium">
                        ${wine.actualPrice.toLocaleString()}
                      </td>
                      <td className="text-center py-3 px-4 font-medium text-purple-600">
                        ${wine.predictedPrice?.toLocaleString()}
                      </td>
                      <td className="text-center py-3 px-4">
                        <span className={`px-2 py-1 rounded text-xs ${
                          error < 20 ? 'bg-green-100 text-green-700' : 
                          error < 40 ? 'bg-yellow-100 text-yellow-700' : 
                          'bg-red-100 text-red-700'
                        }`}>
                          {error.toFixed(1)}%
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          </>
        )}
      </div>

      {/* 커스텀 예측 섹션 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <DollarSign className="w-5 h-5" />
            나만의 와인 가격 예측
          </h3>
          <button
            onClick={() => setShowCustomForm(!showCustomForm)}
            className="text-purple-600 hover:text-purple-700 flex items-center gap-1"
          >
            {showCustomForm ? '접기' : '펼치기'}
            <ChevronRight className={`w-4 h-4 transition-transform ${showCustomForm ? 'rotate-90' : ''}`} />
          </button>
        </div>

        {showCustomForm && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">생산 연도</label>
              <input
                type="number"
                value={customWine.year}
                onChange={(e) => setCustomWine({...customWine, year: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg"
                min="1900"
                max="2024"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">생산 지역</label>
              <select
                value={customWine.region}
                onChange={(e) => setCustomWine({...customWine, region: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
              >
                {Object.keys(REGION_MULTIPLIERS).map(region => (
                  <option key={region} value={region}>{region}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">품종</label>
              <select
                value={customWine.variety}
                onChange={(e) => setCustomWine({...customWine, variety: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
              >
                <option value="Cabernet Sauvignon">Cabernet Sauvignon</option>
                <option value="Merlot">Merlot</option>
                <option value="Pinot Noir">Pinot Noir</option>
                <option value="Shiraz">Shiraz</option>
                <option value="Chardonnay">Chardonnay</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">알코올 도수 (%)</label>
              <input
                type="number"
                value={customWine.alcohol}
                onChange={(e) => setCustomWine({...customWine, alcohol: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg"
                min="10"
                max="16"
                step="0.1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">전문가 평점</label>
              <input
                type="number"
                value={customWine.rating}
                onChange={(e) => setCustomWine({...customWine, rating: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg"
                min="80"
                max="100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">리뷰 수</label>
              <input
                type="number"
                value={customWine.reviews}
                onChange={(e) => setCustomWine({...customWine, reviews: parseInt(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg"
                min="0"
                max="1000"
              />
            </div>

            <div className="md:col-span-2 lg:col-span-3">
              <button
                onClick={predictCustomWine}
                disabled={!modelAccuracy}
                className="w-full px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <TrendingUp className="w-5 h-5" />
                가격 예측하기
              </button>
            </div>
          </div>
        )}
      </div>

      {/* 정보 패널 */}
      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-purple-600 mt-0.5" />
          <div className="text-sm text-purple-700 dark:text-purple-300">
            <p className="font-medium mb-1">와인 가격 예측 모델</p>
            <p>
              이 시뮬레이터는 와인의 다양한 특성을 분석하여 적정 가격을 예측합니다.
              실제 머신러닝에서는 더 많은 특성(테루아, 생산자 명성, 시장 동향 등)을 고려합니다.
            </p>
            <div className="mt-3 space-y-1">
              <p className="font-medium">주요 가격 결정 요인:</p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>빈티지 연도와 숙성 기간</li>
                <li>생산 지역의 명성도</li>
                <li>전문가 평점 (로버트 파커, 와인 스펙테이터 등)</li>
                <li>희소성과 시장 수요</li>
                <li>품종별 특성과 양조 기법</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}