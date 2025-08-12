'use client'

import { useState, useEffect, useRef } from 'react'
import { 
  Gavel, TrendingUp, AlertCircle, Info, Download, 
  BarChart3, DollarSign, Users, Clock, Trophy,
  Activity, Target, Brain, ChevronRight, Play, Pause
} from 'lucide-react'

interface BidData {
  id: number
  itemName: string
  category: string
  startPrice: number
  currentPrice: number
  predictedFinalPrice: number
  bidCount: number
  watcherCount: number
  timeRemaining: number // minutes
  sellerRating: number
  condition: 'new' | 'like-new' | 'good' | 'fair'
  shippingCost: number
  returnPolicy: boolean
}

interface BidHistory {
  timestamp: Date
  bidderCount: number
  price: number
}

// 실제 경매 데이터 (샘플)
const AUCTION_DATASET: BidData[] = [
  { id: 1, itemName: "MacBook Pro 16\" M3", category: "Electronics", startPrice: 500, currentPrice: 1850, predictedFinalPrice: 2400, bidCount: 45, watcherCount: 120, timeRemaining: 120, sellerRating: 4.8, condition: 'like-new', shippingCost: 20, returnPolicy: true },
  { id: 2, itemName: "Rolex Submariner", category: "Watches", startPrice: 5000, currentPrice: 8500, predictedFinalPrice: 9200, bidCount: 23, watcherCount: 85, timeRemaining: 60, sellerRating: 4.9, condition: 'good', shippingCost: 50, returnPolicy: false },
  { id: 3, itemName: "Herman Miller Aeron Chair", category: "Furniture", startPrice: 200, currentPrice: 680, predictedFinalPrice: 850, bidCount: 32, watcherCount: 67, timeRemaining: 180, sellerRating: 4.5, condition: 'good', shippingCost: 80, returnPolicy: true },
  { id: 4, itemName: "Canon EOS R5", category: "Cameras", startPrice: 1000, currentPrice: 2300, predictedFinalPrice: 2800, bidCount: 38, watcherCount: 95, timeRemaining: 90, sellerRating: 4.7, condition: 'new', shippingCost: 15, returnPolicy: true },
  { id: 5, itemName: "Tesla Model 3 Wheels", category: "Auto Parts", startPrice: 400, currentPrice: 1100, predictedFinalPrice: 1400, bidCount: 28, watcherCount: 52, timeRemaining: 240, sellerRating: 4.6, condition: 'like-new', shippingCost: 120, returnPolicy: false },
  { id: 6, itemName: "PS5 Console Bundle", category: "Gaming", startPrice: 300, currentPrice: 580, predictedFinalPrice: 650, bidCount: 67, watcherCount: 145, timeRemaining: 30, sellerRating: 4.4, condition: 'new', shippingCost: 25, returnPolicy: true },
]

// 카테고리별 승수
const CATEGORY_MULTIPLIERS: { [key: string]: number } = {
  "Electronics": 1.2,
  "Watches": 1.5,
  "Furniture": 0.8,
  "Cameras": 1.3,
  "Auto Parts": 0.9,
  "Gaming": 1.1,
  "Art": 2.0,
  "Collectibles": 1.8
}

export default function BiddingPricePredictor() {
  const [selectedAuction, setSelectedAuction] = useState<BidData | null>(null)
  const [predictions, setPredictions] = useState<BidData[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const [modelAccuracy, setModelAccuracy] = useState(0)
  const [liveMode, setLiveMode] = useState(false)
  const [bidHistories, setBidHistories] = useState<{ [key: number]: BidHistory[] }>({})
  const intervalRef = useRef<NodeJS.Timeout>()

  // 입찰 가격 예측 모델
  const predictFinalPrice = (auction: BidData): number => {
    let predictedPrice = auction.currentPrice

    // 시간 요소 (마감 임박할수록 가격 상승)
    const urgencyFactor = Math.max(0.5, 1 - (auction.timeRemaining / 1440)) // 24시간 기준
    predictedPrice *= (1 + urgencyFactor * 0.3)

    // 입찰자 수와 관심도
    const competitionFactor = auction.bidCount / 20 // 20명 기준
    predictedPrice *= (1 + competitionFactor * 0.2)

    // 관찰자 대비 입찰자 비율
    const engagementRate = auction.bidCount / Math.max(1, auction.watcherCount)
    predictedPrice *= (1 + engagementRate * 0.5)

    // 카테고리 효과
    const categoryMultiplier = CATEGORY_MULTIPLIERS[auction.category] || 1.0
    predictedPrice *= categoryMultiplier

    // 판매자 신뢰도
    const sellerTrust = auction.sellerRating / 5
    predictedPrice *= (0.9 + sellerTrust * 0.2)

    // 상품 상태
    const conditionMultipliers = {
      'new': 1.2,
      'like-new': 1.0,
      'good': 0.85,
      'fair': 0.7
    }
    predictedPrice *= conditionMultipliers[auction.condition]

    // 반품 정책
    if (auction.returnPolicy) predictedPrice *= 1.05

    // 배송비 고려
    predictedPrice += auction.shippingCost * 0.5

    // 시작가 대비 현재가 비율 (모멘텀)
    const momentum = auction.currentPrice / auction.startPrice
    if (momentum > 3) predictedPrice *= 1.1

    return Math.round(predictedPrice)
  }

  // 모델 학습
  const trainModel = async () => {
    setIsTraining(true)
    setPredictions([])
    
    for (let i = 0; i < AUCTION_DATASET.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 300))
      
      const auction = AUCTION_DATASET[i]
      const predicted = predictFinalPrice(auction)
      
      setPredictions(prev => [...prev, {
        ...auction,
        predictedFinalPrice: predicted
      }])
    }

    // 정확도 계산
    const totalError = AUCTION_DATASET.reduce((sum, auction) => {
      const predicted = predictFinalPrice(auction)
      return sum + Math.abs(auction.predictedFinalPrice - predicted) / auction.predictedFinalPrice
    }, 0)
    
    const accuracy = Math.max(0, 100 - (totalError / AUCTION_DATASET.length * 100))
    setModelAccuracy(accuracy)
    setIsTraining(false)
  }

  // 실시간 모드 시뮬레이션
  useEffect(() => {
    if (liveMode) {
      intervalRef.current = setInterval(() => {
        setPredictions(prev => prev.map(auction => {
          // 시간 감소
          const newTimeRemaining = Math.max(0, auction.timeRemaining - 1)
          
          // 랜덤하게 입찰 발생
          const bidChance = Math.random()
          let newBidCount = auction.bidCount
          let newCurrentPrice = auction.currentPrice
          
          if (bidChance > 0.7) {
            newBidCount += 1
            newCurrentPrice = Math.round(auction.currentPrice * (1 + Math.random() * 0.05))
          }

          // 관찰자 수 변동
          const newWatcherCount = auction.watcherCount + Math.floor(Math.random() * 3) - 1

          // 새로운 예측가 계산
          const updatedAuction = {
            ...auction,
            timeRemaining: newTimeRemaining,
            bidCount: newBidCount,
            currentPrice: newCurrentPrice,
            watcherCount: Math.max(1, newWatcherCount)
          }

          return {
            ...updatedAuction,
            predictedFinalPrice: predictFinalPrice(updatedAuction)
          }
        }))
      }, 1000)
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [liveMode])

  // 데이터 다운로드
  const downloadData = (format: 'csv' | 'json') => {
    if (format === 'csv') {
      let csv = 'ID,Item Name,Category,Start Price,Current Price,Predicted Final Price,Bid Count,Watcher Count,Time Remaining,Seller Rating,Condition,Shipping Cost,Return Policy\n'
      
      predictions.forEach(auction => {
        csv += `${auction.id},"${auction.itemName}","${auction.category}",${auction.startPrice},${auction.currentPrice},${auction.predictedFinalPrice},${auction.bidCount},${auction.watcherCount},${auction.timeRemaining},${auction.sellerRating},"${auction.condition}",${auction.shippingCost},${auction.returnPolicy}\n`
      })

      const blob = new Blob([csv], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `auction_predictions_${new Date().toISOString().split('T')[0]}.csv`
      a.click()
      URL.revokeObjectURL(url)
    } else {
      const json = JSON.stringify(predictions, null, 2)
      const blob = new Blob([json], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `auction_predictions_${new Date().toISOString().split('T')[0]}.json`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  // 시간 포맷팅
  const formatTime = (minutes: number): string => {
    const hours = Math.floor(minutes / 60)
    const mins = minutes % 60
    return `${hours}h ${mins}m`
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* 헤더 */}
      <div className="bg-gradient-to-r from-orange-600 to-red-600 text-white rounded-xl p-6">
        <div className="flex items-center gap-3 mb-2">
          <Gavel className="w-8 h-8" />
          <h2 className="text-2xl font-bold">경매 입찰가 예측 AI</h2>
        </div>
        <p className="text-orange-100">
          실시간 경매 데이터를 분석하여 최종 낙찰가를 예측하는 AI 모델
        </p>
      </div>

      {/* 모델 학습 섹션 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Brain className="w-5 h-5" />
            입찰가 예측 모델
          </h3>
          {predictions.length > 0 && (
            <button
              onClick={() => setLiveMode(!liveMode)}
              className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
                liveMode 
                  ? 'bg-red-600 text-white hover:bg-red-700' 
                  : 'bg-green-600 text-white hover:bg-green-700'
              }`}
            >
              {liveMode ? (
                <>
                  <Pause className="w-4 h-4" />
                  실시간 중지
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  실시간 모드
                </>
              )}
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              다양한 경매 요소를 분석하여 최종 낙찰가를 예측합니다.
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-orange-500" />
                <span>남은 시간과 긴급도 분석</span>
              </div>
              <div className="flex items-center gap-2">
                <Users className="w-4 h-4 text-orange-500" />
                <span>입찰자 경쟁도 측정</span>
              </div>
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-orange-500" />
                <span>가격 모멘텀 추적</span>
              </div>
              <div className="flex items-center gap-2">
                <Trophy className="w-4 h-4 text-orange-500" />
                <span>판매자 신뢰도 반영</span>
              </div>
            </div>
          </div>
          
          <div className="flex flex-col items-center justify-center">
            {modelAccuracy > 0 ? (
              <div className="text-center">
                <div className="text-4xl font-bold text-green-600 mb-2">
                  {modelAccuracy.toFixed(1)}%
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">예측 정확도</p>
                {liveMode && (
                  <p className="text-xs text-orange-600 mt-2 animate-pulse">
                    실시간 업데이트 중...
                  </p>
                )}
              </div>
            ) : (
              <button
                onClick={trainModel}
                disabled={isTraining}
                className="px-6 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50 flex items-center gap-2"
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

        {/* 예측 결과 */}
        {predictions.length > 0 && (
          <>
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-medium text-gray-700 dark:text-gray-300">실시간 경매 현황</h4>
              <div className="flex gap-2">
                <button
                  onClick={() => downloadData('csv')}
                  className="px-3 py-1.5 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors flex items-center gap-1"
                >
                  <Download className="w-4 h-4" />
                  CSV
                </button>
                <button
                  onClick={() => downloadData('json')}
                  className="px-3 py-1.5 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-1"
                >
                  <Download className="w-4 h-4" />
                  JSON
                </button>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left py-3 px-4">상품명</th>
                    <th className="text-center py-3 px-4">현재가</th>
                    <th className="text-center py-3 px-4">예상 낙찰가</th>
                    <th className="text-center py-3 px-4">입찰</th>
                    <th className="text-center py-3 px-4">관심</th>
                    <th className="text-center py-3 px-4">남은시간</th>
                    <th className="text-center py-3 px-4">상승률</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map((auction) => {
                    const increaseRate = ((auction.predictedFinalPrice - auction.currentPrice) / auction.currentPrice * 100)
                    return (
                      <tr key={auction.id} className="border-b border-gray-100 dark:border-gray-700">
                        <td className="py-3 px-4">
                          <div>
                            <div className="font-medium">{auction.itemName}</div>
                            <div className="text-xs text-gray-500">{auction.category}</div>
                          </div>
                        </td>
                        <td className="text-center py-3 px-4 font-medium">
                          ${auction.currentPrice.toLocaleString()}
                        </td>
                        <td className="text-center py-3 px-4 font-medium text-orange-600">
                          ${auction.predictedFinalPrice.toLocaleString()}
                        </td>
                        <td className="text-center py-3 px-4">
                          <span className="text-blue-600 font-medium">{auction.bidCount}</span>
                        </td>
                        <td className="text-center py-3 px-4">
                          <span className="text-gray-600">{auction.watcherCount}</span>
                        </td>
                        <td className="text-center py-3 px-4">
                          <span className={`${auction.timeRemaining < 60 ? 'text-red-600 font-medium' : ''}`}>
                            {formatTime(auction.timeRemaining)}
                          </span>
                        </td>
                        <td className="text-center py-3 px-4">
                          <span className={`px-2 py-1 rounded text-xs ${
                            increaseRate > 20 ? 'bg-red-100 text-red-700' : 
                            increaseRate > 10 ? 'bg-orange-100 text-orange-700' : 
                            'bg-green-100 text-green-700'
                          }`}>
                            +{increaseRate.toFixed(1)}%
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

      {/* 정보 패널 */}
      <div className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-orange-600 mt-0.5" />
          <div className="text-sm text-orange-700 dark:text-orange-300">
            <p className="font-medium mb-1">경매 입찰가 예측 알고리즘</p>
            <p>
              이 모델은 다양한 경매 데이터를 실시간으로 분석하여 최종 낙찰가를 예측합니다.
              eBay, Yahoo Auctions 등의 실제 경매 패턴을 학습한 AI 모델을 시뮬레이션합니다.
            </p>
            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="font-medium mb-1">주요 예측 변수:</p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>경매 남은 시간 (긴급도)</li>
                  <li>입찰자 수와 경쟁 강도</li>
                  <li>관찰자 대비 입찰자 비율</li>
                  <li>가격 상승 모멘텀</li>
                </ul>
              </div>
              <div>
                <p className="font-medium mb-1">활용 전략:</p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>적정 입찰 시점 파악</li>
                  <li>최종 예산 설정</li>
                  <li>경쟁 강도 분석</li>
                  <li>스나이핑 전략 수립</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}