'use client'

import { useState, useEffect, useCallback } from 'react'
import { 
  TrendingUp, TrendingDown, DollarSign, Clock,
  Users, BarChart3, Target, Zap, ArrowUp, ArrowDown,
  Coins, Shield, Brain, Sparkles, RefreshCw,
  ChevronRight, AlertCircle, CheckCircle2, Info,
  LineChart, PieChart, Activity, Wallet
} from 'lucide-react'

interface PredictionMarket {
  id: string
  asset: string
  symbol: string
  currentPrice: number
  targetPrice: number
  timeframe: string
  deadline: Date
  totalVolume: number
  participants: number
  yesShares: number
  noShares: number
  probability: number
  status: 'active' | 'resolved' | 'pending'
  description: string
  oracle: string
  rewards: {
    total: number
    winners: number
  }
}

interface UserPosition {
  marketId: string
  side: 'yes' | 'no'
  shares: number
  avgPrice: number
  currentValue: number
  pnl: number
}

interface MarketData {
  timestamp: number
  price: number
  volume: number
  probability: number
}

const CRYPTO_ASSETS = [
  { symbol: 'BTC', name: 'Bitcoin', currentPrice: 43250 },
  { symbol: 'ETH', name: 'Ethereum', currentPrice: 2580 },
  { symbol: 'SOL', name: 'Solana', currentPrice: 98.5 },
  { symbol: 'ADA', name: 'Cardano', currentPrice: 0.52 },
  { symbol: 'AVAX', name: 'Avalanche', currentPrice: 35.8 }
]

export default function CryptoPredictionMarkets() {
  const [markets, setMarkets] = useState<PredictionMarket[]>([])
  const [userPositions, setUserPositions] = useState<UserPosition[]>([])
  const [selectedMarket, setSelectedMarket] = useState<string | null>(null)
  const [tradingAmount, setTradingAmount] = useState(100)
  const [selectedSide, setSelectedSide] = useState<'yes' | 'no'>('yes')
  const [marketData, setMarketData] = useState<Record<string, MarketData[]>>({})
  const [userBalance, setUserBalance] = useState(10000)
  const [activeTab, setActiveTab] = useState<'markets' | 'positions' | 'create'>('markets')
  const [isCreatingMarket, setIsCreatingMarket] = useState(false)
  
  // 새 시장 생성 상태
  const [newMarket, setNewMarket] = useState({
    asset: 'BTC',
    targetPrice: 50000,
    timeframe: '7days',
    description: ''
  })

  // 시뮬레이션 데이터 초기화
  useEffect(() => {
    const initialMarkets: PredictionMarket[] = [
      {
        id: 'm1',
        asset: 'Bitcoin',
        symbol: 'BTC',
        currentPrice: 43250,
        targetPrice: 50000,
        timeframe: '7일',
        deadline: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
        totalVolume: 125000,
        participants: 1247,
        yesShares: 65000,
        noShares: 60000,
        probability: 0.52,
        status: 'active',
        description: 'BTC가 7일 내에 $50,000를 돌파할까요?',
        oracle: 'Chainlink Price Feed',
        rewards: { total: 5000, winners: 0 }
      },
      {
        id: 'm2',
        asset: 'Ethereum',
        symbol: 'ETH',
        currentPrice: 2580,
        targetPrice: 3000,
        timeframe: '14일',
        deadline: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000),
        totalVolume: 87500,
        participants: 892,
        yesShares: 45000,
        noShares: 42500,
        probability: 0.51,
        status: 'active',
        description: 'ETH가 2주 내에 $3,000를 달성할까요?',
        oracle: 'Chainlink Price Feed',
        rewards: { total: 3500, winners: 0 }
      },
      {
        id: 'm3',
        asset: 'Solana',
        symbol: 'SOL',
        currentPrice: 98.5,
        targetPrice: 80,
        timeframe: '3일',
        deadline: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000),
        totalVolume: 42000,
        participants: 534,
        yesShares: 28000,
        noShares: 14000,
        probability: 0.67,
        status: 'active',
        description: 'SOL이 3일 내에 $80 아래로 떨어질까요?',
        oracle: 'Pyth Network',
        rewards: { total: 2100, winners: 0 }
      }
    ]
    
    setMarkets(initialMarkets)
    
    // 시뮬레이션 시장 데이터 생성
    const simulationData: Record<string, MarketData[]> = {}
    initialMarkets.forEach(market => {
      simulationData[market.id] = generateMarketData(market.probability)
    })
    setMarketData(simulationData)
  }, [])

  // 시장 데이터 생성 함수
  const generateMarketData = (baseProbability: number): MarketData[] => {
    const data: MarketData[] = []
    let currentProb = baseProbability
    
    for (let i = 0; i < 24; i++) {
      currentProb += (Math.random() - 0.5) * 0.05
      currentProb = Math.max(0.1, Math.min(0.9, currentProb))
      
      data.push({
        timestamp: Date.now() - (24 - i) * 60 * 60 * 1000,
        price: currentProb,
        volume: Math.random() * 1000 + 500,
        probability: currentProb
      })
    }
    return data
  }

  // 실시간 업데이트 시뮬레이션
  useEffect(() => {
    const interval = setInterval(() => {
      setMarkets(prev => prev.map(market => {
        const change = (Math.random() - 0.5) * 0.02
        const newProbability = Math.max(0.1, Math.min(0.9, market.probability + change))
        
        return {
          ...market,
          probability: newProbability,
          yesShares: market.yesShares + Math.floor(Math.random() * 100),
          noShares: market.noShares + Math.floor(Math.random() * 100),
          participants: market.participants + Math.floor(Math.random() * 5)
        }
      }))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  // 포지션 진입
  const enterPosition = (marketId: string, side: 'yes' | 'no', amount: number) => {
    const market = markets.find(m => m.id === marketId)
    if (!market || userBalance < amount) return

    const sharePrice = side === 'yes' ? market.probability : (1 - market.probability)
    const shares = amount / sharePrice

    // 유저 포지션 업데이트
    setUserPositions(prev => {
      const existing = prev.find(p => p.marketId === marketId)
      if (existing) {
        return prev.map(p => p.marketId === marketId ? {
          ...p,
          shares: p.shares + shares,
          avgPrice: (p.avgPrice * p.shares + sharePrice * shares) / (p.shares + shares)
        } : p)
      } else {
        return [...prev, {
          marketId,
          side,
          shares,
          avgPrice: sharePrice,
          currentValue: shares * sharePrice,
          pnl: 0
        }]
      }
    })

    // 시장 업데이트
    setMarkets(prev => prev.map(m => m.id === marketId ? {
      ...m,
      totalVolume: m.totalVolume + amount,
      yesShares: side === 'yes' ? m.yesShares + shares : m.yesShares,
      noShares: side === 'no' ? m.noShares + shares : m.noShares,
      participants: m.participants + 1
    } : m))

    setUserBalance(prev => prev - amount)
  }

  // 새 시장 생성
  const createMarket = () => {
    if (isCreatingMarket) return
    
    setIsCreatingMarket(true)
    
    setTimeout(() => {
      const asset = CRYPTO_ASSETS.find(a => a.symbol === newMarket.asset)
      if (!asset) return

      const newMarketData: PredictionMarket = {
        id: `m${Date.now()}`,
        asset: asset.name,
        symbol: asset.symbol,
        currentPrice: asset.currentPrice,
        targetPrice: newMarket.targetPrice,
        timeframe: newMarket.timeframe === '7days' ? '7일' : newMarket.timeframe === '14days' ? '14일' : '30일',
        deadline: new Date(Date.now() + (newMarket.timeframe === '7days' ? 7 : newMarket.timeframe === '14days' ? 14 : 30) * 24 * 60 * 60 * 1000),
        totalVolume: 0,
        participants: 0,
        yesShares: 0,
        noShares: 0,
        probability: 0.5,
        status: 'active',
        description: newMarket.description || `${asset.symbol}가 ${newMarket.timeframe}에 $${newMarket.targetPrice.toLocaleString()}를 달성할까요?`,
        oracle: 'Chainlink Price Feed',
        rewards: { total: 1000, winners: 0 }
      }

      setMarkets(prev => [newMarketData, ...prev])
      setIsCreatingMarket(false)
      setActiveTab('markets')
      
      // 폼 리셋
      setNewMarket({
        asset: 'BTC',
        targetPrice: 50000,
        timeframe: '7days',
        description: ''
      })
    }, 2000)
  }

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 2
    }).format(price)
  }

  const formatTimeRemaining = (deadline: Date) => {
    const now = new Date()
    const diff = deadline.getTime() - now.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))
    const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60))
    
    if (days > 0) return `${days}일 ${hours}시간`
    return `${hours}시간`
  }

  const getProbabilityColor = (prob: number) => {
    if (prob > 0.6) return 'text-green-600 dark:text-green-400'
    if (prob < 0.4) return 'text-red-600 dark:text-red-400'
    return 'text-yellow-600 dark:text-yellow-400'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-6 h-6 text-white" />
              </div>
              Crypto Prediction Markets
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              블록체인 기반 암호화폐 가격 예측 시장에서 집단지성을 활용해보세요
            </p>
          </div>
          
          <div className="text-right">
            <div className="text-sm text-gray-500 dark:text-gray-400">잔액</div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {formatPrice(userBalance)}
            </div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-1">
              <Activity className="w-4 h-4 text-blue-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">활성 시장</span>
            </div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {markets.filter(m => m.status === 'active').length}
            </div>
          </div>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-1">
              <Users className="w-4 h-4 text-green-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">총 참여자</span>
            </div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {markets.reduce((sum, m) => sum + m.participants, 0).toLocaleString()}
            </div>
          </div>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-1">
              <DollarSign className="w-4 h-4 text-purple-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">총 거래량</span>
            </div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {formatPrice(markets.reduce((sum, m) => sum + m.totalVolume, 0))}
            </div>
          </div>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-1">
              <Wallet className="w-4 h-4 text-orange-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">내 포지션</span>
            </div>
            <div className="text-xl font-bold text-gray-900 dark:text-white">
              {userPositions.length}
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="border-b border-gray-200 dark:border-gray-700">
          <nav className="flex">
            <button
              onClick={() => setActiveTab('markets')}
              className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'markets'
                  ? 'border-purple-500 text-purple-600 dark:text-purple-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
              }`}
            >
              <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4" />
                예측 시장
              </div>
            </button>
            <button
              onClick={() => setActiveTab('positions')}
              className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'positions'
                  ? 'border-purple-500 text-purple-600 dark:text-purple-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
              }`}
            >
              <div className="flex items-center gap-2">
                <Wallet className="w-4 h-4" />
                내 포지션
              </div>
            </button>
            <button
              onClick={() => setActiveTab('create')}
              className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'create'
                  ? 'border-purple-500 text-purple-600 dark:text-purple-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
              }`}
            >
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                시장 생성
              </div>
            </button>
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'markets' && (
            <div className="space-y-4">
              {markets.map(market => (
                <div key={market.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <div className="w-8 h-8 bg-gradient-to-r from-orange-500 to-yellow-500 rounded-full flex items-center justify-center">
                          <Coins className="w-4 h-4 text-white" />
                        </div>
                        <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                          {market.asset} ({market.symbol})
                        </h3>
                        <span className="px-2 py-1 bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400 rounded text-xs font-medium">
                          {market.status.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-gray-700 dark:text-gray-300 mb-3">
                        {market.description}
                      </p>
                      
                      <div className="flex items-center gap-6 text-sm text-gray-600 dark:text-gray-400">
                        <div className="flex items-center gap-1">
                          <DollarSign className="w-4 h-4" />
                          현재가: {formatPrice(market.currentPrice)}
                        </div>
                        <div className="flex items-center gap-1">
                          <Target className="w-4 h-4" />
                          목표가: {formatPrice(market.targetPrice)}
                        </div>
                        <div className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {formatTimeRemaining(market.deadline)}
                        </div>
                        <div className="flex items-center gap-1">
                          <Users className="w-4 h-4" />
                          {market.participants.toLocaleString()}명
                        </div>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className={`text-2xl font-bold ${getProbabilityColor(market.probability)}`}>
                        {Math.round(market.probability * 100)}%
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        YES 확률
                      </div>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="mb-4">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-green-600 dark:text-green-400">YES ({Math.round(market.probability * 100)}%)</span>
                      <span className="text-red-600 dark:text-red-400">NO ({Math.round((1 - market.probability) * 100)}%)</span>
                    </div>
                    <div className="w-full bg-red-200 dark:bg-red-900/30 rounded-full h-3 overflow-hidden">
                      <div 
                        className="bg-green-500 h-3 transition-all duration-500"
                        style={{ width: `${market.probability * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Trading Interface */}
                  {selectedMarket === market.id ? (
                    <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            투자 금액
                          </label>
                          <input
                            type="number"
                            value={tradingAmount}
                            onChange={(e) => setTradingAmount(Number(e.target.value))}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                            min="1"
                            max={userBalance}
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            예측 방향
                          </label>
                          <div className="flex gap-2">
                            <button
                              onClick={() => setSelectedSide('yes')}
                              className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                                selectedSide === 'yes'
                                  ? 'bg-green-500 text-white'
                                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                              }`}
                            >
                              YES
                            </button>
                            <button
                              onClick={() => setSelectedSide('no')}
                              className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                                selectedSide === 'no'
                                  ? 'bg-red-500 text-white'
                                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                              }`}
                            >
                              NO
                            </button>
                          </div>
                        </div>
                        
                        <div className="flex items-end gap-2">
                          <button
                            onClick={() => {
                              enterPosition(market.id, selectedSide, tradingAmount)
                              setSelectedMarket(null)
                            }}
                            disabled={tradingAmount > userBalance}
                            className="flex-1 px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors"
                          >
                            거래 실행
                          </button>
                          <button
                            onClick={() => setSelectedMarket(null)}
                            className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors"
                          >
                            취소
                          </button>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <button
                      onClick={() => setSelectedMarket(market.id)}
                      className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-lg transition-colors"
                    >
                      거래하기
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}

          {activeTab === 'positions' && (
            <div className="space-y-4">
              {userPositions.length === 0 ? (
                <div className="text-center py-12">
                  <Wallet className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    보유 포지션이 없습니다
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    예측 시장에 참여하여 첫 포지션을 만들어보세요
                  </p>
                </div>
              ) : (
                userPositions.map((position, index) => {
                  const market = markets.find(m => m.id === position.marketId)
                  if (!market) return null

                  const currentPrice = position.side === 'yes' ? market.probability : (1 - market.probability)
                  const currentValue = position.shares * currentPrice
                  const pnl = currentValue - (position.shares * position.avgPrice)
                  const pnlPercent = (pnl / (position.shares * position.avgPrice)) * 100

                  return (
                    <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                            {market.asset} - {position.side.toUpperCase()}
                          </h3>
                          <p className="text-gray-600 dark:text-gray-400 text-sm">
                            {market.description}
                          </p>
                        </div>
                        
                        <div className="text-right">
                          <div className={`text-lg font-bold ${pnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                            {pnl >= 0 ? '+' : ''}{formatPrice(pnl)}
                          </div>
                          <div className={`text-sm ${pnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                            {pnl >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%
                          </div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <div className="text-gray-500 dark:text-gray-400">보유 수량</div>
                          <div className="font-medium text-gray-900 dark:text-white">
                            {position.shares.toFixed(2)}
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-500 dark:text-gray-400">평균 가격</div>
                          <div className="font-medium text-gray-900 dark:text-white">
                            {formatPrice(position.avgPrice)}
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-500 dark:text-gray-400">현재 가격</div>
                          <div className="font-medium text-gray-900 dark:text-white">
                            {formatPrice(currentPrice)}
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-500 dark:text-gray-400">현재 가치</div>
                          <div className="font-medium text-gray-900 dark:text-white">
                            {formatPrice(currentValue)}
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })
              )}
            </div>
          )}

          {activeTab === 'create' && (
            <div className="max-w-2xl mx-auto">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6">
                새로운 예측 시장 생성
              </h3>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    암호화폐 선택
                  </label>
                  <select
                    value={newMarket.asset}
                    onChange={(e) => setNewMarket(prev => ({ ...prev, asset: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                  >
                    {CRYPTO_ASSETS.map(asset => (
                      <option key={asset.symbol} value={asset.symbol}>
                        {asset.name} ({asset.symbol}) - {formatPrice(asset.currentPrice)}
                      </option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    목표 가격
                  </label>
                  <input
                    type="number"
                    value={newMarket.targetPrice}
                    onChange={(e) => setNewMarket(prev => ({ ...prev, targetPrice: Number(e.target.value) }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                    min="1"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    기간
                  </label>
                  <select
                    value={newMarket.timeframe}
                    onChange={(e) => setNewMarket(prev => ({ ...prev, timeframe: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                  >
                    <option value="7days">7일</option>
                    <option value="14days">14일</option>
                    <option value="30days">30일</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    설명 (선택사항)
                  </label>
                  <textarea
                    value={newMarket.description}
                    onChange={(e) => setNewMarket(prev => ({ ...prev, description: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white"
                    rows={3}
                    placeholder="시장에 대한 추가 설명을 입력하세요..."
                  />
                </div>
                
                <button
                  onClick={createMarket}
                  disabled={isCreatingMarket}
                  className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                >
                  {isCreatingMarket ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      시장 생성 중...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      시장 생성하기
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Info Panel */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">
              예측 시장 작동 원리
            </h3>
            <div className="text-sm text-blue-800 dark:text-blue-300 space-y-2">
              <p>• <strong>집단지성</strong>: 많은 참여자들의 예측이 모여 정확한 확률을 형성합니다</p>
              <p>• <strong>시장 메커니즘</strong>: 수요와 공급에 따라 YES/NO 토큰의 가격이 결정됩니다</p>
              <p>• <strong>오라클 검증</strong>: Chainlink 등 신뢰할 수 있는 데이터 소스로 결과를 확인합니다</p>
              <p>• <strong>자동 정산</strong>: 스마트 컨트랙트가 결과에 따라 보상을 자동 분배합니다</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}