'use client'

import { useState, useEffect } from 'react'
import { Dice1, Dice2, Dice3, Dice4, Dice5, Dice6, RefreshCw, Play, Pause } from 'lucide-react'

export default function ProbabilityPlayground() {
  // 주사위 실험
  const [diceCount, setDiceCount] = useState(2)
  const [diceResults, setDiceResults] = useState<number[]>([])
  const [diceHistory, setDiceHistory] = useState<number[]>([])
  const [diceFrequency, setDiceFrequency] = useState<{ [key: number]: number }>({})
  
  // 동전 실험
  const [coinCount, setCoinCount] = useState(10)
  const [coinResults, setCoinResults] = useState<boolean[]>([])
  const [coinHistory, setCoinHistory] = useState<{ heads: number; tails: number }>({ heads: 0, tails: 0 })
  
  // 카드 실험
  const [cardDrawCount, setCardDrawCount] = useState(5)
  const [drawnCards, setDrawnCards] = useState<string[]>([])
  const [remainingDeck, setRemainingDeck] = useState(52)
  
  // 자동 실험
  const [autoMode, setAutoMode] = useState(false)
  const [experimentCount, setExperimentCount] = useState(0)

  // 주사위 아이콘 가져오기
  const getDiceIcon = (value: number) => {
    const icons = [Dice1, Dice2, Dice3, Dice4, Dice5, Dice6]
    const Icon = icons[value - 1] || Dice1
    return <Icon className="w-8 h-8" />
  }

  // 주사위 굴리기
  const rollDice = () => {
    const results = Array(diceCount).fill(0).map(() => Math.floor(Math.random() * 6) + 1)
    setDiceResults(results)
    
    const sum = results.reduce((a, b) => a + b, 0)
    setDiceHistory([...diceHistory, sum])
    
    // 빈도 업데이트
    const newFreq = { ...diceFrequency }
    newFreq[sum] = (newFreq[sum] || 0) + 1
    setDiceFrequency(newFreq)
  }

  // 동전 던지기
  const flipCoins = () => {
    const results = Array(coinCount).fill(0).map(() => Math.random() > 0.5)
    setCoinResults(results)
    
    const heads = results.filter(r => r).length
    const tails = results.length - heads
    
    setCoinHistory({
      heads: coinHistory.heads + heads,
      tails: coinHistory.tails + tails
    })
  }

  // 카드 뽑기
  const drawCards = () => {
    const suits = ['♠', '♥', '♦', '♣']
    const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    const deck: string[] = []
    
    suits.forEach(suit => {
      ranks.forEach(rank => {
        deck.push(`${rank}${suit}`)
      })
    })
    
    // Fisher-Yates 셔플
    for (let i = deck.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[deck[i], deck[j]] = [deck[j], deck[i]]
    }
    
    const drawn = deck.slice(0, cardDrawCount)
    setDrawnCards(drawn)
    setRemainingDeck(52 - cardDrawCount)
  }

  // 자동 실험 모드
  useEffect(() => {
    if (autoMode) {
      const interval = setInterval(() => {
        rollDice()
        flipCoins()
        setExperimentCount(prev => prev + 1)
      }, 1000)
      
      return () => clearInterval(interval)
    }
  }, [autoMode, diceCount, coinCount])

  // 리셋
  const resetExperiments = () => {
    setDiceResults([])
    setDiceHistory([])
    setDiceFrequency({})
    setCoinResults([])
    setCoinHistory({ heads: 0, tails: 0 })
    setDrawnCards([])
    setRemainingDeck(52)
    setExperimentCount(0)
    setAutoMode(false)
  }

  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">확률 실험실</h2>
        <div className="flex gap-2">
          <button
            onClick={() => setAutoMode(!autoMode)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
              autoMode 
                ? 'bg-red-600 hover:bg-red-700 text-white' 
                : 'bg-purple-600 hover:bg-purple-700 text-white'
            }`}
          >
            {autoMode ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {autoMode ? '정지' : '자동 실험'}
          </button>
          <button
            onClick={resetExperiments}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg flex items-center gap-2 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            리셋
          </button>
        </div>
      </div>

      {autoMode && (
        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <p className="text-center">
            자동 실험 진행 중... (실험 횟수: {experimentCount})
          </p>
        </div>
      )}

      {/* 주사위 실험 */}
      <section className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
        <h3 className="text-xl font-bold mb-4">주사위 실험</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">주사위 개수</label>
            <input
              type="range"
              min="1"
              max="6"
              value={diceCount}
              onChange={(e) => setDiceCount(Number(e.target.value))}
              className="w-full"
              disabled={autoMode}
            />
            <div className="text-center mt-1">{diceCount}개</div>
          </div>

          <button
            onClick={rollDice}
            disabled={autoMode}
            className="w-full py-2 px-4 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
          >
            주사위 굴리기
          </button>

          {diceResults.length > 0 && (
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <div className="flex justify-center gap-2 mb-3">
                {diceResults.map((result, i) => (
                  <div key={i} className="text-purple-600 dark:text-purple-400">
                    {getDiceIcon(result)}
                  </div>
                ))}
              </div>
              <p className="text-center font-semibold">
                합계: {diceResults.reduce((a, b) => a + b, 0)}
              </p>
            </div>
          )}

          {Object.keys(diceFrequency).length > 0 && (
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">주사위 합 분포</h4>
              <div className="space-y-1">
                {Object.entries(diceFrequency)
                  .sort(([a], [b]) => Number(a) - Number(b))
                  .map(([sum, count]) => {
                    const percentage = (count / diceHistory.length) * 100
                    return (
                      <div key={sum} className="flex items-center gap-2">
                        <span className="w-8 text-right">{sum}:</span>
                        <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-4 relative">
                          <div
                            className="absolute top-0 left-0 h-full bg-purple-600 rounded-full"
                            style={{ width: `${percentage}%` }}
                          />
                        </div>
                        <span className="w-16 text-sm text-right">
                          {count} ({percentage.toFixed(1)}%)
                        </span>
                      </div>
                    )
                  })}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 동전 실험 */}
      <section className="bg-pink-50 dark:bg-pink-900/20 p-6 rounded-xl">
        <h3 className="text-xl font-bold mb-4">동전 던지기</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">동전 개수</label>
            <input
              type="range"
              min="1"
              max="20"
              value={coinCount}
              onChange={(e) => setCoinCount(Number(e.target.value))}
              className="w-full"
              disabled={autoMode}
            />
            <div className="text-center mt-1">{coinCount}개</div>
          </div>

          <button
            onClick={flipCoins}
            disabled={autoMode}
            className="w-full py-2 px-4 bg-pink-600 hover:bg-pink-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
          >
            동전 던지기
          </button>

          {coinResults.length > 0 && (
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <div className="flex flex-wrap justify-center gap-2 mb-3">
                {coinResults.map((result, i) => (
                  <div
                    key={i}
                    className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                      result 
                        ? 'bg-blue-500 text-white' 
                        : 'bg-gray-500 text-white'
                    }`}
                  >
                    {result ? 'H' : 'T'}
                  </div>
                ))}
              </div>
              <p className="text-center">
                앞면: {coinResults.filter(r => r).length} | 
                뒷면: {coinResults.filter(r => !r).length}
              </p>
            </div>
          )}

          {(coinHistory.heads > 0 || coinHistory.tails > 0) && (
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">누적 결과</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {coinHistory.heads}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">앞면</div>
                  <div className="text-xs">
                    {((coinHistory.heads / (coinHistory.heads + coinHistory.tails)) * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-600 dark:text-gray-400">
                    {coinHistory.tails}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">뒷면</div>
                  <div className="text-xs">
                    {((coinHistory.tails / (coinHistory.heads + coinHistory.tails)) * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 카드 실험 */}
      <section className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
        <h3 className="text-xl font-bold mb-4">카드 뽑기</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">뽑을 카드 수</label>
            <input
              type="range"
              min="1"
              max="10"
              value={cardDrawCount}
              onChange={(e) => setCardDrawCount(Number(e.target.value))}
              className="w-full"
            />
            <div className="text-center mt-1">{cardDrawCount}장</div>
          </div>

          <button
            onClick={drawCards}
            className="w-full py-2 px-4 bg-orange-600 hover:bg-orange-700 text-white rounded-lg transition-colors"
          >
            카드 뽑기
          </button>

          {drawnCards.length > 0 && (
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <div className="flex flex-wrap justify-center gap-2 mb-3">
                {drawnCards.map((card, i) => (
                  <div
                    key={i}
                    className={`w-16 h-24 rounded-lg border-2 flex items-center justify-center font-bold text-xl ${
                      card.includes('♥') || card.includes('♦')
                        ? 'border-red-500 text-red-500 bg-red-50 dark:bg-red-900/20'
                        : 'border-gray-800 text-gray-800 dark:text-gray-200 bg-gray-50 dark:bg-gray-900/20'
                    }`}
                  >
                    {card}
                  </div>
                ))}
              </div>
              <p className="text-center text-sm text-gray-600 dark:text-gray-400">
                남은 카드: {remainingDeck}장
              </p>
            </div>
          )}

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">확률 계산</h4>
            <ul className="space-y-1 text-sm">
              <li>• 특정 카드를 뽑을 확률: 1/52 ≈ 1.92%</li>
              <li>• 하트를 뽑을 확률: 13/52 = 25%</li>
              <li>• 그림 카드(J,Q,K)를 뽑을 확률: 12/52 ≈ 23.08%</li>
              <li>• 에이스를 뽑을 확률: 4/52 ≈ 7.69%</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}