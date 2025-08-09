'use client'

import { useEffect, useRef, useState } from 'react'

interface CoinData {
  symbol: string
  price: number
  change24h: number
  volume: number
  marketCap: number
  trend: number[]
}

interface OrderFlow {
  type: 'buy' | 'sell'
  amount: number
  price: number
  timestamp: number
  x: number
  y: number
  life: number
}

export default function CryptoTradingSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [coins, setCoins] = useState<CoinData[]>([])
  const [orderFlows, setOrderFlows] = useState<OrderFlow[]>([])
  const [isTrading, setIsTrading] = useState(false)
  const [totalVolume, setTotalVolume] = useState(0)
  const [animationFrame, setAnimationFrame] = useState(0)
  const animationRef = useRef<number>()

  // 암호화폐 데이터 초기화
  useEffect(() => {
    const initialCoins: CoinData[] = [
      {
        symbol: 'BTC',
        price: 67500,
        change24h: 2.5,
        volume: 28000000000,
        marketCap: 1320000000000,
        trend: Array.from({ length: 30 }, () => Math.random() * 2000 + 66000)
      },
      {
        symbol: 'ETH',
        price: 3420,
        change24h: -1.2,
        volume: 15000000000,
        marketCap: 411000000000,
        trend: Array.from({ length: 30 }, () => Math.random() * 200 + 3350)
      },
      {
        symbol: 'SOL',
        price: 178,
        change24h: 5.8,
        volume: 2100000000,
        marketCap: 84000000000,
        trend: Array.from({ length: 30 }, () => Math.random() * 20 + 170)
      }
    ]

    setCoins(initialCoins)
  }, [])

  // 실시간 트레이딩 시뮬레이션
  useEffect(() => {
    const animate = () => {
      setAnimationFrame(prev => prev + 1)
      
      if (isTrading) {
        // 가격 업데이트 (매초)
        if (animationFrame % 60 === 0) {
          setCoins(prevCoins => prevCoins.map(coin => {
            const changePercent = (Math.random() - 0.5) * 0.1 // -5% ~ +5%
            const newPrice = coin.price * (1 + changePercent)
            const newTrend = [...coin.trend.slice(1), newPrice]
            
            return {
              ...coin,
              price: newPrice,
              change24h: coin.change24h + changePercent * 100,
              trend: newTrend
            }
          }))
        }

        // 주문 플로우 생성 (랜덤)
        if (Math.random() < 0.3) {
          const canvas = canvasRef.current
          if (canvas) {
            const newOrder: OrderFlow = {
              type: Math.random() > 0.5 ? 'buy' : 'sell',
              amount: Math.random() * 10 + 1,
              price: 67500 + (Math.random() - 0.5) * 5000,
              timestamp: Date.now(),
              x: Math.random() * (canvas.width - 100) + 50,
              y: Math.random() * (canvas.height - 100) + 50,
              life: 1.0
            }
            
            setOrderFlows(prev => [...prev.slice(-20), newOrder])
          }
        }

        // 주문 플로우 생명주기 업데이트
        setOrderFlows(prev => prev.map(order => ({
          ...order,
          life: Math.max(0, order.life - 0.02),
          y: order.y - 1
        })).filter(order => order.life > 0))

        // 거래량 업데이트
        setTotalVolume(prev => prev + Math.random() * 1000000)
      }
      
      drawTradingInterface()
      animationRef.current = requestAnimationFrame(animate)
    }

    animate()
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [coins, orderFlows, isTrading, animationFrame])

  const drawTradingInterface = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width = 400
    const height = canvas.height = 250

    // 배경 (트레이딩 터미널 스타일)
    ctx.fillStyle = '#0d1421'
    ctx.fillRect(0, 0, width, height)

    // 그리드 패턴
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.1)'
    ctx.lineWidth = 1
    
    for (let i = 0; i < width; i += 20) {
      ctx.beginPath()
      ctx.moveTo(i, 0)
      ctx.lineTo(i, height)
      ctx.stroke()
    }
    
    for (let i = 0; i < height; i += 20) {
      ctx.beginPath()
      ctx.moveTo(0, i)
      ctx.lineTo(width, i)
      ctx.stroke()
    }

    // 암호화폐 가격 패널 (좌측)
    let yOffset = 30
    coins.forEach((coin, index) => {
      const x = 20
      const y = yOffset + index * 60
      
      // 코인 심볼
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 16px monospace'
      ctx.textAlign = 'left'
      ctx.fillText(coin.symbol, x, y)
      
      // 가격
      const priceColor = coin.change24h >= 0 ? '#00ff88' : '#ff4757'
      ctx.fillStyle = priceColor
      ctx.font = '14px monospace'
      ctx.fillText(`$${coin.price.toLocaleString()}`, x, y + 20)
      
      // 변동률
      ctx.fillStyle = priceColor
      ctx.font = '12px monospace'
      const changeText = `${coin.change24h >= 0 ? '+' : ''}${coin.change24h.toFixed(2)}%`
      ctx.fillText(changeText, x, y + 35)
      
      // 미니 차트
      const chartWidth = 80
      const chartHeight = 30
      const chartX = x + 100
      const chartY = y - 15
      
      if (coin.trend.length > 1) {
        const minPrice = Math.min(...coin.trend)
        const maxPrice = Math.max(...coin.trend)
        const priceRange = maxPrice - minPrice || 1
        
        ctx.beginPath()
        ctx.strokeStyle = priceColor
        ctx.lineWidth = 1.5
        
        coin.trend.forEach((price, i) => {
          const x = chartX + (chartWidth / (coin.trend.length - 1)) * i
          const y = chartY + chartHeight - ((price - minPrice) / priceRange) * chartHeight
          
          if (i === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        })
        ctx.stroke()
        
        // 현재 가격 포인트
        const lastX = chartX + chartWidth
        const lastY = chartY + chartHeight - ((coin.price - minPrice) / priceRange) * chartHeight
        
        ctx.beginPath()
        ctx.arc(lastX, lastY, 2, 0, 2 * Math.PI)
        ctx.fillStyle = priceColor
        ctx.fill()
      }
    })

    // 주문 플로우 시각화
    orderFlows.forEach(order => {
      const opacity = order.life
      const size = 3 + order.amount * 0.5
      
      // 주문 원
      ctx.beginPath()
      ctx.arc(order.x, order.y, size, 0, 2 * Math.PI)
      
      if (order.type === 'buy') {
        ctx.fillStyle = `rgba(0, 255, 136, ${opacity})`
      } else {
        ctx.fillStyle = `rgba(255, 71, 87, ${opacity})`
      }
      ctx.fill()
      
      // 주문 정보 텍스트
      if (opacity > 0.7) {
        ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`
        ctx.font = '8px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(
          `${order.type.toUpperCase()} ${order.amount.toFixed(1)}`,
          order.x,
          order.y - size - 5
        )
      }
    })

    // 실시간 통계 (우측 상단)
    const statsX = width - 150
    let statsY = 30
    
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
    ctx.font = '12px monospace'
    ctx.textAlign = 'left'
    ctx.fillText('Market Stats', statsX, statsY)
    
    statsY += 20
    ctx.fillStyle = '#00ff88'
    ctx.font = '10px monospace'
    ctx.fillText(`Volume: $${(totalVolume / 1000000).toFixed(1)}M`, statsX, statsY)
    
    statsY += 15
    const activeOrders = orderFlows.length
    ctx.fillText(`Active Orders: ${activeOrders}`, statsX, statsY)
    
    statsY += 15
    const buyOrders = orderFlows.filter(o => o.type === 'buy').length
    const sellOrders = orderFlows.filter(o => o.type === 'sell').length
    ctx.fillStyle = '#00ff88'
    ctx.fillText(`Buy: ${buyOrders}`, statsX, statsY)
    
    statsY += 12
    ctx.fillStyle = '#ff4757'
    ctx.fillText(`Sell: ${sellOrders}`, statsX, statsY)

    // 실시간 표시기 (하단)
    if (isTrading) {
      const indicatorY = height - 20
      
      // 실시간 텍스트
      ctx.fillStyle = '#00ff88'
      ctx.font = 'bold 12px monospace'
      ctx.textAlign = 'left'
      ctx.fillText('🔴 LIVE TRADING', 10, indicatorY)
      
      // 펄스 애니메이션
      const pulseRadius = 3 + 2 * Math.sin(animationFrame * 0.2)
      ctx.beginPath()
      ctx.arc(120, indicatorY - 5, pulseRadius, 0, 2 * Math.PI)
      ctx.fillStyle = '#ff4757'
      ctx.fill()
      
      // 데이터 플로우 라인
      const flowY = indicatorY - 10
      for (let i = 0; i < 5; i++) {
        const x = 150 + i * 30 + (animationFrame % 20)
        ctx.beginPath()
        ctx.rect(x, flowY, 10, 2)
        ctx.fillStyle = `rgba(0, 255, 136, ${0.8 - i * 0.15})`
        ctx.fill()
      }
    }

    // 제목
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
    ctx.font = 'bold 14px monospace'
    ctx.textAlign = 'left'
    ctx.fillText('AI Security Lab', 10, 20)
  }

  const startTrading = () => {
    setIsTrading(true)
    setTotalVolume(0)
  }

  const stopTrading = () => {
    setIsTrading(false)
    setOrderFlows([])
    setTotalVolume(0)
  }

  return (
    <div className="relative bg-slate-900 rounded-lg p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold text-sm">AI Security Lab</h3>
        <div className="flex gap-2">
          {!isTrading ? (
            <button
              onClick={startTrading}
              className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-xs rounded transition-colors"
            >
              Start
            </button>
          ) : (
            <button
              onClick={stopTrading}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors"
            >
              Stop
            </button>
          )}
        </div>
      </div>
      
      <canvas
        ref={canvasRef}
        className="w-full border border-slate-600 rounded"
        style={{ maxHeight: '250px' }}
      />
      
      {isTrading && (
        <div className="absolute top-2 right-2 bg-green-600 text-white px-2 py-1 rounded text-xs animate-pulse">
          Trading...
        </div>
      )}
      
      <div className="mt-2 text-xs text-slate-400">
        Adversarial attacks and defense mechanisms
      </div>
    </div>
  )
}