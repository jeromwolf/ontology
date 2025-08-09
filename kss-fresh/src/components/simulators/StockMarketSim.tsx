'use client'

import { useEffect, useRef, useState } from 'react'

interface StockData {
  price: number
  volume: number
  change: number
  timestamp: number
}

interface OrderBook {
  bids: Array<{ price: number; volume: number }>
  asks: Array<{ price: number; volume: number }>
}

export default function StockMarketSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stockData, setStockData] = useState<StockData[]>([])
  const [currentPrice, setCurrentPrice] = useState(50000)
  const [orderBook, setOrderBook] = useState<OrderBook>({ bids: [], asks: [] })
  const [isTrading, setIsTrading] = useState(false)
  const [volume, setVolume] = useState(0)
  const [animationFrame, setAnimationFrame] = useState(0)
  const animationRef = useRef<number>()

  // 초기 데이터 생성
  useEffect(() => {
    const initialData: StockData[] = []
    let price = 50000
    
    for (let i = 0; i < 50; i++) {
      const change = (Math.random() - 0.5) * 1000
      price += change
      initialData.push({
        price,
        volume: Math.random() * 1000 + 100,
        change,
        timestamp: Date.now() - (50 - i) * 60000
      })
    }
    
    setStockData(initialData)
    setCurrentPrice(price)
    
    // 초기 호가창 생성
    generateOrderBook(price)
  }, [])

  const generateOrderBook = (basePrice: number) => {
    const bids = []
    const asks = []
    
    for (let i = 1; i <= 5; i++) {
      bids.push({
        price: basePrice - i * 100,
        volume: Math.random() * 50 + 10
      })
      asks.push({
        price: basePrice + i * 100,
        volume: Math.random() * 50 + 10
      })
    }
    
    setOrderBook({ bids, asks })
  }

  // 실시간 트레이딩 시뮬레이션
  useEffect(() => {
    const animate = () => {
      setAnimationFrame(prev => prev + 1)
      
      if (isTrading) {
        // 가격 업데이트 (2초마다)
        if (animationFrame % 120 === 0) {
          const change = (Math.random() - 0.5) * 2000
          const newPrice = Math.max(1000, currentPrice + change)
          const newVolume = Math.random() * 1000 + 200
          
          setCurrentPrice(newPrice)
          setVolume(newVolume)
          
          // 새로운 데이터 포인트 추가
          setStockData(prev => {
            const newData = [...prev.slice(1), {
              price: newPrice,
              volume: newVolume,
              change,
              timestamp: Date.now()
            }]
            return newData
          })
          
          // 호가창 업데이트
          generateOrderBook(newPrice)
        }
      }
      
      drawChart()
      animationRef.current = requestAnimationFrame(animate)
    }

    animate()
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [stockData, currentPrice, isTrading, animationFrame])

  const drawChart = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width = 400
    const height = canvas.height = 200

    // 배경
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, width, height)

    if (stockData.length === 0) return

    // 가격 범위 계산
    const prices = stockData.map(d => d.price)
    const maxPrice = Math.max(...prices)
    const minPrice = Math.min(...prices)
    const priceRange = maxPrice - minPrice || 1

    // 그리드 그리기
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    
    for (let i = 0; i <= 4; i++) {
      const y = (height / 4) * i
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // 가격 차트 그리기
    ctx.beginPath()
    ctx.strokeStyle = '#00ff88'
    ctx.lineWidth = 2
    
    stockData.forEach((data, index) => {
      const x = (width / (stockData.length - 1)) * index
      const y = height - ((data.price - minPrice) / priceRange) * height
      
      if (index === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()

    // 실시간 포인트 강조
    if (isTrading && stockData.length > 0) {
      const lastData = stockData[stockData.length - 1]
      const lastX = width - 10
      const lastY = height - ((lastData.price - minPrice) / priceRange) * height
      
      const pulseSize = 2 + 3 * Math.sin(animationFrame * 0.1)
      
      ctx.beginPath()
      ctx.arc(lastX, lastY, pulseSize, 0, 2 * Math.PI)
      ctx.fillStyle = lastData.change >= 0 ? '#00ff88' : '#ff4757'
      ctx.fill()
      
      // 글로우 효과
      ctx.beginPath()
      ctx.arc(lastX, lastY, pulseSize + 5, 0, 2 * Math.PI)
      ctx.fillStyle = `rgba(0, 255, 136, ${0.3 * Math.sin(animationFrame * 0.1)})`
      ctx.fill()
    }

    // 볼륨 바 차트 (하단)
    const volumeHeight = 40
    const volumes = stockData.map(d => d.volume)
    const maxVolume = Math.max(...volumes) || 1
    
    ctx.fillStyle = 'rgba(100, 100, 255, 0.3)'
    stockData.forEach((data, index) => {
      const x = (width / stockData.length) * index
      const barWidth = width / stockData.length - 1
      const barHeight = (data.volume / maxVolume) * volumeHeight
      
      ctx.fillRect(x, height - volumeHeight, barWidth, barHeight)
    })

    // 현재 가격 표시
    ctx.fillStyle = '#ffffff'
    ctx.font = '14px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(`₩${currentPrice.toLocaleString()}`, 10, 25)
    
    if (isTrading && stockData.length > 0) {
      const lastChange = stockData[stockData.length - 1].change
      const changeColor = lastChange >= 0 ? '#00ff88' : '#ff4757'
      const changeText = `${lastChange >= 0 ? '+' : ''}${lastChange.toFixed(0)}`
      
      ctx.fillStyle = changeColor
      ctx.font = '12px monospace'
      ctx.fillText(changeText, 10, 45)
      
      // 거래량
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
      ctx.fillText(`Vol: ${volume.toFixed(0)}`, 10, height - 10)
    }

    // 실시간 틱 표시
    if (isTrading) {
      const tickX = width - 30
      ctx.fillStyle = '#00ff88'
      ctx.font = '10px monospace'
      ctx.textAlign = 'center'
      ctx.fillText('LIVE', tickX, 15)
      
      // 깜빡이는 점
      if (Math.sin(animationFrame * 0.2) > 0) {
        ctx.beginPath()
        ctx.arc(tickX + 20, 10, 3, 0, 2 * Math.PI)
        ctx.fillStyle = '#ff4757'
        ctx.fill()
      }
    }
  }

  const startTrading = () => {
    setIsTrading(true)
  }

  const stopTrading = () => {
    setIsTrading(false)
  }

  return (
    <div className="relative bg-gray-900 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold text-sm">Crypto Prediction Markets</h3>
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
        className="w-full border border-gray-600 rounded"
        style={{ maxHeight: '200px' }}
      />
      
      {isTrading && (
        <div className="absolute top-2 right-2 bg-green-600 text-white px-2 py-1 rounded text-xs animate-pulse">
          Trading...
        </div>
      )}
      
      {/* 호가창 미니 버전 */}
      <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
        <div>
          <div className="text-red-400 mb-1">매도</div>
          {orderBook.asks.slice(0, 3).map((ask, i) => (
            <div key={i} className="flex justify-between text-red-300">
              <span>{ask.price.toLocaleString()}</span>
              <span>{ask.volume.toFixed(0)}</span>
            </div>
          ))}
        </div>
        <div>
          <div className="text-blue-400 mb-1">매수</div>
          {orderBook.bids.slice(0, 3).map((bid, i) => (
            <div key={i} className="flex justify-between text-blue-300">
              <span>{bid.price.toLocaleString()}</span>
              <span>{bid.volume.toFixed(0)}</span>
            </div>
          ))}
        </div>
      </div>
      
      <div className="mt-2 text-xs text-gray-400">
        Blockchain-based prediction with live trading
      </div>
    </div>
  )
}