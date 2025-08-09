'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import { ArrowLeft, Play, Pause, RotateCcw, Database, TrendingUp, Package, Users, DollarSign, Calendar, BarChart3, Activity, Truck, AlertTriangle, CheckCircle, RefreshCw, Sparkles, Building, ShoppingCart } from 'lucide-react'

interface ProductionOrder {
  id: string
  orderNo: string
  product: string
  quantity: number
  completed: number
  status: 'waiting' | 'processing' | 'completed' | 'delayed'
  dueDate: Date
  priority: 'low' | 'normal' | 'high' | 'urgent'
  customer: string
}

interface Resource {
  id: string
  name: string
  type: 'machine' | 'worker' | 'material'
  availability: number
  utilization: number
  status: 'available' | 'busy' | 'maintenance' | 'offline'
}

interface InventoryItem {
  id: string
  material: string
  stockLevel: number
  reorderPoint: number
  leadTime: number
  unit: string
  value: number
}

interface KPIMetric {
  name: string
  value: number
  target: number
  trend: 'up' | 'down' | 'stable'
  unit: string
  icon: any
}

export default function MESERPDashboardPage() {
  const searchParams = useSearchParams()
  const backUrl = searchParams.get('from') || '/modules/smart-factory'
  const [isRunning, setIsRunning] = useState(false)
  const [activeModule, setActiveModule] = useState<'production' | 'inventory' | 'quality' | 'finance'>('production')
  const [dataRefreshRate, setDataRefreshRate] = useState(5) // seconds
  const [showAlerts, setShowAlerts] = useState(true)
  
  const productionChartRef = useRef<HTMLCanvasElement>(null)
  const financialChartRef = useRef<HTMLCanvasElement>(null)
  
  const [productionOrders, setProductionOrders] = useState<ProductionOrder[]>([
    {
      id: 'PO001',
      orderNo: 'ORD-2024-001',
      product: '스마트 센서 A',
      quantity: 1000,
      completed: 750,
      status: 'processing',
      dueDate: new Date('2024-12-20'),
      priority: 'high',
      customer: '삼성전자'
    },
    {
      id: 'PO002',
      orderNo: 'ORD-2024-002',
      product: '컨트롤러 B',
      quantity: 500,
      completed: 500,
      status: 'completed',
      dueDate: new Date('2024-12-18'),
      priority: 'normal',
      customer: 'LG전자'
    },
    {
      id: 'PO003',
      orderNo: 'ORD-2024-003',
      product: '액추에이터 C',
      quantity: 800,
      completed: 200,
      status: 'delayed',
      dueDate: new Date('2024-12-17'),
      priority: 'urgent',
      customer: '현대자동차'
    }
  ])

  const [resources, setResources] = useState<Resource[]>([
    { id: 'R001', name: 'CNC 머신 #1', type: 'machine', availability: 100, utilization: 85, status: 'busy' },
    { id: 'R002', name: '조립 라인 A', type: 'machine', availability: 100, utilization: 92, status: 'busy' },
    { id: 'R003', name: '숙련공 팀 1', type: 'worker', availability: 8, utilization: 87, status: 'available' },
    { id: 'R004', name: 'PCB 재료', type: 'material', availability: 2500, utilization: 65, status: 'available' }
  ])

  const [inventory, setInventory] = useState<InventoryItem[]>([
    { id: 'INV001', material: 'PCB 보드', stockLevel: 2500, reorderPoint: 1000, leadTime: 7, unit: 'EA', value: 125000 },
    { id: 'INV002', material: '마이크로칩', stockLevel: 850, reorderPoint: 500, leadTime: 14, unit: 'EA', value: 425000 },
    { id: 'INV003', material: '알루미늄 케이스', stockLevel: 1200, reorderPoint: 800, leadTime: 5, unit: 'EA', value: 60000 },
    { id: 'INV004', material: '전선', stockLevel: 500, reorderPoint: 1000, leadTime: 3, unit: 'M', value: 25000 }
  ])

  const [kpiMetrics, setKpiMetrics] = useState<KPIMetric[]>([
    { name: 'OEE', value: 85.2, target: 85, trend: 'up', unit: '%', icon: Activity },
    { name: '일일 생산량', value: 2340, target: 2500, trend: 'stable', unit: 'EA', icon: Package },
    { name: '재고 회전율', value: 12.5, target: 10, trend: 'up', unit: '회/년', icon: RefreshCw },
    { name: '납기 준수율', value: 94.8, target: 95, trend: 'down', unit: '%', icon: Calendar },
    { name: '원가 절감', value: 8.2, target: 10, trend: 'stable', unit: '%', icon: DollarSign },
    { name: '품질 수율', value: 98.5, target: 99, trend: 'up', unit: '%', icon: CheckCircle }
  ])

  const [alerts, setAlerts] = useState<{id: string, type: 'warning' | 'error' | 'info', message: string, time: Date}[]>([
    { id: '1', type: 'warning', message: '액추에이터 C 생산 지연 - 자재 부족', time: new Date() },
    { id: '2', type: 'info', message: 'ERP 시스템 정기 점검 예정 (오후 10시)', time: new Date() }
  ])

  // 생산 차트 그리기
  useEffect(() => {
    if (!productionChartRef.current) return
    const canvas = productionChartRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = 300

    const drawChart = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 시간대별 생산량
      const hours = 24
      const barWidth = (canvas.width - 40) / hours
      const maxProduction = 150

      for (let i = 0; i < hours; i++) {
        const production = isRunning ? 
          Math.random() * 100 + (i > 8 && i < 17 ? 50 : 20) : // 주간 시간대 생산량 증가
          80 + Math.random() * 20
        
        const barHeight = (production / maxProduction) * (canvas.height - 60)
        const x = 20 + i * barWidth
        const y = canvas.height - barHeight - 40

        // 막대 그리기
        const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight)
        gradient.addColorStop(0, '#3B82F6')
        gradient.addColorStop(1, '#1E40AF')
        
        ctx.fillStyle = gradient
        ctx.fillRect(x, y, barWidth - 2, barHeight)

        // 시간 라벨
        if (i % 4 === 0) {
          ctx.fillStyle = '#666'
          ctx.font = '10px Arial'
          ctx.textAlign = 'center'
          ctx.fillText(`${i}시`, x + barWidth/2, canvas.height - 25)
        }
      }

      // 축 그리기
      ctx.strokeStyle = '#E5E7EB'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(20, canvas.height - 40)
      ctx.lineTo(canvas.width - 20, canvas.height - 40)
      ctx.stroke()

      // Y축 라벨
      ctx.fillStyle = '#666'
      ctx.font = '12px Arial'
      ctx.textAlign = 'right'
      ctx.fillText('150', 15, 20)
      ctx.fillText('100', 15, canvas.height/3)
      ctx.fillText('50', 15, canvas.height*2/3)
      ctx.fillText('0', 15, canvas.height - 40)

      // 제목
      ctx.fillStyle = '#000'
      ctx.font = 'bold 14px Arial'
      ctx.textAlign = 'left'
      ctx.fillText('시간대별 생산량 (단위: EA)', 20, 20)
    }

    const interval = setInterval(drawChart, 1000)
    return () => clearInterval(interval)
  }, [isRunning])

  // 재무 차트 그리기
  useEffect(() => {
    if (!financialChartRef.current) return
    const canvas = financialChartRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = 250

    const drawFinancialChart = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // 매출/비용 추이
      const months = 6
      const dataPoints = []
      const costPoints = []
      
      for (let i = 0; i < months; i++) {
        dataPoints.push({
          x: (canvas.width / months) * i + 40,
          revenue: 200 + Math.random() * 100 + (isRunning ? i * 10 : 0),
          cost: 150 + Math.random() * 50
        })
      }

      // 매출 라인
      ctx.strokeStyle = '#10B981'
      ctx.lineWidth = 3
      ctx.beginPath()
      dataPoints.forEach((point, index) => {
        const y = canvas.height - 40 - (point.revenue / 400) * (canvas.height - 80)
        if (index === 0) ctx.moveTo(point.x, y)
        else ctx.lineTo(point.x, y)
      })
      ctx.stroke()

      // 비용 라인
      ctx.strokeStyle = '#EF4444'
      ctx.lineWidth = 3
      ctx.beginPath()
      dataPoints.forEach((point, index) => {
        const y = canvas.height - 40 - (point.cost / 400) * (canvas.height - 80)
        if (index === 0) ctx.moveTo(point.x, y)
        else ctx.lineTo(point.x, y)
      })
      ctx.stroke()

      // 점 그리기
      dataPoints.forEach(point => {
        // 매출 점
        ctx.fillStyle = '#10B981'
        ctx.beginPath()
        ctx.arc(point.x, canvas.height - 40 - (point.revenue / 400) * (canvas.height - 80), 5, 0, Math.PI * 2)
        ctx.fill()

        // 비용 점
        ctx.fillStyle = '#EF4444'
        ctx.beginPath()
        ctx.arc(point.x, canvas.height - 40 - (point.cost / 400) * (canvas.height - 80), 5, 0, Math.PI * 2)
        ctx.fill()
      })

      // 축과 라벨
      ctx.strokeStyle = '#E5E7EB'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(40, canvas.height - 40)
      ctx.lineTo(canvas.width - 20, canvas.height - 40)
      ctx.moveTo(40, 20)
      ctx.lineTo(40, canvas.height - 40)
      ctx.stroke()

      // 범례
      ctx.fillStyle = '#10B981'
      ctx.fillRect(canvas.width - 100, 20, 20, 3)
      ctx.fillStyle = '#000'
      ctx.font = '12px Arial'
      ctx.fillText('매출', canvas.width - 75, 25)
      
      ctx.fillStyle = '#EF4444'
      ctx.fillRect(canvas.width - 100, 35, 20, 3)
      ctx.fillStyle = '#000'
      ctx.fillText('비용', canvas.width - 75, 40)
    }

    const interval = setInterval(drawFinancialChart, 2000)
    return () => clearInterval(interval)
  }, [isRunning])

  // 실시간 데이터 업데이트
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      // 생산 주문 업데이트
      setProductionOrders(prev => prev.map(order => {
        if (order.status === 'processing') {
          const progress = Math.min(order.quantity, order.completed + Math.random() * 20)
          return {
            ...order,
            completed: Math.round(progress),
            status: progress >= order.quantity ? 'completed' : 'processing'
          }
        }
        return order
      }))

      // 자원 활용도 업데이트
      setResources(prev => prev.map(resource => ({
        ...resource,
        utilization: Math.max(0, Math.min(100, resource.utilization + (Math.random() - 0.5) * 10))
      })))

      // 재고 업데이트
      setInventory(prev => prev.map(item => ({
        ...item,
        stockLevel: Math.max(0, item.stockLevel + (Math.random() - 0.6) * 50)
      })))

      // KPI 업데이트
      setKpiMetrics(prev => prev.map(kpi => ({
        ...kpi,
        value: kpi.value + (Math.random() - 0.5) * 2,
        trend: Math.random() > 0.7 ? (Math.random() > 0.5 ? 'up' : 'down') : kpi.trend
      })))

      // 새 알림 생성
      if (Math.random() > 0.8) {
        const alertTypes = [
          { type: 'warning' as const, message: '재고 수준 경고: 마이크로칩 재주문 필요' },
          { type: 'info' as const, message: '생산 목표 달성: 스마트 센서 A' },
          { type: 'error' as const, message: '장비 오류: CNC 머신 #2 점검 필요' }
        ]
        const newAlert = alertTypes[Math.floor(Math.random() * alertTypes.length)]
        
        setAlerts(prev => [...prev.slice(-4), {
          id: Date.now().toString(),
          ...newAlert,
          time: new Date()
        }])
      }
    }, dataRefreshRate * 1000)

    return () => clearInterval(interval)
  }, [isRunning, dataRefreshRate])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
      case 'processing': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
      case 'delayed': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
      case 'waiting': return 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return 'bg-red-500'
      case 'high': return 'bg-orange-500'
      case 'normal': return 'bg-blue-500'
      case 'low': return 'bg-gray-500'
      default: return 'bg-gray-500'
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link 
                href={backUrl}
                className="flex items-center gap-2 text-amber-600 dark:text-amber-400 hover:text-amber-700 dark:hover:text-amber-300"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>학습 페이지로 돌아가기</span>
              </Link>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <RefreshCw className="w-4 h-4 text-gray-500" />
                <select 
                  value={dataRefreshRate}
                  onChange={(e) => setDataRefreshRate(Number(e.target.value))}
                  className="text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1"
                >
                  <option value={1}>1초</option>
                  <option value={5}>5초</option>
                  <option value={10}>10초</option>
                </select>
              </div>
              <button
                onClick={() => setShowAlerts(!showAlerts)}
                className={`px-3 py-1 rounded text-sm ${
                  showAlerts 
                    ? 'bg-yellow-600 text-white' 
                    : 'bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300'
                }`}
              >
                <AlertTriangle className="w-4 h-4 inline mr-1" />
                알림
              </button>
              <button
                onClick={() => setIsRunning(!isRunning)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium ${
                  isRunning 
                    ? 'bg-red-600 text-white hover:bg-red-700' 
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isRunning ? '실시간 중지' : '실시간 시작'}
              </button>
              <button
                onClick={() => {
                  setIsRunning(false)
                  // Reset data
                }}
                className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
              >
                <RotateCcw className="w-4 h-4" />
                리셋
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Title */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-xl flex items-center justify-center">
              <Database className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                MES/ERP 통합 대시보드
                <Sparkles className="w-6 h-6 text-yellow-500" />
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400">실시간 생산 및 경영 정보 통합 모니터링</p>
            </div>
          </div>
        </div>

        {/* Module Tabs */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-1 mb-6">
          <div className="flex">
            <button
              onClick={() => setActiveModule('production')}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition-all ${
                activeModule === 'production'
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <Building className="w-4 h-4" />
              생산 관리
            </button>
            <button
              onClick={() => setActiveModule('inventory')}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition-all ${
                activeModule === 'inventory'
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <Package className="w-4 h-4" />
              재고 관리
            </button>
            <button
              onClick={() => setActiveModule('quality')}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition-all ${
                activeModule === 'quality'
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <CheckCircle className="w-4 h-4" />
              품질 관리
            </button>
            <button
              onClick={() => setActiveModule('finance')}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md transition-all ${
                activeModule === 'finance'
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <DollarSign className="w-4 h-4" />
              재무 관리
            </button>
          </div>
        </div>

        {/* KPI Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
          {kpiMetrics.map((kpi, index) => (
            <div key={index} className={`bg-white dark:bg-gray-800 rounded-xl p-4 border ${
              kpi.value < kpi.target * 0.9 ? 'border-red-500' : 'border-gray-200 dark:border-gray-700'
            }`}>
              <div className="flex items-center gap-2 mb-2">
                <kpi.icon className="w-5 h-5 text-indigo-500" />
                <span className="text-sm text-gray-600 dark:text-gray-400">{kpi.name}</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {kpi.value.toFixed(1)}{kpi.unit}
                </div>
                <TrendingUp className={`w-4 h-4 ${
                  kpi.trend === 'up' ? 'text-green-500' :
                  kpi.trend === 'down' ? 'text-red-500' :
                  'text-gray-500'
                }`} />
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                목표: {kpi.target}{kpi.unit}
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content Area */}
          <div className="lg:col-span-2 space-y-6">
            {activeModule === 'production' && (
              <>
                {/* Production Orders */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">생산 주문 현황</h2>
                  
                  <div className="space-y-4">
                    {productionOrders.map((order) => (
                      <div key={order.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <div className="flex items-center gap-3">
                              <h3 className="font-semibold text-gray-900 dark:text-white">{order.orderNo}</h3>
                              <div className={`w-2 h-2 rounded-full ${getPriorityColor(order.priority)}`}></div>
                              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(order.status)}`}>
                                {order.status === 'completed' ? '완료' :
                                 order.status === 'processing' ? '진행중' :
                                 order.status === 'delayed' ? '지연' : '대기'}
                              </span>
                            </div>
                            <p className="text-sm text-gray-600 dark:text-gray-400">{order.product} - {order.customer}</p>
                          </div>
                          <div className="text-right">
                            <div className="text-sm text-gray-500 dark:text-gray-400">납기</div>
                            <div className="font-medium text-gray-900 dark:text-white">
                              {order.dueDate.toLocaleDateString()}
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-gray-600 dark:text-gray-400">진행률</span>
                            <span className="font-medium text-gray-900 dark:text-white">
                              {order.completed} / {order.quantity} ({Math.round(order.completed / order.quantity * 100)}%)
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full transition-all duration-500 ${
                                order.status === 'delayed' ? 'bg-red-500' :
                                order.status === 'completed' ? 'bg-green-500' :
                                'bg-blue-500'
                              }`}
                              style={{ width: `${(order.completed / order.quantity) * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Production Chart */}
                <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                  <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">생산량 추이</h2>
                  <canvas 
                    ref={productionChartRef}
                    className="w-full"
                    style={{ height: '300px' }}
                  />
                </div>
              </>
            )}

            {activeModule === 'inventory' && (
              <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">재고 현황</h2>
                
                <div className="space-y-4">
                  {inventory.map((item) => (
                    <div key={item.id} className={`border rounded-lg p-4 ${
                      item.stockLevel < item.reorderPoint 
                        ? 'border-red-500 bg-red-50 dark:bg-red-900/20' 
                        : 'border-gray-200 dark:border-gray-700'
                    }`}>
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold text-gray-900 dark:text-white">{item.material}</h3>
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          ₩{item.value.toLocaleString()}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">재고량</span>
                          <div className="font-medium text-gray-900 dark:text-white">
                            {item.stockLevel} {item.unit}
                          </div>
                        </div>
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">재주문점</span>
                          <div className="font-medium text-gray-900 dark:text-white">
                            {item.reorderPoint} {item.unit}
                          </div>
                        </div>
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">리드타임</span>
                          <div className="font-medium text-gray-900 dark:text-white">
                            {item.leadTime}일
                          </div>
                        </div>
                      </div>
                      
                      {item.stockLevel < item.reorderPoint && (
                        <div className="mt-3 flex items-center gap-2 text-red-600 dark:text-red-400">
                          <AlertTriangle className="w-4 h-4" />
                          <span className="text-sm font-medium">재주문 필요</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeModule === 'finance' && (
              <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">재무 현황</h2>
                <canvas 
                  ref={financialChartRef}
                  className="w-full"
                  style={{ height: '250px' }}
                />
              </div>
            )}
          </div>

          {/* Side Panel */}
          <div className="space-y-6">
            {/* Resources */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">자원 현황</h3>
              
              <div className="space-y-3">
                {resources.map((resource) => (
                  <div key={resource.id} className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {resource.type === 'machine' && <Building className="w-4 h-4 text-blue-500" />}
                        {resource.type === 'worker' && <Users className="w-4 h-4 text-green-500" />}
                        {resource.type === 'material' && <Package className="w-4 h-4 text-purple-500" />}
                        <span className="font-medium text-gray-900 dark:text-white text-sm">{resource.name}</span>
                      </div>
                      <span className={`text-xs font-medium ${
                        resource.status === 'available' ? 'text-green-600' :
                        resource.status === 'busy' ? 'text-blue-600' :
                        resource.status === 'maintenance' ? 'text-yellow-600' :
                        'text-red-600'
                      }`}>
                        {resource.status === 'available' ? '가능' :
                         resource.status === 'busy' ? '사용중' :
                         resource.status === 'maintenance' ? '정비' : '오프라인'}
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full ${
                            resource.utilization > 90 ? 'bg-red-500' :
                            resource.utilization > 70 ? 'bg-yellow-500' :
                            'bg-green-500'
                          }`}
                          style={{ width: `${resource.utilization}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-600 dark:text-gray-400">
                        {resource.utilization.toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Alerts */}
            {showAlerts && (
              <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">실시간 알림</h3>
                
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {alerts.slice().reverse().map((alert) => (
                    <div key={alert.id} className={`p-3 rounded-lg text-sm ${
                      alert.type === 'error' ? 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-300' :
                      alert.type === 'warning' ? 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300' :
                      'bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                    }`}>
                      <p className="font-medium">{alert.message}</p>
                      <p className="text-xs opacity-70 mt-1">
                        {alert.time.toLocaleTimeString()}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Quick Actions */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">빠른 작업</h3>
              
              <div className="space-y-3">
                <button className="w-full p-3 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors text-sm font-medium">
                  <ShoppingCart className="w-4 h-4 inline mr-2" />
                  신규 주문 생성
                </button>
                
                <button className="w-full p-3 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors text-sm font-medium">
                  <Truck className="w-4 h-4 inline mr-2" />
                  자재 발주
                </button>
                
                <button className="w-full p-3 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors text-sm font-medium">
                  <BarChart3 className="w-4 h-4 inline mr-2" />
                  리포트 생성
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}