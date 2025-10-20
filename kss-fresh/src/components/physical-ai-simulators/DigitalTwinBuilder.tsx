'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Play, Pause, RotateCcw, Plus, Trash2, Settings, Activity, Wifi, Cpu, Thermometer, Zap, Wind } from 'lucide-react'

interface IoTDevice {
  id: string
  name: string
  type: 'sensor' | 'actuator' | 'gateway' | 'controller'
  x: number
  y: number
  status: 'active' | 'inactive' | 'error'
  value: number
  connections: string[]
  icon: string
  color: string
}

interface DataFlow {
  from: string
  to: string
  data: number
  timestamp: number
}

type ScenarioType = 'smart-factory' | 'smart-building' | 'autonomous-vehicle' | 'custom'

const DEVICE_TEMPLATES: Record<string, Partial<IoTDevice>> = {
  temperature: { type: 'sensor', icon: 'thermometer', color: '#ef4444', name: 'Temperature Sensor' },
  humidity: { type: 'sensor', icon: 'wind', color: '#3b82f6', name: 'Humidity Sensor' },
  pressure: { type: 'sensor', icon: 'activity', color: '#8b5cf6', name: 'Pressure Sensor' },
  motor: { type: 'actuator', icon: 'zap', color: '#f59e0b', name: 'Motor Controller' },
  valve: { type: 'actuator', icon: 'settings', color: '#10b981', name: 'Valve Controller' },
  gateway: { type: 'gateway', icon: 'wifi', color: '#06b6d4', name: 'IoT Gateway' },
  plc: { type: 'controller', icon: 'cpu', color: '#6366f1', name: 'PLC Controller' }
}

const SCENARIOS: Record<ScenarioType, { name: string; devices: Partial<IoTDevice>[] }> = {
  'smart-factory': {
    name: 'Smart Factory',
    devices: [
      { ...DEVICE_TEMPLATES.temperature, x: 100, y: 100 },
      { ...DEVICE_TEMPLATES.pressure, x: 200, y: 100 },
      { ...DEVICE_TEMPLATES.motor, x: 300, y: 100 },
      { ...DEVICE_TEMPLATES.plc, x: 200, y: 200 },
      { ...DEVICE_TEMPLATES.gateway, x: 200, y: 300 }
    ]
  },
  'smart-building': {
    name: 'Smart Building',
    devices: [
      { ...DEVICE_TEMPLATES.temperature, x: 150, y: 80 },
      { ...DEVICE_TEMPLATES.humidity, x: 250, y: 80 },
      { ...DEVICE_TEMPLATES.valve, x: 350, y: 80 },
      { ...DEVICE_TEMPLATES.gateway, x: 250, y: 200 }
    ]
  },
  'autonomous-vehicle': {
    name: 'Autonomous Vehicle',
    devices: [
      { ...DEVICE_TEMPLATES.temperature, x: 100, y: 150 },
      { ...DEVICE_TEMPLATES.pressure, x: 200, y: 150 },
      { ...DEVICE_TEMPLATES.motor, x: 300, y: 150 },
      { ...DEVICE_TEMPLATES.plc, x: 200, y: 250 }
    ]
  },
  'custom': { name: 'Custom', devices: [] }
}

export default function DigitalTwinBuilder() {
  const [scenario, setScenario] = useState<ScenarioType>('smart-factory')
  const [devices, setDevices] = useState<IoTDevice[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null)
  const [dataFlows, setDataFlows] = useState<DataFlow[]>([])
  const [time, setTime] = useState(0)
  const [showConnections, setShowConnections] = useState(true)
  const [showDataFlow, setShowDataFlow] = useState(true)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()

  // Initialize scenario
  useEffect(() => {
    const scenarioData = SCENARIOS[scenario]
    const initialDevices: IoTDevice[] = scenarioData.devices.map((template, index) => ({
      id: `device-${index}`,
      name: template.name || 'Device',
      type: template.type || 'sensor',
      x: template.x || 100,
      y: template.y || 100,
      status: 'active',
      value: Math.random() * 100,
      connections: [],
      icon: template.icon || 'activity',
      color: template.color || '#3b82f6'
    }))

    // Auto-connect devices in a logical way
    if (initialDevices.length > 1) {
      // Connect sensors to controllers/gateways
      const controllers = initialDevices.filter(d => d.type === 'controller' || d.type === 'gateway')
      const sensors = initialDevices.filter(d => d.type === 'sensor')
      const actuators = initialDevices.filter(d => d.type === 'actuator')

      if (controllers.length > 0) {
        const controller = controllers[0]
        sensors.forEach(sensor => {
          sensor.connections.push(controller.id)
        })
        actuators.forEach(actuator => {
          controller.connections.push(actuator.id)
        })
      }
    }

    setDevices(initialDevices)
    setDataFlows([])
    setTime(0)
  }, [scenario])

  // Simulate device data
  const updateDevices = useCallback(() => {
    setDevices(prevDevices =>
      prevDevices.map(device => {
        let newValue = device.value

        switch (device.type) {
          case 'sensor':
            // Simulate sensor readings with noise
            newValue = 50 + 30 * Math.sin(time * 0.1 + Math.random()) + Math.random() * 10
            break
          case 'actuator':
            // Actuators respond to controller commands
            const controllerConnected = prevDevices.find(d =>
              d.connections.includes(device.id) && d.type === 'controller'
            )
            if (controllerConnected) {
              newValue = controllerConnected.value * 0.8 + Math.random() * 5
            }
            break
          case 'controller':
            // Controllers aggregate sensor data
            const connectedSensors = prevDevices.filter(d =>
              d.connections.includes(device.id) && d.type === 'sensor'
            )
            if (connectedSensors.length > 0) {
              newValue = connectedSensors.reduce((sum, s) => sum + s.value, 0) / connectedSensors.length
            }
            break
          case 'gateway':
            // Gateways relay all data
            newValue = Math.random() * 100
            break
        }

        return {
          ...device,
          value: Math.max(0, Math.min(100, newValue)),
          status: newValue > 90 || newValue < 10 ? 'error' : 'active'
        }
      })
    )

    // Generate data flows
    const newFlows: DataFlow[] = []
    devices.forEach(device => {
      device.connections.forEach(targetId => {
        newFlows.push({
          from: device.id,
          to: targetId,
          data: device.value,
          timestamp: time
        })
      })
    })
    setDataFlows(newFlows)
  }, [devices, time])

  // Animation loop
  useEffect(() => {
    if (!isRunning) return

    const animate = () => {
      setTime(t => t + 0.1)
      updateDevices()
      drawCanvas()
      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRunning, updateDevices])

  // Draw canvas
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw grid
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.3)'
    ctx.lineWidth = 1
    for (let x = 0; x < canvas.width; x += 50) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, canvas.height)
      ctx.stroke()
    }
    for (let y = 0; y < canvas.height; y += 50) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(canvas.width, y)
      ctx.stroke()
    }

    // Draw connections
    if (showConnections) {
      devices.forEach(device => {
        device.connections.forEach(targetId => {
          const target = devices.find(d => d.id === targetId)
          if (!target) return

          // Connection line
          ctx.strokeStyle = 'rgba(100, 116, 139, 0.6)'
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.moveTo(device.x, device.y)
          ctx.lineTo(target.x, target.y)
          ctx.stroke()

          // Data flow animation
          if (showDataFlow) {
            const progress = (time % 2) / 2
            const flowX = device.x + (target.x - device.x) * progress
            const flowY = device.y + (target.y - device.y) * progress

            ctx.fillStyle = device.color
            ctx.beginPath()
            ctx.arc(flowX, flowY, 5, 0, Math.PI * 2)
            ctx.fill()
          }
        })
      })
    }

    // Draw devices
    devices.forEach(device => {
      const isSelected = device.id === selectedDevice

      // Device circle
      const gradient = ctx.createRadialGradient(
        device.x - 10,
        device.y - 10,
        5,
        device.x,
        device.y,
        30
      )
      gradient.addColorStop(0, '#ffffff')
      gradient.addColorStop(1, device.color)
      ctx.fillStyle = gradient
      ctx.beginPath()
      ctx.arc(device.x, device.y, 30, 0, Math.PI * 2)
      ctx.fill()

      // Selected outline
      if (isSelected) {
        ctx.strokeStyle = '#fbbf24'
        ctx.lineWidth = 3
        ctx.stroke()
      }

      // Status ring
      const statusColor = device.status === 'active' ? '#10b981' : device.status === 'error' ? '#ef4444' : '#6b7280'
      ctx.strokeStyle = statusColor
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.arc(device.x, device.y, 35, 0, Math.PI * 2)
      ctx.stroke()

      // Activity pulse
      if (device.status === 'active' && isRunning) {
        const pulseRadius = 35 + Math.sin(time * 5) * 5
        ctx.strokeStyle = `${statusColor}66`
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(device.x, device.y, pulseRadius, 0, Math.PI * 2)
        ctx.stroke()
      }

      // Device label
      ctx.font = 'bold 12px Inter, sans-serif'
      ctx.fillStyle = '#e5e7eb'
      ctx.textAlign = 'center'
      ctx.fillText(device.name.split(' ')[0], device.x, device.y - 50)

      // Value
      ctx.font = '10px Inter, sans-serif'
      ctx.fillStyle = '#94a3b8'
      ctx.fillText(`${device.value.toFixed(1)}`, device.x, device.y + 55)
    })
  }, [devices, selectedDevice, showConnections, showDataFlow, time, isRunning])

  // Initial draw
  useEffect(() => {
    drawCanvas()
  }, [devices, drawCanvas])

  const handleAddDevice = (templateKey: string) => {
    const template = DEVICE_TEMPLATES[templateKey]
    const newDevice: IoTDevice = {
      id: `device-${Date.now()}`,
      name: template.name || 'Device',
      type: template.type || 'sensor',
      x: 200 + Math.random() * 200,
      y: 200 + Math.random() * 100,
      status: 'active',
      value: Math.random() * 100,
      connections: [],
      icon: template.icon || 'activity',
      color: template.color || '#3b82f6'
    }
    setDevices([...devices, newDevice])
  }

  const handleRemoveDevice = (deviceId: string) => {
    setDevices(devices.filter(d => d.id !== deviceId))
    // Remove connections
    setDevices(prevDevices =>
      prevDevices.map(d => ({
        ...d,
        connections: d.connections.filter(id => id !== deviceId)
      }))
    )
  }

  const handleDeviceClick = (deviceId: string) => {
    if (selectedDevice && selectedDevice !== deviceId) {
      // Create connection
      setDevices(prevDevices =>
        prevDevices.map(d =>
          d.id === selectedDevice
            ? { ...d, connections: [...d.connections, deviceId] }
            : d
        )
      )
      setSelectedDevice(null)
    } else {
      setSelectedDevice(deviceId)
    }
  }

  const handleStart = () => setIsRunning(true)
  const handlePause = () => setIsRunning(false)
  const handleReset = () => {
    setIsRunning(false)
    setTime(0)
    setDataFlows([])
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Activity className="w-10 h-10 text-purple-400" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Digital Twin Builder
            </h1>
          </div>
          <p className="text-slate-300 text-lg">
            IoT 디바이스와 CPS(Cyber-Physical Systems) 시뮬레이션 플랫폼
          </p>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
          {/* Canvas */}
          <div className="lg:col-span-3 bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
            <canvas
              ref={canvasRef}
              width={800}
              height={500}
              className="w-full cursor-pointer bg-slate-950 rounded-lg"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect()
                const x = e.clientX - rect.left
                const y = e.clientY - rect.top

                // Check if clicked on device
                const clickedDevice = devices.find(d => {
                  const dx = d.x - x
                  const dy = d.y - y
                  return Math.sqrt(dx * dx + dy * dy) < 30
                })

                if (clickedDevice) {
                  handleDeviceClick(clickedDevice.id)
                }
              }}
            />

            {/* Canvas Controls */}
            <div className="mt-4 flex justify-between items-center">
              <div className="flex gap-2">
                <button
                  onClick={handleStart}
                  disabled={isRunning}
                  className="bg-green-600 hover:bg-green-700 disabled:bg-slate-600 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Start
                </button>
                <button
                  onClick={handlePause}
                  disabled={!isRunning}
                  className="bg-yellow-600 hover:bg-yellow-700 disabled:bg-slate-600 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center gap-2"
                >
                  <Pause className="w-4 h-4" />
                  Pause
                </button>
                <button
                  onClick={handleReset}
                  className="bg-slate-700 hover:bg-slate-600 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center gap-2"
                >
                  <RotateCcw className="w-4 h-4" />
                  Reset
                </button>
              </div>

              <div className="flex gap-4 text-sm">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showConnections}
                    onChange={(e) => setShowConnections(e.target.checked)}
                    className="w-4 h-4 accent-purple-500"
                  />
                  <span>Show Connections</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showDataFlow}
                    onChange={(e) => setShowDataFlow(e.target.checked)}
                    className="w-4 h-4 accent-purple-500"
                  />
                  <span>Show Data Flow</span>
                </label>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700 p-6 space-y-6">
            {/* Scenario Selection */}
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-purple-400" />
                Scenario
              </h2>
              <select
                value={scenario}
                onChange={(e) => {
                  setScenario(e.target.value as ScenarioType)
                  setIsRunning(false)
                }}
                className="w-full bg-slate-900/50 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                {Object.entries(SCENARIOS).map(([key, value]) => (
                  <option key={key} value={key}>
                    {value.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Add Devices */}
            <div>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Plus className="w-4 h-4 text-green-400" />
                Add Devices
              </h3>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(DEVICE_TEMPLATES).map(([key, template]) => (
                  <button
                    key={key}
                    onClick={() => handleAddDevice(key)}
                    className="bg-slate-700 hover:bg-slate-600 text-white text-xs font-medium py-2 px-3 rounded-lg transition-colors"
                    style={{ borderLeft: `3px solid ${template.color}` }}
                  >
                    {template.name?.split(' ')[0]}
                  </button>
                ))}
              </div>
            </div>

            {/* Device List */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Connected Devices ({devices.length})</h3>
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {devices.map(device => (
                  <div
                    key={device.id}
                    className={`p-3 rounded-lg cursor-pointer transition-colors ${
                      selectedDevice === device.id
                        ? 'bg-purple-600'
                        : 'bg-slate-700 hover:bg-slate-600'
                    }`}
                    onClick={() => setSelectedDevice(device.id)}
                  >
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-medium text-sm">{device.name}</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleRemoveDevice(device.id)
                        }}
                        className="text-red-400 hover:text-red-300"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                    <div className="flex justify-between text-xs text-slate-300">
                      <span>{device.type}</span>
                      <span>{device.value.toFixed(1)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Info */}
            <div className="bg-slate-900/50 rounded-lg p-4 text-sm text-slate-300">
              <p className="mb-2">
                <strong>Click a device</strong> to select it, then click another to create a connection.
              </p>
              <p>
                Running time: <strong>{time.toFixed(1)}s</strong>
              </p>
              <p>
                Data flows: <strong>{dataFlows.length}</strong>
              </p>
            </div>
          </div>
        </div>

        {/* Info Panel */}
        <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-semibold mb-3 text-purple-400">Digital Twin Concepts</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm text-slate-300">
            <div>
              <h4 className="font-semibold text-white mb-2">Sensors</h4>
              <p>물리적 환경의 데이터를 수집하는 디바이스 (온도, 습도, 압력 등)</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Actuators</h4>
              <p>물리적 환경에 영향을 주는 제어 디바이스 (모터, 밸브 등)</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Controllers</h4>
              <p>센서 데이터를 분석하고 액추에이터를 제어하는 지능형 시스템</p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-2">Gateways</h4>
              <p>디바이스와 클라우드를 연결하는 통신 허브</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
