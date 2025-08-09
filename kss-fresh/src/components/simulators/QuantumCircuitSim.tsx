'use client'

import { useEffect, useRef, useState } from 'react'

interface Qubit {
  id: number
  state: [number, number] // [|0⟩ amplitude, |1⟩ amplitude]
  x: number
  y: number
  phase: number
}

interface Gate {
  type: 'H' | 'X' | 'Y' | 'Z' | 'CNOT'
  qubit: number
  control?: number
  x: number
  y: number
  active: boolean
}

export default function QuantumCircuitSim() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [qubits, setQubits] = useState<Qubit[]>([])
  const [gates, setGates] = useState<Gate[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [animationFrame, setAnimationFrame] = useState(0)
  const animationRef = useRef<number>()

  // 양자 회로 초기화
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const width = canvas.width = 400
    const height = canvas.height = 250

    // 3큐비트 시스템 초기화
    const newQubits: Qubit[] = [
      { id: 0, state: [1, 0], x: 50, y: 60, phase: 0 },  // |0⟩ 상태
      { id: 1, state: [1, 0], x: 50, y: 120, phase: 0 }, // |0⟩ 상태  
      { id: 2, state: [1, 0], x: 50, y: 180, phase: 0 }  // |0⟩ 상태
    ]

    // 게이트 배치
    const newGates: Gate[] = [
      { type: 'H', qubit: 0, x: 120, y: 60, active: false },
      { type: 'CNOT', qubit: 1, control: 0, x: 200, y: 90, active: false },
      { type: 'X', qubit: 2, x: 280, y: 180, active: false },
      { type: 'H', qubit: 1, x: 320, y: 120, active: false }
    ]

    setQubits(newQubits)
    setGates(newGates)
  }, [])

  // 양자 게이트 시뮬레이션
  const applyGate = (gate: Gate, qubits: Qubit[]): Qubit[] => {
    const newQubits = [...qubits]
    
    switch (gate.type) {
      case 'H': // Hadamard gate
        const qubit = newQubits[gate.qubit]
        const norm = 1 / Math.sqrt(2)
        newQubits[gate.qubit] = {
          ...qubit,
          state: [
            norm * (qubit.state[0] + qubit.state[1]),
            norm * (qubit.state[0] - qubit.state[1])
          ]
        }
        break
        
      case 'X': // Pauli-X gate (bit flip)
        const xQubit = newQubits[gate.qubit]
        newQubits[gate.qubit] = {
          ...xQubit,
          state: [xQubit.state[1], xQubit.state[0]]
        }
        break
        
      case 'CNOT': // Controlled-NOT
        if (gate.control !== undefined) {
          const control = newQubits[gate.control]
          const target = newQubits[gate.qubit]
          
          // 제어 큐비트가 |1⟩ 상태일 때만 타겟에 X 게이트 적용
          if (Math.abs(control.state[1]) > 0.5) {
            newQubits[gate.qubit] = {
              ...target,
              state: [target.state[1], target.state[0]]
            }
          }
        }
        break
    }
    
    return newQubits
  }

  // 애니메이션 루프
  useEffect(() => {
    const animate = () => {
      setAnimationFrame(prev => prev + 1)
      
      if (isRunning) {
        // 게이트 단계별 실행 (3초마다)
        if (animationFrame % 180 === 0) {
          setCurrentStep(prev => {
            const nextStep = (prev + 1) % (gates.length + 1)
            
            if (nextStep === 0) {
              // 리셋
              setQubits(prev => prev.map(q => ({ 
                ...q, 
                state: [1, 0], 
                phase: 0 
              })))
              setGates(prev => prev.map(g => ({ ...g, active: false })))
            } else {
              // 게이트 실행
              const currentGate = gates[nextStep - 1]
              setGates(prev => prev.map((g, i) => ({ 
                ...g, 
                active: i === nextStep - 1 
              })))
              
              setQubits(prev => applyGate(currentGate, prev))
            }
            
            return nextStep
          })
        }

        // 큐비트 위상 애니메이션
        setQubits(prev => prev.map(qubit => ({
          ...qubit,
          phase: qubit.phase + 0.02 * Math.abs(qubit.state[1])
        })))
      }
      
      drawCircuit()
      animationRef.current = requestAnimationFrame(animate)
    }

    animate()
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [qubits, gates, isRunning, currentStep, animationFrame])

  const drawCircuit = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // 배경
    ctx.fillStyle = '#0a0a23'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // 큐비트 라인 그리기
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
    ctx.lineWidth = 2
    
    qubits.forEach(qubit => {
      ctx.beginPath()
      ctx.moveTo(30, qubit.y)
      ctx.lineTo(canvas.width - 30, qubit.y)
      ctx.stroke()
    })

    // 게이트 그리기
    gates.forEach(gate => {
      const isActive = gate.active
      
      if (gate.type === 'CNOT' && gate.control !== undefined) {
        // CNOT 게이트
        const controlQubit = qubits[gate.control]
        const targetQubit = qubits[gate.qubit]
        
        // 제어선
        ctx.strokeStyle = isActive ? '#ff6b6b' : 'rgba(255, 255, 255, 0.6)'
        ctx.lineWidth = isActive ? 3 : 2
        ctx.beginPath()
        ctx.moveTo(gate.x, controlQubit.y)
        ctx.lineTo(gate.x, targetQubit.y)
        ctx.stroke()
        
        // 제어 점
        ctx.beginPath()
        ctx.arc(gate.x, controlQubit.y, 5, 0, 2 * Math.PI)
        ctx.fillStyle = isActive ? '#ff6b6b' : '#ffffff'
        ctx.fill()
        
        // 타겟 원
        ctx.beginPath()
        ctx.arc(gate.x, targetQubit.y, 12, 0, 2 * Math.PI)
        ctx.strokeStyle = isActive ? '#ff6b6b' : '#ffffff'
        ctx.lineWidth = 2
        ctx.stroke()
        
        // 타겟 X
        ctx.strokeStyle = isActive ? '#ff6b6b' : '#ffffff'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(gate.x - 6, targetQubit.y - 6)
        ctx.lineTo(gate.x + 6, targetQubit.y + 6)
        ctx.moveTo(gate.x + 6, targetQubit.y - 6)
        ctx.lineTo(gate.x - 6, targetQubit.y + 6)
        ctx.stroke()
        
      } else {
        // 단일 큐비트 게이트
        const qubit = qubits[gate.qubit]
        
        // 게이트 박스
        ctx.beginPath()
        ctx.rect(gate.x - 15, qubit.y - 15, 30, 30)
        ctx.strokeStyle = isActive ? '#4ecdc4' : 'rgba(255, 255, 255, 0.8)'
        ctx.fillStyle = isActive ? 'rgba(78, 205, 196, 0.3)' : 'rgba(255, 255, 255, 0.1)'
        ctx.lineWidth = isActive ? 3 : 2
        ctx.fill()
        ctx.stroke()
        
        // 게이트 레이블
        ctx.fillStyle = isActive ? '#4ecdc4' : '#ffffff'
        ctx.font = 'bold 14px monospace'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(gate.type, gate.x, qubit.y)
        
        // 활성화 글로우
        if (isActive) {
          ctx.beginPath()
          ctx.rect(gate.x - 20, qubit.y - 20, 40, 40)
          ctx.strokeStyle = 'rgba(78, 205, 196, 0.5)'
          ctx.lineWidth = 6
          ctx.stroke()
        }
      }
    })

    // 큐비트 상태 시각화
    qubits.forEach((qubit, index) => {
      const prob0 = Math.abs(qubit.state[0]) ** 2
      const prob1 = Math.abs(qubit.state[1]) ** 2
      
      // 블로흐 구 표현 (단순화)
      const radius = 15
      const centerX = qubit.x
      const centerY = qubit.y
      
      // |0⟩ 상태 (위쪽)
      ctx.beginPath()
      ctx.arc(centerX, centerY - 5, radius * Math.sqrt(prob0), 0, 2 * Math.PI)
      ctx.fillStyle = `rgba(66, 165, 245, ${prob0})`
      ctx.fill()
      
      // |1⟩ 상태 (아래쪽)
      ctx.beginPath()
      ctx.arc(centerX, centerY + 5, radius * Math.sqrt(prob1), 0, 2 * Math.PI)
      ctx.fillStyle = `rgba(239, 83, 80, ${prob1})`
      ctx.fill()
      
      // 큐비트 레이블
      ctx.fillStyle = '#ffffff'
      ctx.font = '12px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(`q${index}`, centerX, centerY + 35)
      
      // 확률 텍스트
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
      ctx.font = '10px monospace'
      ctx.fillText(`|0⟩: ${prob0.toFixed(2)}`, centerX, centerY + 50)
      ctx.fillText(`|1⟩: ${prob1.toFixed(2)}`, centerX, centerY + 65)
    })

    // 제목 및 상태
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
    ctx.font = '12px monospace'
    ctx.textAlign = 'left'
    ctx.fillText('Quantum Circuit Builder', 10, 20)
    
    if (isRunning) {
      ctx.fillStyle = '#4ecdc4'
      ctx.fillText(`Step: ${currentStep}/${gates.length}`, 10, canvas.height - 15)
    }
  }

  const startCircuit = () => {
    setIsRunning(true)
    setCurrentStep(0)
  }

  const stopCircuit = () => {
    setIsRunning(false)
    setCurrentStep(0)
    
    // 상태 리셋
    setQubits(prev => prev.map(q => ({ 
      ...q, 
      state: [1, 0], 
      phase: 0 
    })))
    setGates(prev => prev.map(g => ({ ...g, active: false })))
  }

  return (
    <div className="relative bg-indigo-900 rounded-lg p-4 border border-indigo-700">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold text-sm">Quantum Circuit Builder</h3>
        <div className="flex gap-2">
          {!isRunning ? (
            <button
              onClick={startCircuit}
              className="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white text-xs rounded transition-colors"
            >
              Run
            </button>
          ) : (
            <button
              onClick={stopCircuit}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors"
            >
              Stop
            </button>
          )}
        </div>
      </div>
      
      <canvas
        ref={canvasRef}
        className="w-full border border-indigo-600 rounded"
        style={{ maxHeight: '250px' }}
      />
      
      {isRunning && (
        <div className="absolute top-2 right-2 bg-purple-600 text-white px-2 py-1 rounded text-xs animate-pulse">
          Computing...
        </div>
      )}
      
      <div className="mt-2 text-xs text-indigo-300">
        Build and simulate quantum algorithms
      </div>
    </div>
  )
}