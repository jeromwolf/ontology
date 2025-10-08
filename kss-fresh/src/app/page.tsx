'use client'

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { NotificationBell } from '@/components/common/NotificationBell';

export default function KSSLandingPage() {
  const [activeDemo, setActiveDemo] = useState(0);
  const [language, setLanguage] = useState('ko');

  const demosKo = [
    {
      title: "3D 지식 그래프",
      description: "복잡한 AI 개념의 인터랙티브 시각화",
      tech: "Three.js + WebGL"
    },
    {
      title: "신경망 시뮬레이터", 
      description: "실시간 토큰화와 어텐션 메커니즘",
      tech: "React + D3.js"
    },
    {
      title: "쿼리 처리 엔진",
      description: "SPARQL 실행과 실시간 결과 시각화",
      tech: "GraphQL + Canvas"
    },
    {
      title: "양자 회로 시뮬레이터",
      description: "양자 컴퓨팅 게이트와 얽힘 시각화",
      tech: "WebGL + Quantum.js"
    },
    {
      title: "RAG 파이프라인 시각화",
      description: "문서 검색 및 생성 흐름 실시간 처리",
      tech: "Vector DB + LLM"
    },
    {
      title: "멀티 에이전트 협업",
      description: "복잡한 작업을 해결하는 AI 에이전트 협력",
      tech: "Agent Protocol + WebSocket"
    },
    {
      title: "주식 시장 분석",
      description: "AI 기반 실시간 주식 예측 및 기술적 분석",
      tech: "TensorFlow + FinanceAPI"
    },
    {
      title: "뉴스 온톨로지 네트워크",
      description: "실시간 뉴스 스트림에서 지식 그래프 추출",
      tech: "NLP + Neo4j"
    },
    {
      title: "블록체인 투자 분석",
      description: "DeFi 포트폴리오 최적화 및 온체인 분석",
      tech: "Web3.js + GraphQL"
    }
  ];

  const demosEn = [
    {
      title: "3D Knowledge Graph",
      description: "Interactive visualization of complex AI concepts",
      tech: "Three.js + WebGL"
    },
    {
      title: "Neural Network Simulator", 
      description: "Real-time tokenization and attention mechanisms",
      tech: "React + D3.js"
    },
    {
      title: "Query Processing Engine",
      description: "Visual SPARQL execution with live results",
      tech: "GraphQL + Canvas"
    },
    {
      title: "Quantum Circuit Simulator",
      description: "Quantum computing gates and entanglement visualization",
      tech: "WebGL + Quantum.js"
    },
    {
      title: "RAG Pipeline Visualizer",
      description: "Document retrieval and generation flow in real-time",
      tech: "Vector DB + LLM"
    },
    {
      title: "Multi-Agent Collaboration",
      description: "AI agents working together to solve complex tasks",
      tech: "Agent Protocol + WebSocket"
    },
    {
      title: "Stock Market Analysis",
      description: "Real-time AI-powered stock prediction and technical analysis",
      tech: "TensorFlow + FinanceAPI"
    },
    {
      title: "News Ontology Network",
      description: "Knowledge graph extraction from real-time news streams",
      tech: "NLP + Neo4j"
    },
    {
      title: "Blockchain Investment Analytics",
      description: "DeFi portfolio optimization and on-chain analysis",
      tech: "Web3.js + GraphQL"
    }
  ];

  const demos = language === 'ko' ? demosKo : demosEn;

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveDemo((prev) => (prev + 1) % demos.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-white text-gray-900">
      <style jsx>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        * {
          font-family: 'Inter', 'Noto Sans KR', sans-serif;
        }
        
        .gradient-text {
          background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        
        .glass-card {
          background: rgba(255, 255, 255, 0.8);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(0, 0, 0, 0.1);
        }
        
        .minimal-button {
          background: #000000;
          color: white;
          border: none;
          transition: all 0.2s ease;
        }
        
        .minimal-button:hover {
          background: #1f2937;
          transform: translateY(-1px);
        }
        
        .demo-indicator {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #e5e7eb;
          transition: all 0.3s ease;
        }
        
        .demo-indicator.active {
          background: #000000;
        }
        
        @keyframes bounce {
          0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
          }
          40% {
            transform: translateY(-10px);
          }
          60% {
            transform: translateY(-5px);
          }
        }
        
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes gradient-x {
          0%, 100% {
            transform: translateX(0%);
          }
          50% {
            transform: translateX(-100%);
          }
        }
        
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        @keyframes fadeOut {
          from {
            opacity: 1;
          }
          to {
            opacity: 0;
          }
        }
        
        @keyframes pulse {
          0%, 100% {
            opacity: 0.8;
            transform: scale(1);
          }
          50% {
            opacity: 1;
            transform: scale(1.1);
          }
        }
      `}</style>

      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-black rounded-lg flex items-center justify-center">
                <span className="text-white text-sm font-bold">K</span>
              </div>
              <span className="text-xl font-semibold text-gray-900">KSS</span>
            </div>
            <nav className="hidden md:flex items-center space-x-8">
              <a href="#" className="text-gray-600 hover:text-gray-900 text-sm font-medium">{language === 'ko' ? '플랫폼' : 'Platform'}</a>
              <a href="#" className="text-gray-600 hover:text-gray-900 text-sm font-medium">{language === 'ko' ? '모듈' : 'Modules'}</a>
              <a href="#" className="text-gray-600 hover:text-gray-900 text-sm font-medium">{language === 'ko' ? '문서' : 'Documentation'}</a>
              <a href="#" className="text-gray-600 hover:text-gray-900 text-sm font-medium">{language === 'ko' ? '기업' : 'Enterprise'}</a>
            </nav>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <button 
                  className={`text-sm font-medium transition-colors ${language === 'ko' ? 'text-gray-900' : 'text-gray-600 hover:text-gray-900'}`}
                  onClick={() => setLanguage('ko')}
                >
                  KO
                </button>
                <span className="text-sm text-gray-400">|</span>
                <button 
                  className={`text-sm font-medium transition-colors ${language === 'en' ? 'text-gray-900' : 'text-gray-600 hover:text-gray-900'}`}
                  onClick={() => setLanguage('en')}
                >
                  EN
                </button>
              </div>
              
              {/* Notification Bell */}
              <NotificationBell className="ml-2" />
              
              <Link href="/auth/signin">
                <button className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors">
                  {language === 'ko' ? '로그인' : 'Login'}
                </button>
              </Link>
              <Link href="/auth/signup">
                <button className="minimal-button px-4 py-2 rounded-lg text-sm font-medium">
                  {language === 'ko' ? '회원가입' : 'Sign Up'}
                </button>
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-6xl md:text-8xl font-bold mb-8 gradient-text leading-tight">
            {language === 'ko' ? (
              <>
                지식의
                <br />
                우주
                <br />
                시뮬레이터
              </>
            ) : (
              <>
                Knowledge
                <br />
                Space
                <br />
                Simulator
              </>
            )}
          </h1>
          
          <p className="text-xl md:text-2xl text-gray-600 mb-20 max-w-3xl mx-auto font-light leading-relaxed">
            {language === 'ko' 
              ? '인터랙티브 AI 교육을 위한 전문 플랫폼. 실시간 3D 시각화와 직접 체험을 통해 복잡한 개념을 경험해보세요.'
              : 'The professional platform for interactive AI education. Experience complex concepts through real-time 3D visualization and hands-on simulation.'
            }
          </p>

          {/* Live Demo Section */}
          <div className="glass-card rounded-3xl p-12 mb-20 relative overflow-hidden">
            {/* Background gradient animation */}
            <div className="absolute inset-0 opacity-10">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 animate-gradient-x"></div>
            </div>
            
            <div className="relative z-10">
              <div className="flex justify-center space-x-2 mb-8">
                {demos.map((_, index) => (
                  <div
                    key={index}
                    className={`transition-all duration-300 ${
                      index === activeDemo 
                        ? 'w-16 h-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full' 
                        : 'w-2 h-2 bg-gray-400 rounded-full'
                    }`}
                  />
                ))}
              </div>
              
              <div className="max-w-5xl mx-auto">
                <h3 className="text-4xl font-bold mb-4 text-gray-900">
                  {demos[activeDemo].title}
                </h3>
                <p className="text-xl text-gray-600 mb-6">
                  {demos[activeDemo].description}
                </p>
                <div className="flex items-center gap-4 mb-8">
                  <div className="px-3 py-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-sm rounded-full font-medium">
                    LIVE DEMO
                  </div>
                  <div className="text-sm text-gray-500">
                    Powered by {demos[activeDemo].tech}
                  </div>
                </div>
                
                {/* Interactive Demo Window */}
                <div className="bg-gradient-to-br from-gray-900 to-black rounded-2xl p-1 mb-8 shadow-2xl">
                  <div className="bg-black rounded-xl p-8 relative">
                    {/* Demo 1: Knowledge Graph */}
                    {activeDemo === 0 && (
                      <div className="aspect-video flex items-center justify-center relative">
                        <svg viewBox="0 0 800 400" className="w-full h-full">
                          <defs>
                            <radialGradient id="nodeGradient1">
                              <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.8" />
                              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.3" />
                            </radialGradient>
                            <radialGradient id="nodeGradient2">
                              <stop offset="0%" stopColor="#34d399" stopOpacity="0.8" />
                              <stop offset="100%" stopColor="#10b981" stopOpacity="0.3" />
                            </radialGradient>
                            <radialGradient id="nodeGradient3">
                              <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.8" />
                              <stop offset="100%" stopColor="#d97706" stopOpacity="0.3" />
                            </radialGradient>
                            <filter id="glow">
                              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                              <feMerge>
                                <feMergeNode in="coloredBlur"/>
                                <feMergeNode in="SourceGraphic"/>
                              </feMerge>
                            </filter>
                          </defs>
                          
                          {/* Animated background particles */}
                          {[...Array(20)].map((_, i) => (
                            <circle key={`particle-${i}`} r="1" fill="#3b82f6" opacity="0.3">
                              <animateMotion
                                path={`M${Math.random() * 800},${Math.random() * 400} Q${Math.random() * 800},${Math.random() * 400} ${Math.random() * 800},${Math.random() * 400}`}
                                dur={`${15 + Math.random() * 10}s`}
                                repeatCount="indefinite"
                              />
                              <animate attributeName="opacity" values="0;0.5;0" dur="3s" repeatCount="indefinite" />
                            </circle>
                          ))}
                          
                          {/* Central hub */}
                          <circle cx="400" cy="200" r="40" fill="url(#nodeGradient1)" filter="url(#glow)">
                            <animate attributeName="r" values="35;45;35" dur="3s" repeatCount="indefinite" />
                          </circle>
                          
                          {/* Satellite nodes */}
                          {[
                            { cx: 200, cy: 100, r: 25, gradient: 'nodeGradient2', label: 'LLM', delay: 0 },
                            { cx: 600, cy: 100, r: 25, gradient: 'nodeGradient2', label: 'RAG', delay: 0.5 },
                            { cx: 200, cy: 300, r: 25, gradient: 'nodeGradient3', label: 'Vision', delay: 1 },
                            { cx: 600, cy: 300, r: 25, gradient: 'nodeGradient3', label: 'Audio', delay: 1.5 },
                            { cx: 300, cy: 50, r: 20, gradient: 'nodeGradient1', label: 'NLP', delay: 0.3 },
                            { cx: 500, cy: 50, r: 20, gradient: 'nodeGradient1', label: 'ML', delay: 0.7 },
                            { cx: 300, cy: 350, r: 20, gradient: 'nodeGradient2', label: 'RL', delay: 1.2 },
                            { cx: 500, cy: 350, r: 20, gradient: 'nodeGradient2', label: 'DL', delay: 1.8 }
                          ].map((node, i) => (
                            <g key={`node-${i}`}>
                              <circle cx={node.cx} cy={node.cy} r={node.r} fill={`url(#${node.gradient})`} filter="url(#glow)">
                                <animate attributeName="r" values={`${node.r-5};${node.r+5};${node.r-5}`} dur="4s" begin={`${node.delay}s`} repeatCount="indefinite" />
                              </circle>
                              <text x={node.cx} y={node.cy} textAnchor="middle" dy="0.3em" fill="white" fontSize="12" fontWeight="600">
                                {node.label}
                              </text>
                            </g>
                          ))}
                          
                          {/* Animated connections */}
                          {[
                            { x1: 400, y1: 200, x2: 200, y2: 100 },
                            { x1: 400, y1: 200, x2: 600, y2: 100 },
                            { x1: 400, y1: 200, x2: 200, y2: 300 },
                            { x1: 400, y1: 200, x2: 600, y2: 300 },
                            { x1: 200, y1: 100, x2: 300, y2: 50 },
                            { x1: 600, y1: 100, x2: 500, y2: 50 },
                            { x1: 200, y1: 300, x2: 300, y2: 350 },
                            { x1: 600, y1: 300, x2: 500, y2: 350 }
                          ].map((line, i) => (
                            <g key={`line-${i}`}>
                              <line {...line} stroke="url(#nodeGradient1)" strokeWidth="2" opacity="0.3" />
                              <circle r="3" fill="#60a5fa">
                                <animateMotion
                                  path={`M${line.x1},${line.y1} L${line.x2},${line.y2}`}
                                  dur="2s"
                                  begin={`${i * 0.3}s`}
                                  repeatCount="indefinite"
                                />
                                <animate attributeName="opacity" values="0;1;0" dur="2s" begin={`${i * 0.3}s`} repeatCount="indefinite" />
                              </circle>
                            </g>
                          ))}
                          
                          {/* Central label */}
                          <text x="400" y="200" textAnchor="middle" dy="0.3em" fill="white" fontSize="16" fontWeight="bold">
                            AI Core
                          </text>
                        </svg>
                        
                        <div className="absolute top-4 right-4 flex items-center gap-2">
                          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                          <span className="text-green-400 text-sm font-mono">Real-time Graph Engine</span>
                        </div>
                        
                        <div className="absolute bottom-4 left-4 text-green-400 text-sm font-mono">
                          <div className="mb-1">$ 8 nodes connected</div>
                          <div>$ 12 relationships active</div>
                        </div>
                      </div>
                    )}

                    {/* Demo 2: Neural Network */}
                    {activeDemo === 1 && (
                      <div className="aspect-video flex items-center justify-center relative">
                        <div className="w-full h-full flex items-center justify-center">
                          <svg viewBox="0 0 800 400" className="w-full h-full">
                            <defs>
                              <linearGradient id="neuronGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.8" />
                                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.8" />
                              </linearGradient>
                              <filter id="neuronGlow">
                                <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                                <feMerge>
                                  <feMergeNode in="coloredBlur"/>
                                  <feMergeNode in="SourceGraphic"/>
                                </feMerge>
                              </filter>
                            </defs>
                            
                            {/* Input Layer */}
                            {[0, 1, 2, 3, 4].map((i) => (
                              <g key={`input-${i}`}>
                                <circle cx="150" cy={80 + i * 60} r="20" fill="url(#neuronGradient)" filter="url(#neuronGlow)">
                                  <animate attributeName="opacity" values="0.4;1;0.4" dur="3s" begin={`${i * 0.2}s`} repeatCount="indefinite" />
                                </circle>
                                <text x="150" y={85 + i * 60} textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">
                                  I{i+1}
                                </text>
                              </g>
                            ))}
                            
                            {/* Hidden Layers */}
                            {[0, 1].map((layer) => (
                              <g key={`hidden-layer-${layer}`}>
                                {[0, 1, 2, 3, 4, 5].map((i) => (
                                  <g key={`hidden-${layer}-${i}`}>
                                    <circle cx={300 + layer * 150} cy={50 + i * 55} r="18" fill="url(#neuronGradient)" filter="url(#neuronGlow)">
                                      <animate attributeName="r" values="15;22;15" dur="2s" begin={`${layer * 0.5 + i * 0.1}s`} repeatCount="indefinite" />
                                    </circle>
                                    <text x={300 + layer * 150} y={55 + i * 55} textAnchor="middle" fill="white" fontSize="9" fontWeight="bold">
                                      H{layer+1}.{i+1}
                                    </text>
                                  </g>
                                ))}
                              </g>
                            ))}
                            
                            {/* Output Layer */}
                            {[0, 1, 2].map((i) => (
                              <g key={`output-${i}`}>
                                <circle cx="650" cy={120 + i * 80} r="25" fill="url(#neuronGradient)" filter="url(#neuronGlow)">
                                  <animate attributeName="opacity" values="0.5;1;0.5" dur="2.5s" begin={`${2 + i * 0.3}s`} repeatCount="indefinite" />
                                </circle>
                                <text x="650" y={125 + i * 80} textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">
                                  O{i+1}
                                </text>
                              </g>
                            ))}
                            
                            {/* Synaptic connections with signal flow */}
                            {/* Input to Hidden 1 */}
                            {[0, 1, 2, 3, 4].map((i) => 
                              [0, 1, 2, 3, 4, 5].map((j) => (
                                <g key={`synapse-ih1-${i}-${j}`}>
                                  <line 
                                    x1="170" y1={80 + i * 60} 
                                    x2="280" y2={50 + j * 55} 
                                    stroke="#3b82f6" 
                                    strokeWidth="0.5" 
                                    opacity="0.2"
                                  />
                                  <circle r="2" fill="#60a5fa">
                                    <animateMotion
                                      path={`M170,${80 + i * 60} L280,${50 + j * 55}`}
                                      dur="1.5s"
                                      begin={`${(i + j) * 0.1}s`}
                                      repeatCount="indefinite"
                                    />
                                    <animate attributeName="opacity" values="0;1;0" dur="1.5s" begin={`${(i + j) * 0.1}s`} repeatCount="indefinite" />
                                  </circle>
                                </g>
                              ))
                            )}
                            
                            {/* Signal strength indicators */}
                            <text x="400" y="30" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                              Deep Neural Network Architecture
                            </text>
                            <text x="400" y="380" textAnchor="middle" fill="#94a3b8" fontSize="12">
                              5 inputs → 12 hidden neurons → 3 outputs
                            </text>
                          </svg>
                        </div>
                        
                        <div className="absolute top-4 right-4 flex items-center gap-2">
                          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                          <span className="text-green-400 text-sm font-mono">Training Active</span>
                        </div>
                        
                        <div className="absolute bottom-4 left-4 text-green-400 text-sm font-mono">
                          <div className="mb-1">$ Epoch: 1,247</div>
                          <div>$ Accuracy: 98.7%</div>
                        </div>
                      </div>
                    )}

                    {/* Demo 3: Real-time Data Processing */}
                    {activeDemo === 2 && (
                      <div className="aspect-video flex items-center justify-center relative">
                        <div className="w-full h-full p-8">
                          {/* Multi-stage pipeline visualization */}
                          <div className="flex items-center justify-between h-full">
                            {/* Input Stream */}
                            <div className="flex-1">
                              <div className="bg-gray-800 rounded-lg p-4 border border-blue-500 shadow-lg shadow-blue-500/20">
                                <div className="text-blue-400 text-sm font-mono mb-3">Input Stream</div>
                                <div className="space-y-2">
                                  {['Data packet 1', 'Data packet 2', 'Data packet 3'].map((packet, i) => (
                                    <div 
                                      key={i}
                                      className="bg-gray-900 rounded px-3 py-2 text-xs text-gray-300 font-mono opacity-0"
                                      style={{
                                        animation: `slideIn 0.5s ease-out ${i * 0.3}s forwards, fadeOut 2s ease-out ${i * 0.3 + 2}s forwards`
                                      }}
                                    >
                                      {packet}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </div>

                            {/* Processing Pipeline */}
                            <div className="flex items-center mx-8">
                              <div className="relative">
                                <svg width="200" height="200" className="transform rotate-90">
                                  <circle cx="100" cy="100" r="80" fill="none" stroke="#1f2937" strokeWidth="8" />
                                  <circle 
                                    cx="100" cy="100" r="80" 
                                    fill="none" 
                                    stroke="url(#pipelineGradient)" 
                                    strokeWidth="8"
                                    strokeDasharray="502"
                                    strokeDashoffset="502"
                                    transform="rotate(-90 100 100)"
                                  >
                                    <animate attributeName="stroke-dashoffset" values="502;0" dur="3s" repeatCount="indefinite" />
                                  </circle>
                                  <defs>
                                    <linearGradient id="pipelineGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                      <stop offset="0%" stopColor="#3b82f6" />
                                      <stop offset="50%" stopColor="#8b5cf6" />
                                      <stop offset="100%" stopColor="#ec4899" />
                                    </linearGradient>
                                  </defs>
                                </svg>
                                <div className="absolute inset-0 flex items-center justify-center">
                                  <div className="text-center">
                                    <div className="text-2xl font-bold text-white mb-1">AI</div>
                                    <div className="text-xs text-gray-400">Processing</div>
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* Output Results */}
                            <div className="flex-1">
                              <div className="bg-gray-800 rounded-lg p-4 border border-green-500 shadow-lg shadow-green-500/20">
                                <div className="text-green-400 text-sm font-mono mb-3">Analysis Results</div>
                                <div className="space-y-2">
                                  {[
                                    { label: 'Sentiment', value: 'Positive', color: 'text-green-400' },
                                    { label: 'Confidence', value: '94.2%', color: 'text-blue-400' },
                                    { label: 'Category', value: 'Technical', color: 'text-purple-400' }
                                  ].map((result, i) => (
                                    <div 
                                      key={i}
                                      className="flex justify-between items-center opacity-0"
                                      style={{
                                        animation: `fadeIn 0.5s ease-out ${2.5 + i * 0.2}s forwards`
                                      }}
                                    >
                                      <span className="text-xs text-gray-400">{result.label}:</span>
                                      <span className={`text-xs font-mono font-bold ${result.color}`}>{result.value}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* Performance metrics */}
                          <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end">
                            <div className="text-green-400 text-sm font-mono">
                              <div className="mb-1">$ Throughput: 10K msg/s</div>
                              <div>$ Latency: &lt;50ms</div>
                            </div>
                            <div className="flex gap-4">
                              {['CPU: 45%', 'RAM: 2.1GB', 'GPU: 78%'].map((metric, i) => (
                                <div key={i} className="text-xs text-gray-400 font-mono">
                                  {metric}
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Demo 4: Quantum Circuit Simulator */}
                    {activeDemo === 3 && (
                      <div className="aspect-video flex items-center justify-center relative">
                        <svg viewBox="0 0 800 400" className="w-full h-full">
                          <defs>
                            <linearGradient id="quantumGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                              <stop offset="0%" stopColor="#06b6d4" />
                              <stop offset="50%" stopColor="#8b5cf6" />
                              <stop offset="100%" stopColor="#ec4899" />
                            </linearGradient>
                            <filter id="quantumGlow">
                              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                              <feMerge>
                                <feMergeNode in="coloredBlur"/>
                                <feMergeNode in="SourceGraphic"/>
                              </feMerge>
                            </filter>
                          </defs>
                          
                          {/* Quantum circuit lines */}
                          {[0, 1, 2].map((i) => (
                            <g key={`qubit-${i}`}>
                              <line x1="50" y1={100 + i * 100} x2="750" y2={100 + i * 100} stroke="#374151" strokeWidth="2" />
                              <text x="20" y={105 + i * 100} fill="white" fontSize="14" fontWeight="bold">|q{i}⟩</text>
                            </g>
                          ))}
                          
                          {/* Quantum gates */}
                          {/* Hadamard gates */}
                          {[0, 1].map((i) => (
                            <g key={`h-gate-${i}`}>
                              <rect x={150 + i * 200} y={80 + i * 100} width="40" height="40" 
                                    fill="url(#quantumGradient)" filter="url(#quantumGlow)" rx="5">
                                <animate attributeName="opacity" values="0.5;1;0.5" dur="2s" begin={`${i * 0.5}s`} repeatCount="indefinite" />
                              </rect>
                              <text x={170 + i * 200} y={105 + i * 100} textAnchor="middle" fill="white" fontSize="20" fontWeight="bold">H</text>
                            </g>
                          ))}
                          
                          {/* CNOT gates */}
                          <g>
                            <line x1="350" y1="100" x2="350" y2="200" stroke="#8b5cf6" strokeWidth="2" />
                            <circle cx="350" cy="100" r="8" fill="#8b5cf6" />
                            <circle cx="350" cy="200" r="20" fill="none" stroke="#8b5cf6" strokeWidth="3" />
                            <line x1="330" y1="200" x2="370" y2="200" stroke="#8b5cf6" strokeWidth="3" />
                            <line x1="350" y1="180" x2="350" y2="220" stroke="#8b5cf6" strokeWidth="3" />
                          </g>
                          
                          {/* Measurement */}
                          {[0, 1, 2].map((i) => (
                            <g key={`measure-${i}`}>
                              <rect x="650" y={80 + i * 100} width="40" height="40" 
                                    fill="#10b981" filter="url(#quantumGlow)" rx="5">
                                <animate attributeName="opacity" values="0.3;1;0.3" dur="1.5s" begin={`${2 + i * 0.3}s`} repeatCount="indefinite" />
                              </rect>
                              <path d={`M 670 ${95 + i * 100} Q 670 ${110 + i * 100} 685 ${110 + i * 100}`} 
                                    fill="none" stroke="white" strokeWidth="2" />
                              <circle cx="670" cy={95 + i * 100} r="2" fill="white" />
                            </g>
                          ))}
                          
                          {/* Quantum state visualization */}
                          <g transform="translate(100, 350)">
                            <text x="0" y="0" fill="#94a3b8" fontSize="12">Quantum State: </text>
                            <text x="100" y="0" fill="#06b6d4" fontSize="12" fontFamily="monospace">
                              |ψ⟩ = α|000⟩ + β|111⟩
                            </text>
                          </g>
                          
                          {/* Entanglement visualization */}
                          {[0, 1, 2].map((i) => (
                            <circle key={`entangle-${i}`} r="3" fill="#ec4899">
                              <animateMotion
                                path={`M ${150 + i * 100},${100 + i * 100} Q 400,200 ${650},${100 + ((i + 1) % 3) * 100}`}
                                dur="3s"
                                begin={`${i * 1}s`}
                                repeatCount="indefinite"
                              />
                              <animate attributeName="opacity" values="0;1;0" dur="3s" begin={`${i * 1}s`} repeatCount="indefinite" />
                            </circle>
                          ))}
                        </svg>
                        
                        <div className="absolute top-4 right-4 flex items-center gap-2">
                          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                          <span className="text-cyan-400 text-sm font-mono">Quantum Entanglement Active</span>
                        </div>
                        
                        <div className="absolute bottom-4 left-4 text-cyan-400 text-sm font-mono">
                          <div className="mb-1">$ 3 qubits initialized</div>
                          <div>$ Bell state: 99.8% fidelity</div>
                        </div>
                      </div>
                    )}

                    {/* Demo 5: RAG Pipeline Visualizer */}
                    {activeDemo === 4 && (
                      <div className="aspect-video flex items-center justify-center relative p-8">
                        <div className="w-full h-full flex flex-col justify-center">
                          {/* Document chunks */}
                          <div className="flex justify-around mb-8">
                            {[0, 1, 2, 3, 4].map((i) => (
                              <div key={`doc-${i}`} 
                                   className="w-16 h-20 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg shadow-lg opacity-0"
                                   style={{
                                     animation: `fadeIn 0.5s ease-out ${i * 0.2}s forwards`
                                   }}>
                                <div className="w-full h-full flex items-center justify-center text-white font-bold">
                                  D{i + 1}
                                </div>
                              </div>
                            ))}
                          </div>
                          
                          {/* Vector embeddings animation */}
                          <div className="relative h-32 mb-8">
                            <svg className="absolute inset-0 w-full h-full">
                              {[0, 1, 2, 3, 4].map((i) => (
                                <g key={`vector-${i}`}>
                                  <line x1={`${20 + i * 20}%`} y1="0" x2={`${20 + i * 20}%`} y2="100%" 
                                        stroke="#8b5cf6" strokeWidth="1" opacity="0.3" />
                                  <circle r="3" fill="#8b5cf6">
                                    <animateMotion
                                      path={`M ${(20 + i * 20) * 8},0 L ${(20 + i * 20) * 8},128`}
                                      dur="2s"
                                      begin={`${0.5 + i * 0.2}s`}
                                      repeatCount="indefinite"
                                    />
                                  </circle>
                                </g>
                              ))}
                              <text x="50%" y="50%" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                                Vector Database
                              </text>
                            </svg>
                          </div>
                          
                          {/* Retrieved context + LLM */}
                          <div className="flex items-center justify-center gap-8">
                            <div className="bg-gray-800 rounded-lg p-4 border border-green-500 shadow-lg shadow-green-500/20 opacity-0"
                                 style={{ animation: 'fadeIn 0.5s ease-out 2s forwards' }}>
                              <div className="text-green-400 text-sm font-mono mb-2">Retrieved Context</div>
                              <div className="text-xs text-gray-300 space-y-1">
                                <div>✓ Document 2 (0.95 similarity)</div>
                                <div>✓ Document 4 (0.87 similarity)</div>
                                <div>✓ Document 1 (0.82 similarity)</div>
                              </div>
                            </div>
                            
                            <div className="text-3xl text-white">→</div>
                            
                            <div className="relative">
                              <div className="w-32 h-32 bg-gradient-to-br from-purple-600 to-pink-600 rounded-full flex items-center justify-center shadow-lg shadow-purple-500/30">
                                <span className="text-white font-bold text-xl">LLM</span>
                              </div>
                              <div className="absolute inset-0 rounded-full border-4 border-purple-400 opacity-0"
                                   style={{ animation: 'pulse 2s ease-out 2.5s infinite' }}></div>
                            </div>
                            
                            <div className="text-3xl text-white">→</div>
                            
                            <div className="bg-gray-800 rounded-lg p-4 border border-blue-500 shadow-lg shadow-blue-500/20 opacity-0"
                                 style={{ animation: 'fadeIn 0.5s ease-out 3s forwards' }}>
                              <div className="text-blue-400 text-sm font-mono mb-2">Generated Answer</div>
                              <div className="text-xs text-gray-300">
                                <div className="typewriter">Based on the retrieved context...</div>
                              </div>
                            </div>
                          </div>
                        </div>
                        
                        <div className="absolute top-4 right-4 flex items-center gap-2">
                          <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
                          <span className="text-purple-400 text-sm font-mono">RAG Pipeline Active</span>
                        </div>
                        
                        <div className="absolute bottom-4 left-4 text-purple-400 text-sm font-mono">
                          <div className="mb-1">$ 5 documents indexed</div>
                          <div>$ Retrieval latency: 23ms</div>
                        </div>
                      </div>
                    )}

                    {/* Demo 6: Multi-Agent Collaboration */}
                    {activeDemo === 5 && (
                      <div className="aspect-video flex items-center justify-center relative">
                        <svg viewBox="0 0 800 400" className="w-full h-full">
                          <defs>
                            <radialGradient id="agentGradient1">
                              <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.8" />
                              <stop offset="100%" stopColor="#d97706" stopOpacity="0.3" />
                            </radialGradient>
                            <radialGradient id="agentGradient2">
                              <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
                              <stop offset="100%" stopColor="#059669" stopOpacity="0.3" />
                            </radialGradient>
                            <radialGradient id="agentGradient3">
                              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.8" />
                              <stop offset="100%" stopColor="#2563eb" stopOpacity="0.3" />
                            </radialGradient>
                          </defs>
                          
                          {/* Central task */}
                          <rect x="300" y="150" width="200" height="100" rx="10" 
                                fill="#1f2937" stroke="#374151" strokeWidth="2" />
                          <text x="400" y="190" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                            Complex Task
                          </text>
                          <text x="400" y="210" textAnchor="middle" fill="#94a3b8" fontSize="12">
                            "Build AI System"
                          </text>
                          
                          {/* Agents */}
                          {[
                            { cx: 150, cy: 100, r: 40, gradient: 'agentGradient1', label: 'Planner', role: 'Strategy' },
                            { cx: 650, cy: 100, r: 40, gradient: 'agentGradient2', label: 'Coder', role: 'Implementation' },
                            { cx: 150, cy: 300, r: 40, gradient: 'agentGradient3', label: 'Tester', role: 'Validation' },
                            { cx: 650, cy: 300, r: 40, gradient: 'agentGradient1', label: 'Reviewer', role: 'Quality' }
                          ].map((agent, i) => (
                            <g key={`agent-${i}`}>
                              <circle cx={agent.cx} cy={agent.cy} r={agent.r} 
                                      fill={`url(#${agent.gradient})`} filter="url(#glow)">
                                <animate attributeName="r" values={`${agent.r-5};${agent.r+5};${agent.r-5}`} 
                                         dur="3s" begin={`${i * 0.5}s`} repeatCount="indefinite" />
                              </circle>
                              <text x={agent.cx} y={agent.cy - 5} textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">
                                {agent.label}
                              </text>
                              <text x={agent.cx} y={agent.cy + 10} textAnchor="middle" fill="white" fontSize="10">
                                {agent.role}
                              </text>
                            </g>
                          ))}
                          
                          {/* Communication lines */}
                          {[
                            { x1: 190, y1: 100, x2: 300, y2: 180 },
                            { x1: 610, y1: 100, x2: 500, y2: 180 },
                            { x1: 190, y1: 300, x2: 300, y2: 220 },
                            { x1: 610, y1: 300, x2: 500, y2: 220 },
                            { x1: 150, y1: 140, x2: 150, y2: 260 },
                            { x1: 650, y1: 140, x2: 650, y2: 260 }
                          ].map((line, i) => (
                            <g key={`comm-${i}`}>
                              <line {...line} stroke="#4b5563" strokeWidth="1" strokeDasharray="5,5" />
                              <circle r="4" fill="#60a5fa">
                                <animateMotion
                                  path={`M${line.x1},${line.y1} L${line.x2},${line.y2}`}
                                  dur="2s"
                                  begin={`${i * 0.3}s`}
                                  repeatCount="indefinite"
                                />
                              </circle>
                              <circle r="4" fill="#60a5fa">
                                <animateMotion
                                  path={`M${line.x2},${line.y2} L${line.x1},${line.y1}`}
                                  dur="2s"
                                  begin={`${i * 0.3 + 1}s`}
                                  repeatCount="indefinite"
                                />
                              </circle>
                            </g>
                          ))}
                          
                          {/* Progress indicators */}
                          <g transform="translate(400, 350)">
                            <rect x="-100" y="-10" width="200" height="20" rx="10" fill="#1f2937" stroke="#374151" />
                            <rect x="-98" y="-8" width="196" height="16" rx="8" fill="#059669" opacity="0.3" />
                            <rect x="-98" y="-8" width="0" height="16" rx="8" fill="#059669">
                              <animate attributeName="width" values="0;196" dur="5s" repeatCount="indefinite" />
                            </rect>
                            <text x="0" y="5" textAnchor="middle" fill="white" fontSize="10">Task Progress</text>
                          </g>
                        </svg>
                        
                        <div className="absolute top-4 right-4 flex items-center gap-2">
                          <div className="w-2 h-2 bg-amber-400 rounded-full animate-pulse"></div>
                          <span className="text-amber-400 text-sm font-mono">4 Agents Collaborating</span>
                        </div>
                        
                        <div className="absolute bottom-4 left-4 text-amber-400 text-sm font-mono">
                          <div className="mb-1">$ Messages exchanged: 127</div>
                          <div>$ Task completion: 78%</div>
                        </div>
                      </div>
                    )}

                    {/* Demo 7: Stock Market Analysis */}
                    {activeDemo === 6 && (
                      <div className="aspect-video flex items-center justify-center relative">
                        <div className="w-full h-full p-6">
                          <div className="flex gap-6 h-full">
                            {/* Candlestick Chart */}
                            <div className="flex-1">
                              <div className="bg-gray-900 rounded-lg p-4 h-full border border-gray-700">
                                <div className="text-green-400 text-sm font-mono mb-2">AAPL - Real-time Analysis</div>
                                <svg viewBox="0 0 400 250" className="w-full h-full">
                                  {/* Grid lines */}
                                  {[0, 1, 2, 3, 4].map(i => (
                                    <line key={`grid-${i}`} x1="40" y1={50 + i * 40} x2="380" y2={50 + i * 40} 
                                          stroke="#374151" strokeWidth="0.5" />
                                  ))}
                                  
                                  {/* Candlesticks */}
                                  {[...Array(20)].map((_, i) => {
                                    const x = 50 + i * 17;
                                    const height = 20 + Math.random() * 40;
                                    const y = 100 + (Math.random() - 0.5) * 60;
                                    const isGreen = Math.random() > 0.5;
                                    
                                    return (
                                      <g key={`candle-${i}`}>
                                        <line x1={x} y1={y - height/2 - 10} x2={x} y2={y + height/2 + 10} 
                                              stroke={isGreen ? '#10b981' : '#ef4444'} strokeWidth="1" />
                                        <rect x={x - 4} y={y - height/2} width="8" height={height} 
                                              fill={isGreen ? '#10b981' : '#ef4444'} opacity="0.8">
                                          <animate attributeName="opacity" values="0;0.8" dur="0.5s" begin={`${i * 0.1}s`} fill="freeze" />
                                        </rect>
                                      </g>
                                    );
                                  })}
                                  
                                  {/* Moving averages */}
                                  <polyline points="50,120 100,115 150,110 200,108 250,112 300,109 350,106" 
                                            fill="none" stroke="#3b82f6" strokeWidth="2" opacity="0.7" />
                                  <polyline points="50,130 100,127 150,125 200,123 250,122 300,120 350,118" 
                                            fill="none" stroke="#f59e0b" strokeWidth="2" opacity="0.7" />
                                  
                                  {/* Price labels */}
                                  <text x="20" y="55" fill="#94a3b8" fontSize="10">$180</text>
                                  <text x="20" y="135" fill="#94a3b8" fontSize="10">$170</text>
                                  <text x="20" y="215" fill="#94a3b8" fontSize="10">$160</text>
                                </svg>
                              </div>
                            </div>
                            
                            {/* AI Predictions & Indicators */}
                            <div className="w-80">
                              <div className="space-y-4">
                                {/* AI Prediction */}
                                <div className="bg-gray-800 rounded-lg p-4 border border-purple-500 shadow-lg shadow-purple-500/20">
                                  <div className="text-purple-400 text-sm font-mono mb-3">AI Prediction</div>
                                  <div className="flex items-center justify-between mb-2">
                                    <span className="text-white font-bold text-xl">$178.45</span>
                                    <span className="text-green-400 text-sm">+2.3%</span>
                                  </div>
                                  <div className="w-full bg-gray-700 rounded-full h-2">
                                    <div className="bg-gradient-to-r from-purple-600 to-pink-600 h-2 rounded-full" 
                                         style={{ width: '0%' }}>
                                      <animate attributeName="width" values="0%;78%" dur="2s" fill="freeze" />
                                    </div>
                                  </div>
                                  <div className="text-xs text-gray-400 mt-2">Confidence: 78%</div>
                                </div>
                                
                                {/* Technical Indicators */}
                                <div className="bg-gray-800 rounded-lg p-4 border border-blue-500 shadow-lg shadow-blue-500/20">
                                  <div className="text-blue-400 text-sm font-mono mb-3">Technical Indicators</div>
                                  <div className="space-y-2 text-sm">
                                    <div className="flex justify-between items-center">
                                      <span className="text-gray-400">RSI</span>
                                      <span className="text-green-400 font-mono">68.5</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                      <span className="text-gray-400">MACD</span>
                                      <span className="text-green-400 font-mono">+1.24</span>
                                    </div>
                                    <div className="flex justify-between items-center">
                                      <span className="text-gray-400">Volume</span>
                                      <span className="text-blue-400 font-mono">125M</span>
                                    </div>
                                  </div>
                                </div>
                                
                                {/* Sentiment Analysis */}
                                <div className="bg-gray-800 rounded-lg p-4 border border-green-500 shadow-lg shadow-green-500/20">
                                  <div className="text-green-400 text-sm font-mono mb-3">Market Sentiment</div>
                                  <div className="flex items-center gap-4">
                                    <div className="w-20 h-20 relative">
                                      <svg className="w-full h-full transform -rotate-90">
                                        <circle cx="40" cy="40" r="30" fill="none" stroke="#374151" strokeWidth="6" />
                                        <circle cx="40" cy="40" r="30" fill="none" stroke="#10b981" strokeWidth="6"
                                                strokeDasharray="188" strokeDashoffset="188">
                                          <animate attributeName="stroke-dashoffset" values="188;56" dur="2s" fill="freeze" />
                                        </circle>
                                      </svg>
                                      <div className="absolute inset-0 flex items-center justify-center">
                                        <span className="text-white font-bold text-sm">70%</span>
                                      </div>
                                    </div>
                                    <div className="text-xs text-gray-400 space-y-1">
                                      <div>Bullish: 70%</div>
                                      <div>Neutral: 20%</div>
                                      <div>Bearish: 10%</div>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        
                        <div className="absolute top-4 right-4 flex items-center gap-2">
                          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                          <span className="text-green-400 text-sm font-mono">Live Market Data</span>
                        </div>
                        
                        <div className="absolute bottom-4 left-4 text-green-400 text-sm font-mono">
                          <div className="mb-1">$ Analyzing 1.2M data points</div>
                          <div>$ Next update: 3s</div>
                        </div>
                      </div>
                    )}

                    {/* Demo 8: News Ontology Network */}
                    {activeDemo === 7 && (
                      <div className="aspect-video flex items-center justify-center relative">
                        <svg viewBox="0 0 800 400" className="w-full h-full">
                          <defs>
                            <linearGradient id="newsGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                              <stop offset="0%" stopColor="#06b6d4" />
                              <stop offset="100%" stopColor="#3b82f6" />
                            </linearGradient>
                          </defs>
                          
                          {/* Central news hub */}
                          <circle cx="400" cy="200" r="50" fill="url(#newsGradient)" opacity="0.8">
                            <animate attributeName="r" values="45;55;45" dur="3s" repeatCount="indefinite" />
                          </circle>
                          <text x="400" y="200" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">
                            <tspan x="400" dy="-5">Breaking</tspan>
                            <tspan x="400" dy="15">News</tspan>
                          </text>
                          
                          {/* News categories */}
                          {[
                            { cx: 200, cy: 100, label: 'Tech', color: '#10b981', stories: 127 },
                            { cx: 600, cy: 100, label: 'Finance', color: '#f59e0b', stories: 89 },
                            { cx: 200, cy: 300, label: 'Politics', color: '#ef4444', stories: 156 },
                            { cx: 600, cy: 300, label: 'Science', color: '#8b5cf6', stories: 73 },
                            { cx: 100, cy: 200, label: 'Sports', color: '#3b82f6', stories: 94 },
                            { cx: 700, cy: 200, label: 'Health', color: '#ec4899', stories: 112 }
                          ].map((category, i) => (
                            <g key={`category-${i}`}>
                              <circle cx={category.cx} cy={category.cy} r="35" fill={category.color} opacity="0.6">
                                <animate attributeName="opacity" values="0.4;0.8;0.4" dur="4s" begin={`${i * 0.5}s`} repeatCount="indefinite" />
                              </circle>
                              <text x={category.cx} y={category.cy - 5} textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">
                                {category.label}
                              </text>
                              <text x={category.cx} y={category.cy + 10} textAnchor="middle" fill="white" fontSize="10">
                                {category.stories} stories
                              </text>
                            </g>
                          ))}
                          
                          {/* Knowledge connections */}
                          {[
                            { x1: 400, y1: 200, x2: 200, y2: 100 },
                            { x1: 400, y1: 200, x2: 600, y2: 100 },
                            { x1: 400, y1: 200, x2: 200, y2: 300 },
                            { x1: 400, y1: 200, x2: 600, y2: 300 },
                            { x1: 400, y1: 200, x2: 100, y2: 200 },
                            { x1: 400, y1: 200, x2: 700, y2: 200 },
                            { x1: 200, y1: 100, x2: 600, y2: 100 },
                            { x1: 200, y1: 300, x2: 600, y2: 300 }
                          ].map((line, i) => (
                            <g key={`connection-${i}`}>
                              <line {...line} stroke="#4b5563" strokeWidth="1" opacity="0.3" />
                              <circle r="2" fill="#06b6d4">
                                <animateMotion
                                  path={`M${line.x1},${line.y1} L${line.x2},${line.y2}`}
                                  dur="3s"
                                  begin={`${i * 0.4}s`}
                                  repeatCount="indefinite"
                                />
                              </circle>
                            </g>
                          ))}
                          
                          {/* Real-time news items */}
                          {[...Array(8)].map((_, i) => {
                            const angle = (i / 8) * Math.PI * 2;
                            const x = 400 + Math.cos(angle) * 150;
                            const y = 200 + Math.sin(angle) * 150;
                            
                            return (
                              <g key={`news-${i}`} opacity="0">
                                <animate attributeName="opacity" values="0;1;0" dur="4s" begin={`${i * 0.5}s`} repeatCount="indefinite" />
                                <rect x={x - 30} y={y - 10} width="60" height="20" rx="3" fill="#1f2937" stroke="#374151" />
                                <text x={x} y={y + 4} textAnchor="middle" fill="#94a3b8" fontSize="8">News #{i + 1}</text>
                              </g>
                            );
                          })}
                          
                          {/* Entity extraction indicators */}
                          <g transform="translate(50, 350)">
                            <text x="0" y="0" fill="#94a3b8" fontSize="12">Entities Extracted: </text>
                            <text x="120" y="0" fill="#06b6d4" fontSize="12" fontFamily="monospace">
                              <tspan>2,847</tspan>
                              <animate attributeName="opacity" values="1;0.5;1" dur="2s" repeatCount="indefinite" />
                            </text>
                          </g>
                          
                          <g transform="translate(600, 350)">
                            <text x="0" y="0" fill="#94a3b8" fontSize="12">Relations: </text>
                            <text x="70" y="0" fill="#3b82f6" fontSize="12" fontFamily="monospace">
                              <tspan>5,126</tspan>
                              <animate attributeName="opacity" values="1;0.5;1" dur="2s" repeatCount="indefinite" />
                            </text>
                          </g>
                        </svg>
                        
                        <div className="absolute top-4 right-4 flex items-center gap-2">
                          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                          <span className="text-cyan-400 text-sm font-mono">Real-time News Processing</span>
                        </div>
                        
                        <div className="absolute bottom-4 left-4 text-cyan-400 text-sm font-mono">
                          <div className="mb-1">$ Processing 500+ news/min</div>
                          <div>$ Knowledge graph: 45K nodes</div>
                        </div>
                      </div>
                    )}

                    {/* Demo 9: Blockchain Investment Analytics */}
                    {activeDemo === 8 && (
                      <div className="aspect-video flex items-center justify-center relative p-4">
                        <div className="w-full h-full flex gap-4">
                          {/* DeFi Portfolio */}
                          <div className="flex-1">
                            <div className="bg-gray-900 rounded-lg p-3 h-full border border-purple-500">
                              <div className="text-purple-400 text-xs font-mono mb-3">DeFi Portfolio Optimization</div>
                              
                              {/* Portfolio allocation */}
                              <div className="mb-4">
                                <svg viewBox="0 0 200 200" className="w-32 h-32 mx-auto">
                                  <defs>
                                    <filter id="innerShadow">
                                      <feGaussianBlur in="SourceAlpha" stdDeviation="3"/>
                                      <feOffset dx="0" dy="2"/>
                                      <feComposite operator="out" in2="SourceAlpha"/>
                                      <feComponentTransfer><feFuncA type="linear" slope="0.5"/></feComponentTransfer>
                                      <feMerge>
                                        <feMergeNode/>
                                        <feMergeNode in="SourceGraphic"/>
                                      </feMerge>
                                    </filter>
                                  </defs>
                                  
                                  {/* Pie chart segments */}
                                  <circle cx="100" cy="100" r="80" fill="#1f2937" />
                                  <path d="M 100 100 L 100 20 A 80 80 0 0 1 164 56 Z" fill="#3b82f6" opacity="0.8" filter="url(#innerShadow)">
                                    <animate attributeName="opacity" values="0;0.8" dur="0.5s" fill="freeze" />
                                  </path>
                                  <path d="M 100 100 L 164 56 A 80 80 0 0 1 164 144 Z" fill="#10b981" opacity="0.8" filter="url(#innerShadow)">
                                    <animate attributeName="opacity" values="0;0.8" dur="0.5s" begin="0.3s" fill="freeze" />
                                  </path>
                                  <path d="M 100 100 L 164 144 A 80 80 0 0 1 36 144 Z" fill="#f59e0b" opacity="0.8" filter="url(#innerShadow)">
                                    <animate attributeName="opacity" values="0;0.8" dur="0.5s" begin="0.6s" fill="freeze" />
                                  </path>
                                  <path d="M 100 100 L 36 144 A 80 80 0 0 1 36 56 Z" fill="#8b5cf6" opacity="0.8" filter="url(#innerShadow)">
                                    <animate attributeName="opacity" values="0;0.8" dur="0.5s" begin="0.9s" fill="freeze" />
                                  </path>
                                  <path d="M 100 100 L 36 56 A 80 80 0 0 1 100 20 Z" fill="#ec4899" opacity="0.8" filter="url(#innerShadow)">
                                    <animate attributeName="opacity" values="0;0.8" dur="0.5s" begin="1.2s" fill="freeze" />
                                  </path>
                                  
                                  <circle cx="100" cy="100" r="40" fill="#111827" />
                                  <text x="100" y="100" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold" dy="5">
                                    $125.4K
                                  </text>
                                </svg>
                              </div>
                              
                              {/* Token allocations */}
                              <div className="space-y-1 text-xs">
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-1">
                                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                                    <span className="text-gray-300">ETH</span>
                                  </div>
                                  <span className="text-white font-mono">35%</span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-1">
                                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                                    <span className="text-gray-300">BTC</span>
                                  </div>
                                  <span className="text-white font-mono">25%</span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-1">
                                    <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                                    <span className="text-gray-300">AAVE</span>
                                  </div>
                                  <span className="text-white font-mono">20%</span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-1">
                                    <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                                    <span className="text-gray-300">UNI</span>
                                  </div>
                                  <span className="text-white font-mono">15%</span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-1">
                                    <div className="w-2 h-2 bg-pink-500 rounded-full"></div>
                                    <span className="text-gray-300">Others</span>
                                  </div>
                                  <span className="text-white font-mono">5%</span>
                                </div>
                              </div>
                            </div>
                          </div>
                          
                          {/* On-chain analytics */}
                          <div className="flex-1 space-y-3">
                            {/* Yield farming opportunities */}
                            <div className="bg-gray-800 rounded-lg p-3 border border-green-500 shadow-lg shadow-green-500/20">
                              <div className="text-green-400 text-xs font-mono mb-2">Top Yield Opportunities</div>
                              <div className="space-y-1">
                                <div className="flex justify-between items-center text-xs">
                                  <span className="text-gray-300">ETH-USDC LP</span>
                                  <span className="text-green-400 font-mono">24.5%</span>
                                </div>
                                <div className="flex justify-between items-center text-xs">
                                  <span className="text-gray-300">AAVE Staking</span>
                                  <span className="text-green-400 font-mono">18.2%</span>
                                </div>
                                <div className="flex justify-between items-center text-xs">
                                  <span className="text-gray-300">Curve 3pool</span>
                                  <span className="text-green-400 font-mono">15.8%</span>
                                </div>
                              </div>
                            </div>
                            
                            {/* Risk metrics */}
                            <div className="bg-gray-800 rounded-lg p-3 border border-red-500 shadow-lg shadow-red-500/20">
                              <div className="text-red-400 text-xs font-mono mb-2">Risk Analysis</div>
                              <div className="space-y-2">
                                <div>
                                  <div className="flex justify-between text-xs mb-1">
                                    <span className="text-gray-400">Impermanent Loss</span>
                                    <span className="text-yellow-400 text-xs">Med</span>
                                  </div>
                                  <div className="w-full bg-gray-700 rounded-full h-1.5">
                                    <div className="bg-gradient-to-r from-green-600 via-yellow-600 to-red-600 h-1.5 rounded-full" style={{ width: '60%' }}></div>
                                  </div>
                                </div>
                                <div>
                                  <div className="flex justify-between text-xs mb-1">
                                    <span className="text-gray-400">Contract Risk</span>
                                    <span className="text-green-400 text-xs">Low</span>
                                  </div>
                                  <div className="w-full bg-gray-700 rounded-full h-1.5">
                                    <div className="bg-gradient-to-r from-green-600 via-yellow-600 to-red-600 h-1.5 rounded-full" style={{ width: '25%' }}></div>
                                  </div>
                                </div>
                              </div>
                            </div>
                            
                            {/* Gas optimization */}
                            <div className="bg-gray-800 rounded-lg p-3 border border-blue-500 shadow-lg shadow-blue-500/20">
                              <div className="text-blue-400 text-xs font-mono mb-1">Gas Optimization</div>
                              <div className="flex items-center justify-between">
                                <div className="text-xs text-gray-400">
                                  <div>Now: 45 gwei</div>
                                  <div>Best: 32 gwei</div>
                                </div>
                                <div className="text-xl font-bold text-blue-400">
                                  -28%
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        
                        <div className="absolute top-4 right-4 flex items-center gap-2">
                          <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
                          <span className="text-purple-400 text-sm font-mono">On-chain Analysis Active</span>
                        </div>
                        
                        <div className="absolute bottom-4 left-4 text-purple-400 text-sm font-mono">
                          <div className="mb-1">$ 1,247 DeFi protocols tracked</div>
                          <div>$ TVL: $45.2B across chains</div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center justify-center mt-6">
                  <div className="flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-gray-900 to-black rounded-full border border-gray-700">
                    <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-gray-300 text-sm font-medium">
                      실시간으로 작동하는 라이브 데모입니다 • Live Interactive Simulation
                    </span>
                  </div>
                </div>
                
                <div className="flex items-center justify-center gap-8 mt-6 text-sm text-gray-500">
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"/>
                      <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd"/>
                    </svg>
                    <span>1.2M+ views</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3zM6 8a2 2 0 11-4 0 2 2 0 014 0zM16 18v-3a5.972 5.972 0 00-.75-2.906A3.005 3.005 0 0119 15v3h-3zM4.75 12.094A5.973 5.973 0 004 15v3H1v-3a3 3 0 013.75-2.906z"/>
                    </svg>
                    <span>50K+ active users</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-20">
            <div className="text-center">
              <div className="text-4xl font-bold text-gray-900 mb-2">100+</div>
              <div className="text-gray-600">Interactive Modules</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-gray-900 mb-2">15+</div>
              <div className="text-gray-600">AI Domains</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-gray-900 mb-2">200+</div>
              <div className="text-gray-600">Learning Hours</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-gray-900 mb-2">3D</div>
              <div className="text-gray-600">Visualization</div>
            </div>
          </div>
        </div>
      </section>

      {/* Learning Modules */}
      <section id="learning-modules" className="py-20 px-6 bg-gray-50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              Learning Modules
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Comprehensive AI education modules with interactive simulations and hands-on learning experiences.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* LLM Module */}
            <Link href="/modules/llm" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🧠</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Large Language Models</h3>
                <p className="text-gray-600 text-sm mb-4">Transformer, GPT, Claude 등 최신 LLM 기술 완전 정복</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">6주 과정</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Ontology Module */}
            <Link href="/modules/ontology" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🔗</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Ontology & Semantic Web</h3>
                <p className="text-gray-600 text-sm mb-4">RDF, SPARQL, 지식 그래프를 통한 시맨틱 웹 기술 마스터</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">8주 과정</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Stock Analysis Module */}
            <Link href="/modules/stock-analysis" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-orange-500 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">📈</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">주식투자분석 시뮬레이터</h3>
                <p className="text-gray-600 text-sm mb-4">실전 투자 전략과 심리까지 포함한 종합 투자 마스터 과정</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">16주 과정</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Smart Factory */}
            <Link href="/modules/smart-factory" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-amber-600 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🏭</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Smart Factory</h3>
              <p className="text-gray-600 text-sm mb-4">Industry 4.0 기반 스마트 팩토리 구축과 운영</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">14시간</span>
                <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
              </div>
              </div>
            </Link>

            {/* RAG Systems */}
            <Link href="/modules/rag" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-green-600 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🗃️</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">RAG Systems</h3>
                <p className="text-gray-600 text-sm mb-4">Retrieval-Augmented Generation 시스템 설계와 구현</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">12시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* System Design */}
            <Link href="/modules/system-design" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🏗️</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">System Design</h3>
                <p className="text-gray-600 text-sm mb-4">대규모 분산 시스템 설계의 핵심 원칙과 실전 패턴 학습</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">20시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Autonomous Mobility */}
            <Link href="/modules/autonomous-mobility" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🚗</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">자율주행 & 미래 모빌리티</h3>
                <p className="text-gray-600 text-sm mb-4">AI 기반 자율주행 기술과 차세대 모빌리티 생태계 완전 정복</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">16시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Medical AI */}
            <Link href="/medical-ai" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-pink-500 to-red-500 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🏥</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Medical AI</h3>
                <p className="text-gray-600 text-sm mb-4">의료 영상 분석, 진단 보조, 신약 개발 AI 기술</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">15시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Physical AI */}
            <Link href="/modules/physical-ai" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-slate-600 to-gray-700 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🤖</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Physical AI & 실세계 지능</h3>
                <p className="text-gray-600 text-sm mb-4">현실 세계와 상호작용하는 AI 시스템의 설계와 구현</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">20시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Quantum Computing */}
            <Link href="/modules/quantum-computing" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-violet-500 to-purple-600 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">⚛️</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Quantum Computing</h3>
                <p className="text-gray-600 text-sm mb-4">양자컴퓨팅 기초부터 Qiskit 실습까지 완전 정복</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">18시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Computer Vision */}
            <Link href="/modules/computer-vision" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-teal-500 to-cyan-600 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">👁️</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Computer Vision</h3>
                <p className="text-gray-600 text-sm mb-4">이미지 인식부터 3D 비전까지 컴퓨터 비전 전문가 과정</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">14시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Deep Learning */}
            <Link href="/modules/deep-learning" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-violet-500 to-purple-600 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🧠</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Deep Learning</h3>
                <p className="text-gray-600 text-sm mb-4">신경망 기초부터 CNN, Transformer, GAN까지 완전 정복</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">25시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Semiconductor */}
            <Link href="/modules/semiconductor" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">💎</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">반도체</h3>
                <p className="text-gray-600 text-sm mb-4">반도체 설계부터 제조까지 - 칩의 모든 것</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">40시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Web3 & Blockchain */}
            <Link href="/modules/web3" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-amber-500 to-orange-600 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">⛓️</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Web3 & Blockchain</h3>
                <p className="text-gray-600 text-sm mb-4">블록체인 기술부터 DeFi, NFT까지 Web3 생태계 완전 분석</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">16시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* AI Agent & MCP */}
            <Link href="/modules/agent-mcp" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-emerald-600 to-teal-600 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🤝</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">AI Agent & MCP</h3>
                <p className="text-gray-600 text-sm mb-4">Model Context Protocol과 멀티 에이전트 시스템 구축</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">12시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* Multi-Agent Systems */}
            <Link href="/modules/multi-agent" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🧩</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Multi-Agent Systems</h3>
                <p className="text-gray-600 text-sm mb-4">복잡한 협력 AI 시스템과 분산 지능 아키텍처</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">10시간</span>
                  <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
                </div>
              </div>
            </Link>

            {/* English Conversation */}
            <Link href="/modules/english-conversation" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-rose-500 to-pink-600 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🗣️</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">English Conversation</h3>
              <p className="text-gray-600 text-sm mb-4">AI 튜터와 함께하는 실전 영어 회화 마스터 과정</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">8시간</span>
                <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
              </div>
              </div>
            </Link>

            {/* Neo4j Knowledge Graph */}
            <Link href="/modules/neo4j" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🌐</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Neo4j Knowledge Graph</h3>
              <p className="text-gray-600 text-sm mb-4">그래프 데이터베이스와 Cypher를 활용한 지식 그래프 구축</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">12시간</span>
                <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
              </div>
              </div>
            </Link>

            {/* AI Automation */}
            <Link href="/modules/ai-automation" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-green-600 to-emerald-700 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">⚙️</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">AI Automation</h3>
              <p className="text-gray-600 text-sm mb-4">AI 기반 업무 자동화와 워크플로우 최적화</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">10시간</span>
                <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
              </div>
              </div>
            </Link>

            {/* AI Security */}
            <Link href="/modules/ai-security" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-red-600 to-gray-700 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🛡️</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">AI Security</h3>
              <p className="text-gray-600 text-sm mb-4">AI 시스템의 보안 위협과 방어 기법 학습</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">18시간</span>
                <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
              </div>
              </div>
            </Link>

            {/* Bioinformatics */}
            <Link href="/modules/bioinformatics" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-teal-600 to-green-700 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🧬</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Bioinformatics</h3>
              <p className="text-gray-600 text-sm mb-4">생물정보학과 AI를 활용한 유전체 분석</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">16시간</span>
                <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
              </div>
              </div>
            </Link>

            {/* Probability & Statistics */}
            <Link href="/modules/probability-statistics" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-indigo-600 to-purple-700 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">📊</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Probability & Statistics</h3>
              <p className="text-gray-600 text-sm mb-4">AI의 수학적 기초인 확률론과 통계학 완전 정복</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">20시간</span>
                <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
              </div>
              </div>
            </Link>

            {/* Linear Algebra */}
            <Link href="/linear-algebra" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-purple-600 to-pink-700 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">📐</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Linear Algebra</h3>
              <p className="text-gray-600 text-sm mb-4">머신러닝의 핵심 수학인 선형대수학 집중 과정</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">15시간</span>
                <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
              </div>
              </div>
            </Link>

            {/* AI Ethics & Governance */}
            <Link href="/modules/ai-ethics" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-rose-500 to-pink-600 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🌹</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">AI Ethics & Governance</h3>
              <p className="text-gray-600 text-sm mb-4">책임감 있는 AI 개발과 윤리적 거버넌스 체계</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">16시간</span>
                <span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full text-xs">개발중</span>
              </div>
              </div>
            </Link>

            {/* Cyber Security */}
            <Link href="/modules/cyber-security" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-red-600 to-orange-700 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🔒</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Cyber Security</h3>
              <p className="text-gray-600 text-sm mb-4">해킹 시뮬레이션과 제로트러스트 보안 모델 실습</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">24시간</span>
                <span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full text-xs">개발중</span>
              </div>
              </div>
            </Link>

            {/* Cloud Computing */}
            <Link href="/modules/cloud-computing" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-sky-500 to-blue-600 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">☁️</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Cloud Computing</h3>
              <p className="text-gray-600 text-sm mb-4">AWS, Azure, GCP를 활용한 클라우드 아키텍처 설계</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">30시간</span>
                <span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full text-xs">개발중</span>
              </div>
              </div>
            </Link>

            {/* Data Engineering */}
            <Link href="/modules/data-engineering" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-indigo-600 to-blue-700 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🗃️</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Data Engineering</h3>
              <p className="text-gray-600 text-sm mb-4">ETL 파이프라인과 실시간 스트림 데이터 처리</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">36시간</span>
                <span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full text-xs">개발중</span>
              </div>
              </div>
            </Link>

            {/* Data Science */}
            <Link href="/modules/data-science" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-emerald-600 to-green-700 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">📊</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Data Science</h3>
              <p className="text-gray-600 text-sm mb-4">데이터에서 가치를 창출하는 과학적 접근법</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">40시간</span>
                <span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full text-xs">개발중</span>
              </div>
              </div>
            </Link>

            {/* Creative AI */}
            <Link href="/modules/creative-ai" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">✨</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Creative AI</h3>
              <p className="text-gray-600 text-sm mb-4">Midjourney, DALL-E, Stable Diffusion 실전 활용</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">24시간</span>
                <span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full text-xs">개발중</span>
              </div>
              </div>
            </Link>

            {/* DevOps & CI/CD */}
            <Link href="/modules/devops-cicd" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-gray-600 to-slate-700 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">⚙️</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">DevOps & CI/CD</h3>
              <p className="text-gray-600 text-sm mb-4">Docker, Kubernetes, GitOps 워크플로우 구축</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">24시간</span>
                <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full text-xs">학습 가능</span>
              </div>
              </div>
            </Link>

            {/* High-Performance Computing */}
            <Link href="/modules/hpc-computing" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-yellow-500 to-orange-600 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">⚡</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">High-Performance Computing</h3>
              <p className="text-gray-600 text-sm mb-4">CUDA 프로그래밍과 분산 컴퓨팅 최적화</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">30시간</span>
                <span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full text-xs">개발중</span>
              </div>
              </div>
            </Link>

            {/* Multimodal AI Systems */}
            <Link href="/modules/multimodal-ai" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-violet-600 to-purple-700 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🧠</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Multimodal AI Systems</h3>
              <p className="text-gray-600 text-sm mb-4">Vision-Language 모델과 멀티모달 AI 구현</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">24시간</span>
                <span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full text-xs">개발중</span>
              </div>
              </div>
            </Link>

            {/* Mathematical Optimization */}
            <Link href="/modules/optimization-theory" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-emerald-600 to-teal-700 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">📐</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Mathematical Optimization</h3>
              <p className="text-gray-600 text-sm mb-4">AI 최적화 이론과 메타휴리스틱 알고리즘</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">30시간</span>
                <span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full text-xs">개발중</span>
              </div>
              </div>
            </Link>

            {/* AI Infrastructure & MLOps */}
            <Link href="/modules/ai-infrastructure" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
              <div className="w-12 h-12 bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl flex items-center justify-center mb-4">
                <span className="text-white text-xl">🏗️</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">AI Infrastructure & MLOps</h3>
              <p className="text-gray-600 text-sm mb-4">대규모 AI 인프라와 ML 파이프라인 구축</p>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-500">36시간</span>
                <span className="bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full text-xs">개발중</span>
              </div>
              </div>
            </Link>


          </div>
        </div>
      </section>

      {/* System Tools */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              System Tools
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-6">
              Professional development and analysis tools for advanced AI research and implementation.
            </p>
            <Link 
              href="/system-tools"
              className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:from-indigo-700 hover:to-purple-700 transition-all transform hover:scale-105"
            >
              🔧 System Tools 바로가기
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
            {/* RDF Editor */}
            <Link href="/rdf-editor" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">📝</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">RDF Editor</h3>
                <p className="text-gray-600 text-sm mb-4">Visual RDF triple editor with real-time validation</p>
                <div className="w-full bg-gray-900 text-white py-2 px-4 rounded-lg text-sm text-center hover:bg-gray-800 transition-colors">
                  Launch Tool
                </div>
              </div>
            </Link>

            {/* SPARQL Playground */}
            <Link href="/sparql-playground" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">⚡</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">SPARQL Playground</h3>
                <p className="text-gray-600 text-sm mb-4">Interactive SPARQL query editor and visualizer</p>
                <div className="w-full bg-gray-900 text-white py-2 px-4 rounded-lg text-sm text-center hover:bg-gray-800 transition-colors">
                  Launch Tool
                </div>
              </div>
            </Link>

            {/* Video Creator */}
            <Link href="/video-creator" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🎬</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Video Creator</h3>
                <p className="text-gray-600 text-sm mb-4">AI-powered educational video generation with Remotion</p>
                <div className="w-full bg-gray-900 text-white py-2 px-4 rounded-lg text-sm text-center hover:bg-gray-800 transition-colors">
                  Launch Tool
                </div>
              </div>
            </Link>

            {/* YouTube Summarizer */}
            <Link href="/youtube-summarizer" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-orange-500 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">📺</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">YouTube Summarizer</h3>
                <p className="text-gray-600 text-sm mb-4">AI-powered video content analysis and summarization</p>
                <div className="w-full bg-gray-900 text-white py-2 px-4 rounded-lg text-sm text-center hover:bg-gray-800 transition-colors">
                  Launch Tool
                </div>
              </div>
            </Link>

            {/* AI Image Generator */}
            <Link href="/system-tools/ai-image-generator" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">🎨</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">AI Image Generator</h3>
                <p className="text-gray-600 text-sm mb-4">DALL-E 3 integration for educational content creation</p>
                <div className="w-full bg-gray-900 text-white py-2 px-4 rounded-lg text-sm text-center hover:bg-gray-800 transition-colors">
                  Launch Tool
                </div>
              </div>
            </Link>

            {/* Content Manager */}
            <Link href="/modules/content-manager" className="block">
              <div className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all cursor-pointer">
                <div className="w-12 h-12 bg-gradient-to-r from-gray-600 to-slate-700 rounded-xl flex items-center justify-center mb-4">
                  <span className="text-white text-xl">📋</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Content Manager</h3>
                <p className="text-gray-600 text-sm mb-4">AI-powered content management and automation system</p>
                <div className="w-full bg-gray-900 text-white py-2 px-4 rounded-lg text-sm text-center hover:bg-gray-800 transition-colors">
                  Launch Tool
                </div>
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 px-6 bg-gray-50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              Professional AI Education Platform
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Designed for researchers, engineers, and organizations who need 
              comprehensive understanding of AI systems through interactive learning.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="glass-card rounded-2xl p-8">
              <div className="w-12 h-12 bg-gray-900 rounded-xl flex items-center justify-center mb-6">
                <span className="text-white text-xl">⚡</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Real-time Simulation
              </h3>
              <p className="text-gray-600 leading-relaxed">
                Experience AI concepts through live, interactive simulations. 
                See how neural networks process data in real-time.
              </p>
            </div>

            <div className="glass-card rounded-2xl p-8">
              <div className="w-12 h-12 bg-gray-900 rounded-xl flex items-center justify-center mb-6">
                <span className="text-white text-xl">🎯</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                3D Visualization
              </h3>
              <p className="text-gray-600 leading-relaxed">
                Complex relationships become clear through advanced 3D visualization. 
                Navigate knowledge graphs like never before.
              </p>
            </div>

            <div className="glass-card rounded-2xl p-8">
              <div className="w-12 h-12 bg-gray-900 rounded-xl flex items-center justify-center mb-6">
                <span className="text-white text-xl">📊</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Enterprise Ready
              </h3>
              <p className="text-gray-600 leading-relaxed">
                Built for professional use with comprehensive analytics, 
                team management, and integration capabilities.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-gray-900 mb-6">
            Ready to transform your AI education?
          </h2>
          <p className="text-xl text-gray-600 mb-12">
            Join leading organizations using KSS for advanced AI training and research.
          </p>
          <div className="flex justify-center">
            <button 
              onClick={() => document.getElementById('learning-modules')?.scrollIntoView({ behavior: 'smooth' })}
              className="minimal-button px-8 py-4 rounded-xl text-lg font-medium cursor-pointer">
              Start Free Trial
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-gray-200">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-3 mb-4 md:mb-0">
              <div className="w-8 h-8 bg-black rounded-lg flex items-center justify-center">
                <span className="text-white text-sm font-bold">K</span>
              </div>
              <span className="text-xl font-semibold text-gray-900">Knowledge Space Simulator</span>
            </div>
            <div className="text-gray-500 text-sm">
              © 2025 KSS. Built with Next.js, TypeScript, and Three.js.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}