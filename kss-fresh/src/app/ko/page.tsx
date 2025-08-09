'use client'

import React, { useState, useEffect } from 'react';
import Link from 'next/link';

export default function KSSKoreanLandingPage() {
  const [activeDemo, setActiveDemo] = useState(0);

  const demos = [
    {
      title: "3D 지식 그래프",
      description: "복잡한 AI 개념의 인터랙티브 시각화",
      tech: "Three.js + WebGL"
    },
    {
      title: "신경망 시뮬레이터", 
      description: "실시간 토큰화와 어텐션 메커니즘 체험",
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
      description: "문서 검색 및 생성 흐름의 실시간 시각화",
      tech: "Vector DB + LLM"
    },
    {
      title: "멀티 에이전트 협업",
      description: "복잡한 작업 해결을 위한 AI 에이전트 협력",
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
          background: rgba(0, 0, 0, 0.2);
          transition: all 0.3s ease;
          cursor: pointer;
        }
        
        .demo-indicator.active {
          background: #000000;
          transform: scale(1.2);
        }
        
        .floating-animation {
          animation: float 6s ease-in-out infinite;
        }
        
        .floating-animation:nth-child(2) {
          animation-delay: -2s;
        }
        
        .floating-animation:nth-child(3) {
          animation-delay: -4s;
        }
        
        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          33% { transform: translateY(-10px) rotate(1deg); }
          66% { transform: translateY(5px) rotate(-1deg); }
        }
        
        .tech-badge {
          background: rgba(0, 0, 0, 0.05);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(0, 0, 0, 0.1);
        }
        
        .section-padding {
          padding: 4rem 2rem;
        }
        
        @media (max-width: 768px) {
          .section-padding {
            padding: 2rem 1rem;
          }
        }
      `}</style>

      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-md border-b border-gray-200/50">
        <div className="max-w-7xl mx-auto px-6 sm:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-8">
              <Link href="/ko" className="text-xl font-bold text-gray-900">
                KSS
              </Link>
              <div className="hidden md:flex space-x-8">
                <Link href="/ko/modules" className="text-gray-600 hover:text-gray-900 transition-colors">
                  학습 모듈
                </Link>
                <Link href="/ko/about" className="text-gray-600 hover:text-gray-900 transition-colors">
                  소개
                </Link>
                <Link href="/ko/pricing" className="text-gray-600 hover:text-gray-900 transition-colors">
                  요금제
                </Link>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Link href="/en" className="text-sm text-gray-600 hover:text-gray-900 transition-colors">
                EN
              </Link>
              <span className="text-sm text-gray-400">|</span>
              <span className="text-sm font-medium text-gray-900">KO</span>
              <Link
                href="/ko/auth/login"
                className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                로그인
              </Link>
              <Link
                href="/ko/auth/register"
                className="minimal-button px-4 py-2 rounded-lg text-sm font-medium"
              >
                회원가입
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="section-padding pt-24">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              <div className="space-y-4">
                <h1 className="text-4xl md:text-6xl font-bold leading-tight">
                  <span className="gradient-text">지식의 우주</span>
                  <br />
                  <span className="text-gray-900">시뮬레이션으로</span>
                  <br />
                  <span className="text-gray-900">체험하다</span>
                </h1>
                <p className="text-xl text-gray-600 leading-relaxed max-w-lg">
                  복잡한 기술 개념을 시각적으로 시뮬레이션하며 직접 체험하는 
                  차세대 학습 플랫폼입니다.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <Link
                  href="/ko/modules"
                  className="minimal-button px-8 py-4 rounded-lg font-medium text-center"
                >
                  무료로 시작하기
                </Link>
                <Link
                  href="#demo"
                  className="px-8 py-4 border border-gray-300 rounded-lg font-medium text-gray-900 hover:border-gray-400 transition-colors text-center"
                >
                  라이브 데모 보기
                </Link>
              </div>

              <div className="flex items-center space-x-6 text-sm text-gray-500">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>30+ 전문 모듈</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>100+ 시뮬레이터</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  <span>15,000+ 수강생</span>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="glass-card p-8 rounded-2xl">
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-gray-900">
                      실시간 라이브 데모
                    </h3>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                      <span className="text-sm text-gray-500">실행 중</span>
                    </div>
                  </div>

                  <div className="bg-gray-50 rounded-xl p-6 min-h-[200px] relative overflow-hidden">
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center space-y-4 max-w-sm">
                        <h4 className="text-xl font-semibold text-gray-900">
                          {demos[activeDemo].title}
                        </h4>
                        <p className="text-gray-600">
                          {demos[activeDemo].description}
                        </p>
                        <div className="tech-badge inline-block px-3 py-1 rounded-full">
                          <span className="text-sm text-gray-700">
                            {demos[activeDemo].tech}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Decorative animated elements */}
                    <div className="absolute top-4 right-4 w-12 h-12 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full floating-animation opacity-20"></div>
                    <div className="absolute bottom-4 left-4 w-8 h-8 bg-gradient-to-br from-green-400 to-blue-500 rounded-full floating-animation opacity-30"></div>
                    <div className="absolute top-1/2 right-1/4 w-6 h-6 bg-gradient-to-br from-purple-400 to-pink-500 rounded-full floating-animation opacity-25"></div>
                  </div>

                  <div className="flex justify-center space-x-2">
                    {demos.map((_, index) => (
                      <div
                        key={index}
                        className={`demo-indicator ${index === activeDemo ? 'active' : ''}`}
                        onClick={() => setActiveDemo(index)}
                      />
                    ))}
                  </div>

                  <p className="text-sm text-gray-500 text-center">
                    <span className="inline-flex items-center space-x-2">
                      <span>🔄</span>
                      <span>5초마다 자동 전환되는 라이브 데모를 확인해보세요</span>
                    </span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="demo" className="section-padding bg-gray-50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center space-y-4 mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900">
              혁신적인 학습 경험
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              기존의 이론 중심 학습에서 벗어나, 직접 체험하며 배우는 새로운 방식
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white rounded-2xl p-8 border border-gray-200">
              <div className="w-12 h-12 bg-blue-500 rounded-xl mb-6 flex items-center justify-center">
                <span className="text-2xl">🎮</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                인터랙티브 시뮬레이터
              </h3>
              <p className="text-gray-600 leading-relaxed">
                복잡한 AI 알고리즘을 시각화하여 직접 조작하고 결과를 실시간으로 확인할 수 있습니다.
              </p>
            </div>

            <div className="bg-white rounded-2xl p-8 border border-gray-200">
              <div className="w-12 h-12 bg-green-500 rounded-xl mb-6 flex items-center justify-center">
                <span className="text-2xl">🧠</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                실무 중심 커리큘럼
              </h3>
              <p className="text-gray-600 leading-relaxed">
                Tesla, OpenAI, Google 등 실제 기업의 기술을 분석하고 응용하는 실전 중심 학습.
              </p>
            </div>

            <div className="bg-white rounded-2xl p-8 border border-gray-200">
              <div className="w-12 h-12 bg-purple-500 rounded-xl mb-6 flex items-center justify-center">
                <span className="text-2xl">⚡</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                빠른 학습 효과
              </h3>
              <p className="text-gray-600 leading-relaxed">
                시뮬레이션 기반 학습으로 완료율이 15%에서 90%로 향상된 입증된 효과.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Modules Preview */}
      <section className="section-padding">
        <div className="max-w-7xl mx-auto">
          <div className="text-center space-y-4 mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900">
              인기 학습 모듈
            </h2>
            <p className="text-xl text-gray-600">
              AI부터 블록체인까지, 최신 기술 트렌드를 모두 담았습니다
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg mb-4"></div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Large Language Models</h3>
              <p className="text-sm text-gray-600 mb-4">GPT, Claude 등 최신 LLM 기술 완전 정복</p>
              <div className="flex items-center text-sm text-gray-500">
                <span>8챕터</span>
                <span className="mx-2">•</span>
                <span>6시뮬레이터</span>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="w-10 h-10 bg-gradient-to-r from-emerald-500 to-green-600 rounded-lg mb-4"></div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">RAG 시스템</h3>
              <p className="text-sm text-gray-600 mb-4">검색 기반 생성 AI 시스템 설계와 구현</p>
              <div className="flex items-center text-sm text-gray-500">
                <span>6챕터</span>
                <span className="mx-2">•</span>
                <span>5시뮬레이터</span>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="w-10 h-10 bg-gradient-to-r from-red-500 to-orange-500 rounded-lg mb-4"></div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">주식 투자 분석</h3>
              <p className="text-sm text-gray-600 mb-4">AI 기반 실전 투자 전략과 심리 분석</p>
              <div className="flex items-center text-sm text-gray-500">
                <span>23챕터</span>
                <span className="mx-2">•</span>
                <span>18시뮬레이터</span>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="w-10 h-10 bg-gradient-to-r from-gray-500 to-slate-600 rounded-lg mb-4"></div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">DevOps & CI/CD</h3>
              <p className="text-sm text-gray-600 mb-4">Docker, Kubernetes로 구축하는 현대적 개발 운영</p>
              <div className="flex items-center text-sm text-gray-500">
                <span>8챕터</span>
                <span className="mx-2">•</span>
                <span>6시뮬레이터</span>
              </div>
            </div>
          </div>

          <div className="text-center mt-12">
            <Link
              href="/ko/modules"
              className="minimal-button px-8 py-4 rounded-lg font-medium"
            >
              모든 모듈 보기
            </Link>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="section-padding bg-gray-900 text-white">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-3xl md:text-4xl font-bold mb-2">30+</div>
              <div className="text-gray-300">전문 학습 모듈</div>
            </div>
            <div>
              <div className="text-3xl md:text-4xl font-bold mb-2">150+</div>
              <div className="text-gray-300">인터랙티브 시뮬레이터</div>
            </div>
            <div>
              <div className="text-3xl md:text-4xl font-bold mb-2">500+</div>
              <div className="text-gray-300">학습 시간</div>
            </div>
            <div>
              <div className="text-3xl md:text-4xl font-bold mb-2">15,000+</div>
              <div className="text-gray-300">수강생</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section-padding">
        <div className="max-w-4xl mx-auto text-center">
          <div className="space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900">
              지금 바로 시작해보세요
            </h2>
            <p className="text-xl text-gray-600">
              복잡한 기술을 쉽고 재미있게 배우는 새로운 방법을 경험해보세요
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/ko/auth/register"
                className="minimal-button px-8 py-4 rounded-lg font-medium"
              >
                무료 회원가입
              </Link>
              <Link
                href="/ko/modules"
                className="px-8 py-4 border border-gray-300 rounded-lg font-medium text-gray-900 hover:border-gray-400 transition-colors"
              >
                모듈 둘러보기
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-50 border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="text-xl font-bold text-gray-900 mb-4">KSS</div>
              <p className="text-gray-600 text-sm leading-relaxed">
                지식의 우주를 시뮬레이션으로 체험하는 차세대 학습 플랫폼
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-4">학습</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li><Link href="/ko/modules" className="hover:text-gray-900">전체 모듈</Link></li>
                <li><Link href="/ko/modules/llm" className="hover:text-gray-900">LLM</Link></li>
                <li><Link href="/ko/modules/rag" className="hover:text-gray-900">RAG</Link></li>
                <li><Link href="/ko/modules/stock-analysis" className="hover:text-gray-900">주식 분석</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-4">회사</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li><Link href="/ko/about" className="hover:text-gray-900">소개</Link></li>
                <li><Link href="/ko/contact" className="hover:text-gray-900">연락처</Link></li>
                <li><Link href="/ko/careers" className="hover:text-gray-900">채용</Link></li>
                <li><Link href="/ko/blog" className="hover:text-gray-900">블로그</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-4">지원</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li><Link href="/ko/help" className="hover:text-gray-900">도움말</Link></li>
                <li><Link href="/ko/community" className="hover:text-gray-900">커뮤니티</Link></li>
                <li><Link href="/ko/privacy" className="hover:text-gray-900">개인정보처리방침</Link></li>
                <li><Link href="/ko/terms" className="hover:text-gray-900">이용약관</Link></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-200 mt-8 pt-8 text-center text-sm text-gray-500">
            © 2024 KSS (Knowledge Space Simulator). All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}