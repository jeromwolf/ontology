'use client'

import React, { useState, useEffect } from 'react';
import Link from 'next/link';

export default function KSSEnglishLandingPage() {
  const [activeDemo, setActiveDemo] = useState(0);

  const demos = [
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
          font-family: 'Inter', sans-serif;
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
              <Link href="/en" className="text-xl font-bold text-gray-900">
                KSS
              </Link>
              <div className="hidden md:flex space-x-8">
                <Link href="/en/modules" className="text-gray-600 hover:text-gray-900 transition-colors">
                  Modules
                </Link>
                <Link href="/en/about" className="text-gray-600 hover:text-gray-900 transition-colors">
                  About
                </Link>
                <Link href="/en/pricing" className="text-gray-600 hover:text-gray-900 transition-colors">
                  Pricing
                </Link>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm font-medium text-gray-900">EN</span>
              <span className="text-sm text-gray-400">|</span>
              <Link href="/ko" className="text-sm text-gray-600 hover:text-gray-900 transition-colors">
                KO
              </Link>
              <Link
                href="/en/auth/login"
                className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                Sign In
              </Link>
              <Link
                href="/en/auth/register"
                className="minimal-button px-4 py-2 rounded-lg text-sm font-medium"
              >
                Sign Up
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
                  <span className="gradient-text">Knowledge</span>
                  <br />
                  <span className="text-gray-900">Universe</span>
                  <br />
                  <span className="text-gray-900">Simulator</span>
                </h1>
                <p className="text-xl text-gray-600 leading-relaxed max-w-lg">
                  Experience complex technical concepts through interactive 
                  simulations in our next-generation learning platform.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <Link
                  href="/en/modules"
                  className="minimal-button px-8 py-4 rounded-lg font-medium text-center"
                >
                  Start Learning Free
                </Link>
                <Link
                  href="#demo"
                  className="px-8 py-4 border border-gray-300 rounded-lg font-medium text-gray-900 hover:border-gray-400 transition-colors text-center"
                >
                  Watch Live Demo
                </Link>
              </div>

              <div className="flex items-center space-x-6 text-sm text-gray-500">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>30+ Modules</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>100+ Simulators</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  <span>15,000+ Students</span>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="glass-card p-8 rounded-2xl">
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-gray-900">
                      Live Interactive Demo
                    </h3>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                      <span className="text-sm text-gray-500">Running</span>
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
                      <span>Auto-rotating live demos every 5 seconds</span>
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
              Revolutionary Learning Experience
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Move beyond theory-based learning to hands-on experiential education
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white rounded-2xl p-8 border border-gray-200">
              <div className="w-12 h-12 bg-blue-500 rounded-xl mb-6 flex items-center justify-center">
                <span className="text-2xl">🎮</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Interactive Simulators
              </h3>
              <p className="text-gray-600 leading-relaxed">
                Visualize complex AI algorithms, manipulate them directly, and see real-time results.
              </p>
            </div>

            <div className="bg-white rounded-2xl p-8 border border-gray-200">
              <div className="w-12 h-12 bg-green-500 rounded-xl mb-6 flex items-center justify-center">
                <span className="text-2xl">🧠</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Industry-Focused Curriculum
              </h3>
              <p className="text-gray-600 leading-relaxed">
                Analyze and apply real technologies from Tesla, OpenAI, Google in practical scenarios.
              </p>
            </div>

            <div className="bg-white rounded-2xl p-8 border border-gray-200">
              <div className="w-12 h-12 bg-purple-500 rounded-xl mb-6 flex items-center justify-center">
                <span className="text-2xl">⚡</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Accelerated Learning
              </h3>
              <p className="text-gray-600 leading-relaxed">
                Proven results: completion rates improved from 15% to 90% with simulation-based learning.
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
              Popular Learning Modules
            </h2>
            <p className="text-xl text-gray-600">
              From AI to blockchain, covering all the latest technology trends
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg mb-4"></div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Large Language Models</h3>
              <p className="text-sm text-gray-600 mb-4">Master GPT, Claude and latest LLM technologies</p>
              <div className="flex items-center text-sm text-gray-500">
                <span>8 Chapters</span>
                <span className="mx-2">•</span>
                <span>6 Simulators</span>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="w-10 h-10 bg-gradient-to-r from-emerald-500 to-green-600 rounded-lg mb-4"></div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">RAG Systems</h3>
              <p className="text-sm text-gray-600 mb-4">Design and implement Retrieval-Augmented Generation</p>
              <div className="flex items-center text-sm text-gray-500">
                <span>6 Chapters</span>
                <span className="mx-2">•</span>
                <span>5 Simulators</span>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="w-10 h-10 bg-gradient-to-r from-red-500 to-orange-500 rounded-lg mb-4"></div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Stock Investment Analysis</h3>
              <p className="text-sm text-gray-600 mb-4">AI-powered practical investment strategies and psychology</p>
              <div className="flex items-center text-sm text-gray-500">
                <span>23 Chapters</span>
                <span className="mx-2">•</span>
                <span>18 Simulators</span>
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="w-10 h-10 bg-gradient-to-r from-gray-500 to-slate-600 rounded-lg mb-4"></div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">DevOps & CI/CD</h3>
              <p className="text-sm text-gray-600 mb-4">Modern development operations with Docker & Kubernetes</p>
              <div className="flex items-center text-sm text-gray-500">
                <span>8 Chapters</span>
                <span className="mx-2">•</span>
                <span>6 Simulators</span>
              </div>
            </div>
          </div>

          <div className="text-center mt-12">
            <Link
              href="/en/modules"
              className="minimal-button px-8 py-4 rounded-lg font-medium"
            >
              View All Modules
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
              <div className="text-gray-300">Learning Modules</div>
            </div>
            <div>
              <div className="text-3xl md:text-4xl font-bold mb-2">150+</div>
              <div className="text-gray-300">Interactive Simulators</div>
            </div>
            <div>
              <div className="text-3xl md:text-4xl font-bold mb-2">500+</div>
              <div className="text-gray-300">Learning Hours</div>
            </div>
            <div>
              <div className="text-3xl md:text-4xl font-bold mb-2">15,000+</div>
              <div className="text-gray-300">Students</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section-padding">
        <div className="max-w-4xl mx-auto text-center">
          <div className="space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900">
              Start Your Learning Journey
            </h2>
            <p className="text-xl text-gray-600">
              Experience a new way to learn complex technologies easily and enjoyably
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/en/auth/register"
                className="minimal-button px-8 py-4 rounded-lg font-medium"
              >
                Sign Up Free
              </Link>
              <Link
                href="/en/modules"
                className="px-8 py-4 border border-gray-300 rounded-lg font-medium text-gray-900 hover:border-gray-400 transition-colors"
              >
                Explore Modules
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
                Next-generation learning platform for experiencing the knowledge universe through simulations
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-4">Learning</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li><Link href="/en/modules" className="hover:text-gray-900">All Modules</Link></li>
                <li><Link href="/en/modules/llm" className="hover:text-gray-900">LLM</Link></li>
                <li><Link href="/en/modules/rag" className="hover:text-gray-900">RAG</Link></li>
                <li><Link href="/en/modules/stock-analysis" className="hover:text-gray-900">Stock Analysis</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-4">Company</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li><Link href="/en/about" className="hover:text-gray-900">About</Link></li>
                <li><Link href="/en/contact" className="hover:text-gray-900">Contact</Link></li>
                <li><Link href="/en/careers" className="hover:text-gray-900">Careers</Link></li>
                <li><Link href="/en/blog" className="hover:text-gray-900">Blog</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-4">Support</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li><Link href="/en/help" className="hover:text-gray-900">Help</Link></li>
                <li><Link href="/en/community" className="hover:text-gray-900">Community</Link></li>
                <li><Link href="/en/privacy" className="hover:text-gray-900">Privacy</Link></li>
                <li><Link href="/en/terms" className="hover:text-gray-900">Terms</Link></li>
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