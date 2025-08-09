'use client'

import Link from 'next/link'
import { ArrowLeft, Settings, Code, Sparkles, Video, Youtube, FileImage, Database, Search, Layers } from 'lucide-react'

const systemTools = [
  {
    name: "RDF Editor",
    href: "/rdf-editor",
    icon: "📝",
    gradient: "from-blue-500 to-cyan-500",
    description: "Visual RDF triple editor with real-time validation and semantic graph visualization",
    category: "데이터 모델링",
    status: "active"
  },
  {
    name: "SPARQL Playground", 
    href: "/sparql-playground",
    icon: "⚡",
    gradient: "from-green-500 to-emerald-500", 
    description: "Interactive SPARQL query editor with syntax highlighting and result visualization",
    category: "쿼리 엔진",
    status: "active"
  },
  {
    name: "Video Creator",
    href: "/video-creator", 
    icon: "🎬",
    gradient: "from-purple-500 to-pink-500",
    description: "AI-powered educational video generation with Remotion and automated content creation",
    category: "콘텐츠 제작",
    status: "active"
  },
  {
    name: "YouTube Summarizer",
    href: "/youtube-summarizer",
    icon: "📺", 
    gradient: "from-red-500 to-orange-500",
    description: "AI-powered video content analysis, summarization, and transcript extraction",
    category: "콘텐츠 분석",
    status: "active"
  },
  {
    name: "AI Image Generator",
    href: "/system-tools/ai-image-generator",
    icon: "🎨",
    gradient: "from-indigo-500 to-purple-500",
    description: "DALL-E 3 integration for educational content creation with automatic project integration",
    category: "이미지 생성",
    status: "active"
  },
  {
    name: "Content Manager",
    href: "/modules/content-manager",
    icon: "📋",
    gradient: "from-gray-600 to-slate-700", 
    description: "AI-powered content management and automation system for module updates",
    category: "콘텐츠 관리",
    status: "active"
  }
]

export default function SystemToolsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <Link
            href="/"
            className="inline-flex items-center text-indigo-600 hover:text-indigo-700 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            홈으로 돌아가기
          </Link>
        </div>

        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            <Settings className="w-12 h-12 inline mr-4 text-indigo-600" />
            System Tools
          </h1>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto">
            KSS 플랫폼을 위한 전문 개발 및 분석 도구 모음입니다. 
            AI 연구, 콘텐츠 제작, 데이터 관리를 위한 고급 기능을 제공합니다.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
          {systemTools.map((tool) => (
            <Link key={tool.name} href={tool.href} className="block group">
              <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all transform hover:-translate-y-2 border border-gray-100">
                <div className={`w-16 h-16 bg-gradient-to-r ${tool.gradient} rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                  <span className="text-white text-2xl">{tool.icon}</span>
                </div>
                
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-xl font-bold text-gray-900 group-hover:text-indigo-600 transition-colors">
                      {tool.name}
                    </h3>
                    <span className="px-3 py-1 bg-green-100 text-green-700 text-xs rounded-full">
                      {tool.status === 'active' ? '활성' : '개발중'}
                    </span>
                  </div>
                  <span className="px-3 py-1 bg-gray-100 text-gray-600 text-sm rounded-full">
                    {tool.category}
                  </span>
                </div>
                
                <p className="text-gray-600 text-sm mb-6 leading-relaxed">
                  {tool.description}
                </p>
                
                <div className="flex items-center justify-between">
                  <span className="text-indigo-600 font-semibold group-hover:text-indigo-700 transition-colors">
                    도구 실행 →
                  </span>
                  <Code className="w-5 h-5 text-gray-400 group-hover:text-indigo-600 transition-colors" />
                </div>
              </div>
            </Link>
          ))}
        </div>

        {/* 통계 및 정보 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
          <div className="bg-white rounded-xl p-6 shadow-lg text-center">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center mx-auto mb-4">
              <Layers className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-2">{systemTools.length}</h3>
            <p className="text-gray-600">활성 시스템 도구</p>
          </div>
          
          <div className="bg-white rounded-xl p-6 shadow-lg text-center">
            <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl flex items-center justify-center mx-auto mb-4">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-2">AI</h3>
            <p className="text-gray-600">통합된 AI 서비스</p>
          </div>
          
          <div className="bg-white rounded-xl p-6 shadow-lg text-center">
            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl flex items-center justify-center mx-auto mb-4">
              <FileImage className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-2">자동화</h3>
            <p className="text-gray-600">콘텐츠 생성 파이프라인</p>
          </div>
        </div>

        {/* 사용 가이드 */}
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-8 text-white">
          <h2 className="text-3xl font-bold mb-6 text-center">🚀 시작하기</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold mb-3">개발자용 도구</h3>
              <ul className="space-y-2 text-indigo-100">
                <li>• RDF Editor로 시맨틱 데이터 모델링</li>
                <li>• SPARQL Playground로 지식 그래프 쿼리</li>
                <li>• Content Manager로 모듈 관리</li>
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-semibold mb-3">콘텐츠 제작 도구</h3>
              <ul className="space-y-2 text-indigo-100">
                <li>• AI Image Generator로 교육 이미지 생성</li>
                <li>• Video Creator로 동영상 제작</li>
                <li>• YouTube Summarizer로 콘텐츠 분석</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}