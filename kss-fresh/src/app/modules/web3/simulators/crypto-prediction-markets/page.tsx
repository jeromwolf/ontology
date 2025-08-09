'use client'

import { Suspense } from 'react'
import dynamic from 'next/dynamic'
import Navigation from '@/components/Navigation'
import { ArrowLeft, BarChart3 } from 'lucide-react'
import Link from 'next/link'

// Dynamic import to avoid SSR issues
const CryptoPredictionMarkets = dynamic(
  () => import('@/app/modules/web3/components/CryptoPredictionMarkets'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center min-h-[600px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">예측 시장 로딩 중...</p>
        </div>
      </div>
    )
  }
)

export default function CryptoPredictionMarketsPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Navigation />
      
      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <Link 
            href="/modules/web3"
            className="inline-flex items-center gap-2 text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Web3 모듈로 돌아가기
          </Link>
        </div>
        
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Crypto Prediction Markets
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                블록체인 기반 암호화폐 가격 예측 시장 시뮬레이터
              </p>
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              🎯 학습 목표
            </h2>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 예측 시장의 메커니즘과 집단지성 원리 이해</li>
              <li>• 블록체인 기반 베팅과 토큰 이코노미 체험</li>
              <li>• 오라클을 통한 실제 데이터 검증 과정 학습</li>
              <li>• 스마트 컨트랙트 자동 정산 시스템 이해</li>
            </ul>
          </div>
        </div>
        
        <Suspense fallback={<div>로딩 중...</div>}>
          <CryptoPredictionMarkets />
        </Suspense>
      </main>
    </div>
  )
}