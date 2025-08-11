'use client'

import { Brain, Activity, Settings, Play } from 'lucide-react'
import Link from 'next/link'

export default function MLPlaygroundPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-black">
      <div className="container mx-auto px-6 py-8">
        {/* 헤더 */}
        <div className="mb-8">
          <Link 
            href="/modules/data-science"
            className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 inline-flex items-center gap-2"
          >
            ← 데이터 사이언스로 돌아가기
          </Link>
          <h1 className="text-4xl font-bold mt-4 bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
            머신러닝 플레이그라운드
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            다양한 머신러닝 알고리즘을 실시간으로 실험해보세요
          </p>
        </div>

        {/* 메인 컨텐츠 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
          <div className="text-center py-20">
            <Brain className="w-24 h-24 mx-auto text-purple-500 mb-6" />
            <h2 className="text-2xl font-semibold mb-4">머신러닝 플레이그라운드 준비 중</h2>
            <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto">
              선형 회귀, 로지스틱 회귀, 의사결정 트리, SVM 등 다양한 머신러닝 알고리즘을 
              실시간으로 시각화하고 파라미터를 조정하며 학습 과정을 관찰할 수 있는 
              인터랙티브 시뮬레이터입니다.
            </p>
            
            <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
              <FeatureCard
                icon={<Activity className="w-8 h-8" />}
                title="실시간 학습"
                description="알고리즘의 학습 과정을 실시간으로 시각화"
              />
              <FeatureCard
                icon={<Settings className="w-8 h-8" />}
                title="파라미터 조정"
                description="하이퍼파라미터를 조정하며 결과 비교"
              />
              <FeatureCard
                icon={<Play className="w-8 h-8" />}
                title="데이터셋 선택"
                description="다양한 샘플 데이터셋으로 실험"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function FeatureCard({ icon, title, description }: {
  icon: React.ReactNode
  title: string
  description: string
}) {
  return (
    <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-lg">
      <div className="text-purple-500 mb-3">{icon}</div>
      <h3 className="font-semibold mb-2">{title}</h3>
      <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
    </div>
  )
}