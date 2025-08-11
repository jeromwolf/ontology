'use client'

import { Target, CheckCircle, Brain } from 'lucide-react'

export default function Introduction() {
  return (
    <>
      {/* 챕터 헤더 */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">딥러닝 입문</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          TensorFlow/PyTorch로 신경망 구축하기 - 기초부터 CNN, RNN까지
        </p>
      </div>

      {/* 학습 목표 */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Target className="text-indigo-600" />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">신경망의 기초 이해</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">퍼셉트론부터 다층 신경망까지</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">역전파 알고리즘</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">경사하강법과 최적화</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">딥러닝 프레임워크</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">TensorFlow와 PyTorch 실습</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">CNN과 RNN 기초</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">이미지와 시퀀스 데이터 처리</p>
            </div>
          </div>
        </div>
      </div>

      {/* 1. 딥러닝 개요 */}
      <section className="mt-8">
        <h2 className="text-3xl font-bold mb-6">1. 딥러닝이란?</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Brain className="text-indigo-500" />
            딥러닝의 정의와 발전
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <strong>딥러닝(Deep Learning)</strong>은 인공 신경망을 여러 층으로 쌓아 
            복잡한 패턴을 학습하는 머신러닝의 한 분야입니다. '깊은(Deep)'이라는 말은 
            신경망의 은닉층이 많다는 의미입니다.
          </p>
          
          <div className="grid md:grid-cols-3 gap-4 mt-6">
            <TimelineCard
              period="1950-1980s"
              title="퍼셉트론과 초기 신경망"
              icon="🧠"
              color="blue"
              items={[
                "Rosenblatt 퍼셉트론",
                "역전파 알고리즘 개발",
                "XOR 문제 해결"
              ]}
            />
            
            <TimelineCard
              period="2006-2012"
              title="딥러닝의 부활"
              icon="🚀"
              color="green"
              items={[
                "Hinton의 DBN",
                "AlexNet (2012)",
                "GPU 컴퓨팅 활용"
              ]}
            />
            
            <TimelineCard
              period="2012-현재"
              title="딥러닝의 전성기"
              icon="🌟"
              color="purple"
              items={[
                "Transformer (2017)",
                "GPT, BERT",
                "Stable Diffusion"
              ]}
            />
          </div>
        </div>

        {/* 딥러닝 vs 전통적 ML */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
            <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-3">전통적 머신러닝</h4>
            <ul className="space-y-2 text-sm">
              <li>✓ 수동 특성 추출 필요</li>
              <li>✓ 상대적으로 적은 데이터</li>
              <li>✓ 해석 가능한 모델</li>
              <li>✓ 구조화된 데이터에 적합</li>
            </ul>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl">
            <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-3">딥러닝</h4>
            <ul className="space-y-2 text-sm">
              <li>✓ 자동 특성 학습</li>
              <li>✓ 대량의 데이터 필요</li>
              <li>✓ 블랙박스 모델</li>
              <li>✓ 비구조화 데이터 처리 우수</li>
            </ul>
          </div>
        </div>
      </section>
    </>
  )
}

function TimelineCard({ period, title, icon, color, items }: {
  period: string
  title: string
  icon: string
  color: string
  items: string[]
}) {
  const colorClasses = {
    blue: 'from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20',
    green: 'from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20',
    purple: 'from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20'
  }

  const textColorClasses = {
    blue: 'text-blue-700 dark:text-blue-400',
    green: 'text-green-700 dark:text-green-400',
    purple: 'text-purple-700 dark:text-purple-400'
  }

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} p-4 rounded-lg`}>
      <h4 className={`font-semibold ${textColorClasses[color as keyof typeof textColorClasses]} mb-2`}>
        {icon} {period}
      </h4>
      <p className="text-sm text-gray-700 dark:text-gray-300">
        {title}
      </p>
      <ul className="mt-2 space-y-1 text-xs">
        {items.map((item, index) => (
          <li key={index}>• {item}</li>
        ))}
      </ul>
    </div>
  )
}