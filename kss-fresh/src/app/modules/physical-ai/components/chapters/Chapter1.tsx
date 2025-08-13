'use client'

import React from 'react'

export default function Chapter1() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h2>Physical AI 개요와 미래</h2>
      
      <h3>1. Physical AI란 무엇인가?</h3>
      <p>
        Physical AI는 현실 세계와 직접 상호작용하는 인공지능 시스템을 의미합니다. 
        디지털 환경에서만 작동하는 전통적인 AI와 달리, Physical AI는 센서, 로봇, 
        액추에이터를 통해 물리적 세계를 인식하고 조작합니다.
      </p>

      <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg my-6">
        <h4 className="text-purple-900 dark:text-purple-100 font-semibold mb-3">
          젠슨 황의 COSMOS 비전
        </h4>
        <ul className="space-y-2">
          <li>• <strong>디지털 트윈</strong>: 물리 세계의 완벽한 디지털 복제</li>
          <li>• <strong>시뮬레이션 우선</strong>: 실제 세계 배포 전 가상 환경에서 학습</li>
          <li>• <strong>물리 법칙 통합</strong>: AI가 물리학을 이해하고 활용</li>
          <li>• <strong>실시간 적응</strong>: 환경 변화에 즉각 대응</li>
        </ul>
      </div>

      <h3>2. Digital AI vs Physical AI</h3>
      <div className="grid md:grid-cols-2 gap-6 my-6">
        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <h5 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">Digital AI</h5>
          <ul className="text-sm space-y-1">
            <li>• 데이터와 정보 처리</li>
            <li>• 패턴 인식과 예측</li>
            <li>• 텍스트, 이미지, 음성 처리</li>
            <li>• 소프트웨어 기반 작동</li>
          </ul>
        </div>
        <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
          <h5 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">Physical AI</h5>
          <ul className="text-sm space-y-1">
            <li>• 물리적 상호작용</li>
            <li>• 센서 융합과 제어</li>
            <li>• 로봇, 드론, 자율주행차</li>
            <li>• 하드웨어-소프트웨어 통합</li>
          </ul>
        </div>
      </div>
    </div>
  )
}