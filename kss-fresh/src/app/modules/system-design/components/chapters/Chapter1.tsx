'use client'

import React from 'react'
import { 
  Server, Shield, CheckCircle, Layers, Cpu
} from 'lucide-react'

export default function Chapter1() {
  return (
    <div className="space-y-8">
      {/* Introduction */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Server className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          시스템 설계란?
        </h2>
        
        <div className="prose dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            시스템 설계는 복잡한 소프트웨어 시스템의 아키텍처를 정의하는 과정입니다. 
            확장 가능하고, 신뢰할 수 있으며, 유지보수가 쉬운 시스템을 구축하기 위한 
            청사진을 만드는 것이 목표입니다.
          </p>
          
          <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-6 mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
              핵심 고려사항
            </h3>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                <span className="text-gray-700 dark:text-gray-300">
                  <strong>기능적 요구사항:</strong> 시스템이 수행해야 할 기능
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                <span className="text-gray-700 dark:text-gray-300">
                  <strong>비기능적 요구사항:</strong> 성능, 확장성, 가용성, 보안
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                <span className="text-gray-700 dark:text-gray-300">
                  <strong>제약사항:</strong> 예산, 시간, 기술 스택, 팀 역량
                </span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Scalability */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Layers className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          확장성 (Scalability)
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              수직 확장 (Vertical Scaling)
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 단일 서버의 성능 향상</li>
              <li>• CPU, RAM, Storage 업그레이드</li>
              <li>• 구현이 간단함</li>
              <li>• 하드웨어 한계 존재</li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              수평 확장 (Horizontal Scaling)
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 서버 대수 증가</li>
              <li>• 로드 밸런싱 필요</li>
              <li>• 무한 확장 가능</li>
              <li>• 복잡도 증가</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Reliability & Availability */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Shield className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          신뢰성과 가용성
        </h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              가용성 목표 (SLA)
            </h3>
            <table className="w-full">
              <thead>
                <tr className="border-b dark:border-gray-700">
                  <th className="text-left py-2 text-gray-700 dark:text-gray-300">가용성</th>
                  <th className="text-left py-2 text-gray-700 dark:text-gray-300">연간 다운타임</th>
                  <th className="text-left py-2 text-gray-700 dark:text-gray-300">일반적 용도</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-2 text-gray-600 dark:text-gray-400">99%</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">3.65일</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">개인 프로젝트</td>
                </tr>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-2 text-gray-600 dark:text-gray-400">99.9%</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">8.77시간</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">일반 서비스</td>
                </tr>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-2 text-gray-600 dark:text-gray-400">99.99%</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">52.6분</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">핵심 서비스</td>
                </tr>
                <tr>
                  <td className="py-2 text-gray-600 dark:text-gray-400">99.999%</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">5.26분</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">금융/의료</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Back-of-the-envelope Calculation */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Cpu className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          백오브더엔벨로프 계산
        </h2>
        
        <div className="bg-yellow-50 dark:bg-yellow-950/20 rounded-lg p-6">
          <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
            시스템 용량 추정 예시
          </h3>
          
          <div className="bg-white dark:bg-gray-700 rounded-lg p-4 font-mono text-sm">
            <p className="text-gray-700 dark:text-gray-300">
              <span className="text-purple-600 dark:text-purple-400"># Twitter 타임라인 설계</span><br/>
              <br/>
              DAU (일일 활성 사용자): 150M<br/>
              사용자당 평균 트윗: 2개/일<br/>
              사용자당 평균 팔로우: 200명<br/>
              <br/>
              <span className="text-green-600 dark:text-green-400"># 일일 트윗 수</span><br/>
              150M × 2 = 300M 트윗/일<br/>
              <br/>
              <span className="text-green-600 dark:text-green-400"># 초당 트윗 (TPS)</span><br/>
              300M / 86,400초 ≈ 3,500 TPS (평균)<br/>
              피크 시간: 3,500 × 2 = 7,000 TPS<br/>
              <br/>
              <span className="text-green-600 dark:text-green-400"># 타임라인 읽기 요청</span><br/>
              150M × 50 (일일 조회) = 7.5B 읽기/일<br/>
              7.5B / 86,400 ≈ 87,000 RPS<br/>
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}