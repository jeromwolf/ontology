'use client'

import { Table } from 'lucide-react'

export default function Section4() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center">
          <Table className="text-indigo-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.4 테이블 및 구조화된 데이터 RAG</h2>
          <p className="text-gray-600 dark:text-gray-400">정형 데이터와 테이블 기반 검색</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl border border-indigo-200 dark:border-indigo-700">
          <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">테이블 이해 및 검색 시스템</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>테이블 RAG는 정형화된 데이터의 구조적 관계를 이해하고 복잡한 분석 질의를 처리합니다.</strong>
              기존 텍스트 기반 RAG가 처리하기 어려운 수치적 추론, 비교 분석, 트렌드 파악 등을 테이블의 행/열 구조를 활용해 정확히 수행합니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>핵심 기술 구성요소:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>테이블 구조 파싱</strong>: 헤더 계층, 병합 셀, 서브 테이블 인식</li>
              <li><strong>스키마 이해</strong>: 컬럼 타입 자동 추론 (숫자, 날짜, 카테고리)</li>
              <li><strong>관계형 추론</strong>: 행 간 비교, 집계, 그룹핑 연산 지원</li>
              <li><strong>자연어→SQL 변환</strong>: "가장 높은 매출을 기록한 월은?" → 구조화 쿼리</li>
            </ul>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-200 dark:border-purple-700 mt-4">
              <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-2">📈 성능 비교</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-purple-300 dark:border-purple-600">
                      <th className="text-left py-2 text-purple-800 dark:text-purple-200">질의 타입</th>
                      <th className="text-left py-2 text-purple-800 dark:text-purple-200">텍스트 RAG</th>
                      <th className="text-left py-2 text-purple-800 dark:text-purple-200">테이블 RAG</th>
                      <th className="text-left py-2 text-purple-800 dark:text-purple-200">향상률</th>
                    </tr>
                  </thead>
                  <tbody className="text-purple-700 dark:text-purple-300">
                    <tr>
                      <td className="py-1">단순 사실 검색</td>
                      <td className="py-1">95%</td>
                      <td className="py-1">97%</td>
                      <td className="py-1">+2%</td>
                    </tr>
                    <tr>
                      <td className="py-1">수치 비교</td>
                      <td className="py-1">72%</td>
                      <td className="py-1">94%</td>
                      <td className="py-1">+31%</td>
                    </tr>
                    <tr>
                      <td className="py-1">집계 연산</td>
                      <td className="py-1">45%</td>
                      <td className="py-1">89%</td>
                      <td className="py-1">+98%</td>
                    </tr>
                    <tr>
                      <td className="py-1">트렌드 분석</td>
                      <td className="py-1">38%</td>
                      <td className="py-1">85%</td>
                      <td className="py-1">+124%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">📊 처리 가능한 테이블 타입</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 재무제표 및 수치 데이터</li>
                <li>• 제품 카탈로그 및 스펙</li>
                <li>• 연구 결과 및 통계</li>
                <li>• 일정 및 시간표</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">🔍 검색 방식</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 셀 값 기반 정확 매칭</li>
                <li>• 컬럼 헤더 의미적 검색</li>
                <li>• 수치 범위 및 조건 검색</li>
                <li>• 행/열 관계 기반 추론</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
