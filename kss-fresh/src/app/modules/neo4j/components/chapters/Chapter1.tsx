'use client';

import { Database } from 'lucide-react';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-blue-800 dark:text-blue-200 mb-4 flex items-center gap-2">
          <Database className="w-6 h-6" />
          그래프 데이터베이스란?
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 mb-6">
          <p className="text-lg text-gray-700 dark:text-gray-300 mb-4">
            <strong>그래프 데이터베이스</strong>는 데이터를 노드(Node), 관계(Relationship), 속성(Property)으로 
            표현하는 NoSQL 데이터베이스입니다. 연결된 데이터를 효율적으로 저장하고 탐색할 수 있습니다.
          </p>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">노드 (Node)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">엔티티나 객체를 표현 (사람, 제품, 장소)</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">관계 (Relationship)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">노드 간의 연결과 상호작용</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">속성 (Property)</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">노드와 관계의 세부 정보</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">관계형 DB vs 그래프 DB</h3>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse border border-gray-300 dark:border-gray-600">
            <thead>
              <tr className="bg-gray-50 dark:bg-gray-800">
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">특징</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">관계형 DB</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">그래프 DB</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">데이터 모델</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">테이블, 행, 열</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">노드, 관계, 속성</td>
              </tr>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">관계 표현</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">외래 키, JOIN</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">직접적인 관계</td>
              </tr>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">성능</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">JOIN 증가 시 저하</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">관계 탐색 일정</td>
              </tr>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">스키마</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">고정 스키마</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">유연한 스키마</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Neo4j의 핵심 특징</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-3">ACID 트랜잭션</h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• <strong>Atomicity</strong>: 전체 성공 또는 전체 실패</li>
              <li>• <strong>Consistency</strong>: 데이터 일관성 보장</li>
              <li>• <strong>Isolation</strong>: 트랜잭션 격리</li>
              <li>• <strong>Durability</strong>: 영구 저장 보장</li>
            </ul>
          </div>
          <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-lg p-6">
            <h4 className="font-semibold text-cyan-700 dark:text-cyan-300 mb-3">Native Graph Storage</h4>
            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>• Index-free adjacency</li>
              <li>• 포인터 기반 직접 연결</li>
              <li>• O(1) 관계 탐색</li>
              <li>• 메모리 최적화</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">활용 사례</h3>
        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">추천 시스템</h4>
            <p className="text-gray-600 dark:text-gray-400">
              Netflix, Amazon: 사용자-아이템 관계 분석으로 개인화 추천
            </p>
          </div>
          <div className="border-l-4 border-blue-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">소셜 네트워크</h4>
            <p className="text-gray-600 dark:text-gray-400">
              Facebook, LinkedIn: 친구 관계, 팔로우 네트워크 관리
            </p>
          </div>
          <div className="border-l-4 border-blue-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">금융 사기 탐지</h4>
            <p className="text-gray-600 dark:text-gray-400">
              PayPal, 은행: 거래 패턴 분석으로 이상 거래 실시간 탐지
            </p>
          </div>
          <div className="border-l-4 border-blue-500 pl-4">
            <h4 className="font-semibold text-gray-900 dark:text-white">지식 그래프</h4>
            <p className="text-gray-600 dark:text-gray-400">
              Google, Microsoft: 엔티티 관계로 검색 품질 향상
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}