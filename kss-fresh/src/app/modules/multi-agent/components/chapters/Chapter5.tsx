'use client';

import React from 'react';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      {/* 합의 알고리즘 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          분산 합의 알고리즘
        </h2>
        <div className="prose prose-lg dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            멀티 에이전트 시스템에서 <strong>합의(Consensus)</strong>는 분산된 에이전트들이 
            공통의 결정에 도달하는 과정입니다. 중앙 조정자 없이도 일관된 의사결정을 가능하게 합니다.
          </p>
        </div>
      </section>

      <section className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          주요 합의 알고리즘
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-3">Voting Mechanisms</h4>
            <ul className="space-y-2 text-sm">
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>Majority Vote:</strong> 과반수 득표
              </li>
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>Weighted Vote:</strong> 가중치 투표
              </li>
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>Ranked Choice:</strong> 선호도 순위
              </li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-3">Byzantine Consensus</h4>
            <ul className="space-y-2 text-sm">
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>PBFT:</strong> Practical Byzantine Fault Tolerance
              </li>
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>Raft:</strong> 리더 기반 합의
              </li>
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>Paxos:</strong> 분산 합의 프로토콜
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          경매 기반 조정 메커니즘
        </h3>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
              <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">English Auction</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                가격이 점진적으로 상승하는 공개 경매
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
              <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">Dutch Auction</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                높은 가격에서 시작해 하락하는 경매
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
              <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">Vickrey Auction</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                비공개 입찰, 차순위 가격 지불
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-green-100 to-blue-100 dark:from-green-900/20 dark:to-blue-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          🎯 실전: 분산 자원 할당
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
            클라우드 컴퓨팅 자원 할당 시나리오
          </h4>
          <div className="space-y-2 text-sm">
            <p className="text-gray-600 dark:text-gray-400">
              여러 에이전트가 제한된 컴퓨팅 자원(CPU, 메모리, 스토리지)을 경쟁
            </p>
            <div className="grid md:grid-cols-2 gap-2 mt-3">
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>문제:</strong> 자원 경쟁과 공정성
              </div>
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>해결:</strong> 경매 메커니즘 적용
              </div>
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>최적화:</strong> 전체 시스템 효율
              </div>
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>공정성:</strong> 비례 할당 보장
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}