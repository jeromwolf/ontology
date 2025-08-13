'use client';

import React from 'react';
import { Users, Network, GitBranch, Layers } from 'lucide-react';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      {/* 멀티 에이전트 시스템 개요 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          멀티 에이전트 시스템의 핵심 개념
        </h2>
        <div className="prose prose-lg dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            멀티 에이전트 시스템(MAS)은 <strong>여러 개의 자율적인 에이전트가 협력하여 복잡한 문제를 해결</strong>하는 
            분산 인공지능 시스템입니다. 각 에이전트는 독립적인 의사결정 능력을 가지며, 
            다른 에이전트와 통신하고 협력하여 단일 에이전트로는 불가능한 작업을 수행합니다.
          </p>
        </div>
      </section>

      <section className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Users className="w-6 h-6 text-orange-600 dark:text-orange-400" />
          왜 멀티 에이전트인가?
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">단일 에이전트의 한계</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 복잡한 문제의 단일 처리 부담</li>
              <li>• 제한된 전문성과 관점</li>
              <li>• 병목 현상과 확장성 문제</li>
              <li>• 단일 실패 지점(SPOF)</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">멀티 에이전트의 강점</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 작업 분할과 병렬 처리</li>
              <li>• 전문화된 역할 분담</li>
              <li>• 높은 확장성과 유연성</li>
              <li>• 내결함성과 견고성</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          멀티 에이전트 아키텍처 패턴
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
            <Network className="w-8 h-8 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Centralized</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              중앙 조정자가 모든 에이전트를 관리하는 구조
            </p>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <GitBranch className="w-8 h-8 text-green-600 dark:text-green-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Decentralized</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              에이전트가 자율적으로 협력하는 P2P 구조
            </p>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <Layers className="w-8 h-8 text-purple-600 dark:text-purple-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Hierarchical</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              계층적 조직 구조로 운영되는 시스템
            </p>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-orange-100 to-red-100 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          💡 실전 예시: 스마트 물류 시스템
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="space-y-3 text-sm">
            <div className="flex items-start gap-3">
              <span className="w-8 h-8 bg-orange-600 text-white rounded-full flex items-center justify-center flex-shrink-0">1</span>
              <div>
                <p className="font-semibold text-gray-900 dark:text-white">Inventory Agent</p>
                <p className="text-gray-600 dark:text-gray-400">재고 수준 모니터링 및 보충 요청</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="w-8 h-8 bg-orange-600 text-white rounded-full flex items-center justify-center flex-shrink-0">2</span>
              <div>
                <p className="font-semibold text-gray-900 dark:text-white">Route Agent</p>
                <p className="text-gray-600 dark:text-gray-400">최적 배송 경로 계산 및 조정</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="w-8 h-8 bg-orange-600 text-white rounded-full flex items-center justify-center flex-shrink-0">3</span>
              <div>
                <p className="font-semibold text-gray-900 dark:text-white">Vehicle Agent</p>
                <p className="text-gray-600 dark:text-gray-400">차량 상태 관리 및 배송 실행</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="w-8 h-8 bg-orange-600 text-white rounded-full flex items-center justify-center flex-shrink-0">4</span>
              <div>
                <p className="font-semibold text-gray-900 dark:text-white">Customer Agent</p>
                <p className="text-gray-600 dark:text-gray-400">고객 요구사항 처리 및 상태 업데이트</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}