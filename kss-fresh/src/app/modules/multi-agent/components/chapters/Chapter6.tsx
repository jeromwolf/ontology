'use client';

import React from 'react';
import { Activity, Settings } from 'lucide-react';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      {/* 오케스트레이션 패턴 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          대규모 에이전트 시스템 오케스트레이션
        </h2>
        <div className="prose prose-lg dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            오케스트레이션은 <strong>수십에서 수천 개의 에이전트를 효율적으로 관리</strong>하고 
            조정하는 기술입니다. 복잡한 워크플로우, 자원 관리, 모니터링을 포함합니다.
          </p>
        </div>
      </section>

      <section className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          오케스트레이션 아키텍처
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Activity className="w-6 h-6 text-indigo-600 dark:text-indigo-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Orchestrator 컴포넌트</h4>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• Task Scheduler</li>
              <li>• Resource Manager</li>
              <li>• Load Balancer</li>
              <li>• Health Monitor</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Settings className="w-6 h-6 text-indigo-600 dark:text-indigo-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">관리 기능</h4>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• Agent Lifecycle Management</li>
              <li>• Configuration Management</li>
              <li>• Version Control</li>
              <li>• Rollback Mechanism</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          확장성 패턴
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">Horizontal Scaling</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              에이전트 인스턴스 수를 동적으로 증감
            </p>
          </div>
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">Sharding</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              작업을 논리적 그룹으로 분할 처리
            </p>
          </div>
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">Federation</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              독립적인 클러스터 간 연합 구성
            </p>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          모니터링과 관측성
        </h3>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-6">
          <div className="grid md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">247</div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Active Agents</p>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 dark:text-green-400">98.5%</div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Success Rate</p>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">1.2s</div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Avg Response</p>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">12K</div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Messages/min</p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-indigo-100 to-purple-100 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          🚀 Enterprise 사례: 금융 거래 시스템
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              <strong>Market Data Agents:</strong> 실시간 시장 데이터 수집 (500+ agents)
            </div>
            <div className="flex items-center gap-3">
              <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>
              <strong>Analysis Agents:</strong> 기술적/기본적 분석 수행 (200+ agents)
            </div>
            <div className="flex items-center gap-3">
              <span className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></span>
              <strong>Trading Agents:</strong> 자동 매매 실행 (100+ agents)
            </div>
            <div className="flex items-center gap-3">
              <span className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></span>
              <strong>Risk Agents:</strong> 리스크 모니터링 및 관리 (50+ agents)
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}