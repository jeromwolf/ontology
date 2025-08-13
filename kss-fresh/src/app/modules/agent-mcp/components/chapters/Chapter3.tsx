'use client';

import React from 'react';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Multi-Agent 시스템 아키텍처
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            복잡한 문제를 해결하기 위해 여러 Agent가 협력하는 시스템입니다. 
            각 Agent는 특정 역할과 전문성을 가지고 있으며, 서로 통신하며 작업을 수행합니다.
          </p>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h3 className="font-semibold mb-3">Multi-Agent 패턴</h3>
            <div className="space-y-3">
              <div>
                <strong>1. Pipeline Pattern</strong>
                <p className="text-sm mt-1">Agent들이 순차적으로 작업을 처리 (A → B → C)</p>
              </div>
              <div>
                <strong>2. Committee Pattern</strong>
                <p className="text-sm mt-1">여러 Agent가 투표를 통해 의사결정</p>
              </div>
              <div>
                <strong>3. Hierarchical Pattern</strong>
                <p className="text-sm mt-1">Manager Agent가 Worker Agent들을 조율</p>
              </div>
              <div>
                <strong>4. Collaborative Pattern</strong>
                <p className="text-sm mt-1">Agent들이 평등하게 협업하며 문제 해결</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Agent 간 통신 프로토콜
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            Agent들이 효과적으로 협업하기 위한 표준화된 메시지 형식:
          </p>
          
          <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`interface AgentMessage {
  from: string;        // 발신 Agent ID
  to: string | string[]; // 수신 Agent ID(s)
  type: 'request' | 'response' | 'broadcast';
  content: {
    task: string;      // 작업 설명
    data: any;         // 전달 데이터
    priority: number;  // 우선순위
    deadline?: Date;   // 마감시간
  };
  metadata: {
    timestamp: Date;
    messageId: string;
    correlationId?: string; // 관련 메시지 추적
  };
}`}
          </pre>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          🎮 A2A Orchestrator 시뮬레이터
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          여러 Agent가 협력하여 복잡한 작업을 수행하는 과정을 시각화합니다.
        </p>
        <div className="text-center p-8 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            시뮬레이터를 보려면 전체 시뮬레이터 페이지를 방문하세요.
          </p>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Consensus 메커니즘
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            여러 Agent가 합의에 도달하는 방법:
          </p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Voting Systems</h4>
              <ul className="text-sm space-y-1">
                <li>• Simple Majority (과반수)</li>
                <li>• Weighted Voting (가중 투표)</li>
                <li>• Consensus Threshold (합의 임계값)</li>
                <li>• Veto Power (거부권)</li>
              </ul>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Conflict Resolution</h4>
              <ul className="text-sm space-y-1">
                <li>• Priority-based (우선순위 기반)</li>
                <li>• Expertise-based (전문성 기반)</li>
                <li>• Random Selection (무작위 선택)</li>
                <li>• Human Arbitration (인간 중재)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}