'use client';

import React from 'react';
import { Activity, Settings } from 'lucide-react';
import References from '@/components/common/References';

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

      <References
        sections={[
          {
            title: 'Orchestration Platforms & Tools',
            icon: 'book',
            color: 'border-orange-500',
            items: [
              {
                title: 'Kubernetes: Container Orchestration',
                description: 'Production-grade container orchestration platform',
                link: 'https://kubernetes.io/'
              },
              {
                title: 'Apache Mesos: Distributed Systems Kernel',
                description: '대규모 클러스터 자원 관리 플랫폼',
                link: 'https://mesos.apache.org/'
              },
              {
                title: 'Docker Swarm: Native Clustering',
                description: 'Docker 네이티브 오케스트레이션',
                link: 'https://docs.docker.com/engine/swarm/'
              },
              {
                title: 'Nomad: Workload Orchestrator',
                description: 'HashiCorp의 워크로드 오케스트레이터',
                link: 'https://www.nomadproject.io/'
              }
            ]
          },
          {
            title: 'Scalability Research',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Large-Scale Distributed Systems: Architecture and Implementation',
                authors: 'Google',
                year: '2021',
                description: 'Google의 대규모 분산 시스템 아키텍처',
                link: 'https://research.google/pubs/pub51877/'
              },
              {
                title: 'Borg, Omega, and Kubernetes',
                authors: 'Brendan Burns, Brian Grant, et al.',
                year: '2016',
                description: 'Google의 컨테이너 오케스트레이션 진화',
                link: 'https://research.google/pubs/pub44843/'
              },
              {
                title: 'Scaling Distributed Machine Learning with the Parameter Server',
                authors: 'Mu Li, David G. Andersen, et al.',
                year: '2014',
                description: '분산 ML 확장을 위한 파라미터 서버',
                link: 'https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf'
              },
              {
                title: 'Ray: A Distributed Framework for Emerging AI Applications',
                authors: 'Philipp Moritz, Robert Nishihara, et al.',
                year: '2018',
                description: 'AI 애플리케이션을 위한 분산 프레임워크',
                link: 'https://arxiv.org/abs/1712.05889'
              }
            ]
          },
          {
            title: 'Monitoring & Observability',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'Prometheus: Monitoring System',
                description: 'CNCF 표준 모니터링 및 알람 시스템',
                link: 'https://prometheus.io/'
              },
              {
                title: 'Grafana: Observability Platform',
                description: '통합 가시성 및 대시보드 플랫폼',
                link: 'https://grafana.com/'
              },
              {
                title: 'OpenTelemetry: Observability Framework',
                description: '분산 추적 및 메트릭 수집 표준',
                link: 'https://opentelemetry.io/'
              },
              {
                title: 'Jaeger: Distributed Tracing',
                description: 'CNCF 분산 추적 시스템',
                link: 'https://www.jaegertracing.io/'
              }
            ]
          },
          {
            title: 'Enterprise Use Cases',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Trading Systems at Scale: Financial Services',
                description: '대규모 금융 거래 시스템 아키텍처',
                link: 'https://www.nasdaq.com/articles/how-nasdaq-uses-kubernetes-and-the-cloud'
              },
              {
                title: 'Agent Orchestration for Robotics Fleets',
                description: '로봇 플릿 관리를 위한 agent 오케스트레이션',
                link: 'https://www.inceptivemind.com/fleet-management-multi-robot-systems/23456/'
              },
              {
                title: 'Smart Grid Agent Systems',
                description: '스마트 그리드를 위한 multi-agent 오케스트레이션',
                link: 'https://ieeexplore.ieee.org/document/8387595'
              },
              {
                title: 'Uber: Microservices at Scale',
                description: 'Uber의 마이크로서비스 오케스트레이션 사례',
                link: 'https://www.uber.com/blog/microservice-architecture/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}