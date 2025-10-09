'use client'

import Link from 'next/link'
import { ArrowLeft, CheckCircle2, Shield } from 'lucide-react'
import References from '@/components/common/References'

export default function Section4() {
  return (
    <>
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
            <Shield className="text-red-600" size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.4 재해 복구 계획</h2>
            <p className="text-gray-600 dark:text-gray-400">RPO 15분, RTO 30분 달성</p>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-red-800 dark:text-red-200 mb-3">복구 목표</h3>
            <div className="space-y-2 text-red-700 dark:text-red-300">
              <p>🎯 <strong>RPO (Recovery Point Objective)</strong>: 15분 - 최대 데이터 손실</p>
              <p>⏱️ <strong>RTO (Recovery Time Objective)</strong>: 30분 - 최대 다운타임</p>
              <p>📊 <strong>SLA (Service Level Agreement)</strong>: 99.9% 가용성</p>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl">
            <h3 className="font-bold text-gray-800 dark:text-gray-200 mb-4">자동화된 재해 복구 시스템</h3>
            <pre className="bg-black text-green-400 p-4 rounded-lg overflow-x-auto">
              <code>{`# Production 체크리스트 템플릿
disaster_recovery_checklist = {
    "사전 준비": [
        "✅ 멀티 리전 아키텍처 구성",
        "✅ 실시간 데이터 복제 설정",
        "✅ 자동 페일오버 스크립트",
        "✅ 백업 검증 자동화",
        "✅ 복구 절차 문서화"
    ],

    "모니터링": [
        "✅ 실시간 헬스 체크 (5초 간격)",
        "✅ 리전간 레이턴시 모니터링",
        "✅ 데이터 동기화 지연 추적",
        "✅ 자동 알림 시스템",
        "✅ 대시보드 구축"
    ],

    "복구 절차": [
        "1️⃣ 장애 감지 (자동, 1분 이내)",
        "2️⃣ 영향 범위 평가 (2분)",
        "3️⃣ 페일오버 결정 (3분)",
        "4️⃣ DNS 업데이트 (5분)",
        "5️⃣ 서비스 검증 (10분)",
        "6️⃣ 사용자 알림 (15분)"
    ],

    "테스트 계획": [
        "🔄 월간 페일오버 드릴",
        "🔄 분기별 전체 복구 테스트",
        "🔄 연간 재해 시뮬레이션",
        "🔄 자동화 스크립트 검증"
    ]
}

# 복구 자동화 스크립트
class AutomatedDisasterRecovery:
    def __init__(self):
        self.regions = {
            'primary': 'us-east-1',
            'secondary': 'us-west-2',
            'tertiary': 'eu-west-1'
        }

        self.recovery_steps = []
        self.start_time = None

    async def execute_failover(self, failed_region: str):
        """자동 페일오버 실행"""
        self.start_time = datetime.now()
        self.recovery_steps = []

        try:
            # 1. 장애 확인
            await self._verify_failure(failed_region)

            # 2. 새 Primary 선택
            new_primary = await self._select_new_primary(failed_region)

            # 3. 데이터 동기화 확인
            await self._verify_data_consistency(new_primary)

            # 4. 트래픽 전환
            await self._switch_traffic(new_primary)

            # 5. 서비스 검증
            await self._verify_services(new_primary)

            # 6. 알림 발송
            await self._notify_stakeholders(failed_region, new_primary)

            recovery_time = (datetime.now() - self.start_time).total_seconds() / 60

            return {
                'success': True,
                'recovery_time_minutes': recovery_time,
                'new_primary': new_primary,
                'steps': self.recovery_steps
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'steps': self.recovery_steps
            }

    async def _verify_failure(self, region: str):
        """장애 확인"""
        # 실제로는 여러 소스에서 확인
        self.recovery_steps.append({
            'step': 'verify_failure',
            'timestamp': datetime.now(),
            'result': f'{region} confirmed down'
        })

    async def _select_new_primary(self, failed_region: str) -> str:
        """새로운 Primary 리전 선택"""
        candidates = [r for r in self.regions.values() if r != failed_region]

        # 데이터 신선도와 가용성 기반 선택
        # 실제로는 복잡한 로직 필요
        new_primary = candidates[0]

        self.recovery_steps.append({
            'step': 'select_primary',
            'timestamp': datetime.now(),
            'result': f'Selected {new_primary} as new primary'
        })

        return new_primary

# 99.9% 가용성 달성 전략
uptime_strategy = {
    "아키텍처": {
        "멀티 리전": "최소 3개 리전에 분산",
        "로드 밸런싱": "지능형 트래픽 분산",
        "데이터 복제": "실시간 크로스 리전 복제",
        "캐싱": "엣지 로케이션 활용"
    },

    "모니터링": {
        "헬스 체크": "5초 간격",
        "메트릭 수집": "1분 간격",
        "알림": "다중 채널 (SMS, Email, Slack)",
        "대시보드": "실시간 상태 표시"
    },

    "자동화": {
        "페일오버": "자동 실행",
        "스케일링": "예측적 오토스케일링",
        "백업": "증분 백업 15분마다",
        "복구": "원클릭 복구"
    },

    "테스트": {
        "카오스 엔지니어링": "주간 실행",
        "부하 테스트": "월간 실행",
        "복구 드릴": "분기별 실행",
        "전체 재해 시뮬레이션": "연간 실행"
    }
}`}</code>
            </pre>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">실전 복구 시나리오</h3>
            <div className="space-y-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">시나리오 1: 단일 서비스 장애</h4>
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 영향: LLM API 불가</li>
                  <li>• 조치: 백업 프로바이더로 자동 전환</li>
                  <li>• 복구 시간: 30초</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">시나리오 2: 전체 리전 장애</h4>
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 영향: Primary 리전 전체 다운</li>
                  <li>• 조치: Secondary 리전으로 완전 페일오버</li>
                  <li>• 복구 시간: 15분</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <h4 className="font-semibold mb-2">시나리오 3: 데이터 센터 재해</h4>
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                  <li>• 영향: 물리적 재해로 데이터센터 손실</li>
                  <li>• 조치: 지리적으로 분산된 백업에서 복구</li>
                  <li>• 복구 시간: 30분</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-green-800 dark:text-green-200 mb-3">Production 준비 체크리스트</h3>
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                <input type="checkbox" className="rounded" />
                <span>멀티 리전 인프라 구성 완료</span>
              </label>
              <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                <input type="checkbox" className="rounded" />
                <span>실시간 데이터 복제 설정</span>
              </label>
              <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                <input type="checkbox" className="rounded" />
                <span>자동 페일오버 스크립트 테스트</span>
              </label>
              <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                <input type="checkbox" className="rounded" />
                <span>모니터링 및 알림 시스템 구축</span>
              </label>
              <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                <input type="checkbox" className="rounded" />
                <span>복구 절차 문서화 및 교육</span>
              </label>
              <label className="flex items-center gap-2 text-green-700 dark:text-green-300">
                <input type="checkbox" className="rounded" />
                <span>정기 복구 훈련 일정 수립</span>
              </label>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 고가용성 & 복구 프레임워크',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'AWS Well-Architected Framework: Reliability',
                authors: 'Amazon Web Services',
                year: '2024',
                description: '고가용성 아키텍처 설계 - Multi-AZ, Auto Scaling, Disaster Recovery',
                link: 'https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/welcome.html'
              },
              {
                title: 'Google SRE Handbook: Eliminating Toil',
                authors: 'Google',
                year: '2024',
                description: 'SRE 원칙 - Error Budget, Incident Management, Postmortem Culture',
                link: 'https://sre.google/sre-book/table-of-contents/'
              },
              {
                title: 'Netflix Chaos Engineering',
                authors: 'Netflix',
                year: '2024',
                description: 'Chaos Monkey - 프로덕션 장애 시뮬레이션, 복원력 테스트',
                link: 'https://netflix.github.io/chaosmonkey/'
              },
              {
                title: 'Kubernetes High Availability',
                authors: 'CNCF',
                year: '2024',
                description: 'K8s HA 아키텍처 - Control Plane HA, Pod Disruption Budget, StatefulSet',
                link: 'https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/ha-topology/'
              },
              {
                title: 'Consul Service Mesh',
                authors: 'HashiCorp',
                year: '2024',
                description: '서비스 디스커버리 - Health Checking, Load Balancing, Failover',
                link: 'https://www.consul.io/'
              }
            ]
          },
          {
            title: '📖 복원력 & 재해복구 연구',
            icon: 'research' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'Patterns for Resilient Architecture',
                authors: 'Nygard, Release It!',
                year: '2024',
                description: '복원력 패턴 - Circuit Breaker, Bulkhead, Timeout, Retry, Fallback',
                link: 'https://www.amazon.com/Release-Design-Deploy-Production-Ready-Software/dp/1680502395'
              },
              {
                title: 'The Mathematics of Reliability',
                authors: 'Barlow & Proschan',
                year: '2024',
                description: '신뢰성 수학 - MTBF, MTTF, MTTR, Availability Calculation',
                link: 'https://doi.org/10.1137/1.9781611971194'
              },
              {
                title: 'Chaos Engineering: System Resiliency in Practice',
                authors: 'Rosenthal et al., O\'Reilly',
                year: '2024',
                description: '카오스 엔지니어링 - Failure Injection, Blast Radius, Steady State',
                link: 'https://www.oreilly.com/library/view/chaos-engineering/9781492043850/'
              },
              {
                title: 'Designing Data-Intensive Applications',
                authors: 'Kleppmann, O\'Reilly',
                year: '2024',
                description: '분산 시스템 복원력 - Replication, Partitioning, Transactions',
                link: 'https://dataintensive.net/'
              }
            ]
          },
          {
            title: '🛠️ 모니터링 & 복구 도구',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Prometheus + Grafana: Monitoring Stack',
                authors: 'CNCF',
                year: '2024',
                description: '메트릭 수집 & 시각화 - Alerting, Service Discovery, Time Series DB',
                link: 'https://prometheus.io/'
              },
              {
                title: 'PagerDuty: Incident Management',
                authors: 'PagerDuty',
                year: '2024',
                description: '인시던트 관리 - On-Call Scheduling, Escalation, Postmortem',
                link: 'https://www.pagerduty.com/'
              },
              {
                title: 'Datadog: Full-Stack Observability',
                authors: 'Datadog',
                year: '2024',
                description: '통합 모니터링 - APM, Logs, Metrics, Synthetic Monitoring',
                link: 'https://www.datadoghq.com/'
              },
              {
                title: 'Velero: Kubernetes Backup & Restore',
                authors: 'VMware',
                year: '2024',
                description: 'K8s 백업 - Disaster Recovery, Cluster Migration, Volume Snapshot',
                link: 'https://velero.io/'
              },
              {
                title: 'Gremlin: Chaos Engineering Platform',
                authors: 'Gremlin',
                year: '2024',
                description: '카오스 실험 플랫폼 - Controlled Failure Injection, Blast Radius Limiting',
                link: 'https://www.gremlin.com/'
              }
            ]
          }
        ]}
      />

      {/* Navigation */}
      <div className="flex justify-between items-center mt-12 pt-8 border-t border-gray-200 dark:border-gray-700">
        <Link
          href="/modules/rag/supplementary/chapter3"
          className="flex items-center gap-2 text-purple-600 hover:text-purple-700 transition-colors"
        >
          <ArrowLeft size={20} />
          이전: Cost Optimization
        </Link>

        <Link
          href="/modules/rag/supplementary"
          className="flex items-center gap-2 text-purple-600 hover:text-purple-700 transition-colors"
        >
          보충 과정 완료
          <CheckCircle2 size={20} />
        </Link>
      </div>
    </>
  )
}
