'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Server } from 'lucide-react'
import dynamic from 'next/dynamic'
import References from '@/components/common/References'

// 동적 임포트로 섹션 컴포넌트들 로드 (성능 최적화)
const Section1MonitoringSystem = dynamic(() => import('./components/Section1MonitoringSystem'), { ssr: false })
const Section2ABTesting = dynamic(() => import('./components/Section2ABTesting'), { ssr: false })
const Section3SecurityPrivacy = dynamic(() => import('./components/Section3SecurityPrivacy'), { ssr: false })
const Section4ScalingStrategies = dynamic(() => import('./components/Section4ScalingStrategies'), { ssr: false })
const Section5APIDeployment = dynamic(() => import('./components/Section5APIDeployment'), { ssr: false })
const Section6MonitoringDashboards = dynamic(() => import('./components/Section6MonitoringDashboards'), { ssr: false })

export default function Chapter6Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/intermediate"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          중급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Server size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 6: Production RAG Systems</h1>
              <p className="text-emerald-100 text-lg">실제 운영 환경에서의 RAG 시스템 구축 및 관리</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content - 각 섹션을 별도 컴포넌트로 분리 */}
      <div className="space-y-8">
        <Section1MonitoringSystem />
        <Section2ABTesting />
        <Section3SecurityPrivacy />
        <Section4ScalingStrategies />
        <Section5APIDeployment />
        <Section6MonitoringDashboards />
      </div>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 프로덕션 RAG 배포 & 운영',
            icon: 'web' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'Docker & Kubernetes for ML',
                authors: 'Kubernetes',
                year: '2025',
                description: 'RAG 컨테이너화 & 오케스트레이션 - K8s 스케일링',
                link: 'https://kubernetes.io/docs/tutorials/stateless-application/'
              },
              {
                title: 'AWS SageMaker Deployment',
                authors: 'AWS',
                year: '2025',
                description: 'LLM 엔드포인트 배포 - 오토스케일링, A/B 테스트',
                link: 'https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html'
              },
              {
                title: 'LangSmith Production Monitoring',
                authors: 'LangChain',
                year: '2025',
                description: 'RAG 체인 추적 - 성능, 비용, 품질 모니터링',
                link: 'https://docs.smith.langchain.com/deployment'
              },
              {
                title: 'MLOps for LLMs',
                authors: 'Google Cloud',
                year: '2025',
                description: 'Vertex AI로 RAG 배포 - CI/CD 파이프라인',
                link: 'https://cloud.google.com/vertex-ai/docs/pipelines/introduction'
              },
              {
                title: 'FastAPI Production Guide',
                authors: 'FastAPI',
                year: '2025',
                description: 'RAG API 서버 - 로드밸런싱, 캐싱, CORS',
                link: 'https://fastapi.tiangolo.com/deployment/'
              }
            ]
          },
          {
            title: '📖 보안, 모니터링 & A/B 테스트',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'OWASP Top 10 for LLM Applications',
                authors: 'OWASP Foundation',
                year: '2024',
                description: 'LLM 보안 위협 - 프롬프트 인젝션, 데이터 유출 방어',
                link: 'https://owasp.org/www-project-top-10-for-large-language-model-applications/'
              },
              {
                title: 'Guardrails AI',
                authors: 'Guardrails',
                year: '2024',
                description: 'LLM 출력 검증 - PII 마스킹, 독성 필터링',
                link: 'https://docs.guardrailsai.com/'
              },
              {
                title: 'A/B Testing for ML Systems',
                authors: 'Netflix',
                year: '2023',
                description: '대규모 실험 설계 - 통계적 유의성, 다변량 테스트',
                link: 'https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15'
              },
              {
                title: 'Observability for ML',
                authors: 'Arize AI',
                year: '2024',
                description: 'RAG 드리프트 감지 - 데이터/모델/개념 드리프트',
                link: 'https://docs.arize.com/arize/'
              }
            ]
          },
          {
            title: '🛠️ 스케일링 & 성능 최적화',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Nginx for RAG APIs',
                authors: 'NGINX',
                year: '2025',
                description: '리버스 프록시 - 로드밸런싱, 속도 제한, SSL',
                link: 'https://docs.nginx.com/nginx/admin-guide/load-balancer/http-load-balancer/'
              },
              {
                title: 'Redis for LLM Caching',
                authors: 'Redis Labs',
                year: '2025',
                description: 'LLM 응답 캐싱 - 비용 80% 절감 사례',
                link: 'https://redis.io/docs/manual/patterns/distributed-locks/'
              },
              {
                title: 'Grafana + Prometheus',
                authors: 'Grafana Labs',
                year: '2025',
                description: 'RAG 메트릭 대시보드 - 실시간 알람, 시각화',
                link: 'https://grafana.com/docs/grafana/latest/getting-started/get-started-grafana-prometheus/'
              },
              {
                title: 'Sentry Error Tracking',
                authors: 'Sentry',
                year: '2025',
                description: 'RAG 에러 모니터링 - 스택 트레이스, 사용자 영향',
                link: 'https://docs.sentry.io/'
              },
              {
                title: 'Load Testing with Locust',
                authors: 'Locust',
                year: '2024',
                description: 'RAG API 부하 테스트 - TPS, 응답시간 벤치마크',
                link: 'https://docs.locust.io/en/stable/'
              }
            ]
          }
        ]}
      />

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/intermediate/chapter5"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 멀티모달 RAG
          </Link>
          
          <Link
            href="/modules/rag/intermediate"
            className="inline-flex items-center gap-2 bg-emerald-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-emerald-600 transition-colors"
          >
            중급 과정 완료!
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}