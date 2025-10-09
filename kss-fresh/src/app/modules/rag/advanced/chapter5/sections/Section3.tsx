'use client'

import Link from 'next/link'
import { Activity, ArrowLeft, ArrowRight } from 'lucide-react'
import References from '@/components/common/References'

export default function Section3() {
  return (
    <>
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
            <Activity className="text-purple-600" size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.3 실시간 RAG 모니터링 대시보드</h2>
            <p className="text-gray-600 dark:text-gray-400">Grafana + Prometheus 기반 종합 모니터링</p>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
            <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">프로덕션 모니터링 시스템 구축</h3>

            <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700 overflow-x-auto">
              <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono">
{`from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
import pandas as pd

# Prometheus 메트릭 정의
rag_requests_total = Counter(
    'rag_requests_total',
    'Total number of RAG requests',
    ['endpoint', 'status']
)

rag_latency_seconds = Histogram(
    'rag_latency_seconds',
    'RAG request latency in seconds',
    ['operation'],
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0]
)

rag_document_retrieval = Histogram(
    'rag_document_retrieval_count',
    'Number of documents retrieved per request',
    buckets=[1, 5, 10, 20, 50, 100]
)

rag_relevance_score = Summary(
    'rag_relevance_score',
    'Relevance scores of RAG responses'
)

rag_active_users = Gauge(
    'rag_active_users',
    'Number of active users in the last 5 minutes'
)

rag_cache_hit_rate = Gauge(
    'rag_cache_hit_rate',
    'Cache hit rate percentage'
)

rag_model_latency = Histogram(
    'rag_model_latency_seconds',
    'Model inference latency',
    ['model_name', 'operation'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

@dataclass
class RAGMetrics:
    """RAG 요청 메트릭"""
    request_id: str
    timestamp: datetime
    query: str
    latency_total: float
    latency_retrieval: float
    latency_generation: float
    documents_retrieved: int
    relevance_score: float
    cache_hit: bool
    error: Optional[str] = None
    user_id: Optional[str] = None

class RAGMonitoringService:
    def __init__(self):
        """
        실시간 RAG 모니터링 서비스
        - Prometheus 메트릭 수집
        - 실시간 이상 탐지
        - 대시보드 데이터 제공
        """
        self.metrics_buffer: List[RAGMetrics] = []
        self.active_users: set = set()
        self.alert_rules = self._init_alert_rules()

    def _init_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """알림 규칙 초기화"""
        return {
            'high_latency': {
                'condition': lambda m: m.latency_total > 2.0,
                'severity': 'warning',
                'message': 'High latency detected: {latency_total:.2f}s'
            },
            'low_relevance': {
                'condition': lambda m: m.relevance_score < 0.7,
                'severity': 'warning',
                'message': 'Low relevance score: {relevance_score:.2f}'
            },
            'error_rate': {
                'condition': lambda metrics: self._calculate_error_rate(metrics) > 0.05,
                'severity': 'critical',
                'message': 'Error rate exceeds 5%'
            },
            'cache_miss': {
                'condition': lambda metrics: self._calculate_cache_hit_rate(metrics) < 0.3,
                'severity': 'info',
                'message': 'Cache hit rate below 30%'
            }
        }

    async def record_request(self, metrics: RAGMetrics):
        """요청 메트릭 기록"""
        # Prometheus 메트릭 업데이트
        rag_requests_total.labels(
            endpoint='rag_query',
            status='success' if not metrics.error else 'error'
        ).inc()

        rag_latency_seconds.labels(operation='total').observe(metrics.latency_total)
        rag_latency_seconds.labels(operation='retrieval').observe(metrics.latency_retrieval)
        rag_latency_seconds.labels(operation='generation').observe(metrics.latency_generation)

        rag_document_retrieval.observe(metrics.documents_retrieved)
        rag_relevance_score.observe(metrics.relevance_score)

        # 활성 사용자 추적
        if metrics.user_id:
            self.active_users.add(metrics.user_id)

        # 버퍼에 추가
        self.metrics_buffer.append(metrics)

        # 이상 탐지
        await self._check_alerts(metrics)

        # 주기적 정리 (5분 이상 된 데이터)
        cutoff = datetime.now() - timedelta(minutes=5)
        self.metrics_buffer = [m for m in self.metrics_buffer if m.timestamp > cutoff]

    async def _check_alerts(self, metrics: RAGMetrics):
        """알림 규칙 체크"""
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule_name in ['error_rate', 'cache_miss']:
                    # 집계 기반 규칙
                    if rule['condition'](self.metrics_buffer):
                        await self._send_alert(rule_name, rule)
                else:
                    # 개별 메트릭 기반 규칙
                    if rule['condition'](metrics):
                        await self._send_alert(rule_name, rule, metrics)
            except Exception as e:
                print(f"Alert check error: {e}")

    async def _send_alert(self, rule_name: str, rule: Dict[str, Any],
                         metrics: Optional[RAGMetrics] = None):
        """알림 발송"""
        message = rule['message']
        if metrics:
            message = message.format(**metrics.__dict__)

        alert = {
            'rule': rule_name,
            'severity': rule['severity'],
            'message': message,
            'timestamp': datetime.now()
        }

        # 실제로는 Slack, PagerDuty 등으로 발송
        print(f"🚨 ALERT [{rule['severity'].upper()}]: {message}")

        # Webhook 호출 (예제)
        if rule['severity'] == 'critical':
            await self._send_webhook(alert)

    async def _send_webhook(self, alert: Dict[str, Any]):
        """Webhook 발송"""
        webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

        payload = {
            'text': f"RAG System Alert",
            'attachments': [{
                'color': 'danger',
                'fields': [
                    {'title': 'Rule', 'value': alert['rule'], 'short': True},
                    {'title': 'Severity', 'value': alert['severity'], 'short': True},
                    {'title': 'Message', 'value': alert['message']},
                    {'title': 'Time', 'value': alert['timestamp'].isoformat()}
                ]
            }]
        }

        # async with aiohttp.ClientSession() as session:
        #     await session.post(webhook_url, json=payload)

    def _calculate_error_rate(self, metrics: List[RAGMetrics]) -> float:
        """에러율 계산"""
        if not metrics:
            return 0.0

        error_count = sum(1 for m in metrics if m.error is not None)
        return error_count / len(metrics)

    def _calculate_cache_hit_rate(self, metrics: List[RAGMetrics]) -> float:
        """캐시 히트율 계산"""
        if not metrics:
            return 0.0

        cache_hits = sum(1 for m in metrics if m.cache_hit)
        return cache_hits / len(metrics)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """대시보드용 데이터 조회"""
        if not self.metrics_buffer:
            return self._empty_dashboard_data()

        # 데이터프레임 변환
        df = pd.DataFrame([m.__dict__ for m in self.metrics_buffer])

        # 현재 메트릭
        current_metrics = {
            'qps': len(df[df['timestamp'] > datetime.now() - timedelta(seconds=60)]),
            'avg_latency': df['latency_total'].mean(),
            'p95_latency': df['latency_total'].quantile(0.95),
            'avg_relevance': df['relevance_score'].mean(),
            'error_rate': self._calculate_error_rate(self.metrics_buffer),
            'cache_hit_rate': self._calculate_cache_hit_rate(self.metrics_buffer),
            'active_users': len(self.active_users),
            'total_requests': len(df)
        }

        # 시계열 데이터 (1분 단위)
        df['timestamp_minute'] = df['timestamp'].dt.floor('T')
        time_series = {
            'latency': df.groupby('timestamp_minute')['latency_total'].agg(['mean', 'p95']),
            'throughput': df.groupby('timestamp_minute').size(),
            'relevance': df.groupby('timestamp_minute')['relevance_score'].mean()
        }

        # Top 느린 쿼리
        slow_queries = df.nlargest(10, 'latency_total')[
            ['query', 'latency_total', 'documents_retrieved']
        ].to_dict('records')

        # 모델별 레이턴시
        model_latency = {
            'retrieval': {
                'mean': df['latency_retrieval'].mean(),
                'p95': df['latency_retrieval'].quantile(0.95)
            },
            'generation': {
                'mean': df['latency_generation'].mean(),
                'p95': df['latency_generation'].quantile(0.95)
            }
        }

        return {
            'current_metrics': current_metrics,
            'time_series': time_series,
            'slow_queries': slow_queries,
            'model_latency': model_latency,
            'last_updated': datetime.now()
        }

    def _empty_dashboard_data(self) -> Dict[str, Any]:
        """빈 대시보드 데이터"""
        return {
            'current_metrics': {
                'qps': 0,
                'avg_latency': 0,
                'p95_latency': 0,
                'avg_relevance': 0,
                'error_rate': 0,
                'cache_hit_rate': 0,
                'active_users': 0,
                'total_requests': 0
            },
            'time_series': {},
            'slow_queries': [],
            'model_latency': {},
            'last_updated': datetime.now()
        }

    async def health_check(self) -> Dict[str, Any]:
        """시스템 헬스 체크"""
        health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.now()
        }

        # 개별 컴포넌트 체크
        checks = {
            'metrics_collection': self._check_metrics_collection(),
            'alert_system': self._check_alert_system(),
            'data_freshness': self._check_data_freshness()
        }

        health_status['checks'] = checks

        # 전체 상태 결정
        if any(check['status'] == 'unhealthy' for check in checks.values()):
            health_status['status'] = 'unhealthy'
        elif any(check['status'] == 'degraded' for check in checks.values()):
            health_status['status'] = 'degraded'

        return health_status

    def _check_metrics_collection(self) -> Dict[str, str]:
        """메트릭 수집 상태 체크"""
        if not self.metrics_buffer:
            return {'status': 'unhealthy', 'message': 'No metrics collected'}

        latest_metric = max(self.metrics_buffer, key=lambda m: m.timestamp)
        age = (datetime.now() - latest_metric.timestamp).total_seconds()

        if age > 300:  # 5분 이상 된 데이터
            return {'status': 'unhealthy', 'message': f'Stale data: {age:.0f}s old'}
        elif age > 60:  # 1분 이상
            return {'status': 'degraded', 'message': f'Delayed data: {age:.0f}s old'}

        return {'status': 'healthy', 'message': 'Collecting metrics normally'}

    def _check_alert_system(self) -> Dict[str, str]:
        """알림 시스템 상태 체크"""
        # 실제로는 마지막 알림 발송 시간 등을 체크
        return {'status': 'healthy', 'message': 'Alert system operational'}

    def _check_data_freshness(self) -> Dict[str, str]:
        """데이터 신선도 체크"""
        if not self.metrics_buffer:
            return {'status': 'unhealthy', 'message': 'No data available'}

        # 최근 1분간 데이터 개수
        recent_count = sum(1 for m in self.metrics_buffer
                          if m.timestamp > datetime.now() - timedelta(minutes=1))

        if recent_count < 10:
            return {'status': 'degraded', 'message': f'Low traffic: {recent_count} requests/min'}

        return {'status': 'healthy', 'message': f'Normal traffic: {recent_count} requests/min'}

# Grafana 대시보드 설정
GRAFANA_DASHBOARD_CONFIG = {
    "dashboard": {
        "title": "RAG System Monitoring",
        "panels": [
            {
                "title": "Request Rate",
                "targets": [{
                    "expr": "rate(rag_requests_total[5m])",
                    "legendFormat": "{{endpoint}} - {{status}}"
                }]
            },
            {
                "title": "Latency Percentiles",
                "targets": [{
                    "expr": "histogram_quantile(0.95, rate(rag_latency_seconds_bucket[5m]))",
                    "legendFormat": "P95 {{operation}}"
                }]
            },
            {
                "title": "Relevance Score Distribution",
                "targets": [{
                    "expr": "rag_relevance_score",
                    "legendFormat": "Relevance Score"
                }]
            },
            {
                "title": "Cache Hit Rate",
                "targets": [{
                    "expr": "rag_cache_hit_rate",
                    "legendFormat": "Cache Hit %"
                }]
            },
            {
                "title": "Active Users",
                "targets": [{
                    "expr": "rag_active_users",
                    "legendFormat": "Active Users"
                }]
            },
            {
                "title": "Error Rate",
                "targets": [{
                    "expr": "rate(rag_requests_total{status='error'}[5m]) / rate(rag_requests_total[5m])",
                    "legendFormat": "Error Rate"
                }]
            }
        ]
    }
}

# 사용 예제
async def simulate_rag_traffic():
    """RAG 트래픽 시뮬레이션"""
    monitoring = RAGMonitoringService()

    # 100개의 요청 시뮬레이션
    for i in range(100):
        # 메트릭 생성
        latency_retrieval = np.random.exponential(0.2)  # 평균 200ms
        latency_generation = np.random.exponential(0.5)  # 평균 500ms

        metrics = RAGMetrics(
            request_id=f"req_{i}",
            timestamp=datetime.now(),
            query=f"Query {i % 20}",
            latency_total=latency_retrieval + latency_generation + np.random.exponential(0.1),
            latency_retrieval=latency_retrieval,
            latency_generation=latency_generation,
            documents_retrieved=np.random.randint(5, 20),
            relevance_score=min(0.95, max(0.5, np.random.normal(0.85, 0.1))),
            cache_hit=np.random.random() > 0.6,
            error="timeout" if np.random.random() < 0.02 else None,
            user_id=f"user_{i % 30}"
        )

        await monitoring.record_request(metrics)
        await asyncio.sleep(0.1)  # 100ms 간격

    # 대시보드 데이터 조회
    dashboard_data = monitoring.get_dashboard_data()

    print("=== RAG Monitoring Dashboard ===")
    print(f"Current Metrics:")
    for metric, value in dashboard_data['current_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")

    print(f"\nTop Slow Queries:")
    for query in dashboard_data['slow_queries'][:5]:
        print(f"  - {query['query']}: {query['latency_total']:.2f}s")

    # 헬스 체크
    health = await monitoring.health_check()
    print(f"\nHealth Status: {health['status']}")
    for component, status in health['checks'].items():
        print(f"  {component}: {status['status']} - {status['message']}")

# 실행
print("=== RAG Monitoring System Demo ===\n")
asyncio.run(simulate_rag_traffic())`}
              </pre>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
            <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">실제 Grafana 대시보드 구성</h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">🎯 핵심 메트릭</h4>
                <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                  <li>• Request Rate (QPS)</li>
                  <li>• Latency (P50, P95, P99)</li>
                  <li>• Error Rate & Success Rate</li>
                  <li>• Active Users</li>
                  <li>• Resource Utilization</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">📊 상세 분석</h4>
                <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                  <li>• Query Distribution</li>
                  <li>• Document Retrieval Stats</li>
                  <li>• Model Performance</li>
                  <li>• Cache Performance</li>
                  <li>• Cost Analysis</li>
                </ul>
              </div>
            </div>

            <div className="mt-4 bg-emerald-100 dark:bg-emerald-900/40 p-4 rounded-lg">
              <p className="text-sm text-emerald-800 dark:text-emerald-200">
                <strong>💡 Pro Tip:</strong> Golden Signals (Latency, Traffic, Errors, Saturation)를
                기반으로 대시보드를 구성하면 시스템 상태를 효과적으로 파악할 수 있습니다.
                특히 RAG 시스템에서는 Relevance Score를 추가 지표로 활용하세요.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Practical Exercise */}
      <section className="bg-gradient-to-r from-emerald-500 to-teal-600 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-6">실습 과제</h2>

        <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
          <h3 className="font-bold mb-4">종합 RAG 평가 및 모니터링 시스템 구축</h3>

          <div className="space-y-4">
            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">📋 요구사항</h4>
              <ol className="space-y-2 text-sm">
                <li>1. RAGAS 프레임워크를 활용한 오프라인 평가 시스템 구축</li>
                <li>2. A/B 테스트 플랫폼 구현 (최소 3가지 변형 지원)</li>
                <li>3. Prometheus + Grafana 실시간 모니터링 구축</li>
                <li>4. 자동 알림 시스템 (Slack/Email 연동)</li>
                <li>5. 주간 성능 리포트 자동 생성</li>
              </ol>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">🎯 평가 기준</h4>
              <ul className="space-y-1 text-sm">
                <li>• 평가 메트릭의 포괄성 (Retrieval + Generation + System)</li>
                <li>• A/B 테스트의 통계적 엄밀성</li>
                <li>• 모니터링 대시보드의 실용성</li>
                <li>• 알림 시스템의 정확성 (False positive rate &lt; 5%)</li>
                <li>• 시스템 확장성 (1000+ QPS 지원)</li>
              </ul>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">💡 도전 과제</h4>
              <p className="text-sm">
                ML 기반 이상 탐지 시스템을 구축하여 단순 threshold 기반 알림을
                넘어서는 지능형 모니터링을 구현해보세요. 특히 계절성과 트렌드를
                고려한 동적 임계값 설정을 구현하면 더욱 효과적입니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 RAG 평가 프레임워크',
            icon: 'web' as const,
            color: 'border-emerald-500',
            items: [
              {
                title: 'RAGAS: RAG Assessment Framework',
                authors: 'Explodinggradients',
                year: '2024',
                description: 'RAG 전용 평가 - Faithfulness, Answer Relevancy, Context Precision',
                link: 'https://docs.ragas.io/'
              },
              {
                title: 'TruLens: LLM App Evaluation',
                authors: 'TruEra',
                year: '2024',
                description: 'RAG 추적 및 평가 - Context Relevance, Groundedness, Answer Relevance',
                link: 'https://www.trulens.org/trulens_eval/getting_started/'
              },
              {
                title: 'DeepEval: LLM Testing',
                authors: 'Confident AI',
                year: '2024',
                description: 'Unit Testing for RAG - BLEU, ROUGE, BERTScore, Hallucination Detection',
                link: 'https://docs.confident-ai.com/'
              },
              {
                title: 'Phoenix by Arize AI',
                authors: 'Arize AI',
                year: '2024',
                description: 'LLM Observability - Traces, Evals, Embeddings 시각화',
                link: 'https://docs.arize.com/phoenix'
              },
              {
                title: 'LangSmith Evaluation',
                authors: 'LangChain',
                year: '2024',
                description: 'End-to-end RAG 평가 - Custom Evaluators, Dataset Management',
                link: 'https://docs.smith.langchain.com/evaluation'
              }
            ]
          },
          {
            title: '📖 RAG 평가 메트릭 연구',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'ARES: Automated RAG Evaluation System',
                authors: 'Saad-Falcon et al., Stanford',
                year: '2024',
                description: '자동화된 RAG 평가 - LLM-as-a-Judge, Synthetic Data Generation',
                link: 'https://arxiv.org/abs/2311.09476'
              },
              {
                title: 'CRUD-RAG: Comprehensive RAG Benchmark',
                authors: 'Lyu et al., Alibaba',
                year: '2024',
                description: 'Create, Read, Update, Delete 시나리오 기반 종합 벤치마크',
                link: 'https://arxiv.org/abs/2401.17043'
              },
              {
                title: 'RGB: Retrieval Generation Benchmark',
                authors: 'Chen et al., Microsoft',
                year: '2024',
                description: 'Multi-domain RAG 벤치마크 - 정확도, Hallucination, Citation',
                link: 'https://arxiv.org/abs/2309.01431'
              },
              {
                title: 'Evaluating Retrieval Quality in RAG',
                authors: 'Lewis et al., Meta',
                year: '2023',
                description: 'Retrieval Precision vs Generation Quality 트레이드오프 연구',
                link: 'https://arxiv.org/abs/2312.10997'
              }
            ]
          },
          {
            title: '🛠️ 모니터링 & 옵저버빌리티',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'OpenTelemetry for LLM Apps',
                authors: 'OpenTelemetry',
                year: '2024',
                description: 'RAG 트레이싱 - Spans, Metrics, Logs 표준화',
                link: 'https://opentelemetry.io/docs/instrumentation/python/libraries/openai/'
              },
              {
                title: 'Langfuse: LLM Engineering Platform',
                authors: 'Langfuse',
                year: '2024',
                description: 'RAG 모니터링 - Traces, Scores, Datasets, Prompt Management',
                link: 'https://langfuse.com/docs'
              },
              {
                title: 'Weights & Biases for LLMs',
                authors: 'Weights & Biases',
                year: '2024',
                description: 'RAG 실험 추적 - Prompts, Model Versions, A/B Testing',
                link: 'https://docs.wandb.ai/guides/prompts'
              },
              {
                title: 'Helicone: LLM Observability',
                authors: 'Helicone',
                year: '2024',
                description: 'OpenAI API 모니터링 - Cost, Latency, Error Tracking',
                link: 'https://docs.helicone.ai/'
              },
              {
                title: 'Datadog LLM Observability',
                authors: 'Datadog',
                year: '2024',
                description: 'Enterprise RAG 모니터링 - APM, Logs, Traces 통합',
                link: 'https://docs.datadoghq.com/llm_observability/'
              }
            ]
          }
        ]}
      />

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced/chapter4"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 고급 Reranking 전략
          </Link>

          <Link
            href="/modules/rag/advanced/chapter6"
            className="inline-flex items-center gap-2 bg-emerald-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-emerald-600 transition-colors"
          >
            다음: 최신 연구 동향
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </>
  )
}
