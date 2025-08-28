import { BarChart } from 'lucide-react'

export default function Section1MonitoringSystem() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-emerald-100 dark:bg-emerald-900/20 flex items-center justify-center">
          <BarChart className="text-emerald-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.1 RAG 시스템 모니터링</h2>
          <p className="text-gray-600 dark:text-gray-400">메트릭 기반 시스템 성능 추적</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-emerald-50 dark:bg-emerald-900/20 p-6 rounded-xl border border-emerald-200 dark:border-emerald-700">
          <h3 className="font-bold text-emerald-800 dark:text-emerald-200 mb-4">통합 모니터링 시스템</h3>
          
          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>Production RAG 시스템의 모니터링은 시스템 안정성과 사용자 경험을 보장하는 핵심 요소입니다.</strong> 
              다층적 메트릭 수집을 통해 성능 병목 지점을 사전에 발견하고, 실시간 알림 시스템으로 
              장애 발생 시 평균 복구 시간(MTTR)을 15분 이내로 단축할 수 있습니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>핵심 모니터링 영역:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>성능 메트릭</strong>: P50/P95/P99 응답 시간, 처리량(QPS), 에러율</li>
              <li><strong>품질 메트릭</strong>: 검색 정확도, 답변 관련성, 사용자 만족도</li>
              <li><strong>인프라 메트릭</strong>: CPU/Memory/GPU 사용률, 네트워크 I/O</li>
              <li><strong>비즈니스 메트릭</strong>: 사용자 참여도, 전환율, 수익 기여도</li>
            </ul>
            
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-700 mt-4">
              <h4 className="font-bold text-green-800 dark:text-green-200 mb-2">🎯 운영 성과 지표</h4>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-green-800 dark:text-green-200">Netflix 수준 SLA</strong>
                  <ul className="list-disc list-inside ml-2 text-green-700 dark:text-green-300 mt-1">
                    <li>99.99% 가용성 (연간 52분 다운타임)</li>
                    <li>P95 응답시간 &lt; 1.5초</li>
                    <li>에러율 &lt; 0.01%</li>
                    <li>평균 복구시간 &lt; 5분</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-green-800 dark:text-green-200">Google 수준 확장성</strong>
                  <ul className="list-disc list-inside ml-2 text-green-700 dark:text-green-300 mt-1">
                    <li>초당 10,000 쿼리 처리</li>
                    <li>자동 스케일링 30초 내 완료</li>
                    <li>글로벌 멀티 리전 배포</li>
                    <li>실시간 A/B 테스트 지원</li>
                  </ul>
                </div>
              </div>
            </div>
            
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-700 mt-4">
              <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-2">📊 프로덕션 모니터링 스택</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-blue-300 dark:border-blue-600">
                      <th className="text-left py-2 text-blue-800 dark:text-blue-200">계층</th>
                      <th className="text-left py-2 text-blue-800 dark:text-blue-200">도구</th>
                      <th className="text-left py-2 text-blue-800 dark:text-blue-200">용도</th>
                      <th className="text-left py-2 text-blue-800 dark:text-blue-200">보존기간</th>
                    </tr>
                  </thead>
                  <tbody className="text-blue-700 dark:text-blue-300">
                    <tr>
                      <td className="py-1">수집</td>
                      <td className="py-1">Prometheus</td>
                      <td className="py-1">메트릭 수집/저장</td>
                      <td className="py-1">90일</td>
                    </tr>
                    <tr>
                      <td className="py-1">시각화</td>
                      <td className="py-1">Grafana</td>
                      <td className="py-1">대시보드/차트</td>
                      <td className="py-1">실시간</td>
                    </tr>
                    <tr>
                      <td className="py-1">알림</td>
                      <td className="py-1">AlertManager</td>
                      <td className="py-1">임계값 기반 알림</td>
                      <td className="py-1">7일</td>
                    </tr>
                    <tr>
                      <td className="py-1">로그</td>
                      <td className="py-1">ELK Stack</td>
                      <td className="py-1">로그 분석/검색</td>
                      <td className="py-1">30일</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          
          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
import asyncio
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

class MetricType(Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"

@dataclass
class RAGMetrics:
    """RAG 시스템 핵심 메트릭"""
    # 성능 메트릭
    response_time: float
    retrieval_time: float
    generation_time: float
    total_request_time: float
    
    # 품질 메트릭
    retrieval_accuracy: float
    relevance_score: float
    answer_completeness: float
    user_satisfaction: Optional[float] = None
    
    # 시스템 메트릭
    cache_hit_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float = 0.0
    
    # 비즈니스 메트릭
    requests_per_minute: int
    unique_users: int
    error_rate: float
    
    # 메타데이터
    timestamp: datetime = datetime.now()
    model_version: str = "v1.0"
    query_type: str = "general"

class RAGMonitoringSystem:
    def __init__(self, service_name: str = "rag-system"):
        self.service_name = service_name
        self.registry = CollectorRegistry()
        
        # Prometheus 메트릭 정의
        self.request_counter = Counter(
            'rag_requests_total',
            'Total RAG requests',
            ['query_type', 'status'],
            registry=self.registry
        )
        
        self.response_time_histogram = Histogram(
            'rag_response_time_seconds',
            'RAG response time',
            ['query_type'],
            registry=self.registry
        )
        
        self.retrieval_accuracy_gauge = Gauge(
            'rag_retrieval_accuracy',
            'Current retrieval accuracy',
            registry=self.registry
        )
        
        self.cache_hit_rate_gauge = Gauge(
            'rag_cache_hit_rate',
            'Cache hit rate',
            registry=self.registry
        )
        
        # 메트릭 저장소
        self.metrics_history: List[RAGMetrics] = []
        self.alerts_config = {
            'response_time_threshold': 5.0,  # 5초
            'error_rate_threshold': 0.05,    # 5%
            'cache_hit_rate_threshold': 0.7,  # 70%
            'cpu_usage_threshold': 80.0       # 80%
        }
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"{service_name}-monitor")
    
    def record_request(self, metrics: RAGMetrics, status: str = "success"):
        """요청 메트릭 기록"""
        # Prometheus 메트릭 업데이트
        self.request_counter.labels(
            query_type=metrics.query_type,
            status=status
        ).inc()
        
        self.response_time_histogram.labels(
            query_type=metrics.query_type
        ).observe(metrics.response_time)
        
        self.retrieval_accuracy_gauge.set(metrics.retrieval_accuracy)
        self.cache_hit_rate_gauge.set(metrics.cache_hit_rate)
        
        # 히스토리 저장 (최근 1000개만 유지)
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        # 알림 확인
        self._check_alerts(metrics)
        
        # 로깅
        self.logger.info(f"Request processed: {asdict(metrics)}")
    
    def _check_alerts(self, metrics: RAGMetrics):
        """임계값 기반 알림 확인"""
        alerts = []
        
        if metrics.response_time > self.alerts_config['response_time_threshold']:
            alerts.append(f"High response time: {metrics.response_time:.2f}s")
        
        if metrics.error_rate > self.alerts_config['error_rate_threshold']:
            alerts.append(f"High error rate: {metrics.error_rate:.2%}")
        
        if metrics.cache_hit_rate < self.alerts_config['cache_hit_rate_threshold']:
            alerts.append(f"Low cache hit rate: {metrics.cache_hit_rate:.2%}")
        
        if metrics.cpu_usage_percent > self.alerts_config['cpu_usage_threshold']:
            alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        for alert in alerts:
            self.logger.warning(f"ALERT: {alert}")
            # 실제 환경에서는 Slack, PagerDuty 등으로 알림 발송
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """성능 요약 정보 반환"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff]
        
        if not recent_metrics:
            return {}
        
        response_times = [m.response_time for m in recent_metrics]
        accuracies = [m.retrieval_accuracy for m in recent_metrics]
        error_rates = [m.error_rate for m in recent_metrics]
        
        return {
            "period_hours": hours,
            "total_requests": len(recent_metrics),
            "avg_response_time": sum(response_times) / len(response_times),
            "p95_response_time": sorted(response_times)[int(0.95 * len(response_times))],
            "avg_accuracy": sum(accuracies) / len(accuracies),
            "avg_error_rate": sum(error_rates) / len(error_rates),
            "last_updated": datetime.now().isoformat()
        }
    
    def export_metrics(self) -> str:
        """Prometheus 메트릭 형식으로 내보내기"""
        return generate_latest(self.registry)

# 간단한 웹 대시보드 (실제로는 Grafana 사용 권장)
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

class MonitoringDashboard(BaseHTTPRequestHandler):
    def __init__(self, monitor: RAGMonitoringSystem, *args, **kwargs):
        self.monitor = monitor
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(self.monitor.export_metrics().encode())
        elif self.path == '/summary':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            summary = self.monitor.get_performance_summary()
            self.wfile.write(json.dumps(summary, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()

# 사용 예제
async def main():
    # 모니터링 시스템 초기화
    monitor = RAGMonitoringSystem("production-rag")
    
    # 대시보드 서버 시작 (별도 스레드)
    def create_handler(*args, **kwargs):
        return MonitoringDashboard(monitor, *args, **kwargs)
    
    server = HTTPServer(('localhost', 8000), create_handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # 테스트 쿼리 실행
    test_queries = [
        ("What is machine learning?", "educational"),
        ("How to implement RAG?", "technical"),
        ("Product pricing information", "business")
    ]
    
    # 요청 처리 및 모니터링
    for query, query_type in test_queries:
        try:
            result = await rag_system.process_query(query, query_type)
            print(f"Query processed: {query[:30]}...")
        except Exception as e:
            print(f"Query failed: {e}")
        
        # 잠시 대기
        await asyncio.sleep(1)
    
    # 성능 요약 출력
    summary = rag_system.monitor.get_performance_summary(hours=1)
    print("\\nPerformance Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # 대시보드 유지 (실제로는 서비스로 실행)
    print("\\nMonitoring dashboard running on http://localhost:8000")
    print("  - Metrics: http://localhost:8000/metrics")
    print("  - Summary: http://localhost:8000/summary")
    
    # 무한 대기 (실제 서비스에서)
    # try:
    #     while True:
    #         await asyncio.sleep(60)
    # except KeyboardInterrupt:
    #     await dashboard.cleanup()

# 실행
# asyncio.run(main())`}
            </pre>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">핵심 모니터링 메트릭</h3>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <h4 className="font-medium text-gray-900 dark:text-white">🚀 성능 메트릭</h4>
              <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                <li>• 전체 응답 시간 (P50, P95, P99)</li>
                <li>• 검색 단계 응답 시간</li>
                <li>• 생성 단계 응답 시간</li>
                <li>• 처리량 (RPS - Requests Per Second)</li>
              </ul>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-medium text-gray-900 dark:text-white">📊 품질 메트릭</h4>
              <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
                <li>• 검색 정확도 (Retrieval Accuracy)</li>
                <li>• 답변 관련성 점수</li>
                <li>• 사용자 만족도 평가</li>
                <li>• 답변 완성도 및 일관성</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}