'use client'

import { Gauge } from 'lucide-react'

export default function Section5() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-amber-100 dark:bg-amber-900/20 flex items-center justify-center">
          <Gauge className="text-amber-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">4.5 성능 프로파일링 및 메모리 관리</h2>
          <p className="text-gray-600 dark:text-gray-400">실시간 성능 모니터링과 최적화</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-amber-50 dark:bg-amber-900/20 p-6 rounded-xl border border-amber-200 dark:border-amber-700">
          <h3 className="font-bold text-amber-800 dark:text-amber-200 mb-4">통합 성능 모니터링 시스템</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>프로덕션 RAG 시스템은 24/7 모니터링이 필수입니다.</strong>
              성능 저하나 오류를 실시간으로 감지하고 자동으로 알림을 받아야 합니다.
              아래의 통합 모니터링 시스템은 RAG에 특화된 메트릭을 추적합니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>핵심 모니터링 지표:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>응답 시간</strong>: P50, P95, P99 백분위 추적</li>
              <li><strong>캐시 히트율</strong>: 캐싱 전략의 효과성 측정</li>
              <li><strong>메모리/CPU 사용량</strong>: 리소스 병목 현상 조기 발견</li>
              <li><strong>에러율</strong>: 검색 실패, 타임아웃 등 추적</li>
              <li><strong>큐 길이</strong>: 시스템 부하 상태 파악</li>
            </ul>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <p className="text-2xl font-bold text-blue-600">2.3초</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">평균 응답시간</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <p className="text-2xl font-bold text-green-600">85%</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">캐시 히트율</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <p className="text-2xl font-bold text-purple-600">512MB</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">메모리 사용량</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
              <p className="text-2xl font-bold text-red-600">99.2%</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">가용성</p>
            </div>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`import psutil
import time
import threading
from dataclasses import dataclass
from typing import Dict, List
import json
from datetime import datetime, timedelta

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    active_connections: int
    queue_length: int

class RAGPerformanceMonitor:
    def __init__(self, log_file: str = "rag_performance.log"):
        self.log_file = log_file
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = 1000
        self.monitoring = False
        self.alert_thresholds = {
            'response_time': 5.0,      # 5초 이상
            'memory_usage': 1000.0,    # 1GB 이상
            'cpu_usage': 80.0,         # 80% 이상
            'cache_hit_rate': 0.7      # 70% 미만
        }

    def start_monitoring(self, interval: float = 10.0):
        """백그라운드 모니터링 시작"""
        self.monitoring = True
        monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        monitor_thread.start()
        print(f"성능 모니터링 시작 (간격: {interval}초)")

    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        print("성능 모니터링 중지")

    def _monitor_loop(self, interval: float):
        """모니터링 루프"""
        while self.monitoring:
            try:
                metrics = self.collect_metrics()
                self.record_metrics(metrics)
                self.check_alerts(metrics)
                time.sleep(interval)
            except Exception as e:
                print(f"모니터링 오류: {e}")
                time.sleep(interval)

    def collect_metrics(self) -> PerformanceMetrics:
        """현재 시스템 메트릭 수집"""
        # 시스템 리소스
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        # 프로세스별 메모리 (현재 프로세스)
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024**2)  # MB

        # RAG 특화 메트릭 (예시 - 실제로는 RAG 시스템에서 수집)
        cache_stats = self.get_cache_stats()
        response_stats = self.get_response_stats()

        return PerformanceMetrics(
            timestamp=datetime.now(),
            response_time=response_stats.get('avg_response_time', 0),
            memory_usage_mb=process_memory,
            cpu_usage_percent=cpu_percent,
            cache_hit_rate=cache_stats.get('hit_rate', 0),
            active_connections=response_stats.get('active_connections', 0),
            queue_length=response_stats.get('queue_length', 0)
        )

    def record_metrics(self, metrics: PerformanceMetrics):
        """메트릭 기록"""
        # 메모리 히스토리 관리
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)

        # 파일 로깅
        with open(self.log_file, 'a', encoding='utf-8') as f:
            log_entry = {
                'timestamp': metrics.timestamp.isoformat(),
                'response_time': metrics.response_time,
                'memory_usage_mb': metrics.memory_usage_mb,
                'cpu_usage_percent': metrics.cpu_usage_percent,
                'cache_hit_rate': metrics.cache_hit_rate,
                'active_connections': metrics.active_connections,
                'queue_length': metrics.queue_length
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\\n')

    def check_alerts(self, metrics: PerformanceMetrics):
        """임계값 기반 알림 확인"""
        alerts = []

        if metrics.response_time > self.alert_thresholds['response_time']:
            alerts.append(f"응답 시간 임계값 초과: {metrics.response_time:.2f}초")

        if metrics.memory_usage_mb > self.alert_thresholds['memory_usage']:
            alerts.append(f"메모리 사용량 임계값 초과: {metrics.memory_usage_mb:.1f}MB")

        if metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage']:
            alerts.append(f"CPU 사용률 임계값 초과: {metrics.cpu_usage_percent:.1f}%")

        if metrics.cache_hit_rate < self.alert_thresholds['cache_hit_rate']:
            alerts.append(f"캐시 히트율 임계값 미달: {metrics.cache_hit_rate:.2%}")

        if alerts:
            self.send_alerts(alerts)

    def send_alerts(self, alerts: List[str]):
        """알림 발송"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\\n🚨 [{timestamp}] 성능 알림:")
        for alert in alerts:
            print(f"  - {alert}")

    def get_performance_report(self, hours: int = 24) -> Dict:
        """성능 보고서 생성"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {"error": "충분한 데이터가 없습니다"}

        # 통계 계산
        response_times = [m.response_time for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
        cache_hits = [m.cache_hit_rate for m in recent_metrics]

        return {
            "period_hours": hours,
            "total_samples": len(recent_metrics),
            "response_time": {
                "avg": sum(response_times) / len(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "p95": sorted(response_times)[int(len(response_times) * 0.95)]
            },
            "memory_usage_mb": {
                "avg": sum(memory_usage) / len(memory_usage),
                "min": min(memory_usage),
                "max": max(memory_usage)
            },
            "cpu_usage_percent": {
                "avg": sum(cpu_usage) / len(cpu_usage),
                "max": max(cpu_usage)
            },
            "cache_hit_rate": {
                "avg": sum(cache_hits) / len(cache_hits),
                "min": min(cache_hits)
            }
        }

    def optimize_recommendations(self) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        recent_metrics = self.metrics_history[-10:]  # 최근 10개 샘플

        if not recent_metrics:
            return ["데이터가 부족합니다"]

        avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)

        if avg_response_time > 3.0:
            recommendations.append("응답 시간 개선을 위해 캐싱 전략을 검토하세요")

        if avg_cache_hit_rate < 0.8:
            recommendations.append("캐시 히트율 향상을 위해 캐시 크기를 늘리거나 TTL을 조정하세요")

        if avg_memory > 800:
            recommendations.append("메모리 사용량이 높습니다. 모델 양자화를 고려하세요")

        return recommendations if recommendations else ["현재 성능이 양호합니다"]

# 사용 예시
monitor = RAGPerformanceMonitor()

# 모니터링 시작
monitor.start_monitoring(interval=30)  # 30초마다

# 실시간 메트릭 수집
current_metrics = monitor.collect_metrics()
print(f"현재 응답 시간: {current_metrics.response_time:.2f}초")
print(f"메모리 사용량: {current_metrics.memory_usage_mb:.1f}MB")

# 성능 보고서
report = monitor.get_performance_report(hours=24)
print("\\n24시간 성능 보고서:")
print(json.dumps(report, indent=2, ensure_ascii=False))

# 최적화 권장사항
recommendations = monitor.optimize_recommendations()
print("\\n최적화 권장사항:")
for rec in recommendations:
    print(f"- {rec}")

# 모니터링 중지
# monitor.stop_monitoring()`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
