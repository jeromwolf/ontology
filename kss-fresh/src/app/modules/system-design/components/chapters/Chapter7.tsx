'use client'

import React from 'react'
import { 
  Activity, Cpu, BookOpen, Network
} from 'lucide-react'

export default function Chapter7() {
  return (
    <div className="space-y-8">
      {/* Observability */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Activity className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          관측가능성 (Observability)
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            관측가능성은 시스템의 외부 출력을 통해 내부 상태를 이해할 수 있는 능력입니다.
          </p>
          
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
              Three Pillars of Observability
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  📊 Metrics
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  시계열 수치 데이터<br/>
                  CPU, Memory, Latency, RPS
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  📝 Logs
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  이벤트 기록<br/>
                  Error, Warning, Info, Debug
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  🔍 Traces
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  요청 흐름 추적<br/>
                  분산 시스템 전체 경로
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Metrics & Monitoring */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Cpu className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          메트릭과 모니터링
        </h2>
        
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              핵심 메트릭 (Golden Signals)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  😦 Latency
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  요청 처리 시간 (P50, P95, P99)
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  📈 Traffic
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  초당 요청 수 (RPS/QPS)
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  ❌ Errors
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  실패율 (4xx, 5xx)
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  💾 Saturation
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  리소스 사용률 (CPU, Memory, Disk)
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Prometheus + Grafana
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs mb-3">
              <span className="text-green-600 dark:text-green-400"># PromQL 예시</span><br/>
              rate(http_requests_total[5m])<br/>
              histogram_quantile(0.95, http_request_duration_seconds)<br/>
              sum(rate(http_requests_total{`{status=~"5.."}`}[5m]))
            </div>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• Pull 기반 메트릭 수집</li>
              <li>• 시계열 데이터베이스</li>
              <li>• 강력한 쿼리 언어</li>
              <li>• 알림 규칙 설정</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Logging */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <BookOpen className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          로깅 시스템
        </h2>
        
        <div className="space-y-6">
          <div className="bg-yellow-50 dark:bg-yellow-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              ELK Stack
            </h3>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-16 font-bold text-purple-600 dark:text-purple-400">E</div>
                <div>
                  <strong>Elasticsearch:</strong> 로그 저장 및 검색
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-16 font-bold text-purple-600 dark:text-purple-400">L</div>
                <div>
                  <strong>Logstash:</strong> 로그 수집 및 처리
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-16 font-bold text-purple-600 dark:text-purple-400">K</div>
                <div>
                  <strong>Kibana:</strong> 시각화 및 대시보드
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              구조화된 로깅
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs">
              {`{`}<br/>
              &nbsp;&nbsp;"timestamp": "2024-01-15T10:30:45Z",<br/>
              &nbsp;&nbsp;"level": "ERROR",<br/>
              &nbsp;&nbsp;"service": "payment-service",<br/>
              &nbsp;&nbsp;"trace_id": "abc123",<br/>
              &nbsp;&nbsp;"user_id": "user_456",<br/>
              &nbsp;&nbsp;"message": "Payment failed",<br/>
              &nbsp;&nbsp;"error": "Insufficient funds"<br/>
              {`}`}
            </div>
          </div>
        </div>
      </section>

      {/* Distributed Tracing */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Network className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          분산 트레이싱
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            마이크로서비스 환경에서 요청이 여러 서비스를 거치는 전체 경로를 추적합니다.
          </p>
          
          <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-950/20 dark:to-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              OpenTelemetry
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• Trace ID: 전체 요청 추적</li>
              <li>• Span ID: 개별 작업 추적</li>
              <li>• Context Propagation: 서비스 간 컨텍스트 전달</li>
              <li>• Auto-instrumentation: 자동 계측</li>
            </ul>
            
            <div className="mt-4 bg-white dark:bg-gray-700 rounded p-3">
              <div className="text-sm font-mono">
                API Gateway → [2ms]<br/>
                └─ Auth Service → [5ms]<br/>
                └─ User Service → [8ms]<br/>
                &nbsp;&nbsp;&nbsp;└─ Database → [15ms]<br/>
                └─ Payment Service → [12ms]<br/>
                Total: 42ms
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}