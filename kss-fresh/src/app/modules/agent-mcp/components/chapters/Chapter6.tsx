'use client';

import React from 'react';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Agent 배포 아키텍처
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            프로덕션 환경에서 Agent를 안정적으로 운영하기 위한 아키텍처:
          </p>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h4 className="font-semibold mb-3">배포 컴포넌트</h4>
            <div className="space-y-2">
              <div>🔄 <strong>Load Balancer</strong>: 트래픽 분산</div>
              <div>🐳 <strong>Container Orchestration</strong>: Kubernetes/Docker</div>
              <div>💾 <strong>State Management</strong>: Redis/PostgreSQL</div>
              <div>📊 <strong>Message Queue</strong>: RabbitMQ/Kafka</div>
              <div>📈 <strong>Monitoring</strong>: Prometheus/Grafana</div>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Scale과 Load Balancing
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            대규모 트래픽을 처리하기 위한 확장 전략:
          </p>
          
          <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
{`# Kubernetes Deployment 예시
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
spec:
  replicas: 5
  selector:
    matchLabels:
      app: agent
  template:
    metadata:
      labels:
        app: agent
    spec:
      containers:
      - name: agent
        image: myregistry/agent:v1.0
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: MAX_WORKERS
          value: "10"
        - name: CACHE_ENABLED
          value: "true"`}
          </pre>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Security와 Rate Limiting
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            Agent 시스템 보안을 위한 필수 조치:
          </p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Security Measures</h4>
              <ul className="text-sm space-y-1">
                <li>🔐 API Key Authentication</li>
                <li>🛡️ Input Validation & Sanitization</li>
                <li>🔒 TLS/SSL Encryption</li>
                <li>📝 Audit Logging</li>
                <li>🚫 Prompt Injection Prevention</li>
              </ul>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Rate Limiting</h4>
              <ul className="text-sm space-y-1">
                <li>⏱️ Requests per minute</li>
                <li>💰 Token-based limits</li>
                <li>👤 User-based quotas</li>
                <li>🔄 Adaptive throttling</li>
                <li>💳 Tier-based access</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Cost 최적화 전략
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            LLM API 비용을 효과적으로 관리하는 방법:
          </p>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
            <h4 className="font-semibold mb-3">비용 절감 기법</h4>
            <div className="space-y-2">
              <div>
                <strong>1. Caching</strong>
                <p className="text-sm">자주 사용되는 응답을 캐싱하여 API 호출 감소</p>
              </div>
              <div>
                <strong>2. Model Selection</strong>
                <p className="text-sm">작업에 적합한 최소 모델 사용 (GPT-3.5 vs GPT-4)</p>
              </div>
              <div>
                <strong>3. Prompt Optimization</strong>
                <p className="text-sm">프롬프트 길이 최적화로 토큰 사용량 감소</p>
              </div>
              <div>
                <strong>4. Batch Processing</strong>
                <p className="text-sm">여러 요청을 묶어서 처리</p>
              </div>
              <div>
                <strong>5. Fallback Strategy</strong>
                <p className="text-sm">비용이 낮은 대체 솔루션 준비</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          실전 체크리스트
        </h2>
        <div className="space-y-4 text-gray-700 dark:text-gray-300">
          <p>
            프로덕션 배포 전 확인 사항:
          </p>
          
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <ul className="space-y-2">
              <li>☑️ 에러 처리 및 복구 메커니즘 구현</li>
              <li>☑️ 모니터링 및 알림 시스템 설정</li>
              <li>☑️ 보안 취약점 스캔 및 패치</li>
              <li>☑️ 부하 테스트 및 성능 최적화</li>
              <li>☑️ 백업 및 재해 복구 계획</li>
              <li>☑️ API 키 및 시크릿 관리</li>
              <li>☑️ 로깅 및 감사 추적 설정</li>
              <li>☑️ 비용 모니터링 및 알림</li>
              <li>☑️ 사용자 피드백 수집 체계</li>
              <li>☑️ 롤백 계획 및 절차</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}