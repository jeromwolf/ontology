'use client';

import React from 'react';
import { Monitor, Activity, FileText, Shield, Lock, AlertTriangle, CheckCircle, Eye, BarChart3 } from 'lucide-react';

export default function Chapter8() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-2xl p-8 mb-8 border border-red-200 dark:border-red-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-red-500 rounded-xl flex items-center justify-center">
            <Monitor className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">모니터링, 로깅, 보안</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          Observability 3대 기둥(메트릭, 로그, 트레이싱)과 Kubernetes 보안 강화 전략
        </p>
      </div>

      {/* Observability Fundamentals */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Eye className="text-blue-600" />
          Observability 3대 기둥
        </h2>

        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border-l-4 border-blue-500">
            <div className="flex items-center gap-2 mb-3">
              <BarChart3 className="text-blue-600" size={24} />
              <h3 className="font-bold m-0">Metrics (메트릭)</h3>
            </div>
            <p className="text-sm mb-2">시계열 수치 데이터로 시스템 상태를 정량적으로 측정</p>
            <ul className="text-xs space-y-1">
              <li>• CPU/메모리 사용률</li>
              <li>• HTTP 요청 수/응답 시간</li>
              <li>• 에러율, 처리량(throughput)</li>
            </ul>
            <p className="text-xs mt-2 font-bold">→ Prometheus + Grafana</p>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg border-l-4 border-green-500">
            <div className="flex items-center gap-2 mb-3">
              <FileText className="text-green-600" size={24} />
              <h3 className="font-bold m-0">Logs (로그)</h3>
            </div>
            <p className="text-sm mb-2">이벤트 기록을 통해 시스템 동작을 상세하게 추적</p>
            <ul className="text-xs space-y-1">
              <li>• 애플리케이션 로그</li>
              <li>• 에러 스택 트레이스</li>
              <li>• 사용자 행동 로그</li>
            </ul>
            <p className="text-xs mt-2 font-bold">→ ELK Stack / Loki</p>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg border-l-4 border-purple-500">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="text-purple-600" size={24} />
              <h3 className="font-bold m-0">Traces (트레이싱)</h3>
            </div>
            <p className="text-sm mb-2">분산 시스템에서 요청의 전체 경로를 추적</p>
            <ul className="text-xs space-y-1">
              <li>• 마이크로서비스 호출 흐름</li>
              <li>• 각 구간의 레이턴시</li>
              <li>• 병목 지점 식별</li>
            </ul>
            <p className="text-xs mt-2 font-bold">→ Jaeger / Zipkin</p>
          </div>
        </div>
      </section>

      {/* Prometheus & Grafana */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <BarChart3 className="text-orange-600" />
          Prometheus & Grafana - 메트릭 모니터링
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">Prometheus 아키텍처</h3>
          <div className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg mb-4">
            <p className="text-sm mb-2"><strong>Pull 기반 메트릭 수집</strong>: 타겟에서 주기적으로 스크랩</p>
            <p className="text-sm mb-2"><strong>PromQL</strong>: 강력한 쿼리 언어로 메트릭 집계 및 분석</p>
            <p className="text-sm"><strong>AlertManager</strong>: 임계값 기반 알림 전송 (Slack, PagerDuty 등)</p>
          </div>

          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Prometheus Operator 설치 (Helm)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \\
  --namespace monitoring --create-namespace

# ServiceMonitor - Kubernetes Service 자동 발견
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: myapp-metrics
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: myapp
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s

# PrometheusRule - 알림 규칙
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: myapp-alerts
  namespace: monitoring
spec:
  groups:
  - name: myapp
    interval: 30s
    rules:
    - alert: HighErrorRate
      expr: |
        rate(http_requests_total{status=~"5.."}[5m]) > 0.05
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }} requests/sec"

    - alert: PodCrashLooping
      expr: |
        rate(kube_pod_container_status_restarts_total[15m]) > 0
      for: 5m
      annotations:
        summary: "Pod {{ $labels.pod }} is crash looping"`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">핵심 PromQL 쿼리</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# CPU 사용률 (%)
100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# 메모리 사용률 (%)
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# HTTP 요청 처리량 (req/sec)
rate(http_requests_total[5m])

# P95 응답 시간
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# 에러율 (%)
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m])) * 100

# Pod 재시작 횟수
kube_pod_container_status_restarts_total

# 네트워크 트래픽 (bytes/sec)
rate(container_network_receive_bytes_total[5m])`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-xl font-bold mb-4">Grafana 대시보드 구성</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded">
              <h4 className="font-bold mb-2">클러스터 레벨</h4>
              <ul className="text-sm space-y-1">
                <li>• 전체 노드 CPU/메모리/디스크</li>
                <li>• Namespace별 리소스 사용</li>
                <li>• Pod 상태 및 재시작 추이</li>
                <li>• 네트워크 I/O</li>
              </ul>
            </div>
            <div className="bg-cyan-50 dark:bg-cyan-900/20 p-4 rounded">
              <h4 className="font-bold mb-2">애플리케이션 레벨</h4>
              <ul className="text-sm space-y-1">
                <li>• 요청 처리량(RPS)</li>
                <li>• 응답 시간(P50/P95/P99)</li>
                <li>• 에러율 및 상태 코드 분포</li>
                <li>• 비즈니스 메트릭(주문 수, 결제 금액)</li>
              </ul>
            </div>
          </div>
          <p className="text-sm mt-4">
            <strong>인기 대시보드</strong>: Grafana Labs에서 미리 제작된 대시보드 Import (예: Node Exporter Full #1860, Kubernetes Cluster Monitoring #7249)
          </p>
        </div>
      </section>

      {/* Logging */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <FileText className="text-green-600" />
          로그 집계 - ELK Stack & Loki
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">ELK Stack (Elasticsearch + Logstash + Kibana)</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Elastic Cloud on Kubernetes (ECK) 설치
kubectl create -f https://download.elastic.co/downloads/eck/2.10.0/crds.yaml
kubectl apply -f https://download.elastic.co/downloads/eck/2.10.0/operator.yaml

# Elasticsearch 클러스터
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: logs
  namespace: logging
spec:
  version: 8.11.0
  nodeSets:
  - name: default
    count: 3
    config:
      node.store.allow_mmap: false

---
# Kibana
apiVersion: kibana.k8s.elastic.co/v1
kind: Kibana
metadata:
  name: logs
  namespace: logging
spec:
  version: 8.11.0
  count: 1
  elasticsearchRef:
    name: logs

---
# Filebeat DaemonSet - 로그 수집
apiVersion: beat.k8s.elastic.co/v1beta1
kind: Beat
metadata:
  name: filebeat
  namespace: logging
spec:
  type: filebeat
  version: 8.11.0
  elasticsearchRef:
    name: logs
  config:
    filebeat.inputs:
    - type: container
      paths:
      - /var/log/containers/*.log
      processors:
      - add_kubernetes_metadata:
          host: \${NODE_NAME}
  daemonSet:
    podTemplate:
      spec:
        serviceAccountName: filebeat
        automountServiceAccountToken: true
        containers:
        - name: filebeat
          volumeMounts:
          - name: varlogcontainers
            mountPath: /var/log/containers
          - name: varlogpods
            mountPath: /var/log/pods
        volumes:
        - name: varlogcontainers
          hostPath:
            path: /var/log/containers
        - name: varlogpods
          hostPath:
            path: /var/log/pods`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">Grafana Loki - 경량 로그 시스템</h3>
          <p className="mb-4">Prometheus에서 영감을 받은 수평 확장 가능한 로그 집계 시스템</p>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Loki Stack 설치 (Helm)
helm repo add grafana https://grafana.github.io/helm-charts
helm install loki grafana/loki-stack \\
  --namespace logging --create-namespace \\
  --set grafana.enabled=true \\
  --set promtail.enabled=true

# Promtail - 로그 수집 에이전트 (DaemonSet)
# 자동으로 Pod 로그를 Loki로 전송

# LogQL 쿼리 예시 (Grafana Explore)
# 특정 Namespace의 에러 로그
{namespace="production"} |= "error"

# HTTP 5xx 에러만 필터링
{app="nginx"} | json | status_code >= 500

# 로그 카운트 (마지막 5분간)
sum(rate({namespace="production"}[5m])) by (pod)

# 정규식 패턴 매칭
{app="api"} |~ "user_id=[0-9]+"`}
            </pre>
          </div>

          <div className="grid md:grid-cols-2 gap-4 mt-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
              <h4 className="font-bold text-sm mb-1">ELK Stack 장점</h4>
              <ul className="text-xs space-y-1">
                <li>✅ 강력한 전문 검색(Full-text search)</li>
                <li>✅ 복잡한 집계 및 분석</li>
                <li>✅ 풍부한 시각화 도구</li>
              </ul>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded">
              <h4 className="font-bold text-sm mb-1">Loki 장점</h4>
              <ul className="text-xs space-y-1">
                <li>✅ 낮은 리소스 사용량</li>
                <li>✅ Prometheus와 유사한 쿼리</li>
                <li>✅ Kubernetes 라벨 기반 인덱싱</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Distributed Tracing */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Activity className="text-purple-600" />
          분산 추적 - Jaeger
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">OpenTelemetry + Jaeger</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Jaeger All-in-One 설치 (개발용)
kubectl create namespace observability
kubectl create -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/crds/jaegertracing.io_jaegers_crd.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/service_account.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/role.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/role_binding.yaml
kubectl create -n observability -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/operator.yaml

# Jaeger 인스턴스
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger
  namespace: observability
spec:
  strategy: production
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: http://elasticsearch:9200

# 애플리케이션에서 OpenTelemetry 사용 (Node.js 예시)
const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');
const { registerInstrumentations } = require('@opentelemetry/instrumentation');
const { HttpInstrumentation } = require('@opentelemetry/instrumentation-http');
const { ExpressInstrumentation } = require('@opentelemetry/instrumentation-express');

const provider = new NodeTracerProvider();
const exporter = new JaegerExporter({
  endpoint: 'http://jaeger-collector:14268/api/traces',
});

provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
provider.register();

registerInstrumentations({
  instrumentations: [
    new HttpInstrumentation(),
    new ExpressInstrumentation(),
  ],
});`}
            </pre>
          </div>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg">
          <h3 className="font-bold mb-3">분산 추적의 핵심 개념</h3>
          <ul className="space-y-2 text-sm">
            <li><strong>Trace</strong>: 하나의 요청이 여러 서비스를 거치는 전체 경로</li>
            <li><strong>Span</strong>: Trace 내의 개별 작업 단위 (예: HTTP 요청, DB 쿼리)</li>
            <li><strong>Context Propagation</strong>: Trace ID를 HTTP 헤더로 전달하여 연결</li>
            <li><strong>활용 사례</strong>: 마이크로서비스 간 레이턴시 분석, 병목 구간 식별, 장애 전파 경로 추적</li>
          </ul>
        </div>
      </section>

      {/* Kubernetes Security */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Shield className="text-red-600" />
          Kubernetes 보안 강화
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">NetworkPolicy - 네트워크 격리</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# 기본 거부 정책 (Default Deny)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
# Frontend → Backend 허용 (Ingress)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080

---
# Backend → Database 허용 (Egress)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-backend-to-db
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:  # DNS 허용 (필수!)
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">Pod Security Standards (PSS)</h3>
          <p className="mb-4">Kubernetes 1.25+에서 PodSecurityPolicy 대체</p>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Namespace 레벨에서 보안 정책 적용
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted

# 3가지 보안 레벨:
# - privileged: 제한 없음
# - baseline: 알려진 권한 상승 방지
# - restricted: 강력한 보안 (권장)

# Restricted 정책 준수 Pod 예시
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
spec:
  securityContext:
    runAsNonRoot: true      # 루트 실행 금지
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    image: myapp:1.0
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      capabilities:
        drop:
        - ALL
    volumeMounts:
    - name: tmp
      mountPath: /tmp
  volumes:
  - name: tmp
    emptyDir: {}`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">Secrets Encryption at Rest</h3>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# EncryptionConfiguration (kube-apiserver)
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
- resources:
  - secrets
  providers:
  - aescbc:
      keys:
      - name: key1
        secret: <BASE64_ENCODED_SECRET>
  - identity: {}  # fallback

# kube-apiserver 플래그 추가
--encryption-provider-config=/path/to/encryption-config.yaml

# Secret 재암호화 (기존 Secret 업데이트)
kubectl get secrets --all-namespaces -o json | kubectl replace -f -`}
            </pre>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-xl font-bold mb-4">Falco - 런타임 보안 감지</h3>
          <p className="mb-4">컨테이너 및 시스템 이상 행동 실시간 탐지</p>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
            <pre className="text-sm">
{`# Falco 설치 (Helm)
helm repo add falcosecurity https://falcosecurity.github.io/charts
helm install falco falcosecurity/falco \\
  --namespace falco --create-namespace \\
  --set falcosidekick.enabled=true \\
  --set falcosidekick.config.slack.webhookurl="https://hooks.slack.com/..."

# Falco 탐지 규칙 예시
- rule: Terminal shell in container
  desc: Detect shell execution in container
  condition: >
    spawned_process and container and
    shell_procs and proc.tty != 0
  output: >
    Shell spawned in container
    (user=%user.name container=%container.name
    shell=%proc.name parent=%proc.pname)
  priority: WARNING

- rule: Write below binary dir
  desc: Detect file modification in /bin, /usr/bin
  condition: >
    open_write and container and
    fd.name pmatch (/bin/*, /usr/bin/*)
  output: File written to binary directory
  priority: CRITICAL

# 실시간 알림:
# - Slack, PagerDuty, Webhook
# - 의심스러운 프로세스 실행
# - 민감한 파일 접근
# - 네트워크 연결 이상`}
            </pre>
          </div>
        </div>
      </section>

      {/* Best Practices */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4">💡 프로덕션 운영 모범 사례</h2>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">SLI/SLO 정의</h3>
                <p className="text-sm">Service Level Indicator(가용성, 레이턴시)와 Objective(목표치) 설정 및 추적</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">알림 피로 방지</h3>
                <p className="text-sm">중요한 알림만 전송, 심각도별 채널 분리, 알림 임계값 조정</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">로그 보존 정책</h3>
                <p className="text-sm">Hot(7일), Warm(30일), Cold(1년) 티어로 분리하여 비용 최적화</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">정기적 보안 감사</h3>
                <p className="text-sm">이미지 스캔(Trivy), RBAC 감사, Secret 순환, 침투 테스트</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">온콜 Runbook 작성</h3>
                <p className="text-sm">각 알림별 대응 절차, 롤백 방법, 에스컬레이션 경로 문서화</p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
            <div className="flex items-start gap-2">
              <CheckCircle className="text-green-500 mt-1 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-bold mb-1">Chaos Engineering</h3>
                <p className="text-sm">Chaos Mesh로 장애 주입 테스트, 복원력 검증</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg border-l-4 border-orange-500">
          <h3 className="font-bold mb-3 flex items-center gap-2">
            <AlertTriangle className="text-orange-500" />
            흔한 실수와 해결 방법
          </h3>
          <ul className="space-y-2 text-sm">
            <li>❌ <strong>메트릭 과다 수집</strong> → 중요한 메트릭만 선별, Cardinality 제한</li>
            <li>❌ <strong>로그 무차별 수집</strong> → 구조화된 로그(JSON), 불필요한 로그 필터링</li>
            <li>❌ <strong>컨테이너 root 실행</strong> → runAsNonRoot, USER 지시어 사용</li>
            <li>❌ <strong>Secret을 환경변수로 노출</strong> → Volume Mount 사용, 프로세스 덤프에서 보호</li>
            <li>❌ <strong>모니터링 시스템 단일 장애점</strong> → Prometheus HA, Thanos 장기 저장소 연동</li>
          </ul>
        </div>
      </section>

      {/* Golden Signals */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4">🎯 Google SRE의 Golden Signals</h2>
        <div className="grid md:grid-cols-4 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg text-center">
            <h3 className="font-bold mb-2">Latency (지연 시간)</h3>
            <p className="text-xs">요청 처리 시간 (P50, P95, P99)</p>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg text-center">
            <h3 className="font-bold mb-2">Traffic (트래픽)</h3>
            <p className="text-xs">초당 요청 수 (RPS)</p>
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg text-center">
            <h3 className="font-bold mb-2">Errors (에러율)</h3>
            <p className="text-xs">실패한 요청 비율 (%)</p>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg text-center">
            <h3 className="font-bold mb-2">Saturation (포화도)</h3>
            <p className="text-xs">리소스 사용률 (CPU, 메모리)</p>
          </div>
        </div>
        <p className="text-sm mt-4 text-center">
          이 4가지 신호를 모니터링하면 대부분의 시스템 문제를 조기 발견할 수 있습니다.
        </p>
      </section>

      {/* Congratulations */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 border-l-4 border-green-500 p-6 rounded-lg my-8">
        <h3 className="text-xl font-bold mb-2">🎉 DevOps & CI/CD 모듈 완료!</h3>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          축하합니다! Docker부터 Kubernetes, GitOps, 그리고 Observability까지 현대적인 DevOps 실무의 전체 스택을 학습하셨습니다.
        </p>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
          <h4 className="font-bold mb-2">핵심 역량 습득</h4>
          <ul className="grid md:grid-cols-2 gap-2 text-sm">
            <li>✅ 컨테이너 빌드 및 최적화</li>
            <li>✅ Kubernetes 운영 및 오케스트레이션</li>
            <li>✅ CI/CD 파이프라인 설계</li>
            <li>✅ GitOps 기반 배포 자동화</li>
            <li>✅ 프로덕션 모니터링 및 로깅</li>
            <li>✅ 보안 강화 및 컴플라이언스</li>
          </ul>
        </div>
        <p className="text-sm mt-4">
          <strong>다음 단계 추천</strong>: Service Mesh(Istio), Platform Engineering, FinOps, SRE 실습 프로젝트
        </p>
      </div>
    </div>
  );
}