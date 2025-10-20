'use client'

import React from 'react'
import { BarChart3, TrendingUp, AlertTriangle, Eye, Activity, Database, Zap, GitBranch } from 'lucide-react'

export default function Chapter10() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 dark:from-gray-900 dark:to-slate-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-slate-700 to-gray-800 rounded-xl shadow-lg">
              <Activity className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-700 to-gray-800 bg-clip-text text-transparent">
                모니터링과 관측성
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Prometheus + Grafana로 ML 시스템 모니터링 및 Data Drift 감지하기
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Eye className="w-6 h-6 text-slate-700" />
              ML 시스템 관측성이 중요한 이유
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                프로덕션 ML 시스템은 <strong>조용히 실패(Silent Failure)</strong>하는 경우가 많습니다.
                모델 정확도가 서서히 하락하거나, 추론 지연이 증가하거나, 데이터 분포가 변하는 상황을
                감지하지 못하면 <strong>비즈니스 손실</strong>로 이어집니다.
                관측성(Observability)은 시스템 내부 상태를 외부에서 이해할 수 있게 하는 능력입니다.
              </p>

              <div className="bg-slate-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-slate-700">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">관측성의 3대 기둥</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <BarChart3 className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>Metrics (메트릭)</strong>: 수치화된 시스템 상태 (CPU, GPU 사용률, 추론 지연 등)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Database className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span><strong>Logs (로그)</strong>: 이벤트 기록 (에러 메시지, 요청 로그)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <GitBranch className="w-5 h-5 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span><strong>Traces (추적)</strong>: 요청의 전체 경로 (분산 시스템에서 병목 지점 파악)</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Prometheus + Grafana */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-slate-700" />
            Prometheus + Grafana: ML 메트릭 모니터링
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Prometheus 메트릭 수집 설정
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# prometheus.yml - Prometheus 설정
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # ML 추론 서버 메트릭
  - job_name: 'ml-inference'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

  # GPU 메트릭 (NVIDIA DCGM Exporter)
  - job_name: 'gpu-metrics'
    static_configs:
      - targets: ['localhost:9400']

  # Node 메트릭 (시스템 리소스)
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Python 애플리케이션에서 메트릭 노출
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Counter: 증가만 하는 메트릭 (요청 수 등)
REQUEST_COUNT = Counter(
    'ml_inference_requests_total',
    'Total ML inference requests',
    ['model_name', 'status']
)

# Histogram: 분포를 추적 (지연시간 등)
INFERENCE_LATENCY = Histogram(
    'ml_inference_duration_seconds',
    'ML inference latency',
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# Gauge: 증가/감소하는 메트릭 (현재 값)
MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Current model accuracy',
    ['model_name']
)

GPU_MEMORY_USED = Gauge(
    'ml_gpu_memory_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

# FastAPI 애플리케이션 예제
from fastapi import FastAPI
import torch

app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    model_name = "resnet50"

    # 지연시간 측정
    with INFERENCE_LATENCY.labels(model_name=model_name).time():
        try:
            # 모델 추론
            result = model.predict(data)

            # 성공 카운트
            REQUEST_COUNT.labels(
                model_name=model_name,
                status='success'
            ).inc()

            return {"prediction": result}

        except Exception as e:
            # 실패 카운트
            REQUEST_COUNT.labels(
                model_name=model_name,
                status='error'
            ).inc()
            raise

# 백그라운드: GPU 메모리 모니터링
def update_gpu_metrics():
    while True:
        for gpu_id in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(gpu_id)
            GPU_MEMORY_USED.labels(gpu_id=str(gpu_id)).set(mem_used)
        time.sleep(10)

# Prometheus 메트릭 엔드포인트 시작 (포트 8000)
start_http_server(8000)`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Grafana 대시보드 구성
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# PromQL 쿼리 예제

# 1. 초당 요청 수 (QPS)
rate(ml_inference_requests_total[5m])

# 2. P95 지연시간
histogram_quantile(0.95,
  rate(ml_inference_duration_seconds_bucket[5m])
)

# 3. 에러율
sum(rate(ml_inference_requests_total{status="error"}[5m]))
/
sum(rate(ml_inference_requests_total[5m]))

# 4. GPU 사용률 (DCGM Exporter)
DCGM_FI_DEV_GPU_UTIL

# 5. GPU 메모리 사용률
ml_gpu_memory_bytes / (80 * 1024^3) * 100  # A100 80GB 기준

# 6. 평균 배치 크기
rate(ml_inference_samples_total[5m])
/
rate(ml_inference_requests_total[5m])`}
                </pre>
              </div>
              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <strong>Grafana 대시보드 구성 요소:</strong> Inference Latency (시계열),
                  QPS (숫자), Error Rate (게이지), GPU Utilization (히트맵), Request Heatmap
                </p>
              </div>
            </div>

            <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">핵심 ML 메트릭</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">성능 메트릭</p>
                  <ul className="text-sm text-slate-200 space-y-1">
                    <li>• P50/P95/P99 Latency</li>
                    <li>• Throughput (QPS)</li>
                    <li>• GPU/CPU Utilization</li>
                  </ul>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">모델 품질 메트릭</p>
                  <ul className="text-sm text-slate-200 space-y-1">
                    <li>• Accuracy/F1/AUC (주기적 평가)</li>
                    <li>• Prediction Confidence</li>
                    <li>• Error Rate</li>
                  </ul>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">데이터 메트릭</p>
                  <ul className="text-sm text-slate-200 space-y-1">
                    <li>• Input Distribution</li>
                    <li>• Feature Statistics (mean, std)</li>
                    <li>• Missing Value Rate</li>
                  </ul>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">시스템 메트릭</p>
                  <ul className="text-sm text-slate-200 space-y-1">
                    <li>• Memory Usage (GPU/RAM)</li>
                    <li>• Network I/O</li>
                    <li>• Disk I/O</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Data Drift Detection */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <AlertTriangle className="w-6 h-6 text-slate-700" />
            Data Drift & Concept Drift 감지
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Data Drift vs Concept Drift
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border-l-4 border-blue-500">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">Data Drift (P(X) 변화)</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
                    입력 데이터 분포가 변하는 현상
                  </p>
                  <p className="text-sm text-slate-700 dark:text-slate-300">
                    <strong>예시:</strong> 이미지 분류 모델에 훈련 때 없던 조명, 각도의 이미지가 입력됨
                  </p>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border-l-4 border-green-500">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">Concept Drift (P(Y|X) 변화)</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
                    입력과 출력 간 관계가 변하는 현상
                  </p>
                  <p className="text-sm text-slate-700 dark:text-slate-300">
                    <strong>예시:</strong> 사용자 선호도가 계절에 따라 변화 (같은 상품에 다른 클릭률)
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Evidently AI로 Drift 감지
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import pandas as pd

# 참조 데이터 (훈련 데이터)
reference_data = pd.read_csv("train.csv")

# 현재 프로덕션 데이터
current_data = pd.read_csv("production_last_week.csv")

# Data Drift 리포트 생성
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset()
])

report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=None
)

# HTML 리포트 저장
report.save_html("drift_report.html")

# 프로그래매틱 접근
drift_results = report.as_dict()
for feature, drift_info in drift_results['metrics'][0]['result']['drift_by_columns'].items():
    if drift_info['drift_detected']:
        print(f"Drift detected in {feature}: {drift_info['drift_score']:.3f}")

# Kolmogorov-Smirnov 테스트 (연속형 변수)
from scipy.stats import ks_2samp

def detect_drift(ref_data, current_data, threshold=0.05):
    drift_features = []
    for col in ref_data.columns:
        statistic, p_value = ks_2samp(ref_data[col], current_data[col])
        if p_value < threshold:
            drift_features.append(col)
            print(f"{col}: p-value={p_value:.4f} (DRIFT)")
    return drift_features`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                WhyLabs로 지속적 모니터링
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter

# WhyLogs 프로파일 생성
results = why.log(pandas=current_data)

# WhyLabs로 전송 (SaaS 모니터링)
results.writer("whylabs").write()

# 로컬 프로파일 비교
from whylogs.core.constraints import ConstraintsBuilder

constraints = ConstraintsBuilder(dataset_profile=reference_profile)
constraints.add_constraint(
    column_name="age",
    constraint_name="mean_within_range",
    params={"min": 25, "max": 45}
)

# 제약 조건 검증
validation_results = constraints.validate(current_profile)
if not validation_results.passed():
    print("Constraints failed!")
    for failure in validation_results.failures:
        print(f"  {failure}")`}
                </pre>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-xl">
                <h3 className="font-bold text-lg mb-3">Drift 대응 전략</h3>
                <ul className="space-y-2 text-sm text-purple-100">
                  <li>• 모델 재훈련 (신규 데이터 포함)</li>
                  <li>• 온라인 학습 (Incremental Learning)</li>
                  <li>• 앙상블 모델 (여러 시점 모델 조합)</li>
                  <li>• A/B 테스트로 새 모델 점진 배포</li>
                </ul>
              </div>
              <div className="bg-gradient-to-r from-orange-500 to-red-600 rounded-xl p-6 text-white shadow-xl">
                <h3 className="font-bold text-lg mb-3">알람 설정 예시</h3>
                <ul className="space-y-2 text-sm text-orange-100">
                  <li>• Drift Score > 0.7 → Slack 알림</li>
                  <li>• Accuracy 5% 하락 → PagerDuty</li>
                  <li>• P95 Latency > 500ms → 자동 스케일업</li>
                  <li>• Error Rate > 1% → 즉시 롤백</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Distributed Tracing */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            분산 추적 (Distributed Tracing)
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                OpenTelemetry + Jaeger 설정
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Tracer 설정
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Jaeger Exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# FastAPI 자동 계측
app = FastAPI()
FastAPIInstrumentor.instrument_app(app)

# 커스텀 span 추가
@app.post("/predict")
async def predict(data: dict):
    with tracer.start_as_current_span("model-preprocessing"):
        preprocessed = preprocess(data)

    with tracer.start_as_current_span("model-inference") as span:
        span.set_attribute("model.name", "resnet50")
        span.set_attribute("batch.size", len(preprocessed))

        result = model.predict(preprocessed)

    with tracer.start_as_current_span("postprocessing"):
        output = postprocess(result)

    return output

# Span에 메타데이터 추가
span.add_event("prediction_complete", {
    "confidence": confidence_score,
    "processing_time_ms": elapsed_time
})`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">Trace 분석으로 병목 찾기</h3>
              <div className="bg-white/10 rounded-lg p-6 font-mono text-sm backdrop-blur">
                <pre className="text-white">
{`전체 요청 경로:
┌─────────────────────────────────────┐
│ API Gateway (5ms)                   │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Feature Store Lookup (120ms) ⚠️    │  ← 병목!
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Model Inference (45ms)              │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Postprocessing (8ms)                │
└─────────────────────────────────────┘

총 지연: 178ms
→ Feature Store 캐싱 추가로 30ms로 단축 가능`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Alerting & SLA */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            알람 & SLA 설정
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
            <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
              Prometheus Alertmanager 규칙
            </h3>
            <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
              <pre className="text-slate-800 dark:text-slate-200">
{`# alert_rules.yml
groups:
  - name: ml_inference_alerts
    interval: 30s
    rules:
      # P95 지연시간 SLA 위반
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95,
                rate(ml_inference_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "P95 latency is {{ $value }}s (SLA: 500ms)"

      # 에러율 급증
      - alert: HighErrorRate
        expr: |
          sum(rate(ml_inference_requests_total{status="error"}[5m]))
          /
          sum(rate(ml_inference_requests_total[5m])) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate above 1%"
          description: "Current error rate: {{ $value | humanizePercentage }}"

      # GPU 메모리 부족
      - alert: GPUMemoryHigh
        expr: ml_gpu_memory_bytes / (80 * 1024^3) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage above 90%"

      # 모델 정확도 하락 (주기적 평가 결과)
      - alert: ModelAccuracyDrop
        expr: ml_model_accuracy < 0.85
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy dropped below threshold"
          description: "Accuracy: {{ $value | humanizePercentage }}"`}
              </pre>
            </div>
          </div>
        </section>

        {/* Key Takeaways */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-8 border border-blue-200 dark:border-blue-800">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4">
              핵심 요점
            </h2>
            <ul className="space-y-3 text-slate-700 dark:text-slate-300">
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">1.</span>
                <span><strong>Prometheus + Grafana</strong>는 ML 시스템의 메트릭을 실시간으로 수집하고 시각화합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">2.</span>
                <span><strong>Data Drift</strong>는 입력 분포 변화, <strong>Concept Drift</strong>는 입출력 관계 변화를 의미합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">3.</span>
                <span><strong>Evidently AI/WhyLabs</strong>는 통계적 테스트로 Drift를 자동 감지합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">4.</span>
                <span><strong>OpenTelemetry + Jaeger</strong>는 분산 추적으로 요청 경로의 병목 지점을 파악합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">5.</span>
                <span><strong>Alertmanager</strong>는 SLA 위반 시 자동으로 알람을 발송하며, Slack/PagerDuty 통합을 지원합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">6.</span>
                <span>핵심 ML 메트릭: <strong>P95 Latency, QPS, Error Rate, GPU Utilization, Model Accuracy</strong></span>
              </li>
            </ul>
          </div>
        </section>

        {/* Next Chapter Preview */}
        <section>
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border-2 border-slate-300 dark:border-gray-600">
            <h3 className="text-lg font-bold text-slate-800 dark:text-white mb-2">
              다음 챕터 미리보기
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              <strong>Chapter 11: CI/CD for ML</strong>
              <br />
              GitHub Actions로 ML 파이프라인을 자동화하고, A/B 테스트, Canary 배포,
              Shadow Mode로 안전하게 모델을 프로덕션에 배포하는 방법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
