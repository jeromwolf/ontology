'use client'

import React from 'react'
import { GitBranch, Workflow, Package, CheckCircle, TrendingUp, Shield, Zap, RefreshCw } from 'lucide-react'

export default function Chapter11() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 dark:from-gray-900 dark:to-slate-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-slate-700 to-gray-800 rounded-xl shadow-lg">
              <GitBranch className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-700 to-gray-800 bg-clip-text text-transparent">
                CI/CD for ML
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                GitHub Actions, Model Registry, A/B Testing으로 자동화된 ML 파이프라인 구축
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Workflow className="w-6 h-6 text-slate-700" />
              ML에서 CI/CD가 중요한 이유
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                전통적 소프트웨어 개발의 CI/CD는 <strong>코드 변경</strong>을 자동으로 테스트하고 배포합니다.
                ML 시스템은 여기에 <strong>데이터 변경, 모델 성능 저하, 재훈련 필요성</strong>이 추가되어
                더 복잡한 파이프라인이 필요합니다. MLOps는 <strong>모델 버전 관리, 자동 검증, 점진적 배포</strong>를 통해
                프로덕션 안정성을 보장합니다.
              </p>

              <div className="bg-slate-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-slate-700">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">ML CI/CD의 핵심 차이점</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <p className="font-bold text-orange-600 dark:text-orange-400 mb-2">전통적 CI/CD</p>
                    <ul className="space-y-1 text-slate-700 dark:text-slate-300 text-sm">
                      <li>• 코드 변경 중심</li>
                      <li>• 단위 테스트, 통합 테스트</li>
                      <li>• 결정론적 동작</li>
                      <li>• Blue-Green 배포</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-bold text-blue-600 dark:text-blue-400 mb-2">ML CI/CD</p>
                    <ul className="space-y-1 text-slate-700 dark:text-slate-300 text-sm">
                      <li>• 코드 + 데이터 + 모델 변경</li>
                      <li>• 성능 테스트, 드리프트 감지</li>
                      <li>• 비결정론적 (확률적)</li>
                      <li>• Canary, Shadow, A/B 배포</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* GitHub Actions for ML */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <GitBranch className="w-6 h-6 text-slate-700" />
            GitHub Actions로 ML 파이프라인 자동화
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                모델 훈련 자동화 워크플로우
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# .github/workflows/train-model.yml
name: Train and Validate Model

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'data/**'
      - 'config/**'
  schedule:
    - cron: '0 2 * * 0'  # 매주 일요일 02:00 UTC

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Download training data
        env:
          AWS_ACCESS_KEY_ID: \${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: \${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          aws s3 sync s3://my-bucket/training-data ./data

      - name: Run data validation
        run: |
          python scripts/validate_data.py \\
            --data-path ./data \\
            --schema-path ./schemas/data_schema.json

      - name: Train model
        run: |
          python train.py \\
            --config config/training.yaml \\
            --output-dir ./models \\
            --mlflow-uri \${{ secrets.MLFLOW_TRACKING_URI }}

      - name: Evaluate model
        id: evaluate
        run: |
          python evaluate.py \\
            --model-path ./models/best_model.pt \\
            --test-data ./data/test \\
            --output metrics.json

          # 메트릭을 GitHub 환경 변수로 저장
          ACCURACY=$(jq -r '.accuracy' metrics.json)
          echo "accuracy=\$ACCURACY" >> $GITHUB_OUTPUT

      - name: Check performance threshold
        run: |
          THRESHOLD=0.85
          if (( \$(echo "\${{ steps.evaluate.outputs.accuracy }} < \$THRESHOLD" | bc -l) )); then
            echo "Model accuracy \${{ steps.evaluate.outputs.accuracy }} below threshold \$THRESHOLD"
            exit 1
          fi

      - name: Upload model to registry
        if: success()
        env:
          MLFLOW_TRACKING_URI: \${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python scripts/register_model.py \\
            --model-path ./models/best_model.pt \\
            --model-name production-classifier \\
            --run-id \${{ github.run_id }} \\
            --accuracy \${{ steps.evaluate.outputs.accuracy }}

      - name: Create deployment PR
        if: success()
        uses: peter-evans/create-pull-request@v5
        with:
          title: "Deploy model v\${{ github.run_number }}"
          body: |
            ## Model Performance
            - Accuracy: \${{ steps.evaluate.outputs.accuracy }}
            - Run ID: \${{ github.run_id }}
            - Commit: \${{ github.sha }}
          branch: deploy/model-v\${{ github.run_number }}`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                모델 테스트 워크플로우
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# .github/workflows/test-model.yml
name: Model Testing Suite

on:
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Run unit tests
        run: |
          pytest tests/unit --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v3

      - name: Test model inference
        run: |
          python -m pytest tests/integration/test_inference.py

      - name: Test model API
        run: |
          docker-compose up -d
          sleep 10
          python tests/integration/test_api.py
          docker-compose down

  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v3

      - name: Benchmark latency
        run: |
          python tests/performance/benchmark_latency.py \\
            --model-path ./models/model.pt \\
            --num-requests 1000 \\
            --output benchmark_results.json

      - name: Check latency SLA
        run: |
          P95_LATENCY=$(jq -r '.p95_latency_ms' benchmark_results.json)
          if (( \$(echo "\$P95_LATENCY > 100" | bc -l) )); then
            echo "P95 latency \${P95_LATENCY}ms exceeds 100ms SLA"
            exit 1
          fi

  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Validate schema
        run: |
          python scripts/validate_schema.py

      - name: Check data drift
        run: |
          python scripts/detect_drift.py \\
            --reference-data ./data/reference \\
            --current-data ./data/latest \\
            --threshold 0.05`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Model Registry */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Package className="w-6 h-6 text-slate-700" />
            Model Registry: 모델 버전 관리
          </h2>

          <div className="space-y-6">
            <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-8 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">Model Registry 핵심 기능</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <CheckCircle className="w-6 h-6 mb-2" />
                  <p className="font-bold mb-1">버전 관리</p>
                  <p className="text-sm text-slate-200">모델, 메타데이터, 하이퍼파라미터 추적</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <Shield className="w-6 h-6 mb-2" />
                  <p className="font-bold mb-1">Staging 워크플로우</p>
                  <p className="text-sm text-slate-200">Dev → Staging → Production 승인 프로세스</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <TrendingUp className="w-6 h-6 mb-2" />
                  <p className="font-bold mb-1">성능 추적</p>
                  <p className="text-sm text-slate-200">A/B 테스트 결과, 메트릭 비교</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <RefreshCw className="w-6 h-6 mb-2" />
                  <p className="font-bold mb-1">롤백 지원</p>
                  <p className="text-sm text-slate-200">문제 발생 시 이전 버전으로 즉시 복구</p>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                MLflow Model Registry 사용
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import mlflow
import mlflow.pytorch

# MLflow 서버 설정
mlflow.set_tracking_uri("http://mlflow.example.com")
mlflow.set_experiment("production-classifier")

# 훈련 실행 로깅
with mlflow.start_run() as run:
    # 파라미터 로깅
    mlflow.log_params({
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "AdamW"
    })

    # 모델 훈련
    model = train_model()

    # 메트릭 로깅
    metrics = evaluate_model(model, test_loader)
    mlflow.log_metrics({
        "accuracy": metrics['accuracy'],
        "f1_score": metrics['f1'],
        "precision": metrics['precision'],
        "recall": metrics['recall']
    })

    # 모델 등록
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name="production-classifier"
    )

    run_id = run.info.run_id

# 모델을 Staging으로 승격
client = mlflow.tracking.MlflowClient()
latest_version = client.get_latest_versions(
    "production-classifier",
    stages=["None"]
)[0].version

client.transition_model_version_stage(
    name="production-classifier",
    version=latest_version,
    stage="Staging",
    archive_existing_versions=False
)

# Production 배포 전 검증
staging_model_uri = f"models:/production-classifier/Staging"
staging_model = mlflow.pytorch.load_model(staging_model_uri)

validation_metrics = validate_model(staging_model, validation_loader)

# 임계값 통과 시 Production으로 승격
if validation_metrics['accuracy'] > 0.90:
    client.transition_model_version_stage(
        name="production-classifier",
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model v{latest_version} promoted to Production!")
else:
    print(f"Model v{latest_version} failed validation")`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                AWS SageMaker Model Registry
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import boto3
import sagemaker
from sagemaker.model import Model

sagemaker_client = boto3.client('sagemaker')
sm_session = sagemaker.Session()

# 모델 패키지 그룹 생성
model_package_group_name = "production-models"
sagemaker_client.create_model_package_group(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageGroupDescription="Production ML models"
)

# 모델 등록
model_package = sagemaker_client.create_model_package(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageDescription="ResNet50 classifier v1.0",
    ModelApprovalStatus="PendingManualApproval",
    InferenceSpecification={
        "Containers": [{
            "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.0-gpu-py38",
            "ModelDataUrl": "s3://my-bucket/models/model.tar.gz"
        }],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"]
    }
)

# 모델 승인
model_package_arn = model_package['ModelPackageArn']
sagemaker_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Approved",
    ApprovalDescription="Passed all validation tests"
)

# Production 배포
model = Model(
    model_data=model_package_arn,
    role=role,
    sagemaker_session=sm_session
)

predictor = model.deploy(
    initial_instance_count=2,
    instance_type="ml.g4dn.xlarge",
    endpoint_name="production-classifier-endpoint"
)`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* A/B Testing */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-slate-700" />
            A/B Testing 인프라
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Canary Deployment (점진적 배포)
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                신규 모델을 <strong>소량의 트래픽(1-5%)</strong>으로 먼저 테스트하고, 문제 없으면 점진적으로 확대합니다.
              </p>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# Kubernetes Deployment with Canary
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model
  ports:
    - port: 8080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-stable
spec:
  replicas: 9  # 90% 트래픽
  selector:
    matchLabels:
      app: model
      version: stable
  template:
    metadata:
      labels:
        app: model
        version: stable
    spec:
      containers:
      - name: model
        image: model:v1.0
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-canary
spec:
  replicas: 1  # 10% 트래픽
  selector:
    matchLabels:
      app: model
      version: canary
  template:
    metadata:
      labels:
        app: model
        version: canary
    spec:
      containers:
      - name: model
        image: model:v2.0  # 새 버전
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Blue-Green Deployment
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# Terraform으로 Blue-Green 배포
resource "aws_lb_target_group" "blue" {
  name     = "model-blue"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 10
  }
}

resource "aws_lb_target_group" "green" {
  name     = "model-green"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 10
  }
}

resource "aws_lb_listener_rule" "production" {
  listener_arn = aws_lb_listener.front_end.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = var.active_target_group  # Blue 또는 Green
  }

  condition {
    path_pattern {
      values = ["/predict/*"]
    }
  }
}

# 배포 스크립트
# 1. Green 환경에 새 모델 배포
# 2. Health check 통과 확인
# 3. 트래픽을 Blue → Green으로 전환
# 4. Blue 환경 종료 (또는 대기)`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Shadow Mode Testing
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                신규 모델을 <strong>프로덕션과 병렬로 실행</strong>하되, 사용자에게 영향을 주지 않고 성능만 측정합니다.
              </p>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import asyncio
import logging

class ShadowModePredictor:
    def __init__(self, production_model, shadow_model):
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.logger = logging.getLogger(__name__)

    async def predict(self, input_data):
        # Production 모델 추론 (사용자에게 반환)
        production_result = await self.production_model.predict(input_data)

        # Shadow 모델 추론 (백그라운드, 비동기)
        asyncio.create_task(
            self._shadow_predict(input_data, production_result)
        )

        return production_result

    async def _shadow_predict(self, input_data, production_result):
        try:
            import time
            start = time.time()
            shadow_result = await self.shadow_model.predict(input_data)
            latency = time.time() - start

            # 결과 비교 및 로깅
            self.logger.info({
                "shadow_latency_ms": latency * 1000,
                "production_prediction": production_result,
                "shadow_prediction": shadow_result,
                "agreement": production_result == shadow_result
            })

            # 메트릭 전송 (Prometheus, DataDog 등)
            metrics.increment('shadow_predictions_total')
            metrics.histogram('shadow_latency_ms', latency * 1000)

            if production_result != shadow_result:
                metrics.increment('shadow_disagreement_total')

        except Exception as e:
            self.logger.error(f"Shadow prediction failed: {e}")
            # Shadow 실패는 Production에 영향 없음`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">배포 전략 비교</h3>
              <div className="space-y-3">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <div className="flex items-center justify-between mb-2">
                    <p className="font-bold">Canary Deployment</p>
                    <span className="text-xs px-2 py-1 bg-blue-500 rounded">점진적</span>
                  </div>
                  <p className="text-sm text-purple-100">
                    1% → 5% → 25% → 100% 단계적 확대. 위험 최소화.
                  </p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <div className="flex items-center justify-between mb-2">
                    <p className="font-bold">Blue-Green</p>
                    <span className="text-xs px-2 py-1 bg-green-500 rounded">즉시 전환</span>
                  </div>
                  <p className="text-sm text-purple-100">
                    두 환경 유지, 순간 전환. 빠른 롤백 가능.
                  </p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <div className="flex items-center justify-between mb-2">
                    <p className="font-bold">Shadow Mode</p>
                    <span className="text-xs px-2 py-1 bg-purple-500 rounded">무위험 테스트</span>
                  </div>
                  <p className="text-sm text-purple-100">
                    사용자 영향 없이 실제 트래픽으로 테스트. 가장 안전.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Automated Validation */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <CheckCircle className="w-6 h-6 text-slate-700" />
            자동 검증 파이프라인
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                모델 검증 체크리스트
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# validate_model.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class ModelValidator:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.checks_passed = []
        self.checks_failed = []

    def validate(self, model, test_loader):
        """모든 검증 단계 실행"""
        self.run_performance_checks(model, test_loader)
        self.run_fairness_checks(model, test_loader)
        self.run_robustness_checks(model)
        self.run_inference_checks(model)

        if self.checks_failed:
            raise ValidationError(
                f"Validation failed: {self.checks_failed}"
            )
        return True

    def run_performance_checks(self, model, test_loader):
        """성능 임계값 검증"""
        y_true, y_pred = [], []
        for batch in test_loader:
            with torch.no_grad():
                predictions = model(batch['input'])
            y_true.extend(batch['label'])
            y_pred.extend(predictions.argmax(dim=1))

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted')
        }

        for metric_name, value in metrics.items():
            threshold = self.thresholds.get(metric_name, 0)
            if value < threshold:
                self.checks_failed.append(
                    f"{metric_name}={value:.3f} < {threshold}"
                )
            else:
                self.checks_passed.append(
                    f"{metric_name}={value:.3f} >= {threshold}"
                )

    def run_fairness_checks(self, model, test_loader):
        """공정성 검증 (그룹 간 성능 격차)"""
        # 예: 성별, 인종 등 보호 속성별 성능 측정
        group_metrics = {}
        for group_name, group_data in test_loader.groups():
            group_accuracy = evaluate_group(model, group_data)
            group_metrics[group_name] = group_accuracy

        max_disparity = max(group_metrics.values()) - min(group_metrics.values())
        if max_disparity > self.thresholds['max_fairness_disparity']:
            self.checks_failed.append(
                f"Fairness disparity {max_disparity:.3f} too high"
            )

    def run_robustness_checks(self, model):
        """적대적 공격 및 노이즈 저항성"""
        # FGSM, PGD 등 적대적 예제 테스트
        pass

    def run_inference_checks(self, model):
        """추론 성능 검증"""
        import time
        dummy_input = torch.randn(1, 3, 224, 224)

        # Latency 측정
        latencies = []
        for _ in range(100):
            start = time.time()
            with torch.no_grad():
                model(dummy_input)
            latencies.append(time.time() - start)

        p95_latency = np.percentile(latencies, 95) * 1000  # ms
        if p95_latency > self.thresholds['max_p95_latency_ms']:
            self.checks_failed.append(
                f"P95 latency {p95_latency:.1f}ms exceeds threshold"
            )`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Complete GitHub Actions Example */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-8 text-white shadow-xl">
            <h2 className="text-2xl font-bold mb-6">완전한 GitHub Actions 워크플로우</h2>
            <div className="bg-white/10 rounded-lg p-6 font-mono text-sm backdrop-blur overflow-x-auto">
              <pre className="text-white">
{`# .github/workflows/mlops-pipeline.yml
name: Complete MLOps Pipeline

on:
  push:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: \${{ github.repository }}

jobs:
  train-and-validate:
    runs-on: ubuntu-latest
    outputs:
      model_version: \${{ steps.version.outputs.version }}
      accuracy: \${{ steps.evaluate.outputs.accuracy }}
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Generate version
        id: version
        run: echo "version=\$(date +%Y%m%d)-\${GITHUB_SHA:0:7}" >> $GITHUB_OUTPUT

      - name: Train model
        run: python train.py --version \${{ steps.version.outputs.version }}

      - name: Validate model
        id: evaluate
        run: |
          python validate_model.py --model models/model.pt
          ACCURACY=$(jq -r '.accuracy' metrics.json)
          echo "accuracy=\$ACCURACY" >> $GITHUB_OUTPUT

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model-\${{ steps.version.outputs.version }}
          path: models/

  build-and-push:
    needs: train-and-validate
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v3
      - uses: docker/login-action@v2
        with:
          registry: \${{ env.REGISTRY }}
          username: \${{ github.actor }}
          password: \${{ secrets.GITHUB_TOKEN }}

      - name: Download model
        uses: actions/download-artifact@v3
        with:
          name: model-\${{ needs.train-and-validate.outputs.model_version }}
          path: models/

      - uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: \${{ env.REGISTRY }}/\${{ env.IMAGE_NAME }}:\${{ needs.train-and-validate.outputs.model_version }}

  deploy-staging:
    needs: [train-and-validate, build-and-push]
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/model-staging \\
            model=\${{ env.REGISTRY }}/\${{ env.IMAGE_NAME }}:\${{ needs.train-and-validate.outputs.model_version }}
          kubectl rollout status deployment/model-staging

  deploy-production:
    needs: [train-and-validate, deploy-staging]
    runs-on: ubuntu-latest
    environment: production
    if: needs.train-and-validate.outputs.accuracy > 0.90
    steps:
      - name: Canary deployment
        run: |
          # 10% 트래픽으로 시작
          kubectl set image deployment/model-canary \\
            model=\${{ env.REGISTRY }}/\${{ env.IMAGE_NAME }}:\${{ needs.train-and-validate.outputs.model_version }}

      - name: Monitor metrics
        run: |
          sleep 300  # 5분 대기
          python scripts/check_canary_metrics.py

      - name: Full rollout
        run: |
          kubectl set image deployment/model-production \\
            model=\${{ env.REGISTRY }}/\${{ env.IMAGE_NAME }}:\${{ needs.train-and-validate.outputs.model_version }}`}
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
                <span><strong>GitHub Actions</strong>는 모델 훈련, 테스트, 배포를 완전 자동화하는 워크플로우를 제공합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">2.</span>
                <span><strong>Model Registry</strong>(MLflow, SageMaker)는 모델 버전 관리, 승인 프로세스, 롤백을 지원합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">3.</span>
                <span><strong>Canary Deployment</strong>는 소량 트래픽으로 신규 모델을 점진적으로 검증합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">4.</span>
                <span><strong>Blue-Green Deployment</strong>는 두 환경을 유지하여 즉시 전환 및 빠른 롤백을 가능하게 합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">5.</span>
                <span><strong>Shadow Mode</strong>는 사용자에게 영향 없이 실제 트래픽으로 신규 모델을 테스트합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">6.</span>
                <span><strong>자동 검증</strong>은 성능, 공정성, 견고성, 추론 지연시간을 자동으로 확인하여 품질을 보장합니다.</span>
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
              <strong>Chapter 12: 프로덕션 MLOps 아키텍처</strong>
              <br />
              Uber Michelangelo, Netflix, Airbnb의 실제 MLOps 시스템 사례 연구 및 End-to-End 아키텍처 설계를 탐구합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
