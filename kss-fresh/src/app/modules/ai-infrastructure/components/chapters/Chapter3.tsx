'use client'

import React from 'react'
import { Box, Container, Layers, Network, Server, Package, Boxes, Workflow } from 'lucide-react'

export default function Chapter3() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 dark:from-gray-900 dark:to-slate-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-slate-700 to-gray-800 rounded-xl shadow-lg">
              <Container className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-700 to-gray-800 bg-clip-text text-transparent">
                컨테이너와 오케스트레이션
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Docker와 Kubernetes로 ML 워크로드를 관리하는 현대적 방법
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Box className="w-6 h-6 text-slate-700" />
              컨테이너가 ML에 필수인 이유
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                머신러닝 프로젝트는 <strong>복잡한 의존성</strong>(Python, CUDA, cuDNN, 프레임워크 등)을 가지며,
                개발 환경과 프로덕션 환경이 다를 경우 "내 컴퓨터에서는 작동했는데..." 문제가 발생합니다.
                컨테이너는 <strong>환경 일관성</strong>을 보장하고, 재현 가능한 ML 파이프라인을 구축합니다.
              </p>

              <div className="bg-slate-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-slate-700">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">컨테이너의 핵심 가치</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <Package className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>재현성(Reproducibility)</strong>: 동일한 환경을 어디서나 실행</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Boxes className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span><strong>격리성(Isolation)</strong>: 의존성 충돌 방지</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Server className="w-5 h-5 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span><strong>확장성(Scalability)</strong>: 컨테이너 단위로 손쉬운 스케일링</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Workflow className="w-5 h-5 text-orange-500 mt-0.5 flex-shrink-0" />
                    <span><strong>이식성(Portability)</strong>: 로컬, 클라우드, 온프레미스 어디든</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Docker Basics */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Container className="w-6 h-6 text-slate-700" />
            Docker 기초
          </h2>

          <div className="space-y-6">
            {/* Dockerfile for ML */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                ML을 위한 Dockerfile 작성
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                PyTorch 훈련 환경을 포함한 Docker 이미지 예제입니다.
              </p>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# PyTorch 공식 이미지 사용 (CUDA 포함)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /workspace

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    vim \\
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 훈련 스크립트 실행
CMD ["python", "train.py"]`}
                </pre>
              </div>
              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <strong>빌드 & 실행:</strong>
                </p>
                <div className="mt-2 font-mono text-sm text-slate-800 dark:text-slate-200">
                  <code>docker build -t my-ml-training .</code>
                  <br />
                  <code>docker run --gpus all -v $(pwd)/data:/workspace/data my-ml-training</code>
                </div>
              </div>
            </div>

            {/* Multi-stage Build */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Multi-stage Build로 이미지 최적화
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                훈련용 무거운 이미지와 추론용 가벼운 이미지를 분리합니다.
              </p>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# Stage 1: Training (Full environment)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel AS trainer
WORKDIR /workspace
COPY . .
RUN pip install -r requirements-train.txt
RUN python train.py --output-dir /models

# Stage 2: Inference (Lightweight)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS serving
WORKDIR /app

# 훈련된 모델만 복사
COPY --from=trainer /models /app/models
COPY inference.py .
RUN pip install --no-cache-dir fastapi uvicorn

# API 서버 실행
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]`}
                </pre>
              </div>
              <div className="mt-4 grid md:grid-cols-2 gap-4">
                <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                  <p className="font-bold text-red-700 dark:text-red-300 mb-1">Training Image</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400">~8GB (개발 도구 포함)</p>
                </div>
                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                  <p className="font-bold text-green-700 dark:text-green-300 mb-1">Serving Image</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400">~2GB (런타임만)</p>
                </div>
              </div>
            </div>

            {/* Docker Compose for ML Stack */}
            <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Layers className="w-6 h-6" />
                Docker Compose: 전체 ML 스택 정의
              </h3>
              <div className="bg-white/10 rounded-lg p-6 font-mono text-sm backdrop-blur overflow-x-auto">
                <pre className="text-white">
{`version: '3.8'

services:
  training:
    image: my-ml-training:latest
    volumes:
      - ./data:/workspace/data
      - ./models:/workspace/models
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Kubernetes Fundamentals */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Network className="w-6 h-6 text-slate-700" />
            Kubernetes 기초
          </h2>

          <div className="space-y-6">
            {/* K8s Core Concepts */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Kubernetes 핵심 개념
              </h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="flex items-center gap-2 mb-2">
                    <Box className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                    <p className="font-bold text-blue-700 dark:text-blue-300">Pod</p>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    컨테이너를 실행하는 최소 단위. 1개 이상의 컨테이너로 구성.
                  </p>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="flex items-center gap-2 mb-2">
                    <Server className="w-5 h-5 text-green-600 dark:text-green-400" />
                    <p className="font-bold text-green-700 dark:text-green-300">Deployment</p>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Pod의 배포 및 업데이트를 선언적으로 관리.
                  </p>
                </div>
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                  <div className="flex items-center gap-2 mb-2">
                    <Network className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                    <p className="font-bold text-purple-700 dark:text-purple-300">Service</p>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Pod 그룹에 대한 네트워크 엔드포인트 제공.
                  </p>
                </div>
                <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
                  <div className="flex items-center gap-2 mb-2">
                    <Package className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                    <p className="font-bold text-orange-700 dark:text-orange-300">ConfigMap/Secret</p>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    설정 및 민감 정보를 분리하여 관리.
                  </p>
                </div>
              </div>
            </div>

            {/* ML Training Job */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Kubernetes Job: 배치 훈련 실행
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`apiVersion: batch/v1
kind: Job
metadata:
  name: pytorch-training-job
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: my-ml-training:v1.0
        resources:
          limits:
            nvidia.com/gpu: 4  # 4 GPUs 요청
            memory: "64Gi"
            cpu: "16"
        volumeMounts:
        - name: data
          mountPath: /workspace/data
        - name: models
          mountPath: /workspace/models
        env:
        - name: MASTER_ADDR
          value: "localhost"
        - name: MASTER_PORT
          value: "29500"
        command: ["torchrun"]
        args:
          - "--nproc_per_node=4"
          - "train.py"
          - "--epochs=100"
          - "--batch-size=32"
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc`}
                </pre>
              </div>
            </div>

            {/* Distributed Training with PyTorchJob */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                PyTorchJob: 분산 훈련 오케스트레이션
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                Kubeflow의 PyTorchJob은 멀티 노드 분산 훈련을 자동으로 설정합니다.
              </p>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: distributed-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.1.0-cuda12.1
            resources:
              limits:
                nvidia.com/gpu: 1
            command:
              - python
              - train.py
              - --backend=nccl

    Worker:
      replicas: 3  # 3개의 Worker 노드
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.1.0-cuda12.1
            resources:
              limits:
                nvidia.com/gpu: 1`}
                </pre>
              </div>
              <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <strong>자동 설정:</strong> MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK 환경변수 자동 주입
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Helm Charts */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Package className="w-6 h-6 text-slate-700" />
            Helm Charts: Kubernetes 패키지 관리
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Helm으로 ML 스택 배포
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                Helm은 Kubernetes의 "패키지 매니저"로, 복잡한 애플리케이션을 차트(Chart)로 정의합니다.
              </p>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto mb-4">
                <pre className="text-slate-800 dark:text-slate-200">
{`# values.yaml - MLflow 배포 설정
replicaCount: 2

image:
  repository: ghcr.io/mlflow/mlflow
  tag: "2.8.0"

service:
  type: LoadBalancer
  port: 5000

postgresql:
  enabled: true
  auth:
    database: mlflow
    username: mlflow

storage:
  artifactRoot: s3://my-bucket/mlflow

ingress:
  enabled: true
  hosts:
    - host: mlflow.example.com
      paths:
        - path: /
          pathType: Prefix`}
                </pre>
              </div>
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-sm font-bold text-slate-800 dark:text-white mb-2">Helm 설치 명령어</p>
                <div className="font-mono text-sm text-slate-700 dark:text-slate-300 space-y-1">
                  <p>helm repo add mlflow https://mlflow.github.io/mlflow</p>
                  <p>helm install mlflow mlflow/mlflow -f values.yaml</p>
                  <p>helm upgrade mlflow mlflow/mlflow -f values.yaml</p>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">인기 ML 관련 Helm Charts</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">JupyterHub</p>
                  <p className="text-sm text-purple-100">멀티 유저 노트북 환경</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Kubeflow</p>
                  <p className="text-sm text-purple-100">End-to-end ML 플랫폼</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Ray Cluster</p>
                  <p className="text-sm text-purple-100">분산 Python 실행 엔진</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Seldon Core</p>
                  <p className="text-sm text-purple-100">모델 서빙 플랫폼</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Container Registries */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Container Registry 전략
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="font-bold text-slate-800 dark:text-white mb-3">Public Registries</h3>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span><strong>Docker Hub</strong>: 공식 이미지 (pytorch, tensorflow)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span><strong>NVIDIA NGC</strong>: GPU 최적화 컨테이너</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span><strong>GitHub Container Registry</strong>: GitHub Actions 통합</span>
                </li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="font-bold text-slate-800 dark:text-white mb-3">Private Registries</h3>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-green-500">•</span>
                  <span><strong>Amazon ECR</strong>: AWS 통합, IAM 인증</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">•</span>
                  <span><strong>Google Artifact Registry</strong>: GCP 통합</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">•</span>
                  <span><strong>Harbor</strong>: 오픈소스, 보안 스캐닝</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="mt-6 bg-slate-50 dark:bg-gray-900 rounded-xl p-6 border-l-4 border-slate-700">
            <h3 className="font-bold text-slate-800 dark:text-white mb-3">이미지 태깅 전략</h3>
            <div className="font-mono text-sm text-slate-700 dark:text-slate-300 space-y-1">
              <p>my-registry.io/ml-training:<strong>v1.2.3</strong> (Semantic versioning)</p>
              <p>my-registry.io/ml-training:<strong>git-abc123f</strong> (Git commit SHA)</p>
              <p>my-registry.io/ml-training:<strong>latest</strong> (최신, 프로덕션 비권장)</p>
              <p>my-registry.io/ml-training:<strong>2024-01-15</strong> (날짜 기반)</p>
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
                <span><strong>Docker</strong>는 ML 환경을 컨테이너화하여 재현성과 이식성을 보장합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">2.</span>
                <span><strong>Multi-stage Build</strong>로 훈련용 무거운 이미지와 추론용 가벼운 이미지를 분리합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">3.</span>
                <span><strong>Kubernetes</strong>는 컨테이너 오케스트레이션으로 대규모 ML 워크로드를 관리합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">4.</span>
                <span><strong>PyTorchJob/TFJob</strong>은 분산 훈련을 위한 환경 변수를 자동 설정합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">5.</span>
                <span><strong>Helm Charts</strong>는 복잡한 ML 스택을 패키지로 배포하고 버전 관리합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">6.</span>
                <span><strong>Container Registry</strong>는 이미지를 중앙 집중식으로 관리하며, 보안 스캐닝을 제공합니다.</span>
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
              <strong>Chapter 4: ML 프레임워크 기초</strong>
              <br />
              PyTorch와 TensorFlow의 설치, 분산 훈련 설정, 체크포인팅 및 로깅 기법을 실습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
