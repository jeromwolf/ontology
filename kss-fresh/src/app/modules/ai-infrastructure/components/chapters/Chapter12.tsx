'use client'

import React from 'react'
import { Server, Cloud, Cpu, Database, Network, Layers, Zap, TrendingUp, Shield, GitBranch } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter12() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 dark:from-gray-900 dark:to-slate-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-slate-700 to-gray-800 rounded-xl shadow-lg">
              <Server className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-700 to-gray-800 bg-clip-text text-transparent">
                프로덕션 MLOps 아키텍처
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                엔드투엔드 MLOps 플랫폼 설계와 실전 사례
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Cloud className="w-6 h-6 text-slate-700" />
              AI 인프라가 중요한 이유
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                현대의 AI 시스템은 수십억 개의 파라미터를 가진 거대한 모델을 훈련하고,
                초당 수천 건의 추론 요청을 처리하며, 페타바이트 규모의 데이터를 관리합니다.
                이러한 복잡성을 다루기 위해서는 <strong>견고하고 확장 가능한 인프라</strong>가 필수적입니다.
              </p>

              <div className="bg-slate-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-slate-700">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">AI 인프라의 핵심 과제</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <Zap className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
                    <span><strong>확장성(Scalability)</strong>: 모델과 데이터 크기 증가에 대응</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <TrendingUp className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span><strong>성능(Performance)</strong>: 훈련 시간 단축 및 추론 지연 최소화</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Shield className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>안정성(Reliability)</strong>: 장애 복구 및 무중단 서비스</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <GitBranch className="w-5 h-5 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span><strong>재현성(Reproducibility)</strong>: 실험 및 배포의 일관성 보장</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* AI Infrastructure Stack */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Layers className="w-6 h-6 text-slate-700" />
            AI 인프라 스택의 계층 구조
          </h2>

          <div className="space-y-4">
            {/* Layer 5: ML Platform */}
            <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">5. ML Platform Layer</h3>
                <Server className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-3 text-purple-100">
                사용자 친화적인 ML 플랫폼 및 AutoML 도구
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">SageMaker</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Vertex AI</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Azure ML</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Databricks</span>
              </div>
            </div>

            {/* Layer 4: ML Framework */}
            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">4. ML Framework Layer</h3>
                <Cpu className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-3 text-blue-100">
                모델 개발 및 훈련을 위한 딥러닝 프레임워크
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">PyTorch</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">TensorFlow</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">JAX</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">ONNX</span>
              </div>
            </div>

            {/* Layer 3: Orchestration */}
            <div className="bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">3. Orchestration Layer</h3>
                <Network className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-3 text-green-100">
                워크플로우 관리 및 파이프라인 오케스트레이션
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Kubernetes</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Kubeflow</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Airflow</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Argo</span>
              </div>
            </div>

            {/* Layer 2: Compute & Storage */}
            <div className="bg-gradient-to-r from-orange-500 to-red-600 rounded-xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">2. Compute & Storage Layer</h3>
                <Database className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-3 text-orange-100">
                컴퓨팅 리소스 및 데이터 스토리지 인프라
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">GPU Clusters</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">S3/GCS</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Distributed FS</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Cache Layer</span>
              </div>
            </div>

            {/* Layer 1: Hardware */}
            <div className="bg-gradient-to-r from-slate-600 to-gray-700 rounded-xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xl font-bold">1. Hardware Layer</h3>
                <Cpu className="w-8 h-8 opacity-80" />
              </div>
              <p className="mb-3 text-slate-100">
                물리적 하드웨어 및 가속기
              </p>
              <div className="flex flex-wrap gap-2">
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">NVIDIA GPUs</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Google TPUs</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">AWS Inferentia</span>
                <span className="px-3 py-1 bg-white/20 rounded-full text-sm">Custom ASICs</span>
              </div>
            </div>
          </div>
        </section>

        {/* Infrastructure Components */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            주요 인프라 컴포넌트
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Training Infrastructure */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700 hover:shadow-xl transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                  <Cpu className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <h3 className="text-xl font-bold text-slate-800 dark:text-white">Training Infrastructure</h3>
              </div>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                대규모 모델 훈련을 위한 분산 컴퓨팅 환경
              </p>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span><strong>분산 훈련</strong>: Data/Model/Pipeline Parallelism</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span><strong>GPU 클러스터</strong>: 멀티 노드 GPU 오케스트레이션</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span><strong>체크포인팅</strong>: 장애 복구 및 실험 재현</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-500">•</span>
                  <span><strong>하이퍼파라미터 튜닝</strong>: Ray Tune, Optuna 통합</span>
                </li>
              </ul>
            </div>

            {/* Serving Infrastructure */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700 hover:shadow-xl transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-green-100 dark:bg-green-900 rounded-lg">
                  <Server className="w-6 h-6 text-green-600 dark:text-green-400" />
                </div>
                <h3 className="text-xl font-bold text-slate-800 dark:text-white">Serving Infrastructure</h3>
              </div>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                프로덕션 환경에서의 모델 서빙 및 추론
              </p>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-green-500">•</span>
                  <span><strong>모델 최적화</strong>: Quantization, Pruning, Distillation</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">•</span>
                  <span><strong>추론 서버</strong>: TorchServe, TensorRT, Triton</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">•</span>
                  <span><strong>Auto-scaling</strong>: 트래픽 기반 동적 스케일링</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500">•</span>
                  <span><strong>모니터링</strong>: Latency, Throughput, Resource 추적</span>
                </li>
              </ul>
            </div>

            {/* Data Infrastructure */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700 hover:shadow-xl transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-purple-100 dark:bg-purple-900 rounded-lg">
                  <Database className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                </div>
                <h3 className="text-xl font-bold text-slate-800 dark:text-white">Data Infrastructure</h3>
              </div>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                대용량 데이터 처리 및 관리 파이프라인
              </p>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-purple-500">•</span>
                  <span><strong>Data Lake</strong>: S3, GCS, Azure Blob Storage</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500">•</span>
                  <span><strong>Feature Store</strong>: Feast, Tecton, Hopsworks</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500">•</span>
                  <span><strong>데이터 버저닝</strong>: DVC, Pachyderm, LakeFS</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500">•</span>
                  <span><strong>ETL 파이프라인</strong>: Spark, Flink, Beam</span>
                </li>
              </ul>
            </div>

            {/* Monitoring & Observability */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700 hover:shadow-xl transition-shadow">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-orange-100 dark:bg-orange-900 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-orange-600 dark:text-orange-400" />
                </div>
                <h3 className="text-xl font-bold text-slate-800 dark:text-white">Monitoring & Observability</h3>
              </div>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                시스템 상태 모니터링 및 성능 추적
              </p>
              <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <span><strong>메트릭 수집</strong>: Prometheus, Grafana, DataDog</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <span><strong>로그 관리</strong>: ELK Stack, Loki, Fluentd</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <span><strong>분산 추적</strong>: Jaeger, Zipkin, OpenTelemetry</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-orange-500">•</span>
                  <span><strong>모델 모니터링</strong>: 드리프트 감지, 성능 저하 알람</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Modern AI Infrastructure Trends */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-2xl p-8 text-white shadow-xl">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Zap className="w-6 h-6" />
              최신 AI 인프라 트렌드
            </h2>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
                <h3 className="font-bold text-lg mb-3">1. MLOps 표준화</h3>
                <p className="text-slate-200 text-sm">
                  DevOps 원칙을 ML에 적용하여 모델 개발부터 배포까지의 전체 라이프사이클을 자동화하고 표준화
                </p>
              </div>

              <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
                <h3 className="font-bold text-lg mb-3">2. GPU 가상화</h3>
                <p className="text-slate-200 text-sm">
                  MIG (Multi-Instance GPU), vGPU 기술을 통해 GPU 리소스를 효율적으로 분할하고 활용
                </p>
              </div>

              <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
                <h3 className="font-bold text-lg mb-3">3. Edge AI</h3>
                <p className="text-slate-200 text-sm">
                  클라우드가 아닌 엣지 디바이스에서 직접 추론을 수행하여 지연시간 단축 및 프라이버시 보호
                </p>
              </div>

              <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
                <h3 className="font-bold text-lg mb-3">4. Green AI</h3>
                <p className="text-slate-200 text-sm">
                  모델 훈련 및 추론의 에너지 효율성을 개선하여 탄소 배출량을 줄이는 친환경 AI 인프라
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Real-world Example */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            실전 예제: GPT 모델 훈련 인프라
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
            <div className="mb-6">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3">
                OpenAI GPT-3 훈련 인프라 구성
              </h3>
              <p className="text-slate-600 dark:text-slate-400">
                175B 파라미터 모델을 훈련하기 위한 실제 인프라 설정 예시
              </p>
            </div>

            <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
              <pre className="text-slate-800 dark:text-slate-200">
{`# 하드웨어 구성
- GPU 클러스터: 10,000+ NVIDIA V100/A100 GPUs
- 네트워크: InfiniBand (200+ Gbps)
- 스토리지: Petabyte-scale distributed file system

# 소프트웨어 스택
- Framework: PyTorch + DeepSpeed/Megatron
- Orchestration: Kubernetes + Slurm
- Data Pipeline: Ray + Dask
- Monitoring: Prometheus + Grafana

# 훈련 설정
- Parallelism: Data + Model + Pipeline
- Batch Size: 3.2M tokens
- Training Time: ~2-3 months
- Checkpointing: Every 1000 steps
- Cost: Estimated $4-12M USD`}
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
                <span>AI 인프라는 <strong>5개 계층</strong>(Hardware, Compute/Storage, Orchestration, Framework, Platform)으로 구성됩니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">2.</span>
                <span><strong>Training Infrastructure</strong>는 분산 훈련, GPU 클러스터 관리, 체크포인팅을 포함합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">3.</span>
                <span><strong>Serving Infrastructure</strong>는 모델 최적화, 추론 서버, Auto-scaling을 제공합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">4.</span>
                <span><strong>Data Infrastructure</strong>는 Data Lake, Feature Store, 데이터 버저닝을 관리합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">5.</span>
                <span>최신 트렌드는 <strong>MLOps 표준화, GPU 가상화, Edge AI, Green AI</strong>를 포함합니다.</span>
              </li>
            </ul>
          </div>
        </section>

        {/* Next Chapter Preview */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border-2 border-slate-300 dark:border-gray-600">
            <h3 className="text-lg font-bold text-slate-800 dark:text-white mb-2">
              다음 챕터 미리보기
            </h3>
            <p className="text-slate-600 dark:text-slate-400">
              <strong>다음 단계: AI Infrastructure 시뮬레이터</strong>
              <br />
              GPU 클러스터 모니터링, 분산 훈련 시각화, MLOps 파이프라인 빌더 등 실전 시뮬레이터에서 직접 체험해보세요!
            </p>
          </div>
        </section>

        {/* References */}
        <References
          sections={[
            {
              title: '📚 AI 인프라 플랫폼 & 도구',
              icon: 'web' as const,
              color: 'border-slate-500',
              items: [
                {
                  title: 'Kubernetes Documentation',
                  authors: 'Cloud Native Computing Foundation (CNCF)',
                  year: '2025',
                  description: '컨테이너 오케스트레이션 플랫폼 공식 문서 - AI 워크로드 배포 및 관리 가이드',
                  link: 'https://kubernetes.io/docs/'
                },
                {
                  title: 'Ray Documentation',
                  authors: 'Anyscale',
                  year: '2025',
                  description: '분산 Python 프레임워크 - 대규모 ML 훈련 및 하이퍼파라미터 튜닝',
                  link: 'https://docs.ray.io/'
                },
                {
                  title: 'Kubeflow Documentation',
                  authors: 'Kubeflow Community',
                  year: '2025',
                  description: 'Kubernetes 기반 ML 플랫폼 - End-to-End MLOps 파이프라인',
                  link: 'https://www.kubeflow.org/docs/'
                },
                {
                  title: 'NVIDIA Triton Inference Server',
                  authors: 'NVIDIA',
                  year: '2025',
                  description: '프로덕션 AI 모델 서빙 플랫폼 - TensorRT, PyTorch, ONNX 지원',
                  link: 'https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/'
                },
                {
                  title: 'MLflow Documentation',
                  authors: 'Databricks',
                  year: '2025',
                  description: 'ML 라이프사이클 관리 플랫폼 - 실험 추적, 모델 레지스트리, 배포',
                  link: 'https://mlflow.org/docs/latest/index.html'
                }
              ]
            },
            {
              title: '📖 핵심 논문 & 기술 문서',
              icon: 'research' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'Hidden Technical Debt in Machine Learning Systems',
                  authors: 'Sculley, D., et al.',
                  year: '2015',
                  description: 'Google 논문 - ML 시스템의 숨겨진 기술 부채와 MLOps 필요성',
                  link: 'https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf'
                },
                {
                  title: 'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models',
                  authors: 'Rajbhandari, S., et al.',
                  year: '2020',
                  description: 'Microsoft DeepSpeed - 대규모 모델 분산 훈련 최적화 기법',
                  link: 'https://arxiv.org/abs/1910.02054'
                },
                {
                  title: 'TorchServe: Serve, Optimize and Scale PyTorch Models in Production',
                  authors: 'Meta AI',
                  year: '2020',
                  description: 'PyTorch 공식 모델 서빙 프레임워크 - 프로덕션 배포 가이드',
                  link: 'https://pytorch.org/serve/'
                },
                {
                  title: 'Scaling Kubernetes to 7,500 nodes',
                  authors: 'OpenAI',
                  year: '2018',
                  description: 'OpenAI 기술 블로그 - 대규모 딥러닝 워크로드를 위한 Kubernetes 스케일링',
                  link: 'https://openai.com/research/scaling-kubernetes-to-7500-nodes'
                },
                {
                  title: 'Megatron-LM: Training Multi-Billion Parameter Language Models',
                  authors: 'Shoeybi, M., et al. (NVIDIA)',
                  year: '2020',
                  description: '대규모 언어 모델 효율적 훈련 - Model Parallelism 구현',
                  link: 'https://arxiv.org/abs/1909.08053'
                }
              ]
            },
            {
              title: '🛠️ 실전 가이드 & 베스트 프랙티스',
              icon: 'tools' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'Google ML Engineering Best Practices',
                  authors: 'Google Cloud',
                  year: '2025',
                  description: 'Google의 프로덕션 ML 엔지니어링 베스트 프랙티스 - 실무 가이드',
                  link: 'https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning'
                },
                {
                  title: 'AWS SageMaker MLOps Workshop',
                  authors: 'Amazon Web Services',
                  year: '2025',
                  description: 'AWS의 End-to-End MLOps 워크샵 - 파이프라인 자동화 실습',
                  link: 'https://catalog.us-east-1.prod.workshops.aws/workshops/7acdc7d8-0ac0-44b1-8e86-6a2e21abcb8a/'
                },
                {
                  title: 'NVIDIA Deep Learning Performance Guide',
                  authors: 'NVIDIA Developer',
                  year: '2025',
                  description: 'GPU 최적화 완벽 가이드 - Mixed Precision, Multi-GPU 훈련',
                  link: 'https://docs.nvidia.com/deeplearning/performance/index.html'
                },
                {
                  title: 'Monitoring Machine Learning Models in Production',
                  authors: 'Chip Huyen',
                  year: '2022',
                  description: '프로덕션 ML 모델 모니터링 가이드 - 드리프트 감지, 성능 추적',
                  link: 'https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html'
                },
                {
                  title: 'Full Stack Deep Learning - MLOps',
                  authors: 'UC Berkeley',
                  year: '2024',
                  description: 'UC Berkeley 코스 - 프로덕션 ML 시스템 구축 실전 가이드',
                  link: 'https://fullstackdeeplearning.com/course/2022/'
                }
              ]
            }
          ]}
        />
      </div>
    </div>
  )
}
