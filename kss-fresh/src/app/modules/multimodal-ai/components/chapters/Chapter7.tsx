'use client'

import React from 'react'
import { Zap, Cpu, Gauge, Smartphone, BookOpen, Settings, Rocket, Timer } from 'lucide-react'

export default function Chapter7() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-purple-50 dark:from-gray-900 dark:to-purple-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl">
              <Zap className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent">
                실시간 멀티모달 AI
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                저지연 멀티모달 파이프라인 구현
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-6 h-6 text-violet-600" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              실시간 멀티모달 AI의 중요성
            </h2>
          </div>

          <div className="prose dark:prose-invert max-w-none">
            <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
              실시간 멀티모달 AI는 밀리초 단위의 저지연(Low Latency)으로 여러 모달리티를 처리하는 시스템입니다.
              자율주행, 로봇 제어, AR/VR, 실시간 번역 등 시간에 민감한 응용 분야에서 필수적입니다.
              단순히 모델을 빠르게 만드는 것을 넘어, 전체 파이프라인의 효율성을 극대화해야 합니다.
            </p>

            <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6 border border-violet-200 dark:border-violet-800">
              <p className="text-violet-900 dark:text-violet-100 font-semibold mb-2">
                💡 실시간의 정의
              </p>
              <p className="text-violet-800 dark:text-violet-200">
                "실시간"의 기준은 응용 분야에 따라 다릅니다.
                자율주행: &lt;100ms, 비디오 스트리밍: &lt;200ms, 챗봇: &lt;1s, AR/VR: &lt;20ms.
                사용자가 지연을 인지하지 못하는 수준이 목표입니다.
              </p>
            </div>
          </div>
        </section>

        {/* 저지연 파이프라인 설계 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Timer className="w-6 h-6 text-violet-600" />
            저지연 파이프라인 설계 원칙
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {[
              {
                icon: <Rocket className="w-8 h-8" />,
                principle: 'Early Termination',
                title: '조기 종료',
                description: '불필요한 계산을 건너뛰고 충분한 신뢰도에서 조기 종료',
                technique: 'Cascade Classifiers, Adaptive Inference',
                example: 'YOLO의 NMS(Non-Maximum Suppression)로 중복 박스 제거',
                savings: '계산량 30-50% 감소',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                icon: <Cpu className="w-8 h-8" />,
                principle: 'Model Compression',
                title: '모델 압축',
                description: '모델 크기와 계산량을 줄이면서 성능 유지',
                technique: 'Quantization, Pruning, Knowledge Distillation',
                example: 'CLIP을 INT8로 양자화하여 4배 빠르게',
                savings: '메모리 75% 감소, 속도 2-4배',
                color: 'from-purple-500 to-pink-500'
              },
              {
                icon: <Zap className="w-8 h-8" />,
                principle: 'Asynchronous Processing',
                title: '비동기 처리',
                description: '모달리티별로 독립적으로 처리하고 결과를 나중에 통합',
                technique: 'Multi-threading, GPU Streaming',
                example: '이미지 인코딩과 텍스트 인코딩을 병렬로 수행',
                savings: '총 지연시간 40% 감소',
                color: 'from-green-500 to-emerald-500'
              },
              {
                icon: <Settings className="w-8 h-8" />,
                principle: 'Caching',
                title: '캐싱',
                description: '반복적으로 사용되는 임베딩을 미리 계산하여 저장',
                technique: 'Embedding Cache, KV Cache (Transformer)',
                example: 'CLIP 텍스트 임베딩을 미리 계산하여 재사용',
                savings: '반복 요청 시 99% 시간 절약',
                color: 'from-orange-500 to-red-500'
              }
            ].map((principle, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${principle.color} text-white mb-4`}>
                  {principle.icon}
                </div>
                <div className="mb-2">
                  <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    {principle.principle}
                  </span>
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {principle.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  {principle.description}
                </p>
                <div className="space-y-2">
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">기법</p>
                    <p className="text-sm text-gray-700 dark:text-gray-300">{principle.technique}</p>
                  </div>
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                    <p className="text-xs text-blue-700 dark:text-blue-300 mb-1">예시</p>
                    <p className="text-sm text-blue-900 dark:text-blue-100">{principle.example}</p>
                  </div>
                  <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                    <p className="text-sm font-semibold text-violet-900 dark:text-violet-100">
                      절감: {principle.savings}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 최적화 기법 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            ⚡ 핵심 최적화 기법
          </h2>

          <div className="space-y-6">
            {[
              {
                name: 'Quantization (양자화)',
                description: '모델 가중치와 활성화를 FP32에서 INT8/INT4로 변환',
                types: [
                  'Post-Training Quantization (PTQ): 학습 후 양자화, 빠르지만 정확도 손실',
                  'Quantization-Aware Training (QAT): 학습 중 양자화 시뮬레이션, 정확도 유지',
                  'Dynamic Quantization: 실행 시간에 동적으로 양자화'
                ],
                code: `# PyTorch INT8 양자화
import torch
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)`,
                impact: 'CLIP 모델 크기 1.7GB → 450MB, 추론 속도 4배',
                color: 'blue'
              },
              {
                name: 'Pruning (가지치기)',
                description: '중요도가 낮은 뉴런이나 연결을 제거하여 모델 경량화',
                types: [
                  'Unstructured Pruning: 개별 가중치 제거 (희소 행렬)',
                  'Structured Pruning: 전체 필터/채널 제거 (dense 연산 유지)',
                  'Lottery Ticket Hypothesis: 초기화부터 sparse 모델 찾기'
                ],
                code: `# PyTorch Pruning
import torch.nn.utils.prune as prune
prune.l1_unstructured(module, name='weight', amount=0.3)`,
                impact: '파라미터 30% 제거해도 정확도 1% 이내 손실',
                color: 'purple'
              },
              {
                name: 'Knowledge Distillation',
                description: '큰 Teacher 모델의 지식을 작은 Student 모델로 전이',
                types: [
                  'Soft Label Distillation: Teacher의 소프트 확률 분포 학습',
                  'Feature Distillation: 중간 레이어 특징 매칭',
                  'Attention Distillation: Attention map 일치시키기'
                ],
                code: `# Distillation Loss
loss = alpha * CE(student_logits, hard_labels) +
       (1-alpha) * KL(student_logits/T, teacher_logits/T)`,
                impact: 'BERT-Large → DistilBERT: 40% 작고 60% 빠르지만 97% 성능 유지',
                color: 'green'
              }
            ].map((technique, idx) => (
              <div key={idx} className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className={`w-10 h-10 rounded-lg bg-${technique.color}-500 text-white flex items-center justify-center font-bold text-xl`}>
                    {idx + 1}
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                    {technique.name}
                  </h3>
                </div>
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  {technique.description}
                </p>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
                  <p className="font-semibold text-gray-900 dark:text-white mb-2">주요 방법</p>
                  <ul className="space-y-2">
                    {technique.types.map((type, i) => (
                      <li key={i} className="text-sm text-gray-600 dark:text-gray-400 flex gap-2">
                        <span className="text-violet-600">•</span>
                        <span>{type}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-gray-900 rounded-lg p-4 mb-3 overflow-x-auto">
                  <pre className="text-sm text-green-400">{technique.code}</pre>
                </div>

                <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                  <p className="text-sm text-violet-900 dark:text-violet-100">
                    <span className="font-semibold">영향:</span> {technique.impact}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 엣지 디바이스 배포 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Smartphone className="w-6 h-6 text-violet-600" />
            엣지 디바이스 배포 전략
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                엣지 AI의 장점과 도전 과제
              </h3>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-5 border border-green-200 dark:border-green-800">
                  <h4 className="font-bold text-green-900 dark:text-green-100 mb-3 flex items-center gap-2">
                    <span className="text-green-600">✅</span>
                    장점
                  </h4>
                  <ul className="space-y-2 text-green-800 dark:text-green-200">
                    <li className="flex gap-2">
                      <span>•</span>
                      <span><strong>저지연:</strong> 클라우드 왕복 시간 제거 (10-100ms 절약)</span>
                    </li>
                    <li className="flex gap-2">
                      <span>•</span>
                      <span><strong>프라이버시:</strong> 데이터가 디바이스를 떠나지 않음</span>
                    </li>
                    <li className="flex gap-2">
                      <span>•</span>
                      <span><strong>오프라인:</strong> 인터넷 없이도 작동</span>
                    </li>
                    <li className="flex gap-2">
                      <span>•</span>
                      <span><strong>비용 절감:</strong> 클라우드 요금 감소</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-red-50 dark:bg-red-900/20 rounded-xl p-5 border border-red-200 dark:border-red-800">
                  <h4 className="font-bold text-red-900 dark:text-red-100 mb-3 flex items-center gap-2">
                    <span className="text-red-600">⚠️</span>
                    도전 과제
                  </h4>
                  <ul className="space-y-2 text-red-800 dark:text-red-200">
                    <li className="flex gap-2">
                      <span>•</span>
                      <span><strong>제한된 자원:</strong> 메모리, 배터리, 계산 능력 부족</span>
                    </li>
                    <li className="flex gap-2">
                      <span>•</span>
                      <span><strong>모델 크기:</strong> 수백 MB ~ 수 GB 모델 압축 필요</span>
                    </li>
                    <li className="flex gap-2">
                      <span>•</span>
                      <span><strong>다양한 하드웨어:</strong> CPU, GPU, NPU 각기 다른 최적화</span>
                    </li>
                    <li className="flex gap-2">
                      <span>•</span>
                      <span><strong>업데이트:</strong> 모델 배포 및 버전 관리 복잡</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                엣지 AI 프레임워크 비교
              </h3>

              <div className="grid gap-6">
                {[
                  {
                    name: 'TensorFlow Lite',
                    description: 'Google의 모바일/임베디드 ML 프레임워크',
                    platforms: 'Android, iOS, Raspberry Pi, Microcontrollers',
                    features: ['INT8/FP16 양자화', 'GPU/NNAPI Delegate', '8MB 이하 모델 최적화'],
                    use_case: '스마트폰 앱, IoT 디바이스',
                    color: 'from-blue-500 to-cyan-500'
                  },
                  {
                    name: 'ONNX Runtime',
                    description: 'Microsoft의 크로스 플랫폼 추론 엔진',
                    platforms: 'CPU, GPU, Web (WASM), Mobile',
                    features: ['다양한 프레임워크 지원 (PyTorch, TF, ONNX)', '그래프 최적화', 'DirectML/CoreML 가속'],
                    use_case: '범용 엣지 배포, 하이브리드 클라우드-엣지',
                    color: 'from-purple-500 to-pink-500'
                  },
                  {
                    name: 'Core ML (Apple)',
                    description: 'Apple 생태계 전용 ML 프레임워크',
                    platforms: 'iOS, macOS, watchOS, tvOS',
                    features: ['Neural Engine 최적화', 'On-device Training', 'Privacy 최우선'],
                    use_case: 'iPhone, iPad, Mac 앱',
                    color: 'from-green-500 to-emerald-500'
                  },
                  {
                    name: 'TensorRT (NVIDIA)',
                    description: 'NVIDIA GPU 전용 고성능 추론 엔진',
                    platforms: 'NVIDIA GPU (Jetson, Data Center)',
                    features: ['FP16/INT8 최적화', '레이어 퓨전', '동적 텐서 메모리'],
                    use_case: '자율주행, 로봇, 엣지 서버',
                    color: 'from-orange-500 to-red-500'
                  }
                ].map((framework, idx) => (
                  <div key={idx} className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-5">
                    <div className="flex items-start gap-3 mb-3">
                      <div className={`flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br ${framework.color} flex items-center justify-center text-white font-bold`}>
                        {idx + 1}
                      </div>
                      <div>
                        <h4 className="font-bold text-gray-900 dark:text-white">{framework.name}</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">{framework.description}</p>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">플랫폼</p>
                        <p className="text-sm text-gray-700 dark:text-gray-300">{framework.platforms}</p>
                      </div>
                      <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">주요 기능</p>
                        <ul className="space-y-1">
                          {framework.features.map((feature, i) => (
                            <li key={i} className="text-sm text-gray-700 dark:text-gray-300 flex gap-2">
                              <span className="text-violet-600">•</span>
                              <span>{feature}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3">
                        <p className="text-xs text-violet-700 dark:text-violet-300 mb-1">활용 사례</p>
                        <p className="text-sm text-violet-900 dark:text-violet-100">{framework.use_case}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* 스트리밍 처리 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🌊 스트리밍 처리 (Streaming Processing)
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              비디오, 오디오 등 연속적인 데이터를 실시간으로 처리하려면 스트리밍 파이프라인이 필요합니다.
              전체 데이터를 기다리지 않고 청크 단위로 처리하여 지연을 최소화합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                실시간 비디오 분석 파이프라인
              </h3>
              <div className="space-y-4">
                {[
                  {
                    stage: 'Frame Capture',
                    desc: '카메라에서 30fps로 프레임 수집 (33ms 간격)',
                    optimization: 'H.264 하드웨어 디코딩으로 CPU 부하 감소'
                  },
                  {
                    stage: 'Preprocessing',
                    desc: '리사이즈, 정규화, 배치 구성 (병렬 처리)',
                    optimization: 'GPU에서 직접 전처리하여 CPU-GPU 전송 최소화'
                  },
                  {
                    stage: 'Inference',
                    desc: 'YOLO로 객체 탐지, CLIP으로 분류',
                    optimization: 'TensorRT로 FP16 추론, Batch 크기 최적화'
                  },
                  {
                    stage: 'Post-processing',
                    desc: 'NMS, 추적(Tracking), 결과 통합',
                    optimization: 'CUDA 커널로 GPU에서 NMS 수행'
                  },
                  {
                    stage: 'Output',
                    desc: '결과를 화면에 렌더링 또는 클라우드로 전송',
                    optimization: 'WebSocket/gRPC로 저지연 스트리밍'
                  }
                ].map((stage, idx) => (
                  <div key={idx} className="flex gap-4 items-start">
                    <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white font-bold">
                      {idx + 1}
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold text-gray-900 dark:text-white mb-1">{stage.stage}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{stage.desc}</p>
                      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-2">
                        <p className="text-xs text-blue-900 dark:text-blue-100">
                          <span className="font-semibold">최적화:</span> {stage.optimization}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-5 border border-blue-200 dark:border-blue-800">
                <h4 className="font-bold text-blue-900 dark:text-blue-100 mb-3">
                  Frame Skipping
                </h4>
                <p className="text-sm text-blue-800 dark:text-blue-200 mb-3">
                  모든 프레임을 처리하지 않고 일부 건너뛰어 계산량 감소.
                </p>
                <p className="text-xs text-blue-700 dark:text-blue-300">
                  예: 30fps → 10fps 처리로 3배 속도 향상, 시각적 품질은 유지
                </p>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-5 border border-green-200 dark:border-green-800">
                <h4 className="font-bold text-green-900 dark:text-green-100 mb-3">
                  Temporal Consistency
                </h4>
                <p className="text-sm text-green-800 dark:text-green-200 mb-3">
                  이전 프레임 결과를 활용하여 현재 프레임 예측 가속.
                </p>
                <p className="text-xs text-green-700 dark:text-green-300">
                  예: Object Tracking으로 매번 전체 detection 대신 위치만 업데이트
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 실전 응용 사례 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🚀 실전 응용 사례
          </h2>

          <div className="grid gap-6">
            {[
              {
                application: '자율주행 인지 시스템',
                latency: '< 100ms',
                description: '카메라, LiDAR, 레이더 데이터를 실시간 통합하여 객체 탐지 및 경로 예측',
                tech: 'TensorRT INT8 + CUDA Streaming + Multi-GPU Pipeline',
                challenge: '안전성 critical, 99.999% 신뢰도 필요',
                solution: 'Redundancy (다중 모델 앙상블), Fallback 메커니즘',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                application: '실시간 AR 번역',
                latency: '< 200ms',
                description: '카메라로 촬영한 텍스트를 실시간 번역하여 화면에 오버레이',
                tech: 'Google Translate AR: OCR + NMT + Text Rendering on-device',
                challenge: '다양한 폰트, 조명, 각도에서 작동',
                solution: 'On-device OCR (TFLite) + Cloud NMT (필요시)',
                color: 'from-purple-500 to-pink-500'
              },
              {
                application: '스마트 스피커 (Alexa, Google Home)',
                latency: '< 1s',
                description: '음성 명령을 인식하고 응답 생성 (wake word → ASR → NLU → TTS)',
                tech: 'Edge Wake Word Detection + Cloud ASR/TTS',
                challenge: '항상 대기 상태, 낮은 전력 소비',
                solution: 'Wake word는 on-device, 나머지는 클라우드로 분산',
                color: 'from-green-500 to-emerald-500'
              },
              {
                application: 'VR 헤드셋 (Meta Quest)',
                latency: '< 20ms (Motion-to-Photon)',
                description: '헤드 움직임 추적, 손 제스처 인식, 공간 매핑',
                tech: 'SLAM + Hand Tracking on Snapdragon XR2',
                challenge: '20ms 이상 지연 시 VR sickness 발생',
                solution: '경량 모델 + 예측 렌더링 + Asynchronous Timewarp',
                color: 'from-orange-500 to-red-500'
              }
            ].map((app, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className="flex items-start gap-4 mb-4">
                  <div className={`flex-shrink-0 w-12 h-12 rounded-lg bg-gradient-to-br ${app.color} flex items-center justify-center text-white font-bold text-xl`}>
                    {idx + 1}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                      {app.application}
                    </h3>
                    <div className="inline-block px-3 py-1 rounded-full bg-red-100 dark:bg-red-900/30 text-red-900 dark:text-red-100 text-xs font-bold mt-1">
                      지연시간 요구: {app.latency}
                    </div>
                  </div>
                </div>

                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  {app.description}
                </p>

                <div className="space-y-2">
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      <span className="font-semibold">기술:</span> {app.tech}
                    </p>
                  </div>
                  <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-3">
                    <p className="text-sm text-amber-900 dark:text-amber-100">
                      <span className="font-semibold">도전:</span> {app.challenge}
                    </p>
                  </div>
                  <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                    <p className="text-sm text-violet-900 dark:text-violet-100">
                      <span className="font-semibold">해결책:</span> {app.solution}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 학습 목표 요약 */}
        <section className="bg-gradient-to-br from-violet-600 to-purple-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">📚 이 챕터에서 배운 내용</h2>
          <ul className="space-y-3">
            {[
              '저지연 파이프라인 설계 원칙 (조기 종료, 압축, 비동기 처리, 캐싱)',
              '최적화 기법: Quantization (INT8), Pruning, Knowledge Distillation',
              '엣지 디바이스 배포 프레임워크 (TFLite, ONNX Runtime, Core ML, TensorRT)',
              '스트리밍 처리: Frame Skipping, Temporal Consistency',
              '실전 응용: 자율주행 (<100ms), AR 번역 (<200ms), VR (<20ms)',
              '클라우드-엣지 하이브리드 아키텍처'
            ].map((item, idx) => (
              <li key={idx} className="flex items-start gap-3">
                <span className="text-violet-200 mt-1">✓</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>

          <div className="mt-8 pt-6 border-t border-violet-400">
            <p className="text-violet-100">
              <span className="font-semibold">다음 챕터:</span> 멀티모달 응용 분야를 살펴봅니다.
              VQA(Visual Question Answering), 이미지 캡셔닝, 비디오 이해 등 실전 태스크를 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
