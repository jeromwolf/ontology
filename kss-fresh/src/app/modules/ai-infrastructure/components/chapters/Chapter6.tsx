'use client'

import React from 'react'
import { Server, Zap, Gauge, Package, Layers, TrendingDown, BarChart3, Settings } from 'lucide-react'

export default function Chapter6() {
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
                모델 서빙과 최적화
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                TorchServe, TensorRT, ONNX Runtime으로 추론 성능 극대화하기
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Gauge className="w-6 h-6 text-slate-700" />
              추론 최적화가 중요한 이유
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                훈련된 모델을 프로덕션에 배포할 때, <strong>추론 지연시간(Latency)</strong>과
                <strong> 처리량(Throughput)</strong>이 비즈니스 성공을 좌우합니다.
                검색 엔진은 100ms 이내, 챗봇은 1초 이내 응답이 필수적입니다.
                모델 서빙 인프라는 <strong>Quantization, Pruning, 배치 처리</strong>로 성능을 극대화합니다.
              </p>

              <div className="bg-slate-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-slate-700">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">추론 성능 지표</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <p className="font-bold text-blue-600 dark:text-blue-400 mb-2">Latency (지연시간)</p>
                    <ul className="space-y-1 text-slate-700 dark:text-slate-300 text-sm">
                      <li>• 단일 요청 처리 시간</li>
                      <li>• 실시간 서비스에 중요</li>
                      <li>• P50, P95, P99 측정</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-bold text-green-600 dark:text-green-400 mb-2">Throughput (처리량)</p>
                    <ul className="space-y-1 text-slate-700 dark:text-slate-300 text-sm">
                      <li>• 초당 처리 요청 수 (QPS)</li>
                      <li>• 배치 처리에 유리</li>
                      <li>• GPU 활용률 최적화</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* TorchServe */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Server className="w-6 h-6 text-slate-700" />
            TorchServe: PyTorch 모델 서빙
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                TorchServe 설치 및 기본 설정
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto mb-4">
                <pre className="text-slate-800 dark:text-slate-200">
{`# TorchServe 설치
pip install torchserve torch-model-archiver torch-workflow-archiver

# 모델 아카이브 생성
torch-model-archiver \\
  --model-name resnet50 \\
  --version 1.0 \\
  --model-file model.py \\
  --serialized-file resnet50.pth \\
  --handler image_classifier \\
  --extra-files index_to_name.json

# 모델 스토어 생성
mkdir model_store
mv resnet50.mar model_store/

# TorchServe 시작
torchserve \\
  --start \\
  --ncs \\
  --model-store model_store \\
  --models resnet50.mar`}
                </pre>
              </div>
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-sm font-bold text-slate-800 dark:text-white mb-2">추론 API 호출</p>
                <div className="font-mono text-sm text-slate-700 dark:text-slate-300">
                  curl -X POST http://localhost:8080/predictions/resnet50 -T image.jpg
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Custom Handler 작성
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# custom_handler.py
from ts.torch_handler.base_handler import BaseHandler
import torch
import json

class CustomHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        """모델 로딩 (서버 시작시 1회 실행)"""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # 모델 로드
        self.model = torch.jit.load(f"{model_dir}/model.pt")
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        """입력 데이터 전처리"""
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            # 이미지 디코딩, 리사이징, 정규화
            image = self.transform(image)
            images.append(image)
        return torch.stack(images)

    def inference(self, data):
        """추론 실행"""
        with torch.no_grad():
            outputs = self.model(data)
        return outputs

    def postprocess(self, inference_output):
        """결과 후처리"""
        probabilities = torch.nn.functional.softmax(
            inference_output, dim=1
        )
        top5_prob, top5_idx = torch.topk(probabilities, 5)

        results = []
        for prob, idx in zip(top5_prob, top5_idx):
            result = {
                "predictions": [
                    {"class": self.mapping[i.item()], "score": p.item()}
                    for p, i in zip(prob, idx)
                ]
            }
            results.append(result)
        return results`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Settings className="w-6 h-6" />
                TorchServe 성능 튜닝
              </h3>
              <div className="bg-white/10 rounded-lg p-6 font-mono text-sm backdrop-blur overflow-x-auto">
                <pre className="text-white">
{`# config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081

# Worker 설정
default_workers_per_model=4  # GPU당 Worker 수
job_queue_size=100           # 대기열 크기
max_request_size=10485760    # 10MB
max_response_size=10485760

# Batching 설정
batch_size=8                 # 동적 배치 크기
max_batch_delay=100          # 100ms 대기

# GPU 설정
number_of_gpu=4
number_of_netty_threads=32

# 메트릭 수집
metrics_mode=prometheus`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* TensorRT */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Zap className="w-6 h-6 text-slate-700" />
            TensorRT: NVIDIA GPU 최적화
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                TensorRT로 모델 최적화
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                TensorRT는 <strong>레이어 융합, 정밀도 감소, 커널 최적화</strong>로 추론 속도를 5-10배 향상시킵니다.
              </p>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import torch
import torch_tensorrt

# PyTorch 모델 로드
model = torch.load("resnet50.pth").eval().cuda()

# TensorRT로 컴파일
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        shape=[1, 3, 224, 224],
        dtype=torch.half  # FP16 사용
    )],
    enabled_precisions={torch.half},  # FP16 활성화
    workspace_size=1 << 30  # 1GB workspace
)

# 추론 실행 (5-10배 빠름!)
with torch.no_grad():
    input_data = torch.randn(1, 3, 224, 224).half().cuda()
    output = trt_model(input_data)

# 최적화된 모델 저장
torch.jit.save(trt_model, "resnet50_trt.ts")`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                TensorRT 최적화 기법
              </h3>
              <div className="space-y-3">
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <p className="font-bold text-purple-700 dark:text-purple-300 mb-2">Layer Fusion</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Conv + BatchNorm + ReLU를 하나의 커널로 융합하여 메모리 대역폭 절감
                  </p>
                </div>
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <p className="font-bold text-blue-700 dark:text-blue-300 mb-2">Precision Calibration</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    INT8 Quantization을 위한 자동 캘리브레이션으로 정확도 유지
                  </p>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <p className="font-bold text-green-700 dark:text-green-300 mb-2">Kernel Auto-Tuning</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    특정 GPU 아키텍처에 최적화된 CUDA 커널 자동 선택
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ONNX Runtime */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Package className="w-6 h-6 text-slate-700" />
            ONNX Runtime: 크로스 플랫폼 추론
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                PyTorch → ONNX 변환
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import torch
import onnx
import onnxruntime as ort

# PyTorch 모델 로드
model = torch.load("model.pth").eval()
dummy_input = torch.randn(1, 3, 224, 224)

# ONNX로 변환
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,  # 상수 폴딩 최적화
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},  # 동적 배치 크기
        'output': {0: 'batch_size'}
    }
)

# ONNX 모델 검증
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# ONNX Runtime으로 추론
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

input_data = dummy_input.numpy()
outputs = session.run(None, {'input': input_data})
print(f"Output shape: {outputs[0].shape}")`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">ONNX Runtime Execution Providers</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">CUDAExecutionProvider</p>
                  <p className="text-sm text-blue-100">NVIDIA GPU 가속 (가장 빠름)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">TensorrtExecutionProvider</p>
                  <p className="text-sm text-blue-100">TensorRT 통합 최적화</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">OpenVINOExecutionProvider</p>
                  <p className="text-sm text-blue-100">Intel CPU/GPU 최적화</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">CoreMLExecutionProvider</p>
                  <p className="text-sm text-blue-100">Apple Silicon 지원</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Quantization */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <TrendingDown className="w-6 h-6 text-slate-700" />
            Quantization: 정밀도 감소 최적화
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Quantization 기법 비교
              </h3>
              <div className="grid md:grid-cols-3 gap-4 mb-6">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <p className="font-bold text-blue-700 dark:text-blue-300 mb-2">Dynamic Quantization</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    실행 시점에 활성화만 INT8 변환
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-500">
                    ✓ 설정 간단<br/>
                    ✓ 캘리브레이션 불필요<br/>
                    △ 속도 향상 중간
                  </p>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <p className="font-bold text-green-700 dark:text-green-300 mb-2">Static Quantization</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    파라미터와 활성화 모두 INT8
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-500">
                    ✓ 최대 속도 향상<br/>
                    △ 캘리브레이션 필요<br/>
                    △ 정확도 약간 하락
                  </p>
                </div>
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                  <p className="font-bold text-purple-700 dark:text-purple-300 mb-2">Quantization-Aware Training</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    훈련 중 양자화 시뮬레이션
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-500">
                    ✓ 정확도 유지<br/>
                    △ 재훈련 필요<br/>
                    △ 시간 소요
                  </p>
                </div>
              </div>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import torch
from torch.quantization import quantize_dynamic, get_default_qconfig

# 1. Dynamic Quantization (가장 간단)
model_fp32 = MyModel().eval()
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.LSTM},  # 양자화할 레이어
    dtype=torch.qint8
)

# 2. Static Quantization (캘리브레이션 필요)
model_fp32.qconfig = get_default_qconfig('x86')  # 또는 'qnnpack' (모바일)
torch.quantization.prepare(model_fp32, inplace=True)

# 캘리브레이션 (대표 데이터로 실행)
with torch.no_grad():
    for data in calibration_dataloader:
        model_fp32(data)

# 양자화 완료
model_int8 = torch.quantization.convert(model_fp32, inplace=True)

# 3. Quantization-Aware Training
model_fp32.qconfig = get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model_fp32)

# 훈련 (양자화 aware)
for epoch in range(num_epochs):
    train(model_prepared, ...)

# 양자화 변환
model_int8 = torch.quantization.convert(model_prepared.eval())`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                성능 비교: FP32 vs FP16 vs INT8
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-red-50 dark:bg-red-900/20 rounded">
                  <span className="text-slate-700 dark:text-slate-300">FP32 (기본)</span>
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-slate-600 dark:text-slate-400">1.0x 속도</span>
                    <span className="font-bold text-red-600 dark:text-red-400">100% 메모리</span>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                  <span className="text-slate-700 dark:text-slate-300">FP16 (Half Precision)</span>
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-slate-600 dark:text-slate-400">2-3x 속도</span>
                    <span className="font-bold text-orange-600 dark:text-orange-400">50% 메모리</span>
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded">
                  <span className="text-slate-700 dark:text-slate-300">INT8 (Quantized)</span>
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-slate-600 dark:text-slate-400">3-4x 속도</span>
                    <span className="font-bold text-green-600 dark:text-green-400">25% 메모리</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Model Pruning */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Layers className="w-6 h-6 text-slate-700" />
            Model Pruning: 가지치기 최적화
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                PyTorch Pruning 실습
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import torch
import torch.nn.utils.prune as prune

# 모델 정의
model = MyModel()

# 1. Unstructured Pruning (개별 파라미터)
module = model.conv1
prune.l1_unstructured(
    module,
    name='weight',
    amount=0.3  # 30% 파라미터 제거
)

# 2. Structured Pruning (필터 단위)
prune.ln_structured(
    module,
    name='weight',
    amount=0.5,  # 50% 필터 제거
    n=2,
    dim=0  # 출력 채널 방향
)

# 3. Global Pruning (전체 모델)
parameters_to_prune = [
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc, 'weight'),
]

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2  # 전체의 20% 제거
)

# Pruning 영구 적용 (mask 제거)
for module, name in parameters_to_prune:
    prune.remove(module, name)

# Sparsity 확인
total_params = 0
zero_params = 0
for param in model.parameters():
    total_params += param.numel()
    zero_params += (param == 0).sum().item()

sparsity = 100. * zero_params / total_params
print(f'Model Sparsity: {sparsity:.2f}%')`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">Pruning 전략</h3>
              <div className="space-y-3">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Magnitude-based Pruning</p>
                  <p className="text-sm text-purple-100">
                    가중치 절댓값이 작은 연결 제거. 가장 단순하고 효과적.
                  </p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Iterative Pruning</p>
                  <p className="text-sm text-purple-100">
                    점진적으로 가지치기 → Fine-tuning 반복. 정확도 유지.
                  </p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Lottery Ticket Hypothesis</p>
                  <p className="text-sm text-purple-100">
                    초기화된 서브네트워크를 찾아 처음부터 재훈련.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Batch Inference */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-slate-700" />
            배치 추론 최적화
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                동적 배칭 (Dynamic Batching)
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                여러 요청을 묶어 한 번에 처리하여 <strong>GPU 활용률을 극대화</strong>합니다.
              </p>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import asyncio
import torch

class DynamicBatcher:
    def __init__(self, model, max_batch_size=32, max_wait_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []
        self.lock = asyncio.Lock()

    async def predict(self, input_data):
        """단일 요청 처리"""
        future = asyncio.Future()

        async with self.lock:
            self.queue.append((input_data, future))

            # 배치가 차거나 타임아웃 도달 시 처리
            if len(self.queue) >= self.max_batch_size:
                await self._process_batch()

        # 타임아웃 설정
        try:
            result = await asyncio.wait_for(
                future,
                timeout=self.max_wait_ms / 1000
            )
            return result
        except asyncio.TimeoutError:
            async with self.lock:
                if len(self.queue) > 0:
                    await self._process_batch()
            return await future

    async def _process_batch(self):
        """배치 처리"""
        if not self.queue:
            return

        # 큐에서 배치 추출
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]

        inputs = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        # 배치 추론
        batch_input = torch.stack(inputs)
        with torch.no_grad():
            batch_output = self.model(batch_input)

        # 결과 분배
        for i, future in enumerate(futures):
            future.set_result(batch_output[i])

# 사용 예시
batcher = DynamicBatcher(model, max_batch_size=16, max_wait_ms=100)
result = await batcher.predict(input_tensor)`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Latency vs Throughput 트레이드오프
              </h3>
              <div className="space-y-3">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <p className="font-bold text-blue-700 dark:text-blue-300">Batch Size = 1</p>
                    <span className="text-xs px-2 py-1 bg-blue-200 dark:bg-blue-800 rounded">Low Latency</span>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    지연시간 최소, GPU 활용률 낮음 (~20%), 실시간 서비스
                  </p>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <p className="font-bold text-green-700 dark:text-green-300">Batch Size = 32</p>
                    <span className="text-xs px-2 py-1 bg-green-200 dark:bg-green-800 rounded">High Throughput</span>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    처리량 최대, 지연시간 증가, GPU 활용률 높음 (~80%), 배치 처리
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* GPU Memory Optimization */}
        <section className="mb-12">
          <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-8 text-white shadow-xl">
            <h2 className="text-2xl font-bold mb-6">GPU 메모리 최적화 기법</h2>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                <p className="font-bold mb-2">Gradient Checkpointing</p>
                <p className="text-sm text-slate-200">
                  활성화를 저장하지 않고 필요시 재계산. 메모리 ↓, 시간 ↑
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                <p className="font-bold mb-2">Model Parallelism</p>
                <p className="text-sm text-slate-200">
                  거대 모델을 여러 GPU에 분할. Latency 약간 증가.
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                <p className="font-bold mb-2">KV Cache Optimization</p>
                <p className="text-sm text-slate-200">
                  Transformer의 Key/Value 캐싱으로 반복 계산 제거.
                </p>
              </div>
              <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                <p className="font-bold mb-2">Flash Attention</p>
                <p className="text-sm text-slate-200">
                  메모리 효율적인 Attention 커널. 2-4배 속도 향상.
                </p>
              </div>
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
                <span><strong>TorchServe</strong>는 PyTorch 모델을 위한 프로덕션급 서빙 프레임워크로 동적 배칭을 지원합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">2.</span>
                <span><strong>TensorRT</strong>는 NVIDIA GPU에서 레이어 융합과 커널 최적화로 5-10배 속도 향상을 제공합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">3.</span>
                <span><strong>ONNX Runtime</strong>은 크로스 플랫폼 추론을 위한 프레임워크 중립적 솔루션입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">4.</span>
                <span><strong>Quantization</strong>은 INT8/FP16으로 정밀도를 낮춰 메모리 50-75% 절감 및 3-4배 속도 향상을 달성합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">5.</span>
                <span><strong>Model Pruning</strong>은 중요하지 않은 파라미터를 제거하여 모델 크기를 30-90% 줄입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">6.</span>
                <span><strong>배치 추론</strong>은 Latency와 Throughput 간 트레이드오프를 조절하여 GPU 활용률을 최적화합니다.</span>
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
              <strong>Chapter 7: 데이터 파이프라인 구축</strong>
              <br />
              Feature Store, Data Versioning, ETL 파이프라인 설계 및 대용량 데이터 처리 기법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
