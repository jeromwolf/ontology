'use client'

import React from 'react'
import { Cpu, Zap, Layers, Box, Grid, BarChart3, CircuitBoard, Gauge } from 'lucide-react'

export default function Chapter2() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 dark:from-gray-900 dark:to-slate-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-slate-700 to-gray-800 rounded-xl shadow-lg">
              <Cpu className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-700 to-gray-800 bg-clip-text text-transparent">
                GPU 아키텍처와 병렬 컴퓨팅
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                NVIDIA GPU 구조와 대규모 병렬 처리 전략의 이해
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <CircuitBoard className="w-6 h-6 text-slate-700" />
              GPU가 AI를 가속하는 이유
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                딥러닝 모델 훈련은 <strong>대규모 행렬 연산</strong>을 필요로 합니다.
                CPU는 순차적 처리에 최적화된 반면, GPU는 <strong>수천 개의 코어</strong>를 활용해
                동일한 연산을 동시에 수행할 수 있어 AI 워크로드에 이상적입니다.
              </p>

              <div className="bg-slate-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-slate-700">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">CPU vs GPU 비교</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <p className="font-bold text-blue-600 dark:text-blue-400 mb-2">CPU (Central Processing Unit)</p>
                    <ul className="space-y-1 text-slate-700 dark:text-slate-300 text-sm">
                      <li>• 4-64 코어 (고성능)</li>
                      <li>• 고속 순차 처리</li>
                      <li>• 복잡한 분기 처리</li>
                      <li>• 범용 컴퓨팅</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-bold text-green-600 dark:text-green-400 mb-2">GPU (Graphics Processing Unit)</p>
                    <ul className="space-y-1 text-slate-700 dark:text-slate-300 text-sm">
                      <li>• 1,000-10,000+ 코어</li>
                      <li>• 대규모 병렬 처리</li>
                      <li>• 행렬 연산 최적화</li>
                      <li>• AI/그래픽 특화</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* NVIDIA GPU Architecture */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Layers className="w-6 h-6 text-slate-700" />
            NVIDIA GPU 아키텍처
          </h2>

          <div className="space-y-6">
            {/* Architecture Generations */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                GPU 세대별 발전
              </h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-4 bg-slate-50 dark:bg-gray-900 rounded-lg">
                  <div className="font-mono text-sm font-bold text-slate-600 dark:text-slate-400 w-24">2017</div>
                  <div className="flex-1">
                    <p className="font-bold text-slate-800 dark:text-white">Volta (V100)</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">Tensor Core 도입, 125 TFLOPS (FP16)</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-slate-50 dark:bg-gray-900 rounded-lg">
                  <div className="font-mono text-sm font-bold text-slate-600 dark:text-slate-400 w-24">2020</div>
                  <div className="flex-1">
                    <p className="font-bold text-slate-800 dark:text-white">Ampere (A100)</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">3세대 Tensor Core, 312 TFLOPS (FP16), MIG 지원</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-slate-50 dark:bg-gray-900 rounded-lg">
                  <div className="font-mono text-sm font-bold text-slate-600 dark:text-slate-400 w-24">2022</div>
                  <div className="flex-1">
                    <p className="font-bold text-slate-800 dark:text-white">Hopper (H100)</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">4세대 Tensor Core, 1,979 TFLOPS (FP8), Transformer Engine</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-slate-50 dark:bg-gray-900 rounded-lg">
                  <div className="font-mono text-sm font-bold text-slate-600 dark:text-slate-400 w-24">2024</div>
                  <div className="flex-1">
                    <p className="font-bold text-slate-800 dark:text-white">Blackwell (B100/B200)</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">5세대 Tensor Core, 20 petaFLOPS, 2x AI 성능</p>
                  </div>
                </div>
              </div>
            </div>

            {/* SM (Streaming Multiprocessor) */}
            <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-6 text-white shadow-lg">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Box className="w-6 h-6" />
                SM (Streaming Multiprocessor) 구조
              </h3>
              <p className="text-slate-200 mb-4">
                GPU의 기본 연산 단위. 여러 개의 CUDA 코어와 Tensor Core를 포함합니다.
              </p>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">CUDA Cores</p>
                  <p className="text-sm text-slate-200">범용 FP32/FP64 연산</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Tensor Cores</p>
                  <p className="text-sm text-slate-200">행렬 곱셈 가속 (FP16/BF16/FP8)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Shared Memory</p>
                  <p className="text-sm text-slate-200">블록 내 고속 캐시 (192KB)</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CUDA Programming Basics */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            CUDA 프로그래밍 기초
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                CUDA 실행 모델
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                CUDA는 <strong>Grid → Block → Thread</strong> 계층 구조로 작업을 조직화합니다.
              </p>

              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto mb-4">
                <pre className="text-slate-800 dark:text-slate-200">
{`// CUDA 커널 함수 예제: 벡터 덧셈
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    // 고유한 스레드 ID 계산
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 범위 체크 후 연산 수행
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// 호스트 코드에서 커널 실행
int main() {
    int N = 1000000;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 커널 실행
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // GPU 동기화
    cudaDeviceSynchronize();
    return 0;
}`}
                </pre>
              </div>

              <div className="grid md:grid-cols-3 gap-4">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <p className="font-bold text-blue-700 dark:text-blue-300 mb-2">Grid</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    전체 작업을 담는 최상위 컨테이너. 여러 블록으로 구성.
                  </p>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <p className="font-bold text-green-700 dark:text-green-300 mb-2">Block</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    하나의 SM에서 실행. 최대 1024 스레드 포함.
                  </p>
                </div>
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                  <p className="font-bold text-purple-700 dark:text-purple-300 mb-2">Thread</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    실제 연산을 수행하는 최소 실행 단위.
                  </p>
                </div>
              </div>
            </div>

            {/* Memory Hierarchy */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                GPU 메모리 계층 구조
              </h3>
              <div className="space-y-3">
                <div className="flex items-start gap-3 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                  <Gauge className="w-6 h-6 text-red-600 dark:text-red-400 flex-shrink-0 mt-1" />
                  <div>
                    <p className="font-bold text-slate-800 dark:text-white">Registers (가장 빠름)</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">스레드 전용, ~1 cycle 지연, 매우 작은 용량</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                  <Gauge className="w-6 h-6 text-orange-600 dark:text-orange-400 flex-shrink-0 mt-1" />
                  <div>
                    <p className="font-bold text-slate-800 dark:text-white">Shared Memory</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">블록 공유, ~20 cycles, 192KB (A100 기준)</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <Gauge className="w-6 h-6 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-1" />
                  <div>
                    <p className="font-bold text-slate-800 dark:text-white">L1/L2 Cache</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">자동 관리, ~100 cycles, 수 MB</p>
                  </div>
                </div>
                <div className="flex items-start gap-3 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <Gauge className="w-6 h-6 text-green-600 dark:text-green-400 flex-shrink-0 mt-1" />
                  <div>
                    <p className="font-bold text-slate-800 dark:text-white">Global Memory (가장 느림)</p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">모든 스레드 접근, ~400-800 cycles, 40-80GB (A100)</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Tensor Cores */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Zap className="w-6 h-6 text-slate-700" />
            Tensor Core: AI 가속의 핵심
          </h2>

          <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-8 text-white shadow-xl mb-6">
            <h3 className="text-xl font-bold mb-4">Tensor Core란?</h3>
            <p className="text-purple-100 mb-6">
              행렬 곱셈-누적(Matrix Multiply-Accumulate) 연산을 <strong>하드웨어 레벨</strong>에서 가속하는 특수 코어.
              FP16/BF16/FP8 같은 혼합 정밀도 연산으로 Transformer 모델 훈련을 크게 가속합니다.
            </p>
            <div className="bg-white/10 rounded-lg p-6 font-mono text-sm backdrop-blur">
              <pre className="text-white">
{`D = A × B + C

여기서:
- A: M×K 행렬 (FP16/BF16/FP8)
- B: K×N 행렬 (FP16/BF16/FP8)
- C: M×N 행렬 (FP32 누산기)
- D: M×N 결과 (FP32 또는 FP16)

Tensor Core는 이 연산을 한 사이클에 수행!`}
              </pre>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="font-bold text-slate-800 dark:text-white mb-3">Tensor Core 활용 예제</h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import torch

# Automatic Mixed Precision (AMP)
from torch.cuda.amp import autocast

model = MyModel().cuda()
optimizer = Adam(model.parameters())
scaler = GradScaler()

for batch in dataloader:
    with autocast():  # FP16 활성화
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="font-bold text-slate-800 dark:text-white mb-3">성능 향상 예시</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                  <span className="text-slate-700 dark:text-slate-300">FP32 (CUDA Core)</span>
                  <span className="font-bold text-blue-600 dark:text-blue-400">19.5 TFLOPS</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded">
                  <span className="text-slate-700 dark:text-slate-300">FP16 (Tensor Core)</span>
                  <span className="font-bold text-green-600 dark:text-green-400">312 TFLOPS</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                  <span className="text-slate-700 dark:text-slate-300">INT8 (Tensor Core)</span>
                  <span className="font-bold text-purple-600 dark:text-purple-400">624 TOPS</span>
                </div>
                <p className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                  * A100 GPU 기준, FP16은 FP32 대비 <strong>16배 빠름</strong>
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Parallelism Strategies */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Grid className="w-6 h-6 text-slate-700" />
            분산 훈련 병렬화 전략
          </h2>

          <div className="space-y-6">
            {/* Data Parallelism */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700 hover:shadow-xl transition-shadow">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-blue-100 dark:bg-blue-900 rounded-lg flex-shrink-0">
                  <BarChart3 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3">
                    1. Data Parallelism (데이터 병렬화)
                  </h3>
                  <p className="text-slate-600 dark:text-slate-400 mb-4">
                    <strong>모델 전체를 각 GPU에 복사</strong>하고, 데이터를 분할하여 병렬 처리합니다.
                    가장 단순하고 널리 사용되는 방식입니다.
                  </p>
                  <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-4 mb-4">
                    <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono overflow-x-auto">
{`# PyTorch DistributedDataParallel (DDP)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 분산 초기화
dist.init_process_group(backend='nccl')
model = MyModel().cuda()
model = DDP(model, device_ids=[local_rank])

# 데이터 샘플러 설정
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)

# 각 GPU가 배치의 일부를 처리
for batch in dataloader:
    output = model(batch)  # Forward
    loss = criterion(output, target)
    loss.backward()  # Backward
    optimizer.step()  # 그래디언트 All-Reduce`}
                    </pre>
                  </div>
                  <div className="flex items-start gap-2 text-sm">
                    <span className="text-green-500 font-bold">✓</span>
                    <div>
                      <p className="text-slate-700 dark:text-slate-300"><strong>장점:</strong> 구현 간단, 거의 선형적 확장성</p>
                      <p className="text-slate-700 dark:text-slate-300 mt-1"><strong>단점:</strong> 모델이 GPU 메모리에 들어가야 함</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Model Parallelism */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700 hover:shadow-xl transition-shadow">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-green-100 dark:bg-green-900 rounded-lg flex-shrink-0">
                  <Layers className="w-6 h-6 text-green-600 dark:text-green-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3">
                    2. Model Parallelism (모델 병렬화)
                  </h3>
                  <p className="text-slate-600 dark:text-slate-400 mb-4">
                    <strong>모델을 레이어별로 분할</strong>하여 여러 GPU에 배치합니다.
                    거대 모델이 단일 GPU 메모리를 초과할 때 필수적입니다.
                  </p>
                  <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-4 mb-4">
                    <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono overflow-x-auto">
{`# 간단한 모델 병렬화 예제
class ModelParallelResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # GPU 0: 초기 레이어
        self.seq1 = nn.Sequential(
            nn.Conv2d(...),
            nn.ReLU()
        ).to('cuda:0')

        # GPU 1: 후반 레이어
        self.seq2 = nn.Sequential(
            nn.Linear(...),
            nn.Softmax()
        ).to('cuda:1')

    def forward(self, x):
        x = self.seq1(x.to('cuda:0'))
        x = self.seq2(x.to('cuda:1'))  # GPU 간 이동
        return x`}
                    </pre>
                  </div>
                  <div className="flex items-start gap-2 text-sm">
                    <span className="text-green-500 font-bold">✓</span>
                    <div>
                      <p className="text-slate-700 dark:text-slate-300"><strong>장점:</strong> 초대형 모델 훈련 가능</p>
                      <p className="text-slate-700 dark:text-slate-300 mt-1"><strong>단점:</strong> GPU 간 통신 오버헤드, 파이프라인 버블</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Pipeline Parallelism */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700 hover:shadow-xl transition-shadow">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-purple-100 dark:bg-purple-900 rounded-lg flex-shrink-0">
                  <Zap className="w-6 h-6 text-purple-600 dark:text-purple-400" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-3">
                    3. Pipeline Parallelism (파이프라인 병렬화)
                  </h3>
                  <p className="text-slate-600 dark:text-slate-400 mb-4">
                    모델을 <strong>파이프라인 스테이지</strong>로 나누고, 마이크로 배치를 연속적으로 처리하여
                    GPU 유휴 시간을 최소화합니다.
                  </p>
                  <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-4 mb-4">
                    <pre className="text-sm text-slate-800 dark:text-slate-200 font-mono overflow-x-auto">
{`# GPipe 스타일 파이프라인
from torch.distributed.pipeline.sync import Pipe

# 모델을 4개 스테이지로 분할
model = nn.Sequential(
    layer1,  # GPU 0
    layer2,  # GPU 1
    layer3,  # GPU 2
    layer4   # GPU 3
)

# 파이프라인 래핑
model = Pipe(model, chunks=8)  # 배치를 8개 마이크로배치로 분할

# 순방향/역방향 자동 파이프라이닝
output = model(input)
loss = criterion(output, target)
loss.backward()`}
                    </pre>
                  </div>
                  <div className="flex items-start gap-2 text-sm">
                    <span className="text-green-500 font-bold">✓</span>
                    <div>
                      <p className="text-slate-700 dark:text-slate-300"><strong>장점:</strong> GPU 활용률 향상, 버블 시간 감소</p>
                      <p className="text-slate-700 dark:text-slate-300 mt-1"><strong>단점:</strong> 메모리 오버헤드 (중간 활성화 저장)</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* 3D Parallelism */}
            <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Grid className="w-6 h-6" />
                3D Parallelism: 하이브리드 접근
              </h3>
              <p className="text-slate-200 mb-4">
                Data + Model + Pipeline 병렬화를 <strong>동시에 사용</strong>하여 초거대 모델(100B+ 파라미터)을 훈련합니다.
              </p>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Data Parallel</p>
                  <p className="text-sm text-slate-200">노드 간 (Inter-node)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Pipeline Parallel</p>
                  <p className="text-sm text-slate-200">레이어 간 (Layer-wise)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-bold mb-2">Tensor Parallel</p>
                  <p className="text-sm text-slate-200">레이어 내 (Intra-layer)</p>
                </div>
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
                <span>GPU는 <strong>수천 개의 코어</strong>로 행렬 연산을 병렬 처리하여 AI 훈련을 가속합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">2.</span>
                <span><strong>Tensor Core</strong>는 FP16/BF16 혼합 정밀도 연산으로 FP32 대비 최대 16배 성능 향상을 제공합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">3.</span>
                <span><strong>Data Parallelism</strong>은 모델을 복제하고 데이터를 분할하여 가장 단순한 확장성을 제공합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">4.</span>
                <span><strong>Model Parallelism</strong>은 모델을 레이어별로 분할하여 단일 GPU 메모리를 초과하는 모델을 훈련합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">5.</span>
                <span><strong>Pipeline Parallelism</strong>은 마이크로배치를 활용해 GPU 유휴 시간을 줄입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">6.</span>
                <span><strong>3D Parallelism</strong>은 세 가지 전략을 결합하여 GPT-3/PaLM 같은 초거대 모델을 훈련합니다.</span>
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
              <strong>Chapter 3: 컨테이너와 오케스트레이션</strong>
              <br />
              Docker, Kubernetes를 활용한 ML 워크로드 컨테이너화 및 클러스터 관리 방법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
