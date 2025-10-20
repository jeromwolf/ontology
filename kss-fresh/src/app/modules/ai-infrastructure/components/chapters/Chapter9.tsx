'use client'

import React from 'react'
import { Cpu, Zap, Network, Layers, Server, TrendingUp, GitBranch, Database } from 'lucide-react'

export default function Chapter9() {
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
                대규모 모델 훈련
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Multi-node Multi-GPU 분산 훈련과 ZeRO/FSDP 최적화 기법
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Cpu className="w-6 h-6 text-slate-700" />
              초거대 모델 훈련의 도전과제
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                GPT-3 (175B), PaLM (540B), LLaMA-2 (70B) 같은 <strong>초거대 언어 모델</strong>은
                단일 GPU 메모리(40-80GB)를 훨씬 초과하는 파라미터를 가집니다.
                예를 들어 <strong>175B 파라미터 모델</strong>은 FP32로 700GB, FP16으로도 350GB의 메모리가 필요합니다.
                이러한 모델을 훈련하려면 <strong>수백~수천 개의 GPU</strong>를 효율적으로 활용하는 고급 기법이 필수입니다.
              </p>

              <div className="bg-slate-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-slate-700">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">메모리 병목의 3대 요인</h3>
                <ul className="space-y-2 text-slate-700 dark:text-slate-300">
                  <li className="flex items-start gap-2">
                    <Layers className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
                    <span><strong>모델 파라미터</strong>: FP16 기준 2 bytes × 파라미터 수</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <TrendingUp className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <span><strong>그래디언트</strong>: 파라미터와 동일한 크기</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Database className="w-5 h-5 text-purple-500 mt-0.5 flex-shrink-0" />
                    <span><strong>옵티마이저 상태</strong>: Adam 기준 파라미터의 2배 (momentum + variance)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Network className="w-5 h-5 text-orange-500 mt-0.5 flex-shrink-0" />
                    <span><strong>중간 활성화</strong>: Forward pass의 중간 결과 (배치 크기에 비례)</span>
                  </li>
                </ul>
                <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 rounded">
                  <p className="text-sm text-slate-700 dark:text-slate-300">
                    <strong>예시:</strong> 175B 파라미터 모델 + Adam 옵티마이저 =
                    175B × 2 (params) + 175B × 2 (gradients) + 175B × 4 (optimizer) =
                    <strong className="text-red-600 dark:text-red-400"> ~1.4TB 메모리 필요!</strong>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Multi-node Multi-GPU Setup */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Network className="w-6 h-6 text-slate-700" />
            Multi-node Multi-GPU 분산 훈련 설정
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                PyTorch DDP Multi-node 설정
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# train.py - 분산 훈련 스크립트
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    """분산 환경 초기화"""
    os.environ['MASTER_ADDR'] = 'node-0.cluster.local'
    os.environ['MASTER_PORT'] = '29500'

    # NCCL 백엔드 초기화 (GPU 통신)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # 모델 생성 및 GPU로 이동
    model = MyLargeModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # 데이터 로더 (각 GPU가 다른 데이터 샘플)
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 훈련 루프
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 매 에폭마다 다른 셔플
        model.train()

        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(rank)
            labels = labels.to(rank)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Rank 0에서만 체크포인트 저장
        if rank == 0:
            torch.save(model.module.state_dict(), f'checkpoint_epoch_{epoch}.pt')

    cleanup()

# 실행 (각 노드에서)
# torchrun --nproc_per_node=8 \\
#          --nnodes=4 \\
#          --node_rank=$NODE_RANK \\
#          --master_addr=node-0.cluster.local \\
#          --master_port=29500 \\
#          train.py`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">분산 훈련 환경 변수</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-mono text-sm mb-1">WORLD_SIZE</p>
                  <p className="text-xs text-slate-200">전체 프로세스 수 (노드수 × GPU수/노드)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-mono text-sm mb-1">RANK</p>
                  <p className="text-xs text-slate-200">글로벌 프로세스 순위 (0 ~ WORLD_SIZE-1)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-mono text-sm mb-1">LOCAL_RANK</p>
                  <p className="text-xs text-slate-200">노드 내 GPU 순위 (0 ~ 7)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <p className="font-mono text-sm mb-1">MASTER_ADDR</p>
                  <p className="text-xs text-slate-200">마스터 노드 주소 (랑데부 포인트)</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ZeRO Optimizer */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Zap className="w-6 h-6 text-slate-700" />
            ZeRO: Zero Redundancy Optimizer
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                ZeRO의 3단계 최적화
              </h3>
              <div className="space-y-4">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border-l-4 border-blue-500">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">ZeRO Stage 1: 옵티마이저 상태 분할</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    Adam의 momentum과 variance를 GPU들에 분산 저장
                  </p>
                  <p className="text-sm text-green-600 dark:text-green-400">
                    <strong>메모리 절감:</strong> 4× (옵티마이저 상태가 전체 메모리의 대부분)
                  </p>
                </div>

                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border-l-4 border-green-500">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">ZeRO Stage 2: 그래디언트 분할</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    각 GPU가 파라미터의 일부에 대한 그래디언트만 저장
                  </p>
                  <p className="text-sm text-green-600 dark:text-green-400">
                    <strong>메모리 절감:</strong> 8× (Stage 1 + 그래디언트 분할)
                  </p>
                </div>

                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border-l-4 border-purple-500">
                  <p className="font-bold text-slate-800 dark:text-white mb-2">ZeRO Stage 3: 파라미터 분할</p>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    모델 파라미터 자체도 분산 저장, 필요시에만 All-gather
                  </p>
                  <p className="text-sm text-green-600 dark:text-green-400">
                    <strong>메모리 절감:</strong> N× (N = GPU 수, 거의 선형 확장)
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                DeepSpeed ZeRO 설정
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# ds_config.json - DeepSpeed 설정 파일
{
  "train_batch_size": 512,
  "train_micro_batch_size_per_gpu": 16,
  "gradient_accumulation_steps": 4,
  "steps_per_print": 100,

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },

  "zero_optimization": {
    "stage": 3,  # ZeRO Stage 3
    "offload_optimizer": {
      "device": "cpu",  # 옵티마이저 상태를 CPU로
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",  # 파라미터도 CPU로 오프로드
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "gather_16bit_weights_on_model_save": true
  },

  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  }
}

# train.py
import deepspeed

# 모델 초기화
model = MyLargeModel()

# DeepSpeed 엔진 생성
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config='ds_config.json'
)

# 훈련 루프
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()

# 실행
# deepspeed --num_gpus=8 train.py`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* FSDP */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            FSDP: Fully Sharded Data Parallel
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                PyTorch FSDP 구현
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

# Mixed Precision 설정
mp_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

# Auto Wrap Policy (Transformer 블록 단위 분할)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={GPT2Block}
)

# FSDP 래핑
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 상당
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    cpu_offload=CPUOffload(offload_params=True),
    device_id=torch.cuda.current_device(),
)

# 체크포인트 저장 (FSDP 전용 방식)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)

save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state_dict, "checkpoint.pt")`}
                </pre>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-xl">
                <h3 className="font-bold text-lg mb-3">FSDP 장점</h3>
                <ul className="space-y-2 text-sm text-purple-100">
                  <li>• PyTorch 네이티브 (별도 라이브러리 불필요)</li>
                  <li>• Transformer 최적화 (auto_wrap_policy)</li>
                  <li>• Backward prefetch로 통신 오버헤드 감소</li>
                  <li>• CPU offload 지원</li>
                </ul>
              </div>
              <div className="bg-gradient-to-r from-blue-500 to-cyan-600 rounded-xl p-6 text-white shadow-xl">
                <h3 className="font-bold text-lg mb-3">DeepSpeed ZeRO 장점</h3>
                <ul className="space-y-2 text-sm text-blue-100">
                  <li>• 더 세밀한 최적화 옵션</li>
                  <li>• 1D/2D/3D Tensor/Pipeline 병렬화 통합</li>
                  <li>• Activation checkpointing 고급 기능</li>
                  <li>• 커뮤니티 레시피 풍부</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Gradient Accumulation */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Gradient Accumulation: 큰 배치를 작게 나누기
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
            <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
              Gradient Accumulation 구현
            </h3>
            <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
              <pre className="text-slate-800 dark:text-slate-200">
{`# 목표: 배치 크기 512로 훈련하고 싶지만, GPU 메모리는 64만 가능
# 해결: 64 크기로 8번 반복 후 옵티마이저 업데이트

accumulation_steps = 8
effective_batch_size = 64 * 8  # = 512

model.train()
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    inputs, labels = batch

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 손실을 accumulation_steps로 나눔 (평균)
    loss = loss / accumulation_steps

    # Backward pass (그래디언트 누적)
    loss.backward()

    # accumulation_steps 마다 옵티마이저 업데이트
    if (i + 1) % accumulation_steps == 0:
        # 그래디언트 클리핑 (선택 사항)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

# 주의: 배치 정규화는 마이크로 배치(64) 통계를 사용하므로
# Layer Normalization이 더 안정적일 수 있음`}
              </pre>
            </div>
            <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <p className="text-sm text-slate-700 dark:text-slate-300">
                <strong>트레이드오프:</strong> 메모리 절약 ↑, 하지만 업데이트 빈도 ↓ (수렴 속도에 영향)
              </p>
            </div>
          </div>
        </section>

        {/* Mixed Precision Training */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            Mixed Precision Training: FP16/BF16
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Automatic Mixed Precision (AMP)
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()  # 동적 손실 스케일링

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch

        # Autocast 컨텍스트 (FP16으로 자동 변환)
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 손실 스케일링 (언더플로우 방지)
        scaler.scale(loss).backward()

        # Gradient unscaling & clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 옵티마이저 업데이트
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

# FP16 vs BF16 비교
# FP16: 5 exponent bits, 10 mantissa bits (범위 좁음, 정밀도 높음)
# BF16: 8 exponent bits, 7 mantissa bits (범위 넓음, FP32와 호환성 ↑)
# BF16은 A100/H100에서 권장 (Transformer 훈련에 안정적)`}
                </pre>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="font-bold text-slate-800 dark:text-white mb-2">FP32</p>
                <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                  <li>• 4 bytes</li>
                  <li>• 기준 정밀도</li>
                  <li>• 안정적</li>
                </ul>
              </div>
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <p className="font-bold text-slate-800 dark:text-white mb-2">FP16</p>
                <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                  <li>• 2 bytes (50% ↓)</li>
                  <li>• 2× 빠름</li>
                  <li>• 언더플로우 위험</li>
                </ul>
              </div>
              <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                <p className="font-bold text-slate-800 dark:text-white mb-2">BF16</p>
                <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                  <li>• 2 bytes (50% ↓)</li>
                  <li>• 2× 빠름</li>
                  <li>• 안정적 (넓은 범위)</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Communication Optimization */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6">
            통신 최적화: NCCL, Gloo, InfiniBand
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                NCCL 환경 변수 튜닝
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# 성능 최적화 NCCL 설정
export NCCL_DEBUG=INFO  # 디버깅용 (프로덕션에서는 WARN)
export NCCL_IB_DISABLE=0  # InfiniBand 활성화
export NCCL_NET_GDR_LEVEL=3  # GPU Direct RDMA
export NCCL_SOCKET_IFNAME=eth0  # 네트워크 인터페이스
export NCCL_MIN_NCHANNELS=16  # 통신 채널 수
export NCCL_P2P_DISABLE=0  # P2P 활성화 (NVLink)
export NCCL_BUFFSIZE=2097152  # 버퍼 크기 (2MB)

# 토폴로지 인식
export NCCL_TOPO_FILE=/path/to/topo.xml  # 커스텀 토폴로지

# 다중 노드 환경
export NCCL_CROSS_NIC=1  # 여러 NIC 활용
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1  # InfiniBand HCA 지정`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-orange-500 to-red-600 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">실전 GPT 모델 훈련 사례</h3>
              <div className="bg-white/10 rounded-lg p-6 font-mono text-sm backdrop-blur overflow-x-auto">
                <pre className="text-white">
{`# 175B 파라미터 GPT 모델 훈련 구성

하드웨어:
- 1024 × NVIDIA A100 80GB (128 노드 × 8 GPU)
- InfiniBand HDR 200 Gbps 네트워크
- NVLink 600 GB/s (노드 내 GPU 간)

소프트웨어 스택:
- DeepSpeed ZeRO Stage 3 + CPU offload
- 3D Parallelism:
  * Data Parallel: 16-way (노드 간)
  * Tensor Parallel: 8-way (노드 내)
  * Pipeline Parallel: 8-way (레이어 분할)
- BF16 Mixed Precision
- Gradient Accumulation: 32 steps
- Activation Checkpointing

성능:
- Effective Batch Size: 4M tokens
- Throughput: ~150 TFLOPS/GPU (50% MFU)
- 훈련 시간: 2-3개월
- 전력 소비: ~2MW`}
                </pre>
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
                <span><strong>ZeRO Stage 3</strong>는 파라미터, 그래디언트, 옵티마이저 상태를 모두 분산하여 거의 선형적인 메모리 절감을 제공합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">2.</span>
                <span><strong>FSDP</strong>는 PyTorch 네이티브 솔루션으로 Transformer 모델에 최적화된 auto_wrap_policy를 제공합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">3.</span>
                <span><strong>Gradient Accumulation</strong>은 메모리 제약 하에서 큰 배치 크기의 효과를 얻게 합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">4.</span>
                <span><strong>BF16</strong>은 FP16보다 넓은 동적 범위로 대규모 모델 훈련에 더 안정적입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">5.</span>
                <span><strong>NCCL</strong>은 GPU 간 All-Reduce 통신을 최적화하며, InfiniBand와 결합 시 최고 성능을 발휘합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">6.</span>
                <span><strong>3D Parallelism</strong> (Data + Tensor + Pipeline)은 1000+ GPU 규모에서 필수적인 전략입니다.</span>
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
              <strong>Chapter 10: 모니터링과 관측성</strong>
              <br />
              Prometheus + Grafana로 ML 시스템을 모니터링하고,
              Data Drift/Concept Drift를 감지하며, 분산 추적으로 성능 병목을 찾는 방법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
