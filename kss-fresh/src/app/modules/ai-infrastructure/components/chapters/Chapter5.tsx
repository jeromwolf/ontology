'use client'

import React from 'react'
import { Zap, Layers, Server, Network, Cpu, BarChart3, GitBranch, Boxes } from 'lucide-react'

export default function Chapter5() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 dark:from-gray-900 dark:to-slate-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-slate-700 to-gray-800 rounded-xl shadow-lg">
              <Zap className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-700 to-gray-800 bg-clip-text text-transparent">
                분산 훈련 전략
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                DeepSpeed, Megatron-LM, ZeRO 최적화로 초거대 모델 훈련하기
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg border border-slate-200 dark:border-gray-700">
            <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-4 flex items-center gap-2">
              <Server className="w-6 h-6 text-slate-700" />
              왜 고급 분산 전략이 필요한가?
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed mb-4">
                GPT-3(175B), PaLM(540B), LLaMA 2(70B) 같은 초거대 언어 모델은
                <strong>단일 GPU 메모리를 훨씬 초과</strong>합니다.
                기본 DDP로는 A100 80GB GPU에도 7B 파라미터 모델조차 훈련하기 어렵습니다.
                고급 최적화 기법은 <strong>메모리 효율성과 통신 오버헤드</strong>를 동시에 해결합니다.
              </p>

              <div className="bg-slate-50 dark:bg-gray-900 rounded-xl p-6 my-6 border-l-4 border-slate-700">
                <h3 className="font-bold text-slate-800 dark:text-white mb-3">메모리 병목 현상</h3>
                <div className="space-y-2 text-slate-700 dark:text-slate-300">
                  <p className="flex items-start gap-2">
                    <span className="text-red-500 font-bold">1.</span>
                    <span><strong>모델 파라미터</strong>: FP32로 1B = 4GB (70B = 280GB!)</span>
                  </p>
                  <p className="flex items-start gap-2">
                    <span className="text-orange-500 font-bold">2.</span>
                    <span><strong>그래디언트</strong>: 파라미터와 동일한 크기</span>
                  </p>
                  <p className="flex items-start gap-2">
                    <span className="text-yellow-500 font-bold">3.</span>
                    <span><strong>옵티마이저 상태</strong>: Adam은 2배 (momentum + variance)</span>
                  </p>
                  <p className="flex items-start gap-2">
                    <span className="text-green-500 font-bold">4.</span>
                    <span><strong>활성화(Activation)</strong>: 배치 크기에 비례, 수십 GB</span>
                  </p>
                </div>
                <div className="mt-4 p-3 bg-red-100 dark:bg-red-900/30 rounded">
                  <p className="text-sm font-bold text-red-800 dark:text-red-200">
                    총 메모리: 모델 × 16배 (FP32 기준) → 70B 모델은 ~4.5TB 필요!
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ZeRO Optimization */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Layers className="w-6 h-6 text-slate-700" />
            ZeRO: Zero Redundancy Optimizer
          </h2>

          <div className="space-y-6">
            {/* ZeRO Stages */}
            <div className="bg-gradient-to-r from-slate-700 to-gray-800 rounded-xl p-8 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">ZeRO의 3단계 메모리 최적화</h3>
              <div className="space-y-4">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <div className="flex items-center justify-between mb-2">
                    <p className="font-bold text-lg">Stage 1: 옵티마이저 상태 분할</p>
                    <span className="px-3 py-1 bg-blue-500 rounded-full text-xs">4배 감소</span>
                  </div>
                  <p className="text-sm text-slate-200">
                    각 GPU가 Adam의 momentum/variance를 일부만 보유. 통신 비용 최소.
                  </p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <div className="flex items-center justify-between mb-2">
                    <p className="font-bold text-lg">Stage 2: + 그래디언트 분할</p>
                    <span className="px-3 py-1 bg-green-500 rounded-full text-xs">8배 감소</span>
                  </div>
                  <p className="text-sm text-slate-200">
                    그래디언트도 파티셔닝. 각 GPU는 자신의 파라미터에 대한 그래디언트만 저장.
                  </p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <div className="flex items-center justify-between mb-2">
                    <p className="font-bold text-lg">Stage 3: + 파라미터 분할</p>
                    <span className="px-3 py-1 bg-purple-500 rounded-full text-xs">N배 감소</span>
                  </div>
                  <p className="text-sm text-slate-200">
                    모델 파라미터도 샤딩. 필요시 All-Gather로 가져옴 (FSDP와 유사).
                  </p>
                </div>
              </div>
            </div>

            {/* DeepSpeed Code */}
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                DeepSpeed로 ZeRO Stage 3 사용하기
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto mb-4">
                <pre className="text-slate-800 dark:text-slate-200">
{`# deepspeed_config.json
{
  "train_batch_size": 64,
  "gradient_accumulation_steps": 4,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",  # CPU로 옵티마이저 오프로드
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",  # 파라미터도 CPU로
      "pin_memory": true
    },
    "overlap_comm": true,  # 통신과 계산 오버랩
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 100
}`}
                </pre>
              </div>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`import deepspeed

# 모델과 옵티마이저 준비
model = MyLargeModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# DeepSpeed 엔진 초기화
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config='deepspeed_config.json'
)

# 훈련 루프는 거의 동일!
for batch in dataloader:
    inputs, labels = batch
    outputs = model_engine(inputs)
    loss = criterion(outputs, labels)

    model_engine.backward(loss)  # .backward() 대신
    model_engine.step()  # optimizer.step() 대신

# 체크포인트 저장
model_engine.save_checkpoint('checkpoints', tag='epoch_10')`}
                </pre>
              </div>
              <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <p className="text-sm text-slate-700 dark:text-slate-300">
                  <strong>실행:</strong> <code className="font-mono">deepspeed --num_gpus=8 train.py</code>
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Megatron-LM */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Network className="w-6 h-6 text-slate-700" />
            Megatron-LM: Tensor & Pipeline Parallelism
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Tensor Parallelism: 레이어 내부 분할
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                Transformer의 Attention과 MLP를 <strong>열(column) 단위로 분할</strong>하여 여러 GPU에 배치합니다.
              </p>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 mb-4">
                <div className="text-sm text-slate-800 dark:text-slate-200 space-y-3">
                  <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                    <p className="font-bold mb-1">Self-Attention 분할</p>
                    <p className="text-xs text-slate-600 dark:text-slate-400 font-mono">
                      Q, K, V 행렬을 헤드 단위로 분할 → 각 GPU가 일부 헤드만 계산
                    </p>
                  </div>
                  <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                    <p className="font-bold mb-1">MLP 분할</p>
                    <p className="text-xs text-slate-600 dark:text-slate-400 font-mono">
                      Linear(d_model, 4*d_model)을 4개 GPU에 나누면<br/>
                      각 GPU는 Linear(d_model, d_model) 계산
                    </p>
                  </div>
                </div>
              </div>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# Megatron-style Tensor Parallelism (개념)
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group):
        super().__init__()
        self.tp_size = dist.get_world_size(group=tp_group)
        self.tp_rank = dist.get_rank(group=tp_group)

        # 출력 차원을 TP 크기로 분할
        assert out_features % self.tp_size == 0
        self.out_features_per_partition = out_features // self.tp_size

        self.weight = nn.Parameter(torch.randn(
            self.out_features_per_partition,
            in_features
        ))

    def forward(self, x):
        # 각 GPU가 일부 출력만 계산
        output = F.linear(x, self.weight)
        # All-Reduce 불필요 (다음 레이어에서 All-Gather)
        return output

class RowParallelLinear(nn.Module):
    def forward(self, x):
        output = F.linear(x, self.weight)
        # 결과를 All-Reduce로 합산
        dist.all_reduce(output, group=self.tp_group)
        return output`}
                </pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                Pipeline Parallelism: 레이어 단위 분할
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                모델을 <strong>레이어 그룹</strong>으로 나누고, 마이크로배치를 파이프라인처럼 흘려보냅니다.
              </p>
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <p className="font-bold text-purple-700 dark:text-purple-300 mb-2">GPipe (Synchronous)</p>
                  <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• Forward 완료 후 Backward 시작</li>
                    <li>• 버블 시간이 큼 (~50%)</li>
                    <li>• 메모리 효율적 (활성화 재계산)</li>
                  </ul>
                </div>
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <p className="font-bold text-blue-700 dark:text-blue-300 mb-2">PipeDream (Asynchronous)</p>
                  <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• Forward와 Backward 교차 실행</li>
                    <li>• 버블 시간 감소 (~10%)</li>
                    <li>• 더 많은 메모리 필요</li>
                  </ul>
                </div>
              </div>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`# Megatron-LM 실행 예시
WORLD_SIZE=64  # 총 64 GPUs
TP=8           # Tensor Parallelism = 8
PP=4           # Pipeline Parallelism = 4
DP=2           # Data Parallelism = 2 (64 = 8*4*2)

python -m torch.distributed.launch \\
  --nproc_per_node=8 \\
  --nnodes=8 \\
  pretrain_gpt.py \\
  --tensor-model-parallel-size 8 \\
  --pipeline-model-parallel-size 4 \\
  --num-layers 96 \\
  --hidden-size 12288 \\
  --num-attention-heads 96 \\
  --seq-length 2048 \\
  --micro-batch-size 1 \\
  --global-batch-size 1536`}
                </pre>
              </div>
            </div>

            <div className="bg-gradient-to-r from-purple-500 to-indigo-600 rounded-xl p-6 text-white shadow-xl">
              <h3 className="text-xl font-bold mb-4">3D Parallelism 조합 전략</h3>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <Cpu className="w-8 h-8 mb-2" />
                  <p className="font-bold mb-1">Data Parallel</p>
                  <p className="text-xs text-purple-100">노드 간 (Inter-node)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <Layers className="w-8 h-8 mb-2" />
                  <p className="font-bold mb-1">Tensor Parallel</p>
                  <p className="text-xs text-purple-100">노드 내 (Intra-node, NVLink)</p>
                </div>
                <div className="bg-white/10 rounded-lg p-4 backdrop-blur">
                  <GitBranch className="w-8 h-8 mb-2" />
                  <p className="font-bold mb-1">Pipeline Parallel</p>
                  <p className="text-xs text-purple-100">레이어 그룹 간</p>
                </div>
              </div>
              <div className="mt-4 p-4 bg-white/10 rounded-lg backdrop-blur">
                <p className="text-sm">
                  <strong>예시:</strong> 175B GPT-3를 128 GPUs로 훈련
                  <br />
                  → DP=4 × TP=8 × PP=4 = 128 GPUs
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* FSDP */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-2">
            <Boxes className="w-6 h-6 text-slate-700" />
            FSDP: Fully Sharded Data Parallel
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                FSDP vs ZeRO Stage 3 비교
              </h3>
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                PyTorch FSDP는 ZeRO Stage 3와 개념적으로 유사하지만, PyTorch 네이티브 통합이 장점입니다.
              </p>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <p className="font-bold text-blue-700 dark:text-blue-300 mb-2">공통점</p>
                  <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• 파라미터, 그래디언트, 옵티마이저 샤딩</li>
                    <li>• All-Gather로 필요시 파라미터 수집</li>
                    <li>• Reduce-Scatter로 그래디언트 분산</li>
                  </ul>
                </div>
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <p className="font-bold text-green-700 dark:text-green-300 mb-2">FSDP 장점</p>
                  <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                    <li>• PyTorch 네이티브 (별도 설치 불필요)</li>
                    <li>• 세밀한 샤딩 제어 (auto_wrap_policy)</li>
                    <li>• Mixed Precision 자동 관리</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-slate-200 dark:border-gray-700">
              <h3 className="text-xl font-bold text-slate-800 dark:text-white mb-4">
                FSDP 고급 설정
              </h3>
              <div className="bg-slate-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm overflow-x-auto">
                <pre className="text-slate-800 dark:text-slate-200">
{`from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

# Transformer 블록 기준 자동 래핑
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock}
)

# Mixed Precision 설정
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,  # 파라미터 BF16
    reduce_dtype=torch.float32,  # 그래디언트 리덕션 FP32
    buffer_dtype=torch.bfloat16  # 버퍼 BF16
)

model = FSDP(
    MyLargeModel(),
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    cpu_offload=CPUOffload(offload_params=True),  # CPU 오프로드
    device_id=torch.cuda.current_device(),
    sync_module_states=True,  # 초기 파라미터 동기화
    forward_prefetch=True,  # 미리 가져오기로 통신 숨김
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    use_orig_params=True  # optimizer.param_groups 접근 허용
)

# 체크포인트 저장 (Rank 0만)
if dist.get_rank() == 0:
    state_dict = model.state_dict()
    torch.save(state_dict, 'model.pt')`}
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
                <span><strong>ZeRO</strong>는 3단계로 옵티마이저, 그래디언트, 파라미터를 순차적으로 샤딩하여 메모리를 절감합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">2.</span>
                <span><strong>DeepSpeed</strong>는 ZeRO + CPU 오프로드로 1.5TB 메모리를 가상으로 확보할 수 있습니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">3.</span>
                <span><strong>Tensor Parallelism</strong>은 레이어 내부를 분할하여 통신 대역폭이 높은 NVLink 환경에 최적입니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">4.</span>
                <span><strong>Pipeline Parallelism</strong>은 마이크로배치로 버블 시간을 줄이고 레이어 간 분산을 가능하게 합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">5.</span>
                <span><strong>3D Parallelism</strong>은 DP × TP × PP 조합으로 수백 GPU 클러스터에서 초거대 모델을 훈련합니다.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-500 text-xl">6.</span>
                <span><strong>FSDP</strong>는 PyTorch 네이티브 구현으로 ZeRO-3와 유사한 성능을 제공하며, 더 간편한 API를 가집니다.</span>
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
              <strong>Chapter 6: 모델 서빙과 최적화</strong>
              <br />
              TorchServe, TensorRT, ONNX Runtime을 활용한 추론 최적화 및 Quantization, Pruning 기법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
