export default function Chapter10() {
  return (
    <div className="space-y-8">
      {/* 챕터 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          AI 가속화 (AI Acceleration)
        </h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 border-l-4 border-yellow-500 p-6 rounded-r-lg">
          <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
            대규모 딥러닝 모델 훈련은 HPC의 가장 중요한 응용 분야가 되었습니다.
            GPT-4, DALL-E 3 같은 모델은 수천 개의 GPU와 엑사플롭급 연산이 필요합니다.
          </p>
        </div>
      </section>

      {/* 분산 훈련 전략 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          1. 분산 훈련 전략
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">데이터 병렬화 (Data Parallelism)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                각 GPU가 동일한 모델을 복제하고 서로 다른 배치 데이터를 처리합니다.
              </p>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`# PyTorch DistributedDataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 초기화
dist.init_process_group(backend='nccl')  # NVIDIA GPUs
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 모델을 DDP로 래핑
model = MyLargeModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 훈련 루프
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()  # 그래디언트 자동 all-reduce
        optimizer.step()

# 8 GPU × 512 batch = 4096 effective batch size`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">모델 병렬화 (Model Parallelism)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                거대한 모델을 여러 GPU에 분할합니다. GPT-3 (175B 파라미터)는 단일 GPU에 올릴 수 없습니다.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b-2 border-yellow-500">
                      <th className="p-2">기법</th>
                      <th className="p-2">설명</th>
                      <th className="p-2">프레임워크</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-semibold">Pipeline Parallelism</td>
                      <td className="p-2">레이어를 GPU에 순차 배치</td>
                      <td className="p-2 font-mono">GPipe, PipeDream</td>
                    </tr>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-semibold">Tensor Parallelism</td>
                      <td className="p-2">행렬 연산을 분할</td>
                      <td className="p-2 font-mono">Megatron-LM</td>
                    </tr>
                    <tr>
                      <td className="p-2 font-semibold">Zero Redundancy (ZeRO)</td>
                      <td className="p-2">옵티마이저 상태 분산</td>
                      <td className="p-2 font-mono">DeepSpeed</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">3D 병렬화 (DeepSpeed ZeRO)</h5>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>Data + Model + Pipeline 병렬화 조합</strong>:<br/>
                  • ZeRO-1: Optimizer 상태 분산 (4× 메모리 절감)<br/>
                  • ZeRO-2: + Gradients 분산 (8× 절감)<br/>
                  • ZeRO-3: + Parameters 분산 (16× 절감)<br/><br/>
                  <strong>성과</strong>: 1 Trillion 파라미터 모델 훈련 가능 (800 GPUs)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 최적화 기법 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          2. 훈련 최적화 기법
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">혼합 정밀도 훈련 (Mixed Precision)</h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                FP16 (16-bit)과 FP32 (32-bit)를 혼합하여 메모리와 속도를 최적화합니다.
              </p>
              <div className="bg-gray-900 rounded p-4 mb-2">
                <pre className="text-sm text-gray-100">
                  <code>{`# PyTorch Automatic Mixed Precision (AMP)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in FP16
    with autocast():
        loss = model(batch)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 성능: 2-3배 빠름, 메모리 50% 절감`}</code>
                </pre>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Gradient Checkpointing</h5>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded mb-2">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>원리</strong>: Forward pass 중 일부 activation만 저장, backward 시 재계산<br/>
                  <strong>트레이드오프</strong>: 메모리 절감 vs 30% 속도 저하<br/>
                  <strong>효과</strong>: 8배 큰 배치 또는 2배 큰 모델 훈련 가능<br/>
                  <strong>라이브러리</strong>: torch.utils.checkpoint
                </p>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">Flash Attention</h5>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>문제</strong>: Transformer Attention은 O(N²) 메모리 (N = 시퀀스 길이)<br/>
                  <strong>해결</strong>: 타일 기반 알고리즘으로 메모리 접근 최적화<br/>
                  <strong>성능</strong>: 2-4배 빠름, 긴 시퀀스 (16K tokens) 지원<br/>
                  <strong>사용</strong>: xFormers, FlashAttention-2 라이브러리
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 대규모 모델 훈련 사례 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          3. 대규모 모델 훈련 사례
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="border-b-2 border-yellow-500">
                    <th className="p-2">모델</th>
                    <th className="p-2">파라미터</th>
                    <th className="p-2">HPC 리소스</th>
                    <th className="p-2">훈련 시간</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="p-2 font-semibold">GPT-3</td>
                    <td className="p-2">175B</td>
                    <td className="p-2">10,000 V100 GPUs</td>
                    <td className="p-2">~1 month</td>
                  </tr>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="p-2 font-semibold">PaLM</td>
                    <td className="p-2">540B</td>
                    <td className="p-2">6,144 TPU v4 chips</td>
                    <td className="p-2">~50 days</td>
                  </tr>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <td className="p-2 font-semibold">LLaMA 65B</td>
                    <td className="p-2">65B</td>
                    <td className="p-2">2,048 A100 GPUs</td>
                    <td className="p-2">~21 days</td>
                  </tr>
                  <tr>
                    <td className="p-2 font-semibold">Stable Diffusion</td>
                    <td className="p-2">890M</td>
                    <td className="p-2">256 A100 GPUs</td>
                    <td className="p-2">~150K GPU-hours</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400 mt-4">비용 분석 예시 (GPT-3)</h5>
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>클라우드 비용</strong> (AWS p4d.24xlarge @ $32.77/hour):<br/>
                  10,000 GPUs ÷ 8 (per instance) = 1,250 인스턴스<br/>
                  1,250 × $32.77 × 24 hours × 30 days = <strong>$29.5 Million</strong><br/><br/>
                  <strong>실제 비용</strong>: OpenAI는 Microsoft Azure 스폰서십으로 훈련<br/>
                  <strong>추정 전력 소비</strong>: ~1,287 MWh
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 인퍼런스 최적화 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          4. 인퍼런스 최적화
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="space-y-4">
            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">양자화 (Quantization)</h5>
              <div className="overflow-x-auto mb-3">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b-2 border-yellow-500">
                      <th className="p-2">정밀도</th>
                      <th className="p-2">메모리</th>
                      <th className="p-2">속도</th>
                      <th className="p-2">정확도 손실</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-mono">FP32</td>
                      <td className="p-2">100%</td>
                      <td className="p-2">1×</td>
                      <td className="p-2 text-green-600 dark:text-green-400">None</td>
                    </tr>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-mono">FP16</td>
                      <td className="p-2">50%</td>
                      <td className="p-2">2-3×</td>
                      <td className="p-2 text-green-600 dark:text-green-400">&lt;1%</td>
                    </tr>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <td className="p-2 font-mono">INT8</td>
                      <td className="p-2">25%</td>
                      <td className="p-2">4×</td>
                      <td className="p-2 text-yellow-600 dark:text-yellow-400">1-2%</td>
                    </tr>
                    <tr>
                      <td className="p-2 font-mono">INT4</td>
                      <td className="p-2">12.5%</td>
                      <td className="p-2">8×</td>
                      <td className="p-2 text-orange-600 dark:text-orange-400">2-5%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h5 className="font-semibold mb-2 text-yellow-600 dark:text-yellow-400">모델 압축 기법</h5>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>Pruning (가지치기)</strong>: 불필요한 가중치 제거 → 50% 압축<br/>
                  <strong>Knowledge Distillation</strong>: 큰 모델 → 작은 모델 지식 전이<br/>
                  <strong>LoRA (Low-Rank Adaptation)</strong>: 파인튜닝 시 0.1% 파라미터만 업데이트<br/>
                  <strong>TensorRT</strong>: NVIDIA 인퍼런스 최적화 엔진 (5-10배 가속)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 요약 */}
      <section>
        <h3 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          📚 핵심 요약
        </h3>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6 border-l-4 border-yellow-500">
          <ul className="space-y-3 text-gray-800 dark:text-gray-200">
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">1.</span>
              <span>데이터 + 모델 + 파이프라인 병렬화로 초대형 모델 훈련이 가능하다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">2.</span>
              <span>Mixed Precision과 Flash Attention으로 훈련 속도를 2-4배 가속할 수 있다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">3.</span>
              <span>GPT-3급 모델 훈련에는 수천 개 GPU와 수백만 달러의 비용이 필요하다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">4.</span>
              <span>양자화와 압축으로 인퍼런스 비용을 10배 이상 절감할 수 있다</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-yellow-600 dark:text-yellow-400 font-bold">5.</span>
              <span>HPC 기술이 AI의 발전을 이끌고, AI가 HPC의 가장 큰 응용 분야가 되었다</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}
