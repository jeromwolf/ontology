'use client';

import References from '@/components/common/References';

export default function Chapter6() {
  return (
    <div className="space-y-8">
      {/* 1. 최적화 알고리즘 소개 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          최적화 & 정규화: 효율적인 학습의 핵심
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          딥러닝 모델의 성능은 아키텍처뿐만 아니라 최적화 알고리즘과 정규화 기법에 크게 의존합니다.
          적절한 Optimizer와 Regularization을 선택하면 학습 속도와 일반화 성능을 모두 향상시킬 수 있습니다.
        </p>

        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-2xl p-6 border border-blue-200 dark:border-blue-700 mb-6">
          <h3 className="text-lg font-semibold mb-3 text-blue-900 dark:text-blue-300">
            💡 이 챕터에서 다룰 내용
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li><strong>Optimizer</strong>: SGD, Momentum, Adam 등의 최적화 알고리즘</li>
            <li><strong>Learning Rate</strong>: 학습률 스케줄링 전략</li>
            <li><strong>Regularization</strong>: L1, L2, Dropout, Batch Normalization</li>
            <li><strong>실전 기법</strong>: Early Stopping, Gradient Clipping</li>
          </ul>
        </div>
      </section>

      {/* 2. Optimizer 비교 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Optimizer: 최적화 알고리즘
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          Optimizer는 손실 함수를 최소화하기 위해 가중치를 업데이트하는 알고리즘입니다.
        </p>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          {/* SGD */}
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4 border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-300">🔵 SGD (Stochastic Gradient Descent)</h4>
            <div className="font-mono text-sm bg-white dark:bg-gray-900 rounded p-2 mb-2">
              θ = θ - η · ∇L(θ)
            </div>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 가장 기본적인 최적화 알고리즘</li>
              <li>• 학습률(η) 선택이 매우 중요</li>
              <li>• Local minimum에 갇힐 수 있음</li>
            </ul>
          </div>

          {/* Momentum */}
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-4 border border-purple-200 dark:border-purple-800">
            <h4 className="font-semibold mb-2 text-purple-900 dark:text-purple-300">🟣 Momentum</h4>
            <div className="font-mono text-sm bg-white dark:bg-gray-900 rounded p-2 mb-2">
              v = β·v + ∇L(θ)<br/>
              θ = θ - η·v
            </div>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 이전 gradient의 방향 정보 활용</li>
              <li>• 관성 효과로 수렴 속도 향상</li>
              <li>• β는 일반적으로 0.9 사용</li>
            </ul>
          </div>

          {/* RMSprop */}
          <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-4 border border-green-200 dark:border-green-800">
            <h4 className="font-semibold mb-2 text-green-900 dark:text-green-300">🟢 RMSprop</h4>
            <div className="font-mono text-sm bg-white dark:bg-gray-900 rounded p-2 mb-2">
              v = β·v + (1-β)·(∇L)²<br/>
              θ = θ - η·∇L / √(v + ε)
            </div>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• Adaptive learning rate 알고리즘</li>
              <li>• 각 파라미터별로 다른 학습률 적용</li>
              <li>• RNN 학습에 효과적</li>
            </ul>
          </div>

          {/* Adam */}
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-4 border border-orange-200 dark:border-orange-800">
            <h4 className="font-semibold mb-2 text-orange-900 dark:text-orange-300">🟠 Adam (최고 인기!)</h4>
            <div className="font-mono text-sm bg-white dark:bg-gray-900 rounded p-2 mb-2 text-xs">
              m = β₁·m + (1-β₁)·∇L<br/>
              v = β₂·v + (1-β₂)·(∇L)²<br/>
              θ = θ - η·m̂ / (√v̂ + ε)
            </div>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• Momentum + RMSprop 결합</li>
              <li>• 가장 널리 사용되는 optimizer</li>
              <li>• β₁=0.9, β₂=0.999, η=0.001 (default)</li>
            </ul>
          </div>
        </div>

        {/* Optimizer 선택 가이드 */}
        <div className="bg-gradient-to-br from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 border border-violet-200 dark:border-violet-800">
          <h3 className="text-lg font-semibold mb-3 text-violet-900 dark:text-violet-300">
            📌 Optimizer 선택 가이드
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li><strong>Adam</strong>: 대부분의 경우 첫 번째 선택 (빠른 수렴, 안정적)</li>
            <li><strong>SGD + Momentum</strong>: 최종 성능이 중요하고 시간이 충분할 때</li>
            <li><strong>RMSprop</strong>: RNN/LSTM 학습 시 효과적</li>
            <li><strong>AdamW</strong>: Weight decay를 올바르게 적용한 Adam 변형 (최근 추세)</li>
          </ul>
        </div>
      </section>

      {/* 3. Learning Rate Scheduling */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Learning Rate Scheduling
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          학습률을 학습 과정 중에 동적으로 조절하면 더 나은 성능과 안정적인 수렴을 얻을 수 있습니다.
        </p>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4 border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-300">Step Decay</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              일정 epoch마다 학습률을 줄임
            </p>
            <div className="font-mono text-xs bg-white dark:bg-gray-900 rounded p-2">
              η = η₀ × 0.5^(epoch/10)
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-4 border border-purple-200 dark:border-purple-800">
            <h4 className="font-semibold mb-2 text-purple-900 dark:text-purple-300">Cosine Annealing</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              코사인 함수로 부드럽게 감소
            </p>
            <div className="font-mono text-xs bg-white dark:bg-gray-900 rounded p-2">
              η = η_min + 0.5(η_max-η_min)(1+cos(πt/T))
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-4 border border-green-200 dark:border-green-800">
            <h4 className="font-semibold mb-2 text-green-900 dark:text-green-300">Warm-up + Decay</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              초반 학습률 증가 후 감소
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Transformer 학습에 필수
            </p>
          </div>
        </div>
      </section>

      {/* 4. 정규화 기법 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Regularization: 과적합 방지
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          정규화는 모델이 학습 데이터에 과적합되는 것을 방지하여 일반화 성능을 향상시킵니다.
        </p>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          {/* L1 & L2 Regularization */}
          <div className="bg-pink-50 dark:bg-pink-900/20 rounded-xl p-6 border border-pink-200 dark:border-pink-800">
            <h4 className="font-semibold mb-3 text-pink-900 dark:text-pink-300 text-lg">📐 L1 & L2 Regularization</h4>
            <div className="space-y-3">
              <div>
                <strong className="text-sm text-gray-900 dark:text-gray-100">L2 (Weight Decay)</strong>
                <div className="font-mono text-sm bg-white dark:bg-gray-900 rounded p-2 mt-1">
                  Loss = L₀ + λ·Σ(w²)
                </div>
                <ul className="text-sm text-gray-700 dark:text-gray-300 mt-2 space-y-1">
                  <li>• 가중치를 0에 가깝게 유지</li>
                  <li>• 가장 널리 사용됨 (λ ≈ 0.0001)</li>
                </ul>
              </div>
              <div>
                <strong className="text-sm text-gray-900 dark:text-gray-100">L1 (Lasso)</strong>
                <div className="font-mono text-sm bg-white dark:bg-gray-900 rounded p-2 mt-1">
                  Loss = L₀ + λ·Σ(|w|)
                </div>
                <ul className="text-sm text-gray-700 dark:text-gray-300 mt-2 space-y-1">
                  <li>• Sparse solution (많은 가중치가 정확히 0)</li>
                  <li>• Feature selection 효과</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Dropout */}
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 border border-indigo-200 dark:border-indigo-800">
            <h4 className="font-semibold mb-3 text-indigo-900 dark:text-indigo-300 text-lg">💧 Dropout</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              학습 시 랜덤하게 뉴런을 제거하여 앙상블 효과 획득
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li>• <strong>Training</strong>: 확률 p로 뉴런 제거 (p=0.5 흔함)</li>
              <li>• <strong>Inference</strong>: 모든 뉴런 사용, 출력×(1-p)</li>
              <li>• CNN보다 FC layer에 더 효과적</li>
              <li>• DropConnect: 뉴런 대신 연결(weight) 제거</li>
            </ul>
          </div>

          {/* Batch Normalization */}
          <div className="bg-teal-50 dark:bg-teal-900/20 rounded-xl p-6 border border-teal-200 dark:border-teal-800 md:col-span-2">
            <h4 className="font-semibold mb-3 text-teal-900 dark:text-teal-300 text-lg">📊 Batch Normalization</h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  각 레이어의 입력을 정규화하여 학습 안정성 향상
                </p>
                <div className="font-mono text-sm bg-white dark:bg-gray-900 rounded p-2">
                  x̂ = (x - μ_B) / √(σ²_B + ε)<br/>
                  y = γ·x̂ + β
                </div>
              </div>
              <div>
                <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                  <li>• 높은 학습률 사용 가능</li>
                  <li>• Internal Covariate Shift 완화</li>
                  <li>• 정규화 효과 (Dropout과 유사)</li>
                  <li>• CNN에서 필수적으로 사용</li>
                  <li>• Training/Inference 모드 구분 필요</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Layer Normalization */}
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-4 border border-yellow-200 dark:border-yellow-800">
          <h4 className="font-semibold mb-2 text-yellow-900 dark:text-yellow-300">⚡ Layer Normalization (Transformer용)</h4>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            Batch 차원이 아닌 Feature 차원에서 정규화. Sequence 길이가 가변적인 NLP에 적합.
            Transformer, BERT, GPT 등에서 표준으로 사용.
          </p>
        </div>
      </section>

      {/* 5. 실전 기법 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          실전 최적화 기법
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          {/* Early Stopping */}
          <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6 border border-red-200 dark:border-red-800">
            <h4 className="font-semibold mb-3 text-red-900 dark:text-red-300 text-lg">🛑 Early Stopping</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              검증 손실(validation loss)이 더 이상 개선되지 않으면 학습 중단
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Patience</strong>: 몇 epoch 동안 개선 없으면 중단</li>
              <li>• 과적합 방지의 가장 실용적인 방법</li>
              <li>• 최고 성능 모델 자동 저장</li>
            </ul>
          </div>

          {/* Gradient Clipping */}
          <div className="bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold mb-3 text-blue-900 dark:text-blue-300 text-lg">✂️ Gradient Clipping</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              Gradient가 너무 커지는 것을 방지 (Exploding Gradient 해결)
            </p>
            <div className="font-mono text-sm bg-white dark:bg-gray-900 rounded p-2 mb-2">
              if ||g|| &gt; threshold:<br/>
              &nbsp;&nbsp;g = g × (threshold / ||g||)
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              RNN/LSTM 학습에 필수적 (threshold ≈ 5.0)
            </p>
          </div>

          {/* Mixed Precision Training */}
          <div className="bg-gradient-to-br from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-xl p-6 border border-green-200 dark:border-green-800">
            <h4 className="font-semibold mb-3 text-green-900 dark:text-green-300 text-lg">⚙️ Mixed Precision Training</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              FP16과 FP32를 혼합하여 메모리와 속도 최적화
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• GPU 메모리 사용량 ~50% 감소</li>
              <li>• 학습 속도 2-3배 향상 (Tensor Core 활용)</li>
              <li>• PyTorch AMP, TensorFlow Mixed Precision 지원</li>
            </ul>
          </div>

          {/* Data Augmentation */}
          <div className="bg-gradient-to-br from-pink-50 to-rose-50 dark:from-pink-900/20 dark:to-rose-900/20 rounded-xl p-6 border border-pink-200 dark:border-pink-800">
            <h4 className="font-semibold mb-3 text-pink-900 dark:text-pink-300 text-lg">🎨 Data Augmentation</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              학습 데이터를 인위적으로 증강하여 과적합 방지
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 이미지: 회전, 크롭, 반전, 색상 변환</li>
              <li>• 텍스트: Back Translation, Synonym Replacement</li>
              <li>• 음성: Pitch Shift, Time Stretch, Noise Injection</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 6. 최적화 체크리스트 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          🎯 실전 최적화 체크리스트
        </h2>

        <div className="bg-gradient-to-br from-violet-50 to-indigo-50 dark:from-violet-900/20 dark:to-indigo-900/20 rounded-2xl p-8 border border-violet-200 dark:border-violet-800">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-lg mb-3 text-violet-900 dark:text-violet-300">1️⃣ 기본 설정</h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>✓ Optimizer: Adam (lr=0.001) 또는 AdamW</li>
                <li>✓ Weight Decay: L2 regularization (λ=0.0001)</li>
                <li>✓ Batch Normalization 또는 Layer Normalization</li>
                <li>✓ He/Xavier Initialization</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-lg mb-3 text-violet-900 dark:text-violet-300">2️⃣ 과적합 방지</h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>✓ Dropout (p=0.3~0.5) for FC layers</li>
                <li>✓ Early Stopping (patience=5~10 epochs)</li>
                <li>✓ Data Augmentation</li>
                <li>✓ Cross-validation 또는 Holdout validation</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-lg mb-3 text-violet-900 dark:text-violet-300">3️⃣ 학습 안정화</h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>✓ Gradient Clipping (RNN의 경우)</li>
                <li>✓ Learning Rate Scheduling</li>
                <li>✓ Warm-up (Transformer의 경우)</li>
                <li>✓ Batch size 조정 (크면 안정, 작으면 일반화)</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-lg mb-3 text-violet-900 dark:text-violet-300">4️⃣ 성능 향상</h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>✓ Mixed Precision Training (AMP)</li>
                <li>✓ Gradient Accumulation (큰 배치 시뮬레이션)</li>
                <li>✓ Transfer Learning (사전학습 모델 활용)</li>
                <li>✓ Ensemble (여러 모델 결합)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '🔴 핵심 논문',
            icon: 'paper' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'Adam: A Method for Stochastic Optimization',
                authors: 'Kingma, D. P., & Ba, J.',
                year: '2014',
                description: 'Adam optimizer를 제안한 논문',
                link: 'https://arxiv.org/abs/1412.6980'
              },
              {
                title: 'Batch Normalization: Accelerating Deep Network Training',
                authors: 'Ioffe, S., & Szegedy, C.',
                year: '2015',
                description: 'Batch Normalization의 원리와 효과',
                link: 'https://arxiv.org/abs/1502.03167'
              },
              {
                title: 'Dropout: A Simple Way to Prevent Overfitting',
                authors: 'Srivastava, N., et al.',
                year: '2014',
                description: 'Dropout 정규화 기법',
                link: 'https://jmlr.org/papers/v15/srivastava14a.html'
              }
            ]
          },
          {
            title: '📘 고급 최적화',
            icon: 'paper' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Decoupled Weight Decay Regularization (AdamW)',
                authors: 'Loshchilov, I., & Hutter, F.',
                year: '2017',
                description: 'Weight decay를 올바르게 적용한 Adam 변형',
                link: 'https://arxiv.org/abs/1711.05101'
              },
              {
                title: 'Layer Normalization',
                authors: 'Ba, J. L., et al.',
                year: '2016',
                description: 'Transformer를 위한 정규화 기법',
                link: 'https://arxiv.org/abs/1607.06450'
              },
              {
                title: 'SGDR: Stochastic Gradient Descent with Warm Restarts',
                authors: 'Loshchilov, I., & Hutter, F.',
                year: '2016',
                description: 'Cosine annealing과 warm restart',
                link: 'https://arxiv.org/abs/1608.03983'
              }
            ]
          },
          {
            title: '🛠️ 실전 가이드',
            icon: 'web' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'PyTorch Optimizer Documentation',
                authors: 'PyTorch Team',
                year: '2023',
                description: '모든 optimizer 구현과 사용법',
                link: 'https://pytorch.org/docs/stable/optim.html'
              },
              {
                title: 'CS231n: Training Neural Networks',
                authors: 'Stanford University',
                year: '2023',
                description: '최적화 기법 종합 가이드',
                link: 'http://cs231n.github.io/neural-networks-3/'
              },
              {
                title: 'An Overview of Gradient Descent Optimization Algorithms',
                authors: 'Sebastian Ruder',
                year: '2016',
                description: 'Optimizer 비교 및 설명',
                link: 'https://ruder.io/optimizing-gradient-descent/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
