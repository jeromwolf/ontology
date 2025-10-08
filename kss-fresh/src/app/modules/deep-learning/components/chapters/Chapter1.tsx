'use client';

import References from '@/components/common/References';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">Perceptron: 신경망의 출발점</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Perceptron은 1957년 Frank Rosenblatt이 고안한 최초의 인공신경망 모델입니다.
          생물학적 뉴런을 모방하여 여러 입력을 받아 가중합을 계산하고 활성화 함수를 통해 출력을 생성합니다.
        </p>

        {/* Perceptron 구조 다이어그램 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4 text-center">Perceptron 구조</h3>
          <svg viewBox="0 0 800 400" className="w-full h-auto">
            {/* 입력층 */}
            <circle cx="150" cy="100" r="30" className="fill-blue-200 dark:fill-blue-800 stroke-blue-500" strokeWidth="2" />
            <text x="150" y="105" textAnchor="middle" className="fill-blue-700 dark:fill-blue-300 text-sm font-semibold">x₁</text>

            <circle cx="150" cy="200" r="30" className="fill-blue-200 dark:fill-blue-800 stroke-blue-500" strokeWidth="2" />
            <text x="150" y="205" textAnchor="middle" className="fill-blue-700 dark:fill-blue-300 text-sm font-semibold">x₂</text>

            <circle cx="150" cy="300" r="30" className="fill-blue-200 dark:fill-blue-800 stroke-blue-500" strokeWidth="2" />
            <text x="150" y="305" textAnchor="middle" className="fill-blue-700 dark:fill-blue-300 text-sm font-semibold">x₃</text>

            {/* 가중치 선 */}
            <line x1="180" y1="100" x2="420" y2="200" className="stroke-purple-400" strokeWidth="3" />
            <text x="280" y="140" className="fill-purple-600 dark:fill-purple-400 text-sm font-semibold">w₁</text>

            <line x1="180" y1="200" x2="420" y2="200" className="stroke-purple-400" strokeWidth="3" />
            <text x="280" y="190" className="fill-purple-600 dark:fill-purple-400 text-sm font-semibold">w₂</text>

            <line x1="180" y1="300" x2="420" y2="200" className="stroke-purple-400" strokeWidth="3" />
            <text x="280" y="260" className="fill-purple-600 dark:fill-purple-400 text-sm font-semibold">w₃</text>

            {/* Bias */}
            <circle cx="150" cy="360" r="20" className="fill-gray-200 dark:fill-gray-700 stroke-gray-500" strokeWidth="2" />
            <text x="150" y="365" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">b</text>
            <line x1="170" y1="360" x2="420" y2="220" className="stroke-gray-400" strokeWidth="2" strokeDasharray="4" />

            {/* Summation 노드 */}
            <circle cx="450" cy="200" r="40" className="fill-yellow-100 dark:fill-yellow-900 stroke-yellow-500" strokeWidth="3" />
            <text x="450" y="205" textAnchor="middle" className="fill-gray-800 dark:fill-gray-200 text-lg font-bold">Σ</text>

            {/* Activation Function */}
            <rect x="550" y="170" width="80" height="60" rx="8" className="fill-green-100 dark:fill-green-900 stroke-green-500" strokeWidth="2" />
            <text x="590" y="205" textAnchor="middle" className="fill-green-700 dark:fill-green-300 text-sm font-semibold">σ(z)</text>
            <line x1="490" y1="200" x2="550" y2="200" className="stroke-gray-600" strokeWidth="2" markerEnd="url(#arrowhead)" />

            {/* 출력 */}
            <circle cx="700" cy="200" r="30" className="fill-red-200 dark:fill-red-900 stroke-red-500" strokeWidth="2" />
            <text x="700" y="205" textAnchor="middle" className="fill-red-700 dark:fill-red-300 text-sm font-semibold">y</text>
            <line x1="630" y1="200" x2="670" y2="200" className="stroke-gray-600" strokeWidth="2" markerEnd="url(#arrowhead)" />

            {/* 화살표 정의 */}
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" className="fill-gray-600" />
              </marker>
            </defs>

            {/* 수식 */}
            <text x="450" y="350" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-base font-mono">
              z = w₁x₁ + w₂x₂ + w₃x₃ + b
            </text>
            <text x="590" y="350" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-base font-mono">
              y = σ(z)
            </text>
          </svg>
        </div>

        <div className="bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-800 rounded-lg p-6">
          <h3 className="font-semibold text-violet-900 dark:text-violet-100 mb-3">Perceptron의 한계</h3>
          <p className="text-violet-800 dark:text-violet-200 mb-3">
            단일 Perceptron은 선형 분류만 가능합니다. XOR 문제처럼 비선형 문제는 해결할 수 없습니다.
          </p>
          <ul className="list-disc list-inside space-y-1 text-violet-700 dark:text-violet-300">
            <li>AND, OR 게이트는 구현 가능</li>
            <li>XOR 게이트는 단일 Perceptron으로 불가능</li>
            <li>해결책: 다층 신경망 (Multi-Layer Perceptron, MLP)</li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">다층 신경망 (MLP)</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          MLP는 입력층, 은닉층, 출력층으로 구성된 다층 구조입니다.
          은닉층이 추가됨으로써 비선형 문제를 해결할 수 있게 되었습니다.
        </p>

        {/* MLP 구조 다이어그램 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4 text-center">Multi-Layer Perceptron (MLP)</h3>
          <svg viewBox="0 0 900 500" className="w-full h-auto">
            {/* Input Layer */}
            <text x="100" y="30" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">Input Layer</text>
            <circle cx="100" cy="150" r="25" className="fill-blue-200 dark:fill-blue-800 stroke-blue-500" strokeWidth="2" />
            <circle cx="100" cy="250" r="25" className="fill-blue-200 dark:fill-blue-800 stroke-blue-500" strokeWidth="2" />
            <circle cx="100" cy="350" r="25" className="fill-blue-200 dark:fill-blue-800 stroke-blue-500" strokeWidth="2" />

            {/* Hidden Layer 1 */}
            <text x="350" y="30" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">Hidden Layer 1</text>
            {[100, 175, 250, 325].map((y, i) => (
              <circle key={i} cx="350" cy={y} r="25" className="fill-purple-200 dark:fill-purple-800 stroke-purple-500" strokeWidth="2" />
            ))}

            {/* Hidden Layer 2 */}
            <text x="600" y="30" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">Hidden Layer 2</text>
            {[125, 212, 299].map((y, i) => (
              <circle key={i} cx="600" cy={y} r="25" className="fill-purple-200 dark:fill-purple-800 stroke-purple-500" strokeWidth="2" />
            ))}

            {/* Output Layer */}
            <text x="800" y="30" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">Output Layer</text>
            <circle cx="800" cy="175" r="25" className="fill-red-200 dark:fill-red-900 stroke-red-500" strokeWidth="2" />
            <circle cx="800" cy="275" r="25" className="fill-red-200 dark:fill-red-900 stroke-red-500" strokeWidth="2" />

            {/* Connections - Input to Hidden1 */}
            {[150, 250, 350].map((y1) =>
              [100, 175, 250, 325].map((y2, i) => (
                <line key={`in-h1-${y1}-${i}`} x1="125" y1={y1} x2="325" y2={y2} className="stroke-gray-300 dark:stroke-gray-600" strokeWidth="1" opacity="0.5" />
              ))
            )}

            {/* Connections - Hidden1 to Hidden2 */}
            {[100, 175, 250, 325].map((y1) =>
              [125, 212, 299].map((y2, i) => (
                <line key={`h1-h2-${y1}-${i}`} x1="375" y1={y1} x2="575" y2={y2} className="stroke-gray-300 dark:stroke-gray-600" strokeWidth="1" opacity="0.5" />
              ))
            )}

            {/* Connections - Hidden2 to Output */}
            {[125, 212, 299].map((y1) =>
              [175, 275].map((y2, i) => (
                <line key={`h2-out-${y1}-${i}`} x1="625" y1={y1} x2="775" y2={y2} className="stroke-gray-300 dark:stroke-gray-600" strokeWidth="1" opacity="0.5" />
              ))
            )}

            {/* Labels */}
            <text x="100" y="150" textAnchor="middle" dy="5" className="fill-blue-700 dark:fill-blue-300 text-xs font-semibold">x₁</text>
            <text x="100" y="250" textAnchor="middle" dy="5" className="fill-blue-700 dark:fill-blue-300 text-xs font-semibold">x₂</text>
            <text x="100" y="350" textAnchor="middle" dy="5" className="fill-blue-700 dark:fill-blue-300 text-xs font-semibold">x₃</text>

            <text x="800" y="175" textAnchor="middle" dy="5" className="fill-red-700 dark:fill-red-300 text-xs font-semibold">y₁</text>
            <text x="800" y="275" textAnchor="middle" dy="5" className="fill-red-700 dark:fill-red-300 text-xs font-semibold">y₂</text>

            {/* Annotations */}
            <text x="450" y="450" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-sm">
              각 층마다 비선형 활성화 함수 적용 (ReLU, Sigmoid, Tanh 등)
            </text>
          </svg>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">활성화 함수 (Activation Functions)</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          활성화 함수는 신경망에 비선형성을 부여하여 복잡한 패턴을 학습할 수 있게 합니다.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          {/* Sigmoid */}
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
            <h3 className="font-semibold text-lg mb-3 text-blue-600 dark:text-blue-400">Sigmoid</h3>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 mb-3 font-mono text-sm">
              σ(x) = 1 / (1 + e⁻ˣ)
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">출력 범위: (0, 1)</p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 확률 해석 가능</li>
              <li>✗ Vanishing Gradient</li>
              <li>✗ 출력이 0 중심이 아님</li>
            </ul>
          </div>

          {/* Tanh */}
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
            <h3 className="font-semibold text-lg mb-3 text-green-600 dark:text-green-400">Tanh</h3>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 mb-3 font-mono text-sm">
              tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">출력 범위: (-1, 1)</p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 0 중심 출력</li>
              <li>✓ Sigmoid보다 빠른 수렴</li>
              <li>✗ Vanishing Gradient</li>
            </ul>
          </div>

          {/* ReLU */}
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
            <h3 className="font-semibold text-lg mb-3 text-purple-600 dark:text-purple-400">ReLU</h3>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-3 mb-3 font-mono text-sm">
              ReLU(x) = max(0, x)
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">출력 범위: [0, ∞)</p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 계산 효율적</li>
              <li>✓ Gradient 소실 완화</li>
              <li>✗ Dying ReLU 문제</li>
            </ul>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
          <h3 className="font-semibold text-green-900 dark:text-green-100 mb-3">현대 딥러닝에서의 선택</h3>
          <ul className="space-y-2 text-green-800 dark:text-green-200">
            <li>• <strong>ReLU 계열</strong>: 가장 널리 사용 (ReLU, Leaky ReLU, ELU, GELU)</li>
            <li>• <strong>Swish/Mish</strong>: 최신 모델에서 성능 향상</li>
            <li>• <strong>Sigmoid/Tanh</strong>: 출력층이나 특수한 경우에만 사용</li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Backpropagation: 역전파 알고리즘</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Backpropagation은 신경망의 가중치를 학습하는 핵심 알고리즘입니다.
          Chain Rule을 이용하여 출력층에서 입력층 방향으로 오차를 전파하며 가중치를 업데이트합니다.
        </p>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4">학습 과정</h3>
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-violet-100 dark:bg-violet-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-violet-600 dark:text-violet-400 font-bold">1</span>
              </div>
              <div>
                <h4 className="font-medium mb-1">Forward Pass (순전파)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  입력 데이터를 신경망에 통과시켜 예측값 계산
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-violet-100 dark:bg-violet-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-violet-600 dark:text-violet-400 font-bold">2</span>
              </div>
              <div>
                <h4 className="font-medium mb-1">Loss 계산</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  예측값과 실제값의 차이를 손실 함수로 계산 (MSE, Cross-Entropy 등)
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-violet-100 dark:bg-violet-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-violet-600 dark:text-violet-400 font-bold">3</span>
              </div>
              <div>
                <h4 className="font-medium mb-1">Backward Pass (역전파)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Chain Rule로 각 가중치에 대한 Gradient 계산
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-violet-100 dark:bg-violet-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <span className="text-violet-600 dark:text-violet-400 font-bold">4</span>
              </div>
              <div>
                <h4 className="font-medium mb-1">Weight Update (가중치 업데이트)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  경사하강법으로 가중치 갱신: w = w - η × ∂L/∂w
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6">
          <h3 className="font-semibold text-yellow-900 dark:text-yellow-100 mb-3">Vanishing & Exploding Gradient</h3>
          <div className="space-y-3 text-yellow-800 dark:text-yellow-200">
            <div>
              <strong>Vanishing Gradient (기울기 소실)</strong>
              <p className="text-sm mt-1">깊은 신경망에서 Gradient가 0에 가까워져 학습 정체</p>
              <p className="text-sm text-yellow-700 dark:text-yellow-300">→ 해결: ReLU, Residual Connection, Batch Normalization</p>
            </div>
            <div>
              <strong>Exploding Gradient (기울기 폭발)</strong>
              <p className="text-sm mt-1">Gradient가 너무 커져서 학습 불안정</p>
              <p className="text-sm text-yellow-700 dark:text-yellow-300">→ 해결: Gradient Clipping, Weight Initialization</p>
            </div>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Neural Networks Foundations',
            icon: 'paper' as const,
            color: 'border-violet-500',
            items: [
              {
                title: 'The Perceptron: A Probabilistic Model',
                authors: 'Frank Rosenblatt',
                year: '1957',
                description: '최초의 Perceptron 논문 - 인공신경망의 시작',
                link: 'https://en.wikipedia.org/wiki/Perceptron'
              },
              {
                title: 'Learning representations by back-propagating errors',
                authors: 'David Rumelhart, Geoffrey Hinton, Ronald Williams',
                year: '1986',
                description: 'Backpropagation 알고리즘의 정립',
                link: 'https://www.nature.com/articles/323533a0'
              },
              {
                title: 'Deep Learning (Book)',
                authors: 'Ian Goodfellow, Yoshua Bengio, Aaron Courville',
                year: '2016',
                description: '딥러닝의 바이블 - 신경망 기초부터 고급까지',
                link: 'https://www.deeplearningbook.org/'
              }
            ]
          },
          {
            title: 'Activation Functions',
            icon: 'paper' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Rectified Linear Units Improve Neural Networks',
                authors: 'Vinod Nair, Geoffrey Hinton',
                year: '2010',
                description: 'ReLU의 효과성 입증 - 현대 딥러닝의 핵심',
                link: 'https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf'
              },
              {
                title: 'Searching for Activation Functions',
                authors: 'Prajit Ramachandran, Barret Zoph, Quoc V. Le',
                year: '2017',
                description: 'Swish 활성화 함수 제안 - Neural Architecture Search',
                link: 'https://arxiv.org/abs/1710.05941'
              },
              {
                title: 'GELU: Gaussian Error Linear Units',
                authors: 'Dan Hendrycks, Kevin Gimpel',
                year: '2016',
                description: 'BERT, GPT 등에서 사용되는 GELU',
                link: 'https://arxiv.org/abs/1606.08415'
              }
            ]
          },
          {
            title: 'Optimization & Training',
            icon: 'paper' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'Understanding the difficulty of training deep networks',
                authors: 'Xavier Glorot, Yoshua Bengio',
                year: '2010',
                description: 'Xavier Initialization - 가중치 초기화의 중요성',
                link: 'http://proceedings.mlr.press/v9/glorot10a.html'
              },
              {
                title: 'Batch Normalization',
                authors: 'Sergey Ioffe, Christian Szegedy',
                year: '2015',
                description: '학습 안정화와 속도 향상 - 딥러닝 필수 기법',
                link: 'https://arxiv.org/abs/1502.03167'
              },
              {
                title: 'Adam Optimizer',
                authors: 'Diederik P. Kingma, Jimmy Ba',
                year: '2014',
                description: '가장 널리 쓰이는 최적화 알고리즘',
                link: 'https://arxiv.org/abs/1412.6980'
              }
            ]
          },
          {
            title: 'Tools & Frameworks',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'PyTorch',
                authors: 'Facebook AI Research',
                year: '2024',
                description: '가장 인기 있는 딥러닝 프레임워크',
                link: 'https://pytorch.org/'
              },
              {
                title: 'TensorFlow',
                authors: 'Google Brain',
                year: '2024',
                description: '프로덕션 딥러닝 플랫폼',
                link: 'https://www.tensorflow.org/'
              },
              {
                title: 'Neural Network Playground',
                authors: 'TensorFlow Team',
                year: '2024',
                description: '브라우저에서 신경망 실습',
                link: 'https://playground.tensorflow.org/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
