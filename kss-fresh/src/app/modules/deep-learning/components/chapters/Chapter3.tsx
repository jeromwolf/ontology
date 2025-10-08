'use client';

import References from '@/components/common/References';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">RNN: 순환 신경망</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Recurrent Neural Network (RNN)은 시계열 데이터를 처리하기 위해 설계된 신경망입니다.
          이전 시점의 정보를 기억하고 현재 시점의 입력과 결합하여 출력을 생성합니다.
        </p>

        {/* RNN 구조 다이어그램 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4 text-center">RNN 구조 (펼친 형태)</h3>
          <svg viewBox="0 0 1000 400" className="w-full h-auto">
            {/* Time steps */}
            {[0, 1, 2, 3, 4].map((t) => (
              <g key={`timestep-${t}`}>
                {/* Input */}
                <circle cx={150 + t * 200} cy={320} r="25" className="fill-blue-200 dark:fill-blue-800 stroke-blue-500" strokeWidth="2" />
                <text x={150 + t * 200} y={325} textAnchor="middle" className="fill-blue-700 dark:fill-blue-300 text-sm font-semibold">
                  x<tspan baselineShift="sub" fontSize="0.7em">{t}</tspan>
                </text>

                {/* Hidden State */}
                <rect x={125 + t * 200} y={150} width="50" height="50" rx="8" className="fill-purple-200 dark:fill-purple-800 stroke-purple-500" strokeWidth="2" />
                <text x={150 + t * 200} y={180} textAnchor="middle" className="fill-purple-700 dark:fill-purple-300 text-sm font-semibold">
                  h<tspan baselineShift="sub" fontSize="0.7em">{t}</tspan>
                </text>

                {/* Output */}
                <circle cx={150 + t * 200} cy={50} r="25" className="fill-red-200 dark:fill-red-900 stroke-red-500" strokeWidth="2" />
                <text x={150 + t * 200} y={55} textAnchor="middle" className="fill-red-700 dark:fill-red-300 text-sm font-semibold">
                  y<tspan baselineShift="sub" fontSize="0.7em">{t}</tspan>
                </text>

                {/* Connections */}
                {/* Input to Hidden */}
                <line x1={150 + t * 200} y1={295} x2={150 + t * 200} y2={200} className="stroke-gray-600" strokeWidth="2" markerEnd="url(#arrowhead)" />

                {/* Hidden to Output */}
                <line x1={150 + t * 200} y1={150} x2={150 + t * 200} y2={75} className="stroke-gray-600" strokeWidth="2" markerEnd="url(#arrowhead)" />

                {/* Recurrent connection */}
                {t < 4 && (
                  <path
                    d={`M ${175 + t * 200} 175 Q ${200 + t * 200} 175 ${225 + t * 200} 175`}
                    fill="none"
                    className="stroke-purple-500"
                    strokeWidth="2"
                    markerEnd="url(#arrowhead-purple)"
                  />
                )}

                {/* Time label */}
                <text x={150 + t * 200} y={370} textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  t = {t}
                </text>
              </g>
            ))}

            {/* Arrow markers */}
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" className="fill-gray-600" />
              </marker>
              <marker id="arrowhead-purple" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" className="fill-purple-500" />
              </marker>
            </defs>

            {/* Labels */}
            <text x="500" y="30" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-base font-semibold">
              h<tspan baselineShift="sub" fontSize="0.7em">t</tspan> = tanh(W<tspan baselineShift="sub" fontSize="0.7em">h</tspan> · h<tspan baselineShift="sub" fontSize="0.7em">t-1</tspan> + W<tspan baselineShift="sub" fontSize="0.7em">x</tspan> · x<tspan baselineShift="sub" fontSize="0.7em">t</tspan> + b)
            </text>
          </svg>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6">
          <h3 className="font-semibold text-yellow-900 dark:text-yellow-100 mb-3">RNN의 문제점</h3>
          <ul className="space-y-2 text-yellow-800 dark:text-yellow-200">
            <li>• <strong>Vanishing Gradient</strong>: 긴 시퀀스에서 기울기 소실</li>
            <li>• <strong>단기 기억</strong>: 먼 과거 정보를 잊어버림</li>
            <li>• <strong>학습 불안정</strong>: 긴 의존성 학습 어려움</li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">LSTM: Long Short-Term Memory</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          LSTM은 RNN의 장기 의존성 문제를 해결하기 위해 고안된 구조입니다.
          게이트 메커니즘을 통해 정보를 선택적으로 기억하고 잊습니다.
        </p>

        {/* LSTM Cell 구조 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4 text-center">LSTM Cell 내부 구조</h3>
          <svg viewBox="0 0 900 500" className="w-full h-auto">
            {/* Cell State (위쪽 수평선) */}
            <line x1="50" y1="100" x2="850" y2="100" className="stroke-green-500" strokeWidth="4" />
            <text x="450" y="80" textAnchor="middle" className="fill-green-600 dark:fill-green-400 text-sm font-semibold">
              Cell State (C_t)
            </text>

            {/* Forget Gate */}
            <g>
              <rect x="150" y="200" width="100" height="80" rx="8" className="fill-red-100 dark:fill-red-900/30 stroke-red-500" strokeWidth="2" />
              <text x="200" y="235" textAnchor="middle" className="fill-red-700 dark:fill-red-300 text-sm font-bold">
                σ
              </text>
              <text x="200" y="270" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs">
                Forget Gate
              </text>

              {/* Connection to Cell State */}
              <line x1="200" y1="200" x2="200" y2="100" className="stroke-red-500" strokeWidth="2" />
              <circle cx="200" cy="100" r="15" className="fill-white dark:fill-gray-800 stroke-red-500" strokeWidth="2" />
              <text x="200" y="105" textAnchor="middle" className="fill-red-600 text-lg font-bold">×</text>
            </g>

            {/* Input Gate */}
            <g>
              <rect x="350" y="200" width="100" height="80" rx="8" className="fill-blue-100 dark:fill-blue-900/30 stroke-blue-500" strokeWidth="2" />
              <text x="400" y="235" textAnchor="middle" className="fill-blue-700 dark:fill-blue-300 text-sm font-bold">
                σ
              </text>
              <text x="400" y="270" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs">
                Input Gate
              </text>
            </g>

            {/* Candidate Values */}
            <g>
              <rect x="500" y="200" width="100" height="80" rx="8" className="fill-purple-100 dark:fill-purple-900/30 stroke-purple-500" strokeWidth="2" />
              <text x="550" y="235" textAnchor="middle" className="fill-purple-700 dark:fill-purple-300 text-sm font-bold">
                tanh
              </text>
              <text x="550" y="270" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs">
                Candidate
              </text>

              {/* Multiply with Input Gate */}
              <circle cx="475" cy="100" r="15" className="fill-white dark:fill-gray-800 stroke-blue-500" strokeWidth="2" />
              <text x="475" y="105" textAnchor="middle" className="fill-blue-600 text-lg font-bold">×</text>
              <line x1="400" y1="200" x2="400" y2="130" className="stroke-blue-500" strokeWidth="2" />
              <line x1="400" y1="130" x2="465" y2="105" className="stroke-blue-500" strokeWidth="2" />
              <line x1="550" y1="200" x2="550" y2="130" className="stroke-purple-500" strokeWidth="2" />
              <line x1="550" y1="130" x2="485" y2="105" className="stroke-purple-500" strokeWidth="2" />
            </g>

            {/* Cell State Update (Add) */}
            <circle cx="475" cy="100" r="0" className="fill-transparent" />
            <circle cx="350" cy="100" r="15" className="fill-white dark:fill-gray-800 stroke-green-500" strokeWidth="2" />
            <text x="350" y="105" textAnchor="middle" className="fill-green-600 text-lg font-bold">+</text>

            {/* Output Gate */}
            <g>
              <rect x="650" y="200" width="100" height="80" rx="8" className="fill-orange-100 dark:fill-orange-900/30 stroke-orange-500" strokeWidth="2" />
              <text x="700" y="235" textAnchor="middle" className="fill-orange-700 dark:fill-orange-300 text-sm font-bold">
                σ
              </text>
              <text x="700" y="270" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs">
                Output Gate
              </text>

              {/* tanh of Cell State */}
              <rect x="625" y="120" width="50" height="40" rx="4" className="fill-green-100 dark:fill-green-900/30 stroke-green-500" strokeWidth="1" />
              <text x="650" y="145" textAnchor="middle" className="fill-green-700 dark:fill-green-300 text-xs font-bold">
                tanh
              </text>

              {/* Output */}
              <circle cx="750" cy="350" r="15" className="fill-white dark:fill-gray-800 stroke-orange-500" strokeWidth="2" />
              <text x="750" y="355" textAnchor="middle" className="fill-orange-600 text-lg font-bold">×</text>
              <line x1="650" y1="140" x2="740" y2="345" className="stroke-green-500" strokeWidth="2" />
              <line x1="700" y1="200" x2="700" y2="330" className="stroke-orange-500" strokeWidth="2" />
              <line x1="700" y1="330" x2="740" y2="350" className="stroke-orange-500" strokeWidth="2" />
            </g>

            {/* Input h(t-1) and x_t */}
            <text x="150" y="380" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm">
              h<tspan baselineShift="sub" fontSize="0.7em">t-1</tspan>, x<tspan baselineShift="sub" fontSize="0.7em">t</tspan>
            </text>
            <line x1="150" y1="360" x2="150" y2="290" className="stroke-gray-500" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <line x1="150" y1="320" x2="400" y2="290" className="stroke-gray-500" strokeWidth="1" opacity="0.5" />
            <line x1="150" y1="320" x2="550" y2="290" className="stroke-gray-500" strokeWidth="1" opacity="0.5" />
            <line x1="150" y1="320" x2="700" y2="290" className="stroke-gray-500" strokeWidth="1" opacity="0.5" />

            {/* Output h_t */}
            <text x="750" y="420" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">
              h_t (Hidden State)
            </text>
            <line x1="750" y1="365" x2="750" y2="400" className="stroke-gray-600" strokeWidth="2" markerEnd="url(#arrowhead)" />
          </svg>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
            <h3 className="font-semibold text-lg mb-3 text-red-600 dark:text-red-400">Forget Gate</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              f<sub>t</sub> = σ(W<sub>f</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>f</sub>)
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              어떤 정보를 버릴지 결정 (0 = 완전히 잊음, 1 = 완전히 기억)
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
            <h3 className="font-semibold text-lg mb-3 text-blue-600 dark:text-blue-400">Input Gate</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              i<sub>t</sub> = σ(W<sub>i</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>i</sub>)
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              어떤 새 정보를 저장할지 결정
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
            <h3 className="font-semibold text-lg mb-3 text-orange-600 dark:text-orange-400">Output Gate</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              o<sub>t</sub> = σ(W<sub>o</sub> · [h<sub>t-1</sub>, x<sub>t</sub>] + b<sub>o</sub>)
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Cell State의 어떤 부분을 출력할지 결정
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">GRU: Gated Recurrent Unit</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          GRU는 LSTM을 간소화한 구조로, 2개의 게이트(Reset, Update)만 사용합니다.
          파라미터 수가 적어 학습이 빠르며, 많은 경우 LSTM과 유사한 성능을 보입니다.
        </p>

        <div className="bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-800 rounded-lg p-6">
          <h3 className="font-semibold text-violet-900 dark:text-violet-100 mb-3">GRU vs LSTM</h3>
          <div className="grid md:grid-cols-2 gap-4 text-violet-800 dark:text-violet-200">
            <div>
              <strong>GRU 장점</strong>
              <ul className="mt-2 space-y-1 text-sm">
                <li>✓ 파라미터 수가 적음 (빠른 학습)</li>
                <li>✓ 구조가 간단함</li>
                <li>✓ 작은 데이터셋에서 효과적</li>
              </ul>
            </div>
            <div>
              <strong>LSTM 장점</strong>
              <ul className="mt-2 space-y-1 text-sm">
                <li>✓ 더 강력한 표현력</li>
                <li>✓ 매우 긴 시퀀스에서 유리</li>
                <li>✓ 복잡한 패턴 학습</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Bidirectional RNN</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          양방향 RNN은 순방향과 역방향 두 개의 RNN을 결합하여 과거와 미래의 컨텍스트를 모두 활용합니다.
        </p>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">응용 분야</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-2 text-blue-600 dark:text-blue-400">자연어 처리</h4>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 기계 번역 (Seq2Seq)</li>
                <li>• 감성 분석</li>
                <li>• Named Entity Recognition</li>
                <li>• 품사 태깅</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2 text-green-600 dark:text-green-400">시계열 분석</h4>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 주가 예측</li>
                <li>• 음성 인식</li>
                <li>• 비디오 분석</li>
                <li>• 이상 탐지</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'RNN & LSTM Foundations',
            icon: 'paper' as const,
            color: 'border-violet-500',
            items: [
              {
                title: 'Long Short-Term Memory',
                authors: 'Sepp Hochreiter, Jürgen Schmidhuber',
                year: '1997',
                description: 'LSTM의 원조 논문 - 장기 의존성 문제 해결',
                link: 'https://www.bioinf.jku.at/publications/older/2604.pdf'
              },
              {
                title: 'Learning to Forget: Continual Prediction with LSTM',
                authors: 'Felix A. Gers, et al.',
                year: '2000',
                description: 'Forget Gate 추가 - 현대 LSTM의 완성',
                link: 'https://dl.acm.org/doi/10.1162/089976600300015015'
              },
              {
                title: 'Learning Phrase Representations using RNN Encoder-Decoder',
                authors: 'Kyunghyun Cho, et al.',
                year: '2014',
                description: 'GRU 제안 - LSTM의 간소화된 대안',
                link: 'https://arxiv.org/abs/1406.1078'
              }
            ]
          },
          {
            title: 'Sequence Modeling',
            icon: 'paper' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Sequence to Sequence Learning with Neural Networks',
                authors: 'Ilya Sutskever, et al.',
                year: '2014',
                description: 'Seq2Seq 모델 - 기계 번역의 혁신',
                link: 'https://arxiv.org/abs/1409.3215'
              },
              {
                title: 'Neural Machine Translation by Jointly Learning to Align',
                authors: 'Dzmitry Bahdanau, et al.',
                year: '2015',
                description: 'Attention 메커니즘의 시작',
                link: 'https://arxiv.org/abs/1409.0473'
              },
              {
                title: 'Show and Tell: A Neural Image Caption Generator',
                authors: 'Oriol Vinyals, et al.',
                year: '2015',
                description: 'CNN + LSTM로 이미지 캡셔닝',
                link: 'https://arxiv.org/abs/1411.4555'
              }
            ]
          },
          {
            title: 'Speech & Audio',
            icon: 'paper' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'Deep Speech 2',
                authors: 'Dario Amodei, et al.',
                year: '2016',
                description: 'Baidu - 영어와 중국어 음성 인식',
                link: 'https://arxiv.org/abs/1512.02595'
              },
              {
                title: 'Listen, Attend and Spell',
                authors: 'William Chan, et al.',
                year: '2016',
                description: 'Attention 기반 음성 인식 - Google',
                link: 'https://arxiv.org/abs/1508.01211'
              },
              {
                title: 'WaveNet: A Generative Model for Raw Audio',
                authors: 'Aaron van den Oord, et al.',
                year: '2016',
                description: 'DeepMind - 자연스러운 음성 합성',
                link: 'https://arxiv.org/abs/1609.03499'
              }
            ]
          },
          {
            title: 'Tools & Libraries',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'PyTorch nn.LSTM',
                authors: 'PyTorch Team',
                year: '2024',
                description: 'PyTorch LSTM 공식 문서',
                link: 'https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html'
              },
              {
                title: 'TensorFlow Keras LSTM',
                authors: 'TensorFlow Team',
                year: '2024',
                description: 'Keras LSTM 레이어',
                link: 'https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM'
              },
              {
                title: 'Understanding LSTM Networks',
                authors: 'Christopher Olah',
                year: '2015',
                description: '가장 유명한 LSTM 설명 블로그',
                link: 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
