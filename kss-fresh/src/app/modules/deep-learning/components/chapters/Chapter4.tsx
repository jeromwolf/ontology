'use client';

import { useState } from 'react';
import References from '@/components/common/References';
import { Copy, Check } from 'lucide-react';

export default function Chapter4() {
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});

  const handleCopy = async (text: string, id: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedStates(prev => ({ ...prev, [id]: true }));
    setTimeout(() => {
      setCopiedStates(prev => ({ ...prev, [id]: false }));
    }, 2000);
  };

  return (
    <div className="space-y-8">
      {/* 1. Transformer 소개 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Transformer: Attention의 힘
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          2017년 구글이 발표한 "Attention Is All You Need" 논문은 딥러닝 분야에 혁명을 일으켰습니다.
          RNN의 순차적 처리의 한계를 극복하고, Attention 메커니즘만으로 구성된 완전히 새로운 아키텍처를 제시했습니다.
        </p>

        <div className="bg-gradient-to-br from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-2xl p-6 border border-violet-200 dark:border-violet-700 mb-6">
          <h3 className="text-lg font-semibold mb-3 text-violet-900 dark:text-violet-300">
            💡 Transformer의 핵심 장점
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li><strong>병렬 처리</strong>: RNN과 달리 모든 위치를 동시에 계산 가능</li>
            <li><strong>장거리 의존성</strong>: 문장의 어느 위치든 직접 참조 가능</li>
            <li><strong>확장성</strong>: 대규모 모델(GPT, BERT)의 기반 아키텍처</li>
            <li><strong>범용성</strong>: NLP뿐만 아니라 Vision(ViT), 멀티모달까지 확장</li>
          </ul>
        </div>
      </section>

      {/* 2. Attention 메커니즘 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Attention 메커니즘의 이해
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          Attention은 입력 시퀀스에서 중요한 부분에 더 많은 가중치를 부여하는 메커니즘입니다.
          Query(Q), Key(K), Value(V) 세 가지 벡터를 사용합니다.
        </p>

        {/* Attention 수식 */}
        <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-6 mb-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Scaled Dot-Product Attention</h3>
            <button
              onClick={() => handleCopy('Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V', 'attention-formula')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="수식 복사"
            >
              {copiedStates['attention-formula'] ? (
                <Check size={16} className="text-green-600" />
              ) : (
                <Copy size={16} className="text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <div className="font-mono text-sm text-center p-4 bg-white dark:bg-gray-900 rounded-lg">
            <div className="text-lg mb-2">Attention(Q, K, V) = softmax(<span className="text-blue-600 dark:text-blue-400">QK<sup>T</sup></span> / √d<sub>k</sub>)V</div>
            <div className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              d<sub>k</sub> = Query/Key 벡터의 차원
            </div>
          </div>
        </div>

        {/* Self-Attention 시각화 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Self-Attention 작동 과정</h3>
          <div className="overflow-x-auto">
            <svg viewBox="0 0 1000 600" className="w-full h-auto">
              {/* 입력 문장 "The cat sat on the mat" */}
              <text x="500" y="30" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-semibold text-lg">
                입력: "The cat sat on the mat"
              </text>

              {/* 6개 단어 */}
              {['The', 'cat', 'sat', 'on', 'the', 'mat'].map((word, i) => (
                <g key={`word-${i}`}>
                  {/* 입력 단어 */}
                  <rect
                    x={120 + i * 140}
                    y="60"
                    width="100"
                    height="40"
                    rx="8"
                    className="fill-blue-100 dark:fill-blue-900 stroke-blue-500 dark:stroke-blue-400"
                    strokeWidth="2"
                  />
                  <text
                    x={170 + i * 140}
                    y="85"
                    textAnchor="middle"
                    className="fill-gray-900 dark:fill-gray-100 font-medium text-sm"
                  >
                    {word}
                  </text>
                </g>
              ))}

              {/* Q, K, V 변환 */}
              <text x="50" y="180" className="fill-red-600 dark:fill-red-400 font-bold">Query (Q)</text>
              <text x="50" y="280" className="fill-green-600 dark:fill-green-400 font-bold">Key (K)</text>
              <text x="50" y="380" className="fill-purple-600 dark:fill-purple-400 font-bold">Value (V)</text>

              {/* Q, K, V 벡터들 */}
              {[0, 1, 2, 3, 4, 5].map((i) => (
                <g key={`qkv-${i}`}>
                  {/* Query */}
                  <circle
                    cx={170 + i * 140}
                    cy="170"
                    r="20"
                    className="fill-red-200 dark:fill-red-900 stroke-red-500 dark:stroke-red-400"
                    strokeWidth="2"
                  />
                  <text
                    x={170 + i * 140}
                    y="176"
                    textAnchor="middle"
                    className="fill-gray-900 dark:fill-gray-100 text-xs font-semibold"
                  >
                    Q{i}
                  </text>

                  {/* Key */}
                  <circle
                    cx={170 + i * 140}
                    cy="270"
                    r="20"
                    className="fill-green-200 dark:fill-green-900 stroke-green-500 dark:stroke-green-400"
                    strokeWidth="2"
                  />
                  <text
                    x={170 + i * 140}
                    y="276"
                    textAnchor="middle"
                    className="fill-gray-900 dark:fill-gray-100 text-xs font-semibold"
                  >
                    K{i}
                  </text>

                  {/* Value */}
                  <circle
                    cx={170 + i * 140}
                    cy="370"
                    r="20"
                    className="fill-purple-200 dark:fill-purple-900 stroke-purple-500 dark:stroke-purple-400"
                    strokeWidth="2"
                  />
                  <text
                    x={170 + i * 140}
                    y="376"
                    textAnchor="middle"
                    className="fill-gray-900 dark:fill-gray-100 text-xs font-semibold"
                  >
                    V{i}
                  </text>

                  {/* 입력에서 Q,K,V로 연결선 */}
                  <line
                    x1={170 + i * 140}
                    y1="100"
                    x2={170 + i * 140}
                    y2="150"
                    className="stroke-red-400 dark:stroke-red-600"
                    strokeWidth="1.5"
                    strokeDasharray="4"
                  />
                  <line
                    x1={170 + i * 140}
                    y1="100"
                    x2={170 + i * 140}
                    y2="250"
                    className="stroke-green-400 dark:stroke-green-600"
                    strokeWidth="1.5"
                    strokeDasharray="4"
                  />
                  <line
                    x1={170 + i * 140}
                    y1="100"
                    x2={170 + i * 140}
                    y2="350"
                    className="stroke-purple-400 dark:stroke-purple-600"
                    strokeWidth="1.5"
                    strokeDasharray="4"
                  />
                </g>
              ))}

              {/* Attention Score 계산 (예: "cat" 단어에 대한 attention) */}
              <text x="500" y="440" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-semibold">
                Attention Score = softmax(Q₁ · Kᵀ / √d_k)
              </text>

              {/* "cat"에 대한 attention weights 시각화 */}
              {[0.05, 0.65, 0.15, 0.05, 0.05, 0.05].map((weight, i) => (
                <g key={`attention-${i}`}>
                  <rect
                    x={120 + i * 140}
                    y="470"
                    width="100"
                    height={weight * 100}
                    className={i === 1 ? 'fill-orange-500 dark:fill-orange-600' : 'fill-orange-200 dark:fill-orange-900'}
                    opacity="0.8"
                  />
                  <text
                    x={170 + i * 140}
                    y="490"
                    textAnchor="middle"
                    className="fill-gray-900 dark:fill-gray-100 text-xs font-semibold"
                  >
                    {(weight * 100).toFixed(0)}%
                  </text>
                </g>
              ))}

              <text x="500" y="590" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-sm italic">
                "cat" 단어가 자기 자신에게 가장 높은 attention (65%)을 부여
              </text>
            </svg>
          </div>
        </div>

        {/* Attention 계산 과정 설명 */}
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-xl p-4 border border-red-200 dark:border-red-800">
            <h4 className="font-semibold mb-2 text-red-900 dark:text-red-300">1️⃣ Query (Q)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>질문을 하는 벡터</strong><br/>
              "현재 단어가 다른 단어들에게 무엇을 찾고 있는가?"
            </p>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-4 border border-green-200 dark:border-green-800">
            <h4 className="font-semibold mb-2 text-green-900 dark:text-green-300">2️⃣ Key (K)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>비교를 위한 벡터</strong><br/>
              "각 단어가 제공할 수 있는 정보는 무엇인가?"
            </p>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-4 border border-purple-200 dark:border-purple-800">
            <h4 className="font-semibold mb-2 text-purple-900 dark:text-purple-300">3️⃣ Value (V)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>실제 전달할 정보</strong><br/>
              "각 단어가 가진 실제 의미 표현"
            </p>
          </div>
        </div>
      </section>

      {/* 3. Multi-Head Attention */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Multi-Head Attention
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          단일 attention보다 여러 개의 attention을 병렬로 수행하면 문장의 다양한 측면을 동시에 학습할 수 있습니다.
          예를 들어, 한 head는 문법적 관계를, 다른 head는 의미적 유사성을 학습합니다.
        </p>

        {/* Multi-Head Attention 시각화 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Multi-Head Attention 구조 (8 heads)</h3>
          <div className="overflow-x-auto">
            <svg viewBox="0 0 1000 500" className="w-full h-auto">
              {/* 입력 */}
              <text x="500" y="30" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-semibold">
                Input Embeddings (d_model = 512)
              </text>
              <rect
                x="350"
                y="50"
                width="300"
                height="40"
                rx="8"
                className="fill-blue-100 dark:fill-blue-900 stroke-blue-500 dark:stroke-blue-400"
                strokeWidth="2"
              />

              {/* 8개의 Head로 분할 */}
              {[0, 1, 2, 3, 4, 5, 6, 7].map((head) => (
                <g key={`head-${head}`}>
                  {/* Linear 변환 (Q, K, V) */}
                  <rect
                    x={80 + head * 115}
                    y="140"
                    width="80"
                    height="60"
                    rx="6"
                    className="fill-purple-100 dark:fill-purple-900 stroke-purple-500 dark:stroke-purple-400"
                    strokeWidth="2"
                  />
                  <text
                    x={120 + head * 115}
                    y="165"
                    textAnchor="middle"
                    className="fill-gray-900 dark:fill-gray-100 text-xs font-semibold"
                  >
                    Head {head + 1}
                  </text>
                  <text
                    x={120 + head * 115}
                    y="182"
                    textAnchor="middle"
                    className="fill-gray-600 dark:fill-gray-400 text-xs"
                  >
                    Q,K,V
                  </text>
                  <text
                    x={120 + head * 115}
                    y="194"
                    textAnchor="middle"
                    className="fill-gray-600 dark:fill-gray-400 text-xs"
                  >
                    (d_k=64)
                  </text>

                  {/* 입력에서 각 Head로 연결 */}
                  <line
                    x1="500"
                    y1="90"
                    x2={120 + head * 115}
                    y2="140"
                    className="stroke-blue-400 dark:stroke-blue-600"
                    strokeWidth="1.5"
                    opacity="0.5"
                  />

                  {/* Scaled Dot-Product Attention */}
                  <rect
                    x={80 + head * 115}
                    y="240"
                    width="80"
                    height="50"
                    rx="6"
                    className="fill-orange-100 dark:fill-orange-900 stroke-orange-500 dark:stroke-orange-400"
                    strokeWidth="2"
                  />
                  <text
                    x={120 + head * 115}
                    y="262"
                    textAnchor="middle"
                    className="fill-gray-900 dark:fill-gray-100 text-xs font-semibold"
                  >
                    Attention
                  </text>
                  <text
                    x={120 + head * 115}
                    y="278"
                    textAnchor="middle"
                    className="fill-gray-600 dark:fill-gray-400 text-xs"
                  >
                    softmax
                  </text>

                  {/* Head에서 Attention으로 연결 */}
                  <line
                    x1={120 + head * 115}
                    y1="200"
                    x2={120 + head * 115}
                    y2="240"
                    className="stroke-purple-400 dark:stroke-purple-600"
                    strokeWidth="2"
                  />

                  {/* Attention에서 Concat으로 연결 */}
                  <line
                    x1={120 + head * 115}
                    y1="290"
                    x2="500"
                    y2="360"
                    className="stroke-orange-400 dark:stroke-orange-600"
                    strokeWidth="1.5"
                    opacity="0.5"
                  />
                </g>
              ))}

              {/* Concatenate */}
              <rect
                x="350"
                y="360"
                width="300"
                height="40"
                rx="8"
                className="fill-green-100 dark:fill-green-900 stroke-green-500 dark:stroke-green-400"
                strokeWidth="2"
              />
              <text
                x="500"
                y="385"
                textAnchor="middle"
                className="fill-gray-900 dark:fill-gray-100 font-semibold"
              >
                Concatenate (8 × 64 = 512)
              </text>

              {/* Linear 변환 */}
              <rect
                x="350"
                y="440"
                width="300"
                height="40"
                rx="8"
                className="fill-violet-100 dark:fill-violet-900 stroke-violet-500 dark:stroke-violet-400"
                strokeWidth="2"
              />
              <text
                x="500"
                y="465"
                textAnchor="middle"
                className="fill-gray-900 dark:fill-gray-100 font-semibold"
              >
                Linear (W^O)
              </text>

              {/* Concat에서 Linear로 연결 */}
              <line
                x1="500"
                y1="400"
                x2="500"
                y2="440"
                className="stroke-green-400 dark:stroke-green-600"
                strokeWidth="2"
              />
            </svg>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-4 text-center italic">
            8개의 attention head가 병렬로 작동하여 다양한 관점에서 문맥을 이해
          </p>
        </div>
      </section>

      {/* 4. Positional Encoding */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Positional Encoding
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          Transformer는 순차적으로 처리하지 않기 때문에, 단어의 위치 정보를 명시적으로 추가해야 합니다.
          Positional Encoding은 sin/cos 함수를 사용하여 위치 정보를 인코딩합니다.
        </p>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-6 mb-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Positional Encoding 수식</h3>
            <button
              onClick={() => handleCopy('PE(pos,2i) = sin(pos / 10000^(2i/d_model))\nPE(pos,2i+1) = cos(pos / 10000^(2i/d_model))', 'pe-formula')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="수식 복사"
            >
              {copiedStates['pe-formula'] ? (
                <Check size={16} className="text-green-600" />
              ) : (
                <Copy size={16} className="text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <div className="font-mono text-sm bg-white dark:bg-gray-900 rounded-lg p-4">
            <div className="mb-2">PE(pos, 2i) = sin(pos / 10000<sup>2i/d_model</sup>)</div>
            <div>PE(pos, 2i+1) = cos(pos / 10000<sup>2i/d_model</sup>)</div>
            <div className="text-xs text-gray-600 dark:text-gray-400 mt-3">
              pos = 단어의 위치, i = 차원 인덱스, d_model = 임베딩 차원 (512)
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
          <h3 className="text-lg font-semibold mb-3 text-blue-900 dark:text-blue-300">
            💡 왜 sin/cos 함수를 사용할까?
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li><strong>주기성</strong>: 다양한 주파수로 위치 패턴을 표현</li>
            <li><strong>상대적 위치</strong>: 두 위치 간의 거리를 쉽게 계산 가능</li>
            <li><strong>확장성</strong>: 학습 시보다 긴 문장도 처리 가능</li>
            <li><strong>학습 불필요</strong>: 고정된 함수로 계산 (파라미터 불필요)</li>
          </ul>
        </div>
      </section>

      {/* 5. Transformer 전체 아키텍처 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Transformer 전체 아키텍처
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          Transformer는 Encoder와 Decoder로 구성되며, 각각 여러 층(layer)을 쌓은 구조입니다.
          원논문에서는 6개의 Encoder와 6개의 Decoder를 사용했습니다.
        </p>

        {/* Transformer 아키텍처 시각화 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Encoder-Decoder 구조</h3>
          <div className="overflow-x-auto">
            <svg viewBox="0 0 1000 800" className="w-full h-auto">
              {/* Encoder 부분 */}
              <text x="250" y="40" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-bold text-lg">
                Encoder (×6)
              </text>

              {/* Input Embedding */}
              <rect x="150" y="60" width="200" height="40" rx="8" className="fill-blue-100 dark:fill-blue-900 stroke-blue-500 dark:stroke-blue-400" strokeWidth="2" />
              <text x="250" y="85" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Input Embedding
              </text>

              {/* Positional Encoding */}
              <rect x="150" y="120" width="200" height="40" rx="8" className="fill-green-100 dark:fill-green-900 stroke-green-500 dark:stroke-green-400" strokeWidth="2" />
              <text x="250" y="145" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                + Positional Encoding
              </text>

              {/* Encoder 블록 (N×) */}
              <rect x="130" y="190" width="240" height="350" rx="12" className="fill-violet-50 dark:fill-violet-900/30 stroke-violet-500 dark:stroke-violet-400" strokeWidth="3" strokeDasharray="8" />

              {/* Multi-Head Attention */}
              <rect x="150" y="210" width="200" height="50" rx="8" className="fill-orange-100 dark:fill-orange-900 stroke-orange-500 dark:stroke-orange-400" strokeWidth="2" />
              <text x="250" y="233" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Multi-Head
              </text>
              <text x="250" y="250" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Self-Attention
              </text>

              {/* Add & Norm */}
              <rect x="150" y="280" width="200" height="35" rx="6" className="fill-gray-200 dark:fill-gray-700 stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <text x="250" y="302" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Add & Norm
              </text>

              {/* Feed Forward */}
              <rect x="150" y="335" width="200" height="50" rx="8" className="fill-purple-100 dark:fill-purple-900 stroke-purple-500 dark:stroke-purple-400" strokeWidth="2" />
              <text x="250" y="358" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Feed Forward
              </text>
              <text x="250" y="375" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Network
              </text>

              {/* Add & Norm */}
              <rect x="150" y="405" width="200" height="35" rx="6" className="fill-gray-200 dark:fill-gray-700 stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <text x="250" y="427" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Add & Norm
              </text>

              {/* N× 표시 */}
              <text x="250" y="475" textAnchor="middle" className="fill-violet-600 dark:fill-violet-400 text-sm font-bold italic">
                N = 6 layers
              </text>

              {/* 연결선 (Encoder) */}
              <line x1="250" y1="100" x2="250" y2="120" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="250" y1="160" x2="250" y2="210" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="250" y1="260" x2="250" y2="280" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="250" y1="315" x2="250" y2="335" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="250" y1="385" x2="250" y2="405" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="250" y1="440" x2="250" y2="490" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />

              {/* Encoder 출력 */}
              <rect x="150" y="490" width="200" height="40" rx="8" className="fill-teal-100 dark:fill-teal-900 stroke-teal-500 dark:stroke-teal-400" strokeWidth="2" />
              <text x="250" y="515" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Encoder Output
              </text>

              {/* Decoder 부분 */}
              <text x="650" y="40" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-bold text-lg">
                Decoder (×6)
              </text>

              {/* Output Embedding */}
              <rect x="550" y="60" width="200" height="40" rx="8" className="fill-blue-100 dark:fill-blue-900 stroke-blue-500 dark:stroke-blue-400" strokeWidth="2" />
              <text x="650" y="85" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Output Embedding
              </text>

              {/* Positional Encoding */}
              <rect x="550" y="120" width="200" height="40" rx="8" className="fill-green-100 dark:fill-green-900 stroke-green-500 dark:stroke-green-400" strokeWidth="2" />
              <text x="650" y="145" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                + Positional Encoding
              </text>

              {/* Decoder 블록 (N×) */}
              <rect x="530" y="190" width="240" height="450" rx="12" className="fill-pink-50 dark:fill-pink-900/30 stroke-pink-500 dark:stroke-pink-400" strokeWidth="3" strokeDasharray="8" />

              {/* Masked Multi-Head Attention */}
              <rect x="550" y="210" width="200" height="50" rx="8" className="fill-red-100 dark:fill-red-900 stroke-red-500 dark:stroke-red-400" strokeWidth="2" />
              <text x="650" y="228" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Masked Multi-Head
              </text>
              <text x="650" y="245" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Self-Attention
              </text>

              {/* Add & Norm */}
              <rect x="550" y="280" width="200" height="35" rx="6" className="fill-gray-200 dark:fill-gray-700 stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <text x="650" y="302" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Add & Norm
              </text>

              {/* Cross-Attention (Encoder-Decoder Attention) */}
              <rect x="550" y="335" width="200" height="60" rx="8" className="fill-yellow-100 dark:fill-yellow-900 stroke-yellow-500 dark:stroke-yellow-400" strokeWidth="2" />
              <text x="650" y="355" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Multi-Head
              </text>
              <text x="650" y="372" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Cross-Attention
              </text>
              <text x="650" y="387" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs">
                (Encoder Output)
              </text>

              {/* Encoder에서 Cross-Attention으로 연결 */}
              <line x1="350" y1="510" x2="480" y2="365" className="stroke-teal-500 dark:stroke-teal-400" strokeWidth="2.5" markerEnd="url(#arrowhead)" />
              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" className="fill-teal-500 dark:fill-teal-400">
                  <polygon points="0 0, 10 3.5, 0 7" />
                </marker>
              </defs>
              <text x="415" y="430" className="fill-teal-600 dark:fill-teal-400 text-xs font-semibold">
                K, V
              </text>

              {/* Add & Norm */}
              <rect x="550" y="415" width="200" height="35" rx="6" className="fill-gray-200 dark:fill-gray-700 stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <text x="650" y="437" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Add & Norm
              </text>

              {/* Feed Forward */}
              <rect x="550" y="470" width="200" height="50" rx="8" className="fill-purple-100 dark:fill-purple-900 stroke-purple-500 dark:stroke-purple-400" strokeWidth="2" />
              <text x="650" y="493" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Feed Forward
              </text>
              <text x="650" y="510" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Network
              </text>

              {/* Add & Norm */}
              <rect x="550" y="540" width="200" height="35" rx="6" className="fill-gray-200 dark:fill-gray-700 stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <text x="650" y="562" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Add & Norm
              </text>

              {/* N× 표시 */}
              <text x="650" y="610" textAnchor="middle" className="fill-pink-600 dark:fill-pink-400 text-sm font-bold italic">
                N = 6 layers
              </text>

              {/* 연결선 (Decoder) */}
              <line x1="650" y1="100" x2="650" y2="120" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="650" y1="160" x2="650" y2="210" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="650" y1="260" x2="650" y2="280" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="650" y1="315" x2="650" y2="335" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="650" y1="395" x2="650" y2="415" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="650" y1="450" x2="650" y2="470" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="650" y1="520" x2="650" y2="540" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              <line x1="650" y1="575" x2="650" y2="655" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />

              {/* Linear & Softmax */}
              <rect x="550" y="655" width="200" height="40" rx="8" className="fill-indigo-100 dark:fill-indigo-900 stroke-indigo-500 dark:stroke-indigo-400" strokeWidth="2" />
              <text x="650" y="680" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Linear & Softmax
              </text>

              {/* Output Probabilities */}
              <rect x="550" y="715" width="200" height="40" rx="8" className="fill-emerald-100 dark:fill-emerald-900 stroke-emerald-500 dark:stroke-emerald-400" strokeWidth="2" />
              <text x="650" y="740" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 text-sm font-semibold">
                Output Probabilities
              </text>
            </svg>
          </div>
        </div>

        {/* 주요 구성 요소 설명 */}
        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="bg-violet-50 dark:bg-violet-900/20 rounded-xl p-4 border border-violet-200 dark:border-violet-800">
            <h4 className="font-semibold mb-2 text-violet-900 dark:text-violet-300">🔹 Encoder</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Self-Attention</strong>: 입력 문장 내 단어들 간의 관계 파악</li>
              <li>• <strong>Feed Forward</strong>: 각 위치마다 독립적으로 비선형 변환</li>
              <li>• <strong>Residual + LayerNorm</strong>: 안정적인 학습을 위한 정규화</li>
            </ul>
          </div>

          <div className="bg-pink-50 dark:bg-pink-900/20 rounded-xl p-4 border border-pink-200 dark:border-pink-800">
            <h4 className="font-semibold mb-2 text-pink-900 dark:text-pink-300">🔹 Decoder</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>Masked Self-Attention</strong>: 미래 토큰을 보지 못하도록 마스킹</li>
              <li>• <strong>Cross-Attention</strong>: Encoder 출력(소스)과 Decoder(타겟) 연결</li>
              <li>• <strong>Auto-regressive</strong>: 이전 출력을 다음 입력으로 사용</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 6. Transformer의 응용 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Transformer의 응용 사례
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          Transformer는 NLP를 넘어 다양한 분야에서 혁신을 일으키고 있습니다.
        </p>

        <div className="grid md:grid-cols-2 gap-4">
          {/* BERT */}
          <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-300 text-lg">
              🔵 BERT (2018)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>Bidirectional Encoder Representations from Transformers</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• Encoder만 사용 (양방향 문맥 이해)</li>
              <li>• Masked Language Modeling (MLM)</li>
              <li>• 질의응답, 문서 분류, NER 등에 활용</li>
            </ul>
          </div>

          {/* GPT */}
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 border border-green-200 dark:border-green-800">
            <h4 className="font-semibold mb-2 text-green-900 dark:text-green-300 text-lg">
              🟢 GPT (2018~)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>Generative Pre-trained Transformer</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• Decoder만 사용 (Auto-regressive 생성)</li>
              <li>• 다음 단어 예측 (Next Token Prediction)</li>
              <li>• GPT-3, GPT-4, ChatGPT로 발전</li>
            </ul>
          </div>

          {/* T5 */}
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800">
            <h4 className="font-semibold mb-2 text-purple-900 dark:text-purple-300 text-lg">
              🟣 T5 (2019)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>Text-to-Text Transfer Transformer</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• Encoder-Decoder 모두 사용</li>
              <li>• 모든 NLP 태스크를 Text-to-Text로 통합</li>
              <li>• 번역, 요약, 질의응답 등 범용 모델</li>
            </ul>
          </div>

          {/* Vision Transformer */}
          <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6 border border-orange-200 dark:border-orange-800">
            <h4 className="font-semibold mb-2 text-orange-900 dark:text-orange-300 text-lg">
              🟠 Vision Transformer (2020)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>이미지를 패치로 나누어 Transformer 적용</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 이미지 패치를 토큰처럼 처리</li>
              <li>• CNN 없이도 SOTA 성능 달성</li>
              <li>• DINO, MAE, CLIP 등으로 발전</li>
            </ul>
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
                title: 'Attention Is All You Need',
                authors: 'Vaswani, A., et al.',
                year: '2017',
                description: 'Transformer 아키텍처를 최초로 제안한 혁명적 논문',
                link: 'https://arxiv.org/abs/1706.03762'
              },
              {
                title: 'BERT: Pre-training of Deep Bidirectional Transformers',
                authors: 'Devlin, J., et al.',
                year: '2018',
                description: 'Encoder 기반 사전학습 모델의 효과 입증',
                link: 'https://arxiv.org/abs/1810.04805'
              },
              {
                title: 'Language Models are Few-Shot Learners (GPT-3)',
                authors: 'Brown, T., et al.',
                year: '2020',
                description: '1750억 파라미터 모델의 in-context learning 능력',
                link: 'https://arxiv.org/abs/2005.14165'
              }
            ]
          },
          {
            title: '📘 학습 자료',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'The Illustrated Transformer',
                authors: 'Jay Alammar',
                year: '2018',
                description: 'Transformer를 시각적으로 설명하는 최고의 튜토리얼',
                link: 'http://jalammar.github.io/illustrated-transformer/'
              },
              {
                title: 'Annotated Transformer',
                authors: 'Harvard NLP',
                year: '2018',
                description: 'PyTorch로 구현한 주석 달린 Transformer 코드',
                link: 'http://nlp.seas.harvard.edu/annotated-transformer/'
              },
              {
                title: 'Transformers from Scratch',
                authors: 'Peter Bloem',
                year: '2019',
                description: 'Transformer를 처음부터 구현하는 상세 가이드',
                link: 'http://peterbloem.nl/blog/transformers'
              }
            ]
          },
          {
            title: '🛠️ 라이브러리 & 도구',
            icon: 'github' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Hugging Face Transformers',
                authors: 'Hugging Face',
                year: '2023',
                description: '사전학습된 Transformer 모델 라이브러리 (BERT, GPT, T5 등)',
                link: 'https://github.com/huggingface/transformers'
              },
              {
                title: 'fairseq',
                authors: 'Meta AI',
                year: '2023',
                description: 'Seq2Seq 및 Transformer 연구를 위한 프레임워크',
                link: 'https://github.com/facebookresearch/fairseq'
              },
              {
                title: 'Tensor2Tensor',
                authors: 'Google Brain',
                year: '2023',
                description: '원논문 저자들이 만든 Transformer 구현',
                link: 'https://github.com/tensorflow/tensor2tensor'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
