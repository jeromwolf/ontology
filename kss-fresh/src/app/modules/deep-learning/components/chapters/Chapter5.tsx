'use client';

import { useState } from 'react';
import References from '@/components/common/References';
import { Copy, Check } from 'lucide-react';

export default function Chapter5() {
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
      {/* 1. 생성 모델 소개 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          GAN & 생성 모델의 세계
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          생성 모델(Generative Model)은 데이터의 분포를 학습하여 새로운 데이터를 생성하는 모델입니다.
          2014년 Ian Goodfellow가 제안한 GAN(Generative Adversarial Networks)은 게임 이론의 개념을 도입하여
          생성 모델의 성능을 획기적으로 향상시켰습니다.
        </p>

        <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl p-6 border border-purple-200 dark:border-purple-700 mb-6">
          <h3 className="text-lg font-semibold mb-3 text-purple-900 dark:text-purple-300">
            💡 생성 모델의 목표
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li><strong>데이터 분포 학습</strong>: 실제 데이터의 확률 분포 p(x)를 모델링</li>
            <li><strong>새로운 샘플 생성</strong>: 학습한 분포에서 새로운 데이터 샘플링</li>
            <li><strong>고품질 출력</strong>: 실제 데이터와 구별하기 어려운 결과물 생성</li>
            <li><strong>다양성 확보</strong>: 다양한 종류의 데이터 생성 능력</li>
          </ul>
        </div>

        {/* 판별 모델 vs 생성 모델 */}
        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4 border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-300">판별 모델 (Discriminative)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              <strong>P(Y|X)</strong>: 입력 X가 주어졌을 때 레이블 Y를 예측
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 분류, 회귀 문제</li>
              <li>• CNN, ResNet, Transformer</li>
              <li>• 경계선(decision boundary) 학습</li>
            </ul>
          </div>

          <div className="bg-pink-50 dark:bg-pink-900/20 rounded-xl p-4 border border-pink-200 dark:border-pink-800">
            <h4 className="font-semibold mb-2 text-pink-900 dark:text-pink-300">생성 모델 (Generative)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              <strong>P(X)</strong> 또는 <strong>P(X,Y)</strong>: 데이터 분포 자체를 학습
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 새로운 데이터 생성</li>
              <li>• GAN, VAE, Diffusion Models</li>
              <li>• 데이터 분포 모델링</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 2. GAN 기본 구조 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          GAN의 기본 구조
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          GAN은 두 개의 신경망이 서로 경쟁하며 학습하는 구조입니다.
          위조지폐범(Generator)과 경찰(Discriminator)의 대결로 비유할 수 있습니다.
        </p>

        {/* GAN 구조 시각화 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Generator vs Discriminator</h3>
          <div className="overflow-x-auto">
            <svg viewBox="0 0 1000 600" className="w-full h-auto">
              {/* Random Noise (Z) */}
              <text x="100" y="100" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-semibold">
                Random Noise
              </text>
              <text x="100" y="118" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-sm">
                z ~ N(0, 1)
              </text>
              <circle cx="100" cy="150" r="30" className="fill-purple-200 dark:fill-purple-900 stroke-purple-500 dark:stroke-purple-400" strokeWidth="2" />
              <text x="100" y="157" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-semibold">
                z
              </text>

              {/* Generator */}
              <rect x="50" y="230" width="100" height="120" rx="12" className="fill-gradient-to-br from-purple-100 to-pink-100 dark:from-purple-900 dark:to-pink-900 stroke-purple-500 dark:stroke-purple-400" strokeWidth="3" />
              <text x="100" y="270" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-bold text-lg">
                Generator
              </text>
              <text x="100" y="290" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm">
                G(z)
              </text>
              <text x="100" y="310" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                위조지폐범
              </text>
              <text x="100" y="326" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                (가짜 이미지 생성)
              </text>

              {/* Arrow from Noise to Generator */}
              <line x1="100" y1="180" x2="100" y2="230" className="stroke-purple-500 dark:stroke-purple-400" strokeWidth="2" markerEnd="url(#arrow-purple)" />

              {/* Fake Image */}
              <rect x="50" y="400" width="100" height="80" rx="8" className="fill-pink-100 dark:fill-pink-900 stroke-pink-500 dark:stroke-pink-400" strokeWidth="2" />
              <text x="100" y="435" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-semibold">
                Fake Image
              </text>
              <text x="100" y="455" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-sm">
                G(z)
              </text>

              {/* Arrow from Generator to Fake Image */}
              <line x1="100" y1="350" x2="100" y2="400" className="stroke-pink-500 dark:stroke-pink-400" strokeWidth="2" markerEnd="url(#arrow-pink)" />

              {/* Real Data */}
              <text x="900" y="100" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-semibold">
                Real Data
              </text>
              <rect x="850" y="120" width="100" height="80" rx="8" className="fill-green-100 dark:fill-green-900 stroke-green-500 dark:stroke-green-400" strokeWidth="2" />
              <text x="900" y="155" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-semibold">
                Real Image
              </text>
              <text x="900" y="175" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-sm">
                x ~ p_data(x)
              </text>

              {/* Discriminator */}
              <rect x="400" y="280" width="200" height="140" rx="12" className="fill-gradient-to-br from-blue-100 to-cyan-100 dark:from-blue-900 dark:to-cyan-900 stroke-blue-500 dark:stroke-blue-400" strokeWidth="3" />
              <text x="500" y="330" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-bold text-lg">
                Discriminator
              </text>
              <text x="500" y="350" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm">
                D(x)
              </text>
              <text x="500" y="370" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                경찰
              </text>
              <text x="500" y="386" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                (진짜/가짜 판별)
              </text>
              <text x="500" y="405" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs font-semibold">
                출력: 0~1 확률
              </text>

              {/* Arrows to Discriminator */}
              {/* From Fake */}
              <line x1="150" y1="440" x2="400" y2="330" className="stroke-pink-500 dark:stroke-pink-400" strokeWidth="2" markerEnd="url(#arrow-pink)" strokeDasharray="5" />
              <text x="270" y="375" className="fill-pink-600 dark:fill-pink-400 text-sm font-semibold">
                Fake
              </text>

              {/* From Real */}
              <line x1="850" y1="160" x2="600" y2="310" className="stroke-green-500 dark:stroke-green-400" strokeWidth="2" markerEnd="url(#arrow-green)" />
              <text x="720" y="225" className="fill-green-600 dark:fill-green-400 text-sm font-semibold">
                Real
              </text>

              {/* Discriminator Outputs */}
              {/* Fake label (0) */}
              <circle cx="500" cy="480" r="35" className="fill-red-100 dark:fill-red-900 stroke-red-500 dark:stroke-red-400" strokeWidth="2" />
              <text x="500" y="485" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-bold text-xl">
                0
              </text>
              <text x="500" y="502" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                (Fake)
              </text>

              {/* Real label (1) */}
              <circle cx="700" cy="380" r="35" className="fill-green-100 dark:fill-green-900 stroke-green-500 dark:stroke-green-400" strokeWidth="2" />
              <text x="700" y="385" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-bold text-xl">
                1
              </text>
              <text x="700" y="402" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                (Real)
              </text>

              {/* Loss/Gradient arrows */}
              <path d="M 100 350 Q 100 500 500 480" fill="none" className="stroke-orange-500 dark:stroke-orange-400" strokeWidth="2" strokeDasharray="8" markerEnd="url(#arrow-orange)" />
              <text x="280" y="500" className="fill-orange-600 dark:fill-orange-400 text-xs font-semibold">
                Gradient (속이기 위한 신호)
              </text>

              {/* Arrow markers */}
              <defs>
                <marker id="arrow-purple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" className="fill-purple-500 dark:fill-purple-400" />
                </marker>
                <marker id="arrow-pink" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" className="fill-pink-500 dark:fill-pink-400" />
                </marker>
                <marker id="arrow-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" className="fill-green-500 dark:fill-green-400" />
                </marker>
                <marker id="arrow-orange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" className="fill-orange-500 dark:fill-orange-400" />
                </marker>
              </defs>

              {/* Labels */}
              <text x="500" y="580" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-sm italic">
                Discriminator는 진짜(1)와 가짜(0)를 구별하고, Generator는 Discriminator를 속이려 함
              </text>
            </svg>
          </div>
        </div>

        {/* Generator와 Discriminator 역할 */}
        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-4 border border-purple-200 dark:border-purple-800">
            <h4 className="font-semibold mb-2 text-purple-900 dark:text-purple-300">🎨 Generator (생성자)</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>입력</strong>: Random noise z (일반적으로 N(0,1))</li>
              <li>• <strong>출력</strong>: 가짜 이미지 G(z)</li>
              <li>• <strong>목표</strong>: Discriminator를 속일 만한 진짜같은 이미지 생성</li>
              <li>• <strong>학습</strong>: D(G(z))가 1에 가까워지도록</li>
            </ul>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4 border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-300">🔍 Discriminator (판별자)</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>입력</strong>: 이미지 x (진짜 또는 가짜)</li>
              <li>• <strong>출력</strong>: 확률 D(x) ∈ [0, 1]</li>
              <li>• <strong>목표</strong>: 진짜는 1, 가짜는 0으로 정확히 분류</li>
              <li>• <strong>학습</strong>: D(x_real) → 1, D(G(z)) → 0</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 3. GAN 손실 함수 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          GAN의 손실 함수 (MinMax Game)
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          GAN은 두 플레이어가 경쟁하는 게임 이론의 MinMax 문제로 정의됩니다.
        </p>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-xl p-6 mb-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">GAN Objective Function</h3>
            <button
              onClick={() => handleCopy('min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]', 'gan-loss')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title="수식 복사"
            >
              {copiedStates['gan-loss'] ? (
                <Check size={16} className="text-green-600" />
              ) : (
                <Copy size={16} className="text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <div className="font-mono text-sm bg-white dark:bg-gray-900 rounded-lg p-4">
            <div className="text-center mb-4">
              <span className="text-blue-600 dark:text-blue-400">min<sub>G</sub></span>{' '}
              <span className="text-red-600 dark:text-red-400">max<sub>D</sub></span>{' '}
              V(D, G) =
            </div>
            <div className="text-center mb-2">
              <span className="text-green-600 dark:text-green-400">E<sub>x~p<sub>data</sub>(x)</sub>[log D(x)]</span>
              {' + '}
              <span className="text-pink-600 dark:text-pink-400">E<sub>z~p<sub>z</sub>(z)</sub>[log(1 - D(G(z)))]</span>
            </div>
          </div>

          <div className="mt-4 space-y-2 text-sm text-gray-700 dark:text-gray-300">
            <div className="flex items-start gap-2">
              <span className="text-green-600 dark:text-green-400 font-semibold">•</span>
              <span><strong className="text-green-600 dark:text-green-400">첫 번째 항</strong>: Discriminator가 진짜 데이터를 1로 분류</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-pink-600 dark:text-pink-400 font-semibold">•</span>
              <span><strong className="text-pink-600 dark:text-pink-400">두 번째 항</strong>: Discriminator가 가짜 데이터를 0으로 분류</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-blue-600 dark:text-blue-400 font-semibold">•</span>
              <span><strong className="text-blue-600 dark:text-blue-400">Generator 목표</strong>: V를 최소화 (D를 속이기)</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-red-600 dark:text-red-400 font-semibold">•</span>
              <span><strong className="text-red-600 dark:text-red-400">Discriminator 목표</strong>: V를 최대화 (정확히 판별)</span>
            </div>
          </div>
        </div>

        {/* 학습 과정 */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">GAN 학습 과정 (Alternating Training)</h3>
          <div className="overflow-x-auto">
            <svg viewBox="0 0 1000 500" className="w-full h-auto">
              {/* Step 1: Train Discriminator */}
              <rect x="50" y="50" width="250" height="180" rx="12" className="fill-blue-50 dark:fill-blue-900/30 stroke-blue-500 dark:stroke-blue-400" strokeWidth="2" />
              <text x="175" y="85" textAnchor="middle" className="fill-blue-900 dark:fill-blue-300 font-bold text-lg">
                Step 1: Train D
              </text>
              <text x="175" y="110" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm">
                Discriminator 학습
              </text>

              {/* D training details */}
              <circle cx="100" cy="150" r="15" className="fill-green-200 dark:fill-green-900 stroke-green-500 dark:stroke-green-400" strokeWidth="2" />
              <text x="130" y="155" className="fill-gray-700 dark:fill-gray-300 text-xs">Real → D(x) = 1</text>

              <circle cx="100" cy="185" r="15" className="fill-red-200 dark:fill-red-900 stroke-red-500 dark:stroke-red-400" strokeWidth="2" />
              <text x="130" y="190" className="fill-gray-700 dark:fill-gray-300 text-xs">Fake → D(G(z)) = 0</text>

              <text x="175" y="215" textAnchor="middle" className="fill-blue-600 dark:fill-blue-400 text-xs font-semibold italic">
                G는 고정 (Freeze)
              </text>

              {/* Step 2: Train Generator */}
              <rect x="375" y="50" width="250" height="180" rx="12" className="fill-purple-50 dark:fill-purple-900/30 stroke-purple-500 dark:stroke-purple-400" strokeWidth="2" />
              <text x="500" y="85" textAnchor="middle" className="fill-purple-900 dark:fill-purple-300 font-bold text-lg">
                Step 2: Train G
              </text>
              <text x="500" y="110" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm">
                Generator 학습
              </text>

              {/* G training details */}
              <circle cx="425" cy="150" r="15" className="fill-purple-200 dark:fill-purple-900 stroke-purple-500 dark:stroke-purple-400" strokeWidth="2" />
              <text x="455" y="155" className="fill-gray-700 dark:fill-gray-300 text-xs">z → G(z)</text>

              <circle cx="425" cy="185" r="15" className="fill-orange-200 dark:fill-orange-900 stroke-orange-500 dark:stroke-orange-400" strokeWidth="2" />
              <text x="455" y="190" className="fill-gray-700 dark:fill-gray-300 text-xs">속이기: D(G(z)) → 1</text>

              <text x="500" y="215" textAnchor="middle" className="fill-purple-600 dark:fill-purple-400 text-xs font-semibold italic">
                D는 고정 (Freeze)
              </text>

              {/* Repeat arrow */}
              <path d="M 625 140 Q 750 140 750 280 Q 750 420 175 420 Q 50 420 50 280 Q 50 240 50 230" fill="none" className="stroke-gray-500 dark:stroke-gray-400" strokeWidth="2" strokeDasharray="8" markerEnd="url(#arrow-repeat)" />
              <text x="750" y="280" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-sm font-semibold">
                반복
              </text>

              {/* Convergence */}
              <rect x="700" y="50" width="250" height="180" rx="12" className="fill-green-50 dark:fill-green-900/30 stroke-green-500 dark:stroke-green-400" strokeWidth="2" />
              <text x="825" y="85" textAnchor="middle" className="fill-green-900 dark:fill-green-300 font-bold text-lg">
                수렴 (Convergence)
              </text>

              <text x="825" y="120" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm">
                Nash Equilibrium
              </text>

              <text x="825" y="150" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                D(G(z)) = 0.5
              </text>

              <text x="825" y="175" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                (진짜와 가짜를 구별 불가)
              </text>

              <text x="825" y="205" textAnchor="middle" className="fill-green-600 dark:fill-green-400 text-xs font-semibold">
                ✓ p_g = p_data
              </text>

              {/* Arrow to convergence */}
              <line x1="625" y1="100" x2="700" y2="100" className="stroke-green-500 dark:stroke-green-400" strokeWidth="2" markerEnd="url(#arrow-green2)" />

              {/* Iteration counter */}
              <text x="500" y="300" textAnchor="middle" className="fill-gray-900 dark:fill-gray-100 font-semibold">
                반복 횟수
              </text>

              {/* Timeline */}
              <line x1="100" y1="350" x2="900" y2="350" className="stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />
              {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                <g key={`iter-${i}`}>
                  <circle cx={100 + i * 100} cy="350" r="5" className="fill-gray-600 dark:fill-gray-400" />
                  <text x={100 + i * 100} y="375" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                    {i * 1000}
                  </text>
                </g>
              ))}

              {/* Performance curve */}
              <path d="M 100 450 Q 300 420 500 390 Q 700 370 900 360" fill="none" className="stroke-green-500 dark:stroke-green-400" strokeWidth="3" />
              <text x="500" y="480" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-sm italic">
                생성 품질 개선 →
              </text>

              <defs>
                <marker id="arrow-repeat" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" className="fill-gray-500 dark:fill-gray-400" />
                </marker>
                <marker id="arrow-green2" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" className="fill-green-500 dark:fill-green-400" />
                </marker>
              </defs>
            </svg>
          </div>
        </div>
      </section>

      {/* 4. GAN 변형 모델들 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          주요 GAN 변형 모델들
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          초기 GAN의 학습 불안정성을 개선하고 다양한 응용을 위해 수많은 변형 모델이 제안되었습니다.
        </p>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          {/* DCGAN */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-300 text-lg">
              🔵 DCGAN (2015)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>Deep Convolutional GAN</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• CNN 아키텍처 적용 (Fully connected 제거)</li>
              <li>• Batch Normalization 사용</li>
              <li>• LeakyReLU 활성화 함수</li>
              <li>• 안정적인 학습을 위한 가이드라인 제시</li>
            </ul>
          </div>

          {/* cGAN */}
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800">
            <h4 className="font-semibold mb-2 text-purple-900 dark:text-purple-300 text-lg">
              🟣 cGAN (2014)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>Conditional GAN</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 조건(label) 정보를 G와 D에 입력</li>
              <li>• 원하는 클래스의 데이터 생성 가능</li>
              <li>• Image-to-Image 변환 (Pix2Pix)</li>
              <li>• G(z, y), D(x, y) 형태</li>
            </ul>
          </div>

          {/* StyleGAN */}
          <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6 border border-orange-200 dark:border-orange-800">
            <h4 className="font-semibold mb-2 text-orange-900 dark:text-orange-300 text-lg">
              🟠 StyleGAN (2018)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>Style-Based Generator</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• Style transfer 개념 도입</li>
              <li>• 고해상도 이미지 생성 (1024×1024)</li>
              <li>• Adaptive Instance Normalization (AdaIN)</li>
              <li>• 이 사람은 존재하지 않습니다(thispersondoesnotexist)</li>
            </ul>
          </div>

          {/* CycleGAN */}
          <div className="bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 rounded-xl p-6 border border-teal-200 dark:border-teal-800">
            <h4 className="font-semibold mb-2 text-teal-900 dark:text-teal-300 text-lg">
              🔷 CycleGAN (2017)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>Unpaired Image-to-Image Translation</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 페어 데이터 없이 도메인 변환</li>
              <li>• Cycle Consistency Loss 사용</li>
              <li>• 말↔얼룩말, 여름↔겨울 변환</li>
              <li>• G: X→Y, F: Y→X 두 개의 Generator</li>
            </ul>
          </div>

          {/* WGAN */}
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 border border-green-200 dark:border-green-800">
            <h4 className="font-semibold mb-2 text-green-900 dark:text-green-300 text-lg">
              🟢 WGAN (2017)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>Wasserstein GAN</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• Wasserstein distance 사용</li>
              <li>• Mode collapse 문제 완화</li>
              <li>• 학습 안정성 크게 향상</li>
              <li>• Gradient penalty (WGAN-GP)</li>
            </ul>
          </div>

          {/* ProGAN */}
          <div className="bg-gradient-to-br from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 border border-violet-200 dark:border-violet-800">
            <h4 className="font-semibold mb-2 text-violet-900 dark:text-violet-300 text-lg">
              🟣 ProGAN (2017)
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              <strong>Progressive Growing of GANs</strong>
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 점진적으로 해상도 증가 (4×4 → 1024×1024)</li>
              <li>• 안정적인 고해상도 학습</li>
              <li>• Layer-by-layer 학습 전략</li>
              <li>• NVIDIA의 초고해상도 얼굴 생성</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 5. GAN의 응용 분야 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          GAN의 실전 응용
        </h2>
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          GAN은 이미지 생성을 넘어 다양한 분야에서 활용되고 있습니다.
        </p>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-pink-50 dark:bg-pink-900/20 rounded-xl p-4 border border-pink-200 dark:border-pink-800">
            <h4 className="font-semibold mb-2 text-pink-900 dark:text-pink-300">🎨 이미지 생성</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 사실적인 얼굴 생성</li>
              <li>• 예술 작품 생성</li>
              <li>• 저해상도 → 고해상도 (Super Resolution)</li>
            </ul>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-4 border border-blue-200 dark:border-blue-800">
            <h4 className="font-semibold mb-2 text-blue-900 dark:text-blue-300">🖼️ 이미지 변환</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 스타일 전이 (Style Transfer)</li>
              <li>• 흑백 → 컬러</li>
              <li>• 스케치 → 사진</li>
            </ul>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-4 border border-green-200 dark:border-green-800">
            <h4 className="font-semibold mb-2 text-green-900 dark:text-green-300">🔊 음성/비디오</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 음성 합성</li>
              <li>• 비디오 생성 및 예측</li>
              <li>• Deepfake 기술</li>
            </ul>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-4 border border-purple-200 dark:border-purple-800">
            <h4 className="font-semibold mb-2 text-purple-900 dark:text-purple-300">🧬 의료/과학</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 의료 영상 증강</li>
              <li>• 신약 개발 (분자 생성)</li>
              <li>• 단백질 구조 예측</li>
            </ul>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-4 border border-orange-200 dark:border-orange-800">
            <h4 className="font-semibold mb-2 text-orange-900 dark:text-orange-300">🎮 게임/VR</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 게임 텍스처 생성</li>
              <li>• NPC 캐릭터 생성</li>
              <li>• 가상 환경 구축</li>
            </ul>
          </div>

          <div className="bg-teal-50 dark:bg-teal-900/20 rounded-xl p-4 border border-teal-200 dark:border-teal-800">
            <h4 className="font-semibold mb-2 text-teal-900 dark:text-teal-300">🛡️ 보안/탐지</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• 이상 탐지 (Anomaly Detection)</li>
              <li>• Deepfake 탐지</li>
              <li>• 데이터 증강</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 6. GAN의 도전 과제 */}
      <section>
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          GAN의 도전 과제
        </h2>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-xl p-4 border border-red-200 dark:border-red-800">
            <h4 className="font-semibold mb-2 text-red-900 dark:text-red-300">⚠️ Mode Collapse</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              Generator가 다양성을 잃고 일부 샘플만 생성
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>원인</strong>: G가 D를 속이기 쉬운 샘플만 학습</li>
              <li>• <strong>해결</strong>: Minibatch discrimination, Unrolled GAN, WGAN</li>
            </ul>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-xl p-4 border border-orange-200 dark:border-orange-800">
            <h4 className="font-semibold mb-2 text-orange-900 dark:text-orange-300">⚠️ Training Instability</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              학습이 불안정하고 수렴하지 않음
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>원인</strong>: G와 D의 균형 맞추기 어려움</li>
              <li>• <strong>해결</strong>: Learning rate 조정, Spectral Normalization</li>
            </ul>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-4 border border-yellow-200 dark:border-yellow-800">
            <h4 className="font-semibold mb-2 text-yellow-900 dark:text-yellow-300">⚠️ Vanishing Gradient</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              D가 너무 강해지면 G의 gradient가 소실
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>원인</strong>: D(G(z)) → 0에서 log(1-D(G(z)))의 기울기 소실</li>
              <li>• <strong>해결</strong>: Non-saturating loss, WGAN</li>
            </ul>
          </div>

          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-4 border border-indigo-200 dark:border-indigo-800">
            <h4 className="font-semibold mb-2 text-indigo-900 dark:text-indigo-300">⚠️ Evaluation Metrics</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              생성 품질을 정량적으로 평가하기 어려움
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• <strong>지표</strong>: Inception Score (IS), Fréchet Inception Distance (FID)</li>
              <li>• <strong>한계</strong>: 주관적 품질과 항상 일치하지 않음</li>
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
                title: 'Generative Adversarial Networks',
                authors: 'Goodfellow, I., et al.',
                year: '2014',
                description: 'GAN을 최초로 제안한 혁명적 논문',
                link: 'https://arxiv.org/abs/1406.2661'
              },
              {
                title: 'Unsupervised Representation Learning with DCGANs',
                authors: 'Radford, A., et al.',
                year: '2015',
                description: 'CNN 기반 안정적인 GAN 학습 가이드라인',
                link: 'https://arxiv.org/abs/1511.06434'
              },
              {
                title: 'Conditional Generative Adversarial Nets',
                authors: 'Mirza, M., & Osindero, S.',
                year: '2014',
                description: '조건부 GAN - 원하는 클래스 생성',
                link: 'https://arxiv.org/abs/1411.1784'
              },
              {
                title: 'Wasserstein GAN',
                authors: 'Arjovsky, M., et al.',
                year: '2017',
                description: 'Wasserstein distance로 학습 안정성 향상',
                link: 'https://arxiv.org/abs/1701.07875'
              }
            ]
          },
          {
            title: '📘 고급 GAN 변형',
            icon: 'paper' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Progressive Growing of GANs',
                authors: 'Karras, T., et al.',
                year: '2017',
                description: '점진적 해상도 증가로 고해상도 생성',
                link: 'https://arxiv.org/abs/1710.10196'
              },
              {
                title: 'A Style-Based Generator Architecture (StyleGAN)',
                authors: 'Karras, T., et al.',
                year: '2018',
                description: '스타일 기반 초고해상도 얼굴 생성',
                link: 'https://arxiv.org/abs/1812.04948'
              },
              {
                title: 'Unpaired Image-to-Image Translation (CycleGAN)',
                authors: 'Zhu, J., et al.',
                year: '2017',
                description: '페어 데이터 없이 도메인 변환',
                link: 'https://arxiv.org/abs/1703.10593'
              },
              {
                title: 'Image-to-Image Translation with cGANs (Pix2Pix)',
                authors: 'Isola, P., et al.',
                year: '2016',
                description: '조건부 GAN을 활용한 이미지 변환',
                link: 'https://arxiv.org/abs/1611.07004'
              }
            ]
          },
          {
            title: '🛠️ 라이브러리 & 도구',
            icon: 'github' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'PyTorch GAN Zoo',
                authors: 'Facebook AI',
                year: '2023',
                description: '다양한 GAN 구현 모음',
                link: 'https://github.com/facebookresearch/pytorch_GAN_zoo'
              },
              {
                title: 'StyleGAN2-ADA',
                authors: 'NVIDIA',
                year: '2023',
                description: 'NVIDIA의 공식 StyleGAN2 구현',
                link: 'https://github.com/NVlabs/stylegan2-ada-pytorch'
              },
              {
                title: 'CycleGAN and Pix2Pix',
                authors: 'junyanz',
                year: '2023',
                description: 'Image-to-Image translation 도구',
                link: 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix'
              },
              {
                title: 'This Person Does Not Exist',
                authors: 'Phillip Wang',
                year: '2019',
                description: 'StyleGAN으로 생성한 가상 인물 (데모)',
                link: 'https://thispersondoesnotexist.com/'
              }
            ]
          },
          {
            title: '📚 학습 자료',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'GAN Lab: Understanding GANs',
                authors: 'MIT-IBM Watson AI Lab',
                year: '2018',
                description: '인터랙티브 GAN 시각화 도구',
                link: 'https://poloclub.github.io/ganlab/'
              },
              {
                title: 'The GAN Zoo',
                authors: 'Avinash Hindupur',
                year: '2023',
                description: '500+ GAN 변형 모델 목록',
                link: 'https://github.com/hindupuravinash/the-gan-zoo'
              },
              {
                title: 'How to Train a GAN? Tips and tricks',
                authors: 'Soumith Chintala',
                year: '2016',
                description: 'GAN 학습을 위한 실전 팁',
                link: 'https://github.com/soumith/ganhacks'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
