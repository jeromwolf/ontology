'use client';

import Link from 'next/link';
import { Code, FlaskConical } from 'lucide-react';

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-indigo-800 dark:text-indigo-200 mb-4 flex items-center gap-2">
          <Code className="w-6 h-6" />
          Transformer 아키텍처 완전 분석
        </h2>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-6">
          <p className="text-lg text-gray-700 dark:text-gray-300 mb-4">
            Transformer는 "Attention Is All You Need" 논문에서 소개된 혁신적인 아키텍처로, 
            현재 모든 LLM의 기반이 되고 있습니다.
          </p>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Transformer 전체 구조</h3>
        
        {/* Transformer Architecture Diagram */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border mb-8">
          <img 
            src="/images/llm/transformer-architecture.png" 
            alt="Transformer Architecture" 
            className="w-full max-w-3xl mx-auto rounded-lg shadow-lg"
          />
          <p className="text-sm text-gray-600 dark:text-gray-400 text-center mt-4">
            Transformer 아키텍처 (Attention Is All You Need, 2017)
          </p>
        </div>
        
        {/* Transformer 3D 시뮬레이터 링크 */}
        <div className="mb-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-semibold text-purple-900 dark:text-purple-200 mb-1">🎮 Transformer 3D 시뮬레이터</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Transformer 아키텍처의 구조를 3D로 시각화하고 탐구해보세요
              </p>
            </div>
            <Link 
              href="/modules/llm/simulators/transformer-architecture"
              className="inline-flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              <FlaskConical className="w-4 h-4" />
              시뮬레이터 실행
            </Link>
          </div>
        </div>
        
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 mt-8">핵심 구성 요소</h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
            <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-3">🔵 Encoder (왼쪽)</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 입력 문장을 이해하고 인코딩</li>
              <li>• Self-Attention으로 문맥 파악</li>
              <li>• 각 토큰이 다른 모든 토큰을 볼 수 있음</li>
              <li>• 6개 레이어로 구성 (N×)</li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-3">🟢 Decoder (오른쪽)</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 출력 문장을 생성</li>
              <li>• Masked Self-Attention 사용</li>
              <li>• 미래 토큰은 볼 수 없음 (자기회귀적)</li>
              <li>• Encoder 출력을 참조 (Cross-Attention)</li>
            </ul>
          </div>
        </div>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Self-Attention</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                각 토큰이 다른 모든 토큰과의 관계를 동시에 계산
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Multi-Head Attention</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                여러 개의 attention head로 다양한 관계 패턴 포착
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Feed Forward Network</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                각 위치에서 독립적으로 적용되는 완전연결층
              </p>
            </div>
          </div>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Layer Normalization</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                각 층의 출력을 안정화하여 깊은 네트워크 학습 가능
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Residual Connection</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                그래디언트 소실 문제 해결과 학습 안정성 향상
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
              <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Positional Encoding</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                순서 정보를 모델에 제공하는 위치 인코딩
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Attention 메커니즘 수식</h3>
        
        {/* Attention Visualizer 시뮬레이터 링크 */}
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-semibold text-blue-900 dark:text-blue-200 mb-1">🎮 Attention 메커니즘 시각화</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Self-Attention이 단어 간의 관계를 어떻게 학습하는지 시각적으로 탐구해보세요
              </p>
            </div>
            <Link 
              href="/modules/llm/simulators/attention-visualizer"
              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <FlaskConical className="w-4 h-4" />
              시뮬레이터 실행
            </Link>
          </div>
        </div>
        
        <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-lg">
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Self-Attention 계산</h4>
              <div className="bg-white dark:bg-gray-900 p-4 rounded border font-mono text-sm">
                Attention(Q, K, V) = softmax(QK^T / √d_k)V
              </div>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Multi-Head Attention</h4>
              <div className="bg-white dark:bg-gray-900 p-4 rounded border font-mono text-sm">
                MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}