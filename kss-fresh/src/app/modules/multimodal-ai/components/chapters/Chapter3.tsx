'use client'

import React from 'react'
import { Layers, Code, Zap, GitBranch, BookOpen, Brain, Workflow, Settings } from 'lucide-react'

export default function Chapter3() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-purple-50 dark:from-gray-900 dark:to-purple-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl">
              <Workflow className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent">
                멀티모달 아키텍처
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                크로스모달 어텐션과 퓨전 기법
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-6 h-6 text-violet-600" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              모달리티 퓨전(Fusion)이란?
            </h2>
          </div>

          <div className="prose dark:prose-invert max-w-none">
            <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
              멀티모달 AI의 핵심은 서로 다른 모달리티(텍스트, 이미지, 오디오 등)를 어떻게 통합(Fusion)하느냐입니다.
              퓨전 전략에 따라 모델의 성능, 학습 효율성, 적용 가능한 태스크가 크게 달라집니다.
              이 챕터에서는 Early Fusion, Late Fusion, Cross-Attention을 중심으로 다양한 퓨전 기법을 살펴봅니다.
            </p>

            <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6 border border-violet-200 dark:border-violet-800">
              <p className="text-violet-900 dark:text-violet-100 font-semibold mb-2">
                💡 핵심 질문
              </p>
              <p className="text-violet-800 dark:text-violet-200">
                언제 모달리티를 통합할 것인가? 초기 단계(Early)에서 합칠 것인가,
                각 모달리티를 독립적으로 처리한 후 마지막(Late)에 합칠 것인가,
                아니면 중간 단계(Hybrid)에서 상호작용하게 할 것인가?
              </p>
            </div>
          </div>
        </section>

        {/* 퓨전 전략 분류 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <GitBranch className="w-6 h-6 text-violet-600" />
            3가지 주요 퓨전 전략
          </h2>

          <div className="space-y-6">
            {[
              {
                type: 'Early Fusion',
                description: '입력 단계에서 모달리티를 통합',
                method: '이미지와 텍스트를 토큰화하여 단일 시퀀스로 연결 후 Transformer에 입력',
                pros: ['모달리티 간 복잡한 상호작용 학습 가능', '단순하고 직관적인 구조'],
                cons: ['큰 계산 비용', '각 모달리티의 특성을 살리기 어려움', '모달리티 길이 불균형 문제'],
                examples: 'VisualBERT, VL-BERT',
                color: 'from-blue-500 to-cyan-500',
                icon: <Layers className="w-6 h-6" />
              },
              {
                type: 'Late Fusion',
                description: '각 모달리티를 독립적으로 처리 후 마지막에 통합',
                method: '이미지와 텍스트를 별도의 인코더로 처리하여 임베딩을 얻은 후 연결 또는 투표',
                pros: ['각 모달리티에 최적화된 인코더 사용 가능', '계산 효율적', '모듈화 설계 용이'],
                cons: ['모달리티 간 깊은 상호작용 부족', '간단한 태스크에만 효과적'],
                examples: 'CLIP (코사인 유사도), 앙상블 모델',
                color: 'from-purple-500 to-pink-500',
                icon: <Brain className="w-6 h-6" />
              },
              {
                type: 'Hybrid Fusion (Cross-Attention)',
                description: '중간 레이어에서 모달리티 간 상호작용',
                method: '각 모달리티를 독립 인코딩한 후 Cross-Attention으로 정보 교환',
                pros: ['Early와 Late의 장점 결합', '유연한 상호작용', '효율성과 표현력 균형'],
                cons: ['설계 복잡도 증가', '하이퍼파라미터 튜닝 필요'],
                examples: 'Flamingo, BLIP, LLaVA',
                color: 'from-green-500 to-emerald-500',
                icon: <Workflow className="w-6 h-6" />
              }
            ].map((strategy, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg"
              >
                <div className="flex items-start gap-4 mb-4">
                  <div className={`flex-shrink-0 p-3 rounded-lg bg-gradient-to-br ${strategy.color} text-white`}>
                    {strategy.icon}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                      {strategy.type}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 mb-3">
                      {strategy.description}
                    </p>
                  </div>
                </div>

                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-4 mb-4">
                  <p className="font-semibold text-gray-900 dark:text-white mb-2">방법</p>
                  <p className="text-sm text-gray-700 dark:text-gray-300">{strategy.method}</p>
                </div>

                <div className="grid md:grid-cols-2 gap-4 mb-4">
                  <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border-l-4 border-green-500">
                    <p className="font-semibold text-green-900 dark:text-green-100 mb-2">장점</p>
                    <ul className="space-y-1">
                      {strategy.pros.map((pro, i) => (
                        <li key={i} className="text-sm text-green-800 dark:text-green-200 flex gap-2">
                          <span>✓</span>
                          <span>{pro}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border-l-4 border-red-500">
                    <p className="font-semibold text-red-900 dark:text-red-100 mb-2">단점</p>
                    <ul className="space-y-1">
                      {strategy.cons.map((con, i) => (
                        <li key={i} className="text-sm text-red-800 dark:text-red-200 flex gap-2">
                          <span>✗</span>
                          <span>{con}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                  <p className="text-sm text-violet-900 dark:text-violet-100">
                    <span className="font-semibold">대표 모델:</span> {strategy.examples}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Early Fusion 상세 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🔵 Early Fusion 상세 분석
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              Early Fusion은 입력 단계에서 모든 모달리티를 단일 표현으로 통합합니다.
              VisualBERT와 VL-BERT가 대표적인 예입니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                VisualBERT 아키텍처
              </h3>
              <div className="space-y-4">
                {[
                  {
                    step: '1',
                    title: '이미지 처리',
                    desc: 'Faster R-CNN으로 이미지에서 객체 영역(Region) 추출 → 각 영역을 벡터로 인코딩'
                  },
                  {
                    step: '2',
                    title: '텍스트 처리',
                    desc: 'WordPiece 토크나이저로 텍스트를 서브워드 토큰으로 분할'
                  },
                  {
                    step: '3',
                    title: '통합 시퀀스 생성',
                    desc: '[CLS] text_token1 ... text_tokenN [SEP] image_region1 ... image_regionM'
                  },
                  {
                    step: '4',
                    title: 'Transformer 처리',
                    desc: '통합 시퀀스를 BERT와 동일한 Transformer에 입력, Self-Attention으로 모든 토큰 간 상호작용'
                  },
                  {
                    step: '5',
                    title: '출력',
                    desc: '[CLS] 토큰으로 분류, 각 토큰으로 VQA/캡셔닝 수행'
                  }
                ].map((stage, idx) => (
                  <div key={idx} className="flex gap-4 items-start">
                    <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white font-bold">
                      {stage.step}
                    </div>
                    <div>
                      <p className="font-semibold text-gray-900 dark:text-white">{stage.title}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{stage.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
              <h4 className="font-bold text-blue-900 dark:text-blue-100 mb-3">
                Early Fusion의 핵심 특징
              </h4>
              <ul className="space-y-2 text-blue-800 dark:text-blue-200">
                <li className="flex gap-2">
                  <span>•</span>
                  <span><strong>완전한 Self-Attention:</strong> 모든 텍스트-이미지 토큰 쌍이 서로 어텐션 가능</span>
                </li>
                <li className="flex gap-2">
                  <span>•</span>
                  <span><strong>복잡한 추론:</strong> "빨간 모자를 쓴 사람의 왼쪽에 있는 개" 같은 복잡한 관계 이해</span>
                </li>
                <li className="flex gap-2">
                  <span>•</span>
                  <span><strong>계산 비용:</strong> O(N²) 복잡도로 시퀀스 길이에 민감 (N = 텍스트+이미지 토큰 수)</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Late Fusion 상세 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🟣 Late Fusion 상세 분석
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              Late Fusion은 각 모달리티를 독립적으로 처리한 후 최종 단계에서 통합합니다.
              CLIP이 대표적인 예로, 계산 효율성이 뛰어납니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                CLIP의 Late Fusion 전략
              </h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-4">
                    <p className="font-semibold text-blue-900 dark:text-blue-100 mb-2">Image Encoder</p>
                    <p className="text-sm text-blue-800 dark:text-blue-200">
                      Vision Transformer 또는 ResNet으로 이미지를 512차원 벡터로 인코딩
                    </p>
                    <p className="text-xs text-blue-700 dark:text-blue-300 mt-2">
                      독립적 처리 → 텍스트와 상호작용 없음
                    </p>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg p-4">
                    <p className="font-semibold text-purple-900 dark:text-purple-100 mb-2">Text Encoder</p>
                    <p className="text-sm text-purple-800 dark:text-purple-200">
                      Transformer로 텍스트를 512차원 벡터로 인코딩
                    </p>
                    <p className="text-xs text-purple-700 dark:text-purple-300 mt-2">
                      독립적 처리 → 이미지와 상호작용 없음
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-4 bg-violet-50 dark:bg-violet-900/20 rounded-lg p-4 border-l-4 border-violet-500">
                <p className="font-semibold text-violet-900 dark:text-violet-100 mb-2">Fusion 단계</p>
                <p className="text-sm text-violet-800 dark:text-violet-200">
                  두 벡터의 <strong>코사인 유사도</strong>를 계산하여 매칭 여부 판단.
                  별도의 융합 레이어 없이 내적(dot product)만으로 통합.
                </p>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800">
              <h4 className="font-bold text-purple-900 dark:text-purple-100 mb-3">
                Late Fusion의 장점
              </h4>
              <div className="space-y-3">
                {[
                  {
                    title: '효율성',
                    desc: '각 모달리티를 병렬로 처리 가능, 한 번 인코딩한 벡터 재사용'
                  },
                  {
                    title: '확장성',
                    desc: '새로운 모달리티 추가 시 기존 인코더 수정 불필요'
                  },
                  {
                    title: '해석성',
                    desc: '각 모달리티의 기여도를 독립적으로 분석 가능'
                  }
                ].map((advantage, idx) => (
                  <div key={idx} className="flex gap-3">
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-purple-500 text-white flex items-center justify-center text-sm font-bold">
                      {idx + 1}
                    </div>
                    <div>
                      <p className="font-semibold text-purple-900 dark:text-purple-100">{advantage.title}</p>
                      <p className="text-sm text-purple-800 dark:text-purple-200">{advantage.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Cross-Attention 상세 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🟢 Cross-Attention (Hybrid Fusion) 상세 분석
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              Cross-Attention은 Early와 Late Fusion의 장점을 결합한 방식입니다.
              각 모달리티를 독립적으로 인코딩한 후, 중간 레이어에서 서로의 정보를 참조하며 상호작용합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                Cross-Attention 수식
              </h3>
              <div className="bg-gray-900 rounded-lg p-6 overflow-x-auto mb-4">
                <pre className="text-sm text-gray-100">
{`# Query는 텍스트, Key/Value는 이미지
Q = W_q @ text_features   # [batch, text_len, d_model]
K = W_k @ image_features  # [batch, image_len, d_model]
V = W_v @ image_features  # [batch, image_len, d_model]

# Attention Score 계산
scores = Q @ K.T / sqrt(d_model)  # [batch, text_len, image_len]
attention_weights = softmax(scores, dim=-1)

# 이미지 정보를 텍스트에 융합
output = attention_weights @ V  # [batch, text_len, d_model]`}
                </pre>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                텍스트 토큰이 Query가 되어 이미지 특징(Key/Value)에 어텐션합니다.
                각 텍스트 토큰은 관련 있는 이미지 영역에 집중하여 시각 정보를 가져옵니다.
              </p>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6 border border-green-200 dark:border-green-800">
              <h3 className="font-bold text-green-900 dark:text-green-100 mb-4">
                Flamingo의 Cross-Attention 적용
              </h3>
              <div className="space-y-4">
                {[
                  {
                    component: 'Vision Encoder (Frozen)',
                    desc: 'NFNet으로 이미지/비디오를 고차원 특징으로 변환'
                  },
                  {
                    component: 'Perceiver Resampler',
                    desc: '가변 길이 시각 특징을 고정 개수 토큰(예: 64개)으로 압축'
                  },
                  {
                    component: 'Language Model Layers',
                    desc: 'Chinchilla 70B LLM의 각 레이어에 Cross-Attention 삽입'
                  },
                  {
                    component: 'Cross-Attention Block',
                    desc: 'LLM 텍스트 토큰(Query)이 시각 토큰(Key/Value)에 어텐션하여 시각 정보 융합'
                  }
                ].map((component, idx) => (
                  <div key={idx} className="flex gap-3 items-start">
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-green-500 text-white flex items-center justify-center font-bold">
                      {idx + 1}
                    </div>
                    <div>
                      <p className="font-semibold text-green-900 dark:text-green-100">{component.component}</p>
                      <p className="text-sm text-green-800 dark:text-green-200">{component.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                <h4 className="font-bold text-blue-900 dark:text-blue-100 mb-2">
                  🎯 언제 유리한가?
                </h4>
                <ul className="space-y-2 text-sm text-blue-800 dark:text-blue-200">
                  <li>• 복잡한 시각-언어 추론 (VQA, 이미지 캡셔닝)</li>
                  <li>• 사전 학습된 LLM 활용 (Frozen LLM + Adapter)</li>
                  <li>• Few-shot 학습 (Flamingo)</li>
                </ul>
              </div>
              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
                <h4 className="font-bold text-orange-900 dark:text-orange-100 mb-2">
                  ⚠️ 주의할 점
                </h4>
                <ul className="space-y-2 text-sm text-orange-800 dark:text-orange-200">
                  <li>• Cross-Attention 레이어 위치 선택 중요</li>
                  <li>• 너무 많이 삽입하면 계산 비용 증가</li>
                  <li>• 너무 적게 삽입하면 상호작용 부족</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Transformer 기반 최신 아키텍처 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🚀 최신 Transformer 기반 멀티모달 아키텍처
          </h2>

          <div className="grid gap-6">
            {[
              {
                name: 'LLaVA (Large Language and Vision Assistant)',
                architecture: 'CLIP Vision Encoder + Projection + LLaMA LLM',
                fusion: 'Visual tokens을 텍스트 임베딩 공간으로 projection 후 LLM에 입력',
                innovation: '간단한 projection layer만으로 강력한 비전-언어 대화 가능',
                performance: '높은 VQA 정확도, GPT-4V 수준의 대화 능력',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                name: 'BLIP-2 (Bootstrapping Language-Image Pre-training)',
                architecture: 'Frozen Image Encoder + Q-Former + Frozen LLM',
                fusion: 'Q-Former가 learnable query로 이미지에서 정보 추출 후 LLM에 전달',
                innovation: 'Q-Former로 이미지-텍스트 정렬과 텍스트 생성을 동시에 학습',
                performance: 'COCO 캡셔닝 SOTA, 54B 파라미터 대비 효율적',
                color: 'from-purple-500 to-pink-500'
              },
              {
                name: 'GPT-4V (GPT-4 Vision)',
                architecture: 'Multimodal Transformer (내부 아키텍처 비공개)',
                fusion: '이미지와 텍스트를 통합 토큰 시퀀스로 처리 (추정)',
                innovation: '복잡한 시각 추론, 차트 분석, OCR, 의료 영상 해석',
                performance: 'MMMU 벤치마크 56.8%, 인간 수준 근접',
                color: 'from-green-500 to-emerald-500'
              }
            ].map((model, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className="flex items-start gap-4">
                  <div className={`flex-shrink-0 w-12 h-12 rounded-lg bg-gradient-to-br ${model.color} flex items-center justify-center text-white font-bold text-xl`}>
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                      {model.name}
                    </h3>
                    <div className="space-y-3">
                      <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">아키텍처</p>
                        <p className="text-sm text-gray-700 dark:text-gray-300">{model.architecture}</p>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">퓨전 방식</p>
                        <p className="text-sm text-gray-700 dark:text-gray-300">{model.fusion}</p>
                      </div>
                      <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                        <p className="text-sm text-violet-900 dark:text-violet-100">
                          <span className="font-semibold">혁신:</span> {model.innovation}
                        </p>
                      </div>
                      <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3 border-l-4 border-green-500">
                        <p className="text-sm text-green-900 dark:text-green-100">
                          <span className="font-semibold">성능:</span> {model.performance}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 학습 목표 요약 */}
        <section className="bg-gradient-to-br from-violet-600 to-purple-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">📚 이 챕터에서 배운 내용</h2>
          <ul className="space-y-3">
            {[
              'Early Fusion, Late Fusion, Hybrid Fusion의 개념과 장단점',
              'VisualBERT의 Early Fusion 구조와 완전한 Self-Attention',
              'CLIP의 Late Fusion 전략과 코사인 유사도 기반 통합',
              'Cross-Attention의 수식과 Flamingo의 적용 사례',
              'LLaVA, BLIP-2, GPT-4V 등 최신 Transformer 기반 아키텍처',
              '퓨전 전략 선택 기준 (계산 비용, 태스크 복잡도, 효율성)'
            ].map((item, idx) => (
              <li key={idx} className="flex items-start gap-3">
                <span className="text-violet-200 mt-1">✓</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>

          <div className="mt-8 pt-6 border-t border-violet-400">
            <p className="text-violet-100">
              <span className="font-semibold">다음 챕터:</span> 오디오-비주얼 AI를 살펴봅니다.
              Whisper, Wav2Vec2와 같은 음성 모델과 비디오 통합 방법을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
