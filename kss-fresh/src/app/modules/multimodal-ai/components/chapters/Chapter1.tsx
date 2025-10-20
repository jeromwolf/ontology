'use client'

import React from 'react'
import { BookOpen, Brain, Eye, Mic, Image as ImageIcon, Video, Layers, Zap } from 'lucide-react'

export default function Chapter1() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-purple-50 dark:from-gray-900 dark:to-purple-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent">
                멀티모달 AI 개요
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                다중 모달리티 AI의 개념과 중요성
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-6 h-6 text-violet-600" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              멀티모달 AI란 무엇인가?
            </h2>
          </div>

          <div className="prose dark:prose-invert max-w-none">
            <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
              멀티모달 AI(Multimodal AI)는 여러 종류의 데이터 모달리티(텍스트, 이미지, 음성, 비디오 등)를
              동시에 처리하고 이해할 수 있는 인공지능 시스템입니다. 인간이 다양한 감각을 통해 세상을
              이해하는 것처럼, 멀티모달 AI는 여러 데이터 소스를 통합하여 더 풍부하고 정확한 이해를
              달성합니다.
            </p>

            <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6 border border-violet-200 dark:border-violet-800">
              <p className="text-violet-900 dark:text-violet-100 font-semibold mb-2">
                💡 핵심 개념
              </p>
              <p className="text-violet-800 dark:text-violet-200">
                멀티모달 AI는 단순히 여러 데이터를 동시에 처리하는 것을 넘어,
                각 모달리티 간의 상호 관계와 보완적 정보를 활용하여
                단일 모달리티만으로는 불가능한 복잡한 추론과 이해를 가능하게 합니다.
              </p>
            </div>
          </div>
        </section>

        {/* 모달리티 종류 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Layers className="w-6 h-6 text-violet-600" />
            주요 모달리티(Modality) 유형
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {[
              {
                icon: <BookOpen className="w-8 h-8" />,
                title: '텍스트 (Text)',
                description: '자연어 처리, 문서 이해, 대화형 인터페이스',
                examples: 'GPT-4, BERT, 챗봇, 문서 요약',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                icon: <ImageIcon className="w-8 h-8" />,
                title: '이미지 (Image)',
                description: '시각적 정보 인식, 객체 탐지, 장면 이해',
                examples: 'ResNet, YOLO, 이미지 분류, 세그멘테이션',
                color: 'from-green-500 to-emerald-500'
              },
              {
                icon: <Mic className="w-8 h-8" />,
                title: '오디오 (Audio)',
                description: '음성 인식, 음악 분석, 소리 분류',
                examples: 'Whisper, Wav2Vec2, 음성-텍스트 변환',
                color: 'from-orange-500 to-red-500'
              },
              {
                icon: <Video className="w-8 h-8" />,
                title: '비디오 (Video)',
                description: '시공간 정보, 동작 인식, 이벤트 탐지',
                examples: 'TimeSformer, VideoMAE, 행동 인식',
                color: 'from-purple-500 to-pink-500'
              }
            ].map((modality, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${modality.color} text-white mb-4`}>
                  {modality.icon}
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {modality.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  {modality.description}
                </p>
                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">예시:</p>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    {modality.examples}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 왜 멀티모달 AI가 중요한가? */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Zap className="w-6 h-6 text-violet-600" />
            왜 멀티모달 AI가 중요한가?
          </h2>

          <div className="space-y-6">
            {[
              {
                number: '01',
                title: '인간 수준의 이해',
                description: '인간은 시각, 청각, 언어를 통합하여 세상을 이해합니다. 멀티모달 AI는 이러한 인간의 인지 방식을 모방하여 더 자연스럽고 효과적인 AI 시스템을 구축합니다.',
                example: 'GPT-4V는 이미지를 보고 자연어로 설명하며, 사용자와 대화할 수 있습니다.'
              },
              {
                number: '02',
                title: '컨텍스트 풍부화',
                description: '단일 모달리티로는 놓칠 수 있는 정보를 다른 모달리티가 보완합니다. 텍스트만으로 애매한 의미도 이미지와 결합하면 명확해집니다.',
                example: '"이거 어때?"라는 질문은 사진과 함께 제공되어야 정확한 답변이 가능합니다.'
              },
              {
                number: '03',
                title: '새로운 응용 분야',
                description: '이미지 캡셔닝, 비디오 QA, 크로스모달 검색 등 멀티모달 AI만이 가능하게 하는 혁신적 응용 분야가 탄생합니다.',
                example: 'DALL-E는 텍스트 설명으로 이미지를 생성하고, CLIP은 이미지-텍스트 간 검색을 가능하게 합니다.'
              },
              {
                number: '04',
                title: '견고성(Robustness) 향상',
                description: '하나의 모달리티에 노이즈가 있어도 다른 모달리티가 이를 보완하여 전체 시스템의 신뢰성이 향상됩니다.',
                example: '어두운 환경에서 시각 정보가 불확실해도 음성 정보로 상황을 파악할 수 있습니다.'
              }
            ].map((reason, idx) => (
              <div key={idx} className="flex gap-6 items-start">
                <div className="flex-shrink-0">
                  <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
                    <span className="text-2xl font-bold text-white">{reason.number}</span>
                  </div>
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                    {reason.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-3">
                    {reason.description}
                  </p>
                  <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-4 border-l-4 border-violet-500">
                    <p className="text-sm text-violet-900 dark:text-violet-100">
                      <span className="font-semibold">예시:</span> {reason.example}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 대표적인 멀티모달 AI 모델 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🌟 대표적인 멀티모달 AI 모델
          </h2>

          <div className="grid gap-6">
            {[
              {
                name: 'CLIP (OpenAI, 2021)',
                description: 'Contrastive Language-Image Pre-training',
                capability: '이미지와 텍스트를 동일한 임베딩 공간에 매핑하여 제로샷 분류와 크로스모달 검색 가능',
                impact: '4억 개 이미지-텍스트 쌍으로 학습, ImageNet 제로샷 정확도 76.2%',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                name: 'DALL-E 3 (OpenAI, 2023)',
                description: 'Text-to-Image Generation',
                capability: '자연어 설명을 고품질 이미지로 변환, 복잡한 개념과 스타일 이해',
                impact: '창작, 디자인, 마케팅 분야 혁신, 1024×1024 해상도 지원',
                color: 'from-purple-500 to-pink-500'
              },
              {
                name: 'GPT-4V (OpenAI, 2023)',
                description: 'Multimodal Large Language Model',
                capability: '텍스트와 이미지를 동시에 입력받아 복잡한 시각적 추론과 대화 가능',
                impact: '시각적 질문 응답, 차트 분석, 의료 영상 해석 등 광범위한 응용',
                color: 'from-green-500 to-emerald-500'
              },
              {
                name: 'Flamingo (DeepMind, 2022)',
                description: 'Few-shot Visual Language Model',
                capability: '단 몇 개의 예시만으로 새로운 시각-언어 태스크 수행',
                impact: '80B 파라미터, 16개 벤치마크에서 SOTA 달성',
                color: 'from-orange-500 to-red-500'
              }
            ].map((model, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-all"
              >
                <div className="flex items-start gap-4">
                  <div className={`flex-shrink-0 w-12 h-12 rounded-lg bg-gradient-to-br ${model.color} flex items-center justify-center text-white font-bold text-xl`}>
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-1">
                      {model.name}
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">
                      {model.description}
                    </p>
                    <p className="text-gray-700 dark:text-gray-300 mb-3">
                      <span className="font-semibold">핵심 능력:</span> {model.capability}
                    </p>
                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        <span className="font-semibold">영향:</span> {model.impact}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 멀티모달 AI의 과제 */}
        <section className="mb-12 bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-2xl p-8 border border-amber-200 dark:border-amber-800">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            ⚠️ 멀티모달 AI의 주요 과제
          </h2>

          <div className="space-y-4">
            {[
              {
                challenge: '모달리티 정렬(Alignment)',
                description: '서로 다른 모달리티를 공통의 표현 공간에 매핑하는 것은 기술적으로 어렵습니다. 예를 들어, "귀여운 강아지"라는 텍스트와 실제 강아지 이미지를 어떻게 동일한 임베딩으로 표현할 것인가?'
              },
              {
                challenge: '데이터 불균형',
                description: '텍스트 데이터는 풍부하지만 고품질의 이미지-텍스트 쌍, 비디오-캡션 데이터는 상대적으로 부족합니다. LAION-5B 같은 대규모 데이터셋도 노이즈가 많습니다.'
              },
              {
                challenge: '계산 비용',
                description: '멀티모달 모델은 각 모달리티를 처리하는 인코더와 퓨전 메커니즘이 필요해 단일 모달 모델보다 훨씬 많은 계산 자원을 요구합니다. GPT-4V는 수천 개의 GPU로 학습되었습니다.'
              },
              {
                challenge: '편향과 환각(Hallucination)',
                description: '학습 데이터의 편향이 그대로 반영되며, 존재하지 않는 정보를 생성하는 환각 문제가 심각합니다. DALL-E가 "투명한 코끼리"를 그릴 때 물리 법칙을 무시하는 경우가 있습니다.'
              }
            ].map((item, idx) => (
              <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow">
                <h4 className="font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                  <span className="text-amber-600 dark:text-amber-400">▸</span>
                  {item.challenge}
                </h4>
                <p className="text-gray-600 dark:text-gray-400">
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* 학습 목표 요약 */}
        <section className="bg-gradient-to-br from-violet-600 to-purple-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">📚 이 챕터에서 배운 내용</h2>
          <ul className="space-y-3">
            {[
              '멀티모달 AI의 정의와 핵심 개념',
              '주요 모달리티 유형 (텍스트, 이미지, 오디오, 비디오)',
              '멀티모달 AI의 중요성 (인간 수준 이해, 컨텍스트 풍부화, 견고성 향상)',
              '대표 모델 (CLIP, DALL-E 3, GPT-4V, Flamingo)의 특징과 영향',
              '현재 직면한 기술적 과제 (정렬, 데이터 불균형, 계산 비용, 환각)'
            ].map((item, idx) => (
              <li key={idx} className="flex items-start gap-3">
                <span className="text-violet-200 mt-1">✓</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>

          <div className="mt-8 pt-6 border-t border-violet-400">
            <p className="text-violet-100">
              <span className="font-semibold">다음 챕터:</span> Vision-Language 모델의 아키텍처와
              CLIP, DALL-E의 내부 동작 원리를 심층 분석합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
