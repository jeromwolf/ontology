'use client'

import React from 'react'
import { MessageSquare, ImageIcon, Film, Eye, BookOpen, Sparkles, Target, TrendingUp } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter8() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-purple-50 dark:from-gray-900 dark:to-purple-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl">
              <Target className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent">
                멀티모달 응용
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                VQA, 이미지 캡셔닝, 비디오 이해
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-6 h-6 text-violet-600" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              멀티모달 AI 응용 분야
            </h2>
          </div>

          <div className="prose dark:prose-invert max-w-none">
            <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
              멀티모달 AI는 실세계의 다양한 문제를 해결하는 강력한 도구입니다.
              이미지와 텍스트를 동시에 이해하는 VQA(Visual Question Answering),
              이미지를 자연어로 설명하는 이미지 캡셔닝,
              비디오의 시공간 정보를 파악하는 비디오 이해 등
              인간의 인지 능력을 모방하는 혁신적 응용 분야가 탄생했습니다.
            </p>

            <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6 border border-violet-200 dark:border-violet-800">
              <p className="text-violet-900 dark:text-violet-100 font-semibold mb-2">
                💡 왜 멀티모달 응용이 중요한가?
              </p>
              <p className="text-violet-800 dark:text-violet-200">
                단일 모달리티만으로는 해결할 수 없는 복잡한 태스크를 멀티모달 AI가 가능하게 합니다.
                시각 장애인을 위한 이미지 설명, 의료 영상 진단, 자율주행 인지 시스템 등
                <strong>실세계 문제 해결</strong>에 직접적으로 기여합니다.
              </p>
            </div>
          </div>
        </section>

        {/* VQA (Visual Question Answering) */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <MessageSquare className="w-6 h-6 text-violet-600" />
            Visual Question Answering (VQA)
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg mb-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              VQA란 무엇인가?
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              VQA는 이미지와 자연어 질문을 입력받아 정확한 답변을 생성하는 태스크입니다.
              이미지 이해(객체, 장면, 관계), 언어 이해(질문 의도), 추론(답변 도출)을 모두 필요로 합니다.
            </p>

            <div className="grid md:grid-cols-3 gap-4 mb-6">
              {[
                {
                  type: '객체 인식',
                  question: '"이 사진에 몇 명이 있나요?"',
                  answer: '"3명"',
                  difficulty: '쉬움',
                  color: 'blue'
                },
                {
                  type: '공간 관계',
                  question: '"고양이가 어디에 있나요?"',
                  answer: '"소파 위에"',
                  difficulty: '중간',
                  color: 'purple'
                },
                {
                  type: '추상적 추론',
                  question: '"이 사람의 기분은 어때 보이나요?"',
                  answer: '"행복해 보입니다"',
                  difficulty: '어려움',
                  color: 'red'
                }
              ].map((example, idx) => (
                <div key={idx} className={`bg-${example.color}-50 dark:bg-${example.color}-900/20 rounded-xl p-4 border border-${example.color}-200 dark:border-${example.color}-800`}>
                  <div className={`inline-block px-2 py-1 rounded bg-${example.color}-500 text-white text-xs font-bold mb-2`}>
                    {example.type}
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-1">
                    <span className="font-semibold">Q:</span> {example.question}
                  </p>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                    <span className="font-semibold">A:</span> {example.answer}
                  </p>
                  <p className={`text-xs text-${example.color}-700 dark:text-${example.color}-300`}>
                    난이도: {example.difficulty}
                  </p>
                </div>
              ))}
            </div>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h4 className="font-bold text-gray-900 dark:text-white mb-4">
                VQA 모델 아키텍처
              </h4>
              <div className="space-y-4">
                {[
                  {
                    step: '1',
                    title: 'Image Encoding',
                    desc: 'ResNet/ViT로 이미지를 특징 맵 추출 (예: [196, 768] - 14×14 패치)',
                    model: 'Vision Encoder'
                  },
                  {
                    step: '2',
                    title: 'Question Encoding',
                    desc: 'BERT/GPT로 질문을 임베딩으로 변환 (예: [seq_len, 768])',
                    model: 'Language Encoder'
                  },
                  {
                    step: '3',
                    title: 'Cross-Modal Fusion',
                    desc: '질문 임베딩이 이미지 특징에 Cross-Attention하여 관련 영역 집중',
                    model: 'Attention Module'
                  },
                  {
                    step: '4',
                    title: 'Answer Generation',
                    desc: '통합 표현을 MLP 또는 Decoder로 처리하여 답변 생성',
                    model: 'Classification Head / Decoder'
                  }
                ].map((stage, idx) => (
                  <div key={idx} className="flex gap-4 items-start">
                    <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white font-bold">
                      {stage.step}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <p className="font-semibold text-gray-900 dark:text-white">{stage.title}</p>
                        <span className="text-xs text-gray-500 dark:text-gray-400">({stage.model})</span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{stage.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {[
              {
                model: 'BLIP (Salesforce, 2022)',
                description: 'Bootstrapping Language-Image Pre-training',
                features: [
                  'VQA, 이미지 캡셔닝, 검색을 단일 모델로',
                  'CapFilt로 노이즈 데이터 자동 정제',
                  'VQAv2에서 78.3% 정확도'
                ],
                innovation: 'Unified Vision-Language Understanding',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                model: 'GPT-4V (OpenAI, 2023)',
                description: 'Multimodal Large Language Model',
                features: [
                  '복잡한 추론과 대화형 VQA',
                  '차트, 다이어그램, 수식 이해',
                  'Few-shot 학습으로 새 태스크 적응'
                ],
                innovation: 'Human-level Visual Reasoning',
                color: 'from-purple-500 to-pink-500'
              }
            ].map((model, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg"
              >
                <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${model.color} text-white mb-4`}>
                  <Eye className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-1">
                  {model.model}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  {model.description}
                </p>
                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 mb-3">
                  <p className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">주요 특징</p>
                  <ul className="space-y-1">
                    {model.features.map((feature, i) => (
                      <li key={i} className="text-sm text-gray-600 dark:text-gray-400 flex gap-2">
                        <span className="text-violet-600">•</span>
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                  <p className="text-sm text-violet-900 dark:text-violet-100">
                    <span className="font-semibold">혁신:</span> {model.innovation}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Image Captioning */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <ImageIcon className="w-6 h-6 text-violet-600" />
            이미지 캡셔닝 (Image Captioning)
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              이미지 캡셔닝은 이미지를 입력받아 자연어로 설명을 생성하는 태스크입니다.
              VQA와 달리 질문 없이 이미지의 전체적인 내용을 자동으로 설명합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                이미지 캡셔닝 접근법 발전
              </h3>

              <div className="space-y-4">
                {[
                  {
                    generation: '1세대 (2015)',
                    model: 'Show and Tell (Google)',
                    approach: 'CNN (Inception) + LSTM Decoder',
                    description: 'CNN으로 이미지 특징 추출 후 LSTM이 단어를 순차적으로 생성',
                    limitation: '단순한 문장, 반복적 표현',
                    color: 'blue'
                  },
                  {
                    generation: '2세대 (2017)',
                    model: 'Show, Attend and Tell',
                    approach: 'CNN + Attention + LSTM',
                    description: '매 단어 생성 시 이미지의 관련 영역에 Attention하여 디테일 향상',
                    limitation: '여전히 RNN 기반으로 긴 문장에 약함',
                    color: 'purple'
                  },
                  {
                    generation: '3세대 (2020+)',
                    model: 'Transformer Captioning (COCO, Oscar)',
                    approach: 'ViT + GPT-style Decoder',
                    description: 'Transformer로 병렬 처리, Self-Attention으로 문맥 일관성 향상',
                    limitation: '대규모 데이터와 계산 자원 필요',
                    color: 'green'
                  },
                  {
                    generation: '4세대 (2022+)',
                    model: 'BLIP, GIT (Generative Image-to-text)',
                    approach: 'Unified VLP + Auto-regressive Decoder',
                    description: '캡셔닝, VQA, 검색을 단일 모델로 통합, 노이즈 데이터 자동 정제',
                    limitation: '환각(Hallucination) 문제 여전히 존재',
                    color: 'orange'
                  }
                ].map((gen, idx) => (
                  <div key={idx} className={`bg-${gen.color}-50 dark:bg-${gen.color}-900/20 rounded-lg p-5 border-l-4 border-${gen.color}-500`}>
                    <div className="flex items-center gap-3 mb-2">
                      <div className={`w-8 h-8 rounded-full bg-${gen.color}-500 text-white flex items-center justify-center font-bold text-sm`}>
                        {idx + 1}
                      </div>
                      <div>
                        <p className="font-bold text-gray-900 dark:text-white">{gen.generation}</p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">{gen.model}</p>
                      </div>
                    </div>
                    <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                      <span className="font-semibold">접근법:</span> {gen.approach}
                    </p>
                    <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                      {gen.description}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      <span className="font-semibold">한계:</span> {gen.limitation}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-violet-50 dark:bg-violet-900/20 rounded-xl p-6 border border-violet-200 dark:border-violet-800">
              <h4 className="font-bold text-violet-900 dark:text-violet-100 mb-3">
                💻 간단한 캡셔닝 코드 (BLIP)
              </h4>
              <div className="bg-gray-900 rounded-lg p-6 overflow-x-auto">
                <pre className="text-sm text-gray-100">
{`from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 모델과 프로세서 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 이미지 로드
image = Image.open("photo.jpg")

# 캡션 생성
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs, max_length=50)
caption = processor.decode(out[0], skip_special_tokens=True)

print(caption)  # "A dog sitting on a bench in a park"`}
                </pre>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-green-50 dark:bg-green-900/20 rounded-xl p-5 border border-green-200 dark:border-green-800">
                <h4 className="font-bold text-green-900 dark:text-green-100 mb-3">
                  ✅ 응용 분야
                </h4>
                <ul className="space-y-2 text-green-800 dark:text-green-200">
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>시각 장애인을 위한 이미지 설명 (Accessibility)</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>소셜 미디어 자동 태깅 (Instagram, Pinterest)</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>의료 영상 리포트 자동 생성</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>로봇의 환경 이해 및 설명</span>
                  </li>
                </ul>
              </div>

              <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-5 border border-amber-200 dark:border-amber-800">
                <h4 className="font-bold text-amber-900 dark:text-amber-100 mb-3">
                  ⚠️ 주요 도전 과제
                </h4>
                <ul className="space-y-2 text-amber-800 dark:text-amber-200">
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>환각(Hallucination): 존재하지 않는 객체 생성</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>편향(Bias): 학습 데이터의 고정관념 반영</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>디테일 부족: "사람이 있다" vs "30대 남성이 정장 입고"</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span>평가 어려움: BLEU/CIDEr 점수가 인간 평가와 불일치</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Video Understanding */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Film className="w-6 h-6 text-violet-600" />
            비디오 이해 (Video Understanding)
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg mb-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              비디오 이해의 특수성
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-6">
              비디오는 이미지와 달리 시간적 차원(Temporal Dimension)을 가집니다.
              단순히 프레임을 독립적으로 분석하는 것을 넘어,
              프레임 간의 <strong>시간적 관계</strong>와 <strong>동작(Motion)</strong>을 이해해야 합니다.
            </p>

            <div className="grid md:grid-cols-3 gap-4 mb-6">
              {[
                {
                  task: 'Action Recognition',
                  description: '비디오에서 행동 분류',
                  example: '"달리기", "점프", "춤추기"',
                  model: 'I3D, TimeSformer',
                  color: 'blue'
                },
                {
                  task: 'Video Captioning',
                  description: '비디오 전체를 설명',
                  example: '"남자가 공을 차서 골을 넣고 있다"',
                  model: 'VideoBERT, VIOLET',
                  color: 'purple'
                },
                {
                  task: 'Temporal Grounding',
                  description: '질문에 해당하는 시간 구간 찾기',
                  example: 'Q: "언제 고양이가 점프했나요?" → 3.2s-5.1s',
                  model: 'Moment-DETR',
                  color: 'green'
                }
              ].map((task, idx) => (
                <div key={idx} className={`bg-${task.color}-50 dark:bg-${task.color}-900/20 rounded-xl p-4 border border-${task.color}-200 dark:border-${task.color}-800`}>
                  <div className={`inline-block px-3 py-1 rounded-full bg-${task.color}-500 text-white text-xs font-bold mb-3`}>
                    {task.task}
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                    {task.description}
                  </p>
                  <div className="bg-white dark:bg-gray-800 rounded p-2 mb-2">
                    <p className="text-xs text-gray-600 dark:text-gray-400 italic">
                      {task.example}
                    </p>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    <span className="font-semibold">모델:</span> {task.model}
                  </p>
                </div>
              ))}
            </div>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h4 className="font-bold text-gray-900 dark:text-white mb-4">
                비디오 처리 아키텍처 발전
              </h4>
              <div className="space-y-4">
                {[
                  {
                    approach: '2D CNN + Temporal Pooling',
                    description: '각 프레임을 독립적으로 처리 후 평균/최대 풀링',
                    limitation: '시간적 관계 무시',
                    example: 'Two-Stream Networks'
                  },
                  {
                    approach: '3D CNN',
                    description: '공간과 시간을 동시에 컨볼루션 (3D 커널)',
                    limitation: '계산량 폭증 (O(T×H×W))',
                    example: 'C3D, I3D'
                  },
                  {
                    approach: 'RNN/LSTM',
                    description: '프레임 시퀀스를 순차적으로 처리',
                    limitation: '긴 비디오에서 vanishing gradient',
                    example: 'LRCN'
                  },
                  {
                    approach: 'Video Transformer',
                    description: 'Self-Attention으로 프레임 간 관계 학습',
                    limitation: 'O(T²) 복잡도로 메모리 부담',
                    example: 'TimeSformer, ViViT, VideoMAE'
                  }
                ].map((arch, idx) => (
                  <div key={idx} className="flex gap-3 items-start">
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white font-bold">
                      {idx + 1}
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold text-gray-900 dark:text-white mb-1">{arch.approach}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">{arch.description}</p>
                      <p className="text-xs text-red-600 dark:text-red-400 mb-1">
                        <span className="font-semibold">한계:</span> {arch.limitation}
                      </p>
                      <p className="text-xs text-violet-700 dark:text-violet-300">
                        <span className="font-semibold">예시:</span> {arch.example}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="grid gap-6">
            {[
              {
                application: 'YouTube 콘텐츠 추천',
                description: '비디오 내용을 이해하여 관련 콘텐츠 추천',
                tech: 'Video Embeddings + Collaborative Filtering',
                impact: '시청 시간 20% 증가, 사용자 만족도 향상',
                metric: '일일 10억+ 비디오 처리',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                application: '스포츠 하이라이트 자동 생성',
                description: '축구 경기에서 골, 파울 등 주요 장면 자동 추출',
                tech: 'Action Detection + Temporal Segmentation',
                impact: '편집 시간 90% 단축, 실시간 하이라이트',
                metric: 'ESPN, NBA 등 주요 스포츠 리그 채택',
                color: 'from-purple-500 to-pink-500'
              },
              {
                application: '보안 감시 시스템',
                description: '이상 행동 자동 탐지 (침입, 폭력, 사고)',
                tech: 'Anomaly Detection + Real-time Alerting',
                impact: '보안 요원 업무 효율 2배, 오탐률 50% 감소',
                metric: '공항, 지하철, 쇼핑몰 등 대규모 배치',
                color: 'from-green-500 to-emerald-500'
              },
              {
                application: '의료: 수술 비디오 분석',
                description: '수술 단계 자동 인식 및 위험 상황 경고',
                tech: 'Surgical Workflow Recognition + Phase Detection',
                impact: '수술 시간 15% 단축, 합병증 20% 감소',
                metric: 'Johns Hopkins 등 주요 병원 임상 시험',
                color: 'from-orange-500 to-red-500'
              }
            ].map((app, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className="flex items-start gap-4">
                  <div className={`flex-shrink-0 w-12 h-12 rounded-lg bg-gradient-to-br ${app.color} flex items-center justify-center text-white font-bold text-xl`}>
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                      {app.application}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 mb-3">
                      {app.description}
                    </p>
                    <div className="space-y-2">
                      <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3">
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          <span className="font-semibold">기술:</span> {app.tech}
                        </p>
                      </div>
                      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                        <p className="text-sm text-blue-900 dark:text-blue-100">
                          <span className="font-semibold">영향:</span> {app.impact}
                        </p>
                      </div>
                      <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                        <p className="text-sm text-violet-900 dark:text-violet-100">
                          <span className="font-semibold">규모:</span> {app.metric}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Future Trends */}
        <section className="mb-12 bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-2xl p-8 border border-cyan-200 dark:border-cyan-800">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <TrendingUp className="w-6 h-6 text-cyan-600" />
            미래 트렌드 및 연구 방향
          </h2>

          <div className="space-y-4">
            {[
              {
                trend: 'Embodied AI',
                description: '로봇이 물리 세계에서 멀티모달 입력(카메라, 센서, 음성)을 통합하여 행동 결정',
                impact: '자율주행, 가정용 로봇, 창고 자동화'
              },
              {
                trend: 'Unified Multi-Task Models',
                description: '단일 모델이 VQA, 캡셔닝, 검색, 생성을 모두 수행 (예: Unified-IO, Flamingo)',
                impact: '범용 멀티모달 AI 에이전트 탄생'
              },
              {
                trend: 'Zero-Shot Generalization',
                description: '학습 데이터 없이도 새로운 모달리티 조합 처리 (예: 3D → Text, Audio → 3D)',
                impact: '데이터 부족 영역 돌파구'
              },
              {
                trend: 'Multimodal Chain-of-Thought',
                description: 'LLM의 사고 연쇄를 멀티모달로 확장하여 복잡한 추론 가능',
                impact: '의료 진단, 과학 연구, 법률 분석'
              },
              {
                trend: 'Efficient Multimodal Models',
                description: '모바일/엣지 디바이스에서 실행 가능한 경량 멀티모달 모델',
                impact: 'AR 글래스, 스마트워치, IoT 디바이스'
              }
            ].map((item, idx) => (
              <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow">
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 text-white flex items-center justify-center font-bold">
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <h4 className="font-bold text-gray-900 dark:text-white mb-2">
                      {item.trend}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      {item.description}
                    </p>
                    <div className="bg-cyan-50 dark:bg-cyan-900/20 rounded p-2">
                      <p className="text-xs text-cyan-900 dark:text-cyan-100">
                        <span className="font-semibold">영향:</span> {item.impact}
                      </p>
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
              'VQA: 이미지+질문 → 답변 생성 (BLIP, GPT-4V)',
              'VQA 아키텍처: Image Encoding + Question Encoding + Cross-Modal Fusion',
              '이미지 캡셔닝: CNN+LSTM → Attention → Transformer → Unified VLP',
              '이미지 캡셔닝 응용: 시각 장애인 지원, 의료 리포트 자동 생성',
              '비디오 이해: Action Recognition, Video Captioning, Temporal Grounding',
              '비디오 처리: 2D CNN → 3D CNN → RNN → Video Transformer',
              '실전 응용: YouTube 추천, 스포츠 하이라이트, 보안 감시, 수술 분석',
              '미래 트렌드: Embodied AI, Unified Models, Zero-Shot, Chain-of-Thought'
            ].map((item, idx) => (
              <li key={idx} className="flex items-start gap-3">
                <span className="text-violet-200 mt-1">✓</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>

          <div className="mt-8 pt-6 border-t border-violet-400">
            <p className="text-violet-100 mb-4">
              <span className="font-semibold">축하합니다!</span> 멀티모달 AI 시스템의 전체 여정을 완주하셨습니다.
            </p>
            <p className="text-violet-100">
              이제 멀티모달 AI의 기본 개념부터 최신 모델, 실시간 배포, 실전 응용까지
              체계적으로 이해하게 되었습니다. 이 지식을 바탕으로 혁신적인 멀티모달 AI 프로젝트를 시작해보세요!
            </p>
          </div>
        </section>

        {/* References */}
        <References
          sections={[
            {
              title: '📚 공식 문서 & 플랫폼',
              icon: 'web' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'OpenAI Vision API',
                  authors: 'OpenAI',
                  year: '2024',
                  description: 'GPT-4V API 공식 문서 - VQA와 이미지 캡셔닝 실전 구현',
                  link: 'https://platform.openai.com/docs/guides/vision'
                },
                {
                  title: 'Google Gemini Multimodal',
                  authors: 'Google DeepMind',
                  year: '2024',
                  description: 'Gemini 멀티모달 기능 가이드 - 텍스트, 이미지, 비디오 통합 처리',
                  link: 'https://ai.google.dev/gemini-api/docs/vision'
                },
                {
                  title: 'Anthropic Claude Vision',
                  authors: 'Anthropic',
                  year: '2024',
                  description: 'Claude 3의 Vision 기능 - 차트, 다이어그램, 이미지 분석',
                  link: 'https://docs.anthropic.com/claude/docs/vision'
                },
                {
                  title: 'Meta AI Multimodal',
                  authors: 'Meta',
                  year: '2024',
                  description: 'ImageBind, SAM, CLIP 통합 플랫폼',
                  link: 'https://ai.meta.com/blog/imagebind-six-modalities-binding-ai/'
                },
                {
                  title: 'Hugging Face Multimodal',
                  authors: 'Hugging Face',
                  year: '2024',
                  description: 'BLIP, CLIP, ViT 등 멀티모달 모델 통합 라이브러리',
                  link: 'https://huggingface.co/docs/transformers/tasks/image_text_to_text'
                }
              ]
            },
            {
              title: '📖 핵심 논문',
              icon: 'research' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'CLIP: Connecting Text and Images',
                  authors: 'Radford et al.',
                  year: '2021',
                  description: 'Contrastive Learning으로 텍스트-이미지 정렬 - 멀티모달 AI의 기초',
                  link: 'https://arxiv.org/abs/2103.00020'
                },
                {
                  title: 'Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2)',
                  authors: 'Ramesh et al.',
                  year: '2022',
                  description: 'Diffusion Model 기반 Text-to-Image 생성',
                  link: 'https://arxiv.org/abs/2204.06125'
                },
                {
                  title: 'Flamingo: a Visual Language Model for Few-Shot Learning',
                  authors: 'Alayrac et al.',
                  year: '2022',
                  description: 'Few-shot 멀티모달 학습의 획기적 진전',
                  link: 'https://arxiv.org/abs/2204.14198'
                },
                {
                  title: 'GPT-4 Technical Report',
                  authors: 'OpenAI',
                  year: '2023',
                  description: 'GPT-4V의 멀티모달 능력 - 차트, 수식, 복잡한 추론',
                  link: 'https://arxiv.org/abs/2303.08774'
                }
              ]
            },
            {
              title: '🛠️ 실전 도구',
              icon: 'tools' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'Hugging Face Transformers (Multimodal)',
                  authors: 'Hugging Face',
                  year: '2024',
                  description: 'BLIP, CLIP, ViT 통합 라이브러리 - VQA와 캡셔닝 빠른 구현',
                  link: 'https://huggingface.co/docs/transformers/index'
                },
                {
                  title: 'LangChain Multimodal',
                  authors: 'LangChain',
                  year: '2024',
                  description: 'GPT-4V, Gemini Vision 통합 - 멀티모달 체인 구축',
                  link: 'https://python.langchain.com/docs/integrations/chat/openai#multimodal-inputs'
                },
                {
                  title: 'LlamaIndex Multimodal',
                  authors: 'LlamaIndex',
                  year: '2024',
                  description: '멀티모달 RAG - 이미지/비디오 검색 및 질의응답',
                  link: 'https://docs.llamaindex.ai/en/stable/examples/multi_modal/'
                },
                {
                  title: 'OpenCLIP',
                  authors: 'laion-ai',
                  year: '2024',
                  description: 'CLIP 오픈소스 구현 - 다양한 크기와 학습 데이터셋',
                  link: 'https://github.com/mlfoundations/open_clip'
                },
                {
                  title: 'Video-LLaVA',
                  authors: 'PKU-YuanGroup',
                  year: '2024',
                  description: '비디오 이해 오픈소스 모델 - Video QA, Captioning',
                  link: 'https://github.com/PKU-YuanGroup/Video-LLaVA'
                }
              ]
            }
          ]}
        />
      </div>
    </div>
  )
}
