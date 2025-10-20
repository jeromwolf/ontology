'use client'

import React from 'react'
import { Layers, Brain, Link, Zap, Code, Image as ImageIcon, BookOpen, Sparkles } from 'lucide-react'

export default function Chapter2() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-purple-50 dark:from-gray-900 dark:to-purple-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl">
              <Layers className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent">
                Vision-Language 모델
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                CLIP, DALL-E, Flamingo 아키텍처 분석
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-6 h-6 text-violet-600" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Vision-Language 모델이란?
            </h2>
          </div>

          <div className="prose dark:prose-invert max-w-none">
            <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
              Vision-Language 모델은 시각(Vision)과 언어(Language)를 통합적으로 이해하는 AI 시스템입니다.
              이미지의 시각적 정보와 텍스트의 의미론적 정보를 공통의 표현 공간으로 매핑하여,
              이미지 캡셔닝, 시각적 질문 응답(VQA), 텍스트-이미지 생성 등 다양한 태스크를 수행합니다.
            </p>

            <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6 border border-violet-200 dark:border-violet-800">
              <p className="text-violet-900 dark:text-violet-100 font-semibold mb-2">
                💡 핵심 아이디어
              </p>
              <p className="text-violet-800 dark:text-violet-200">
                Vision-Language 모델의 핵심은 <strong>모달리티 간 정렬(Alignment)</strong>입니다.
                "귀여운 강아지"라는 텍스트와 실제 강아지 사진을 같은 벡터 공간의 가까운 위치에
                배치함으로써, 두 모달리티 간의 의미적 연결을 학습합니다.
              </p>
            </div>
          </div>
        </section>

        {/* CLIP 아키텍처 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Link className="w-6 h-6 text-violet-600" />
            CLIP: Contrastive Language-Image Pre-training
          </h2>

          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg mb-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              🏗️ CLIP 아키텍처 구조
            </h3>

            <div className="space-y-6">
              <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
                <p className="text-gray-700 dark:text-gray-300 mb-4">
                  CLIP은 <strong>Dual Encoder</strong> 구조를 사용합니다:
                </p>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border-l-4 border-blue-500">
                    <h4 className="font-bold text-blue-900 dark:text-blue-100 mb-2">
                      Image Encoder
                    </h4>
                    <p className="text-sm text-blue-800 dark:text-blue-200">
                      Vision Transformer (ViT) 또는 ResNet을 사용하여 이미지를 512차원 벡터로 인코딩
                    </p>
                  </div>
                  <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border-l-4 border-purple-500">
                    <h4 className="font-bold text-purple-900 dark:text-purple-100 mb-2">
                      Text Encoder
                    </h4>
                    <p className="text-sm text-purple-800 dark:text-purple-200">
                      Transformer를 사용하여 텍스트를 512차원 벡터로 인코딩
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 border border-violet-200 dark:border-violet-800">
                <h4 className="font-bold text-violet-900 dark:text-violet-100 mb-3">
                  Contrastive Learning 학습 방식
                </h4>
                <p className="text-violet-800 dark:text-violet-200 mb-4">
                  CLIP은 4억 개의 이미지-텍스트 쌍을 사용하여 다음과 같이 학습됩니다:
                </p>
                <ol className="space-y-3 text-violet-800 dark:text-violet-200">
                  <li className="flex gap-2">
                    <span className="font-bold">1.</span>
                    <span>배치 내 N개의 이미지-텍스트 쌍을 인코딩 (예: N=32,768)</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-bold">2.</span>
                    <span>N × N 코사인 유사도 행렬 계산</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-bold">3.</span>
                    <span>대각선 요소(매칭 쌍)는 높은 유사도, 나머지는 낮은 유사도로 학습</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-bold">4.</span>
                    <span>Symmetric Cross-Entropy Loss로 최적화</span>
                  </li>
                </ol>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h4 className="font-bold text-gray-900 dark:text-white mb-4">
              💻 CLIP 유사도 계산 코드
            </h4>
            <div className="bg-gray-900 rounded-lg p-6 overflow-x-auto">
              <pre className="text-sm text-gray-100">
{`import torch
import torch.nn.functional as F

# Image와 Text 인코더 출력 (배치 크기 N)
image_features = image_encoder(images)  # [N, 512]
text_features = text_encoder(texts)      # [N, 512]

# L2 정규화
image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)

# 코사인 유사도 행렬 계산 (logit scale 학습 파라미터)
logit_scale = model.logit_scale.exp()
logits_per_image = logit_scale * image_features @ text_features.T  # [N, N]
logits_per_text = logits_per_image.T

# Contrastive Loss
labels = torch.arange(N).to(device)  # 정답은 대각선 (i -> i)
loss_i = F.cross_entropy(logits_per_image, labels)
loss_t = F.cross_entropy(logits_per_text, labels)
loss = (loss_i + loss_t) / 2`}
              </pre>
            </div>
          </div>
        </section>

        {/* CLIP의 강력한 능력 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🎯 CLIP의 강력한 제로샷 능력
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {[
              {
                title: '제로샷 분류',
                description: '학습 시 보지 못한 카테고리도 텍스트 프롬프트로 분류 가능',
                example: '"a photo of a [class]" 형식으로 클래스 텍스트 임베딩을 생성하고, 이미지와 유사도를 계산하여 분류',
                metric: 'ImageNet 제로샷 정확도 76.2% (ResNet-50 감독학습 76.3%와 동등)',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                title: '크로스모달 검색',
                description: '텍스트로 이미지를 검색하거나, 이미지로 관련 텍스트를 찾을 수 있음',
                example: '"sunset over the ocean"이라는 텍스트로 관련 이미지들을 유사도 순으로 검색',
                metric: 'Flickr30K에서 이미지-텍스트 검색 Recall@1 88.0%',
                color: 'from-green-500 to-emerald-500'
              },
              {
                title: '도메인 일반화',
                description: '다양한 도메인과 스타일의 이미지에 대해 견고하게 작동',
                example: '스케치, 만화, 사진, 그림 등 다양한 스타일의 이미지를 동일한 텍스트와 매칭',
                metric: '27개 분류 벤치마크에서 평균적으로 우수한 성능',
                color: 'from-purple-500 to-pink-500'
              },
              {
                title: '프롬프트 엔지니어링',
                description: '텍스트 프롬프트를 정교하게 설계하여 성능 향상',
                example: '"a photo of a {class}" 대신 "a photo of a {class}, a type of pet" 같은 맥락 추가',
                metric: '프롬프트 앙상블로 정확도 3.5% 향상',
                color: 'from-orange-500 to-red-500'
              }
            ].map((capability, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${capability.color} text-white mb-4`}>
                  <Sparkles className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {capability.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-3">
                  {capability.description}
                </p>
                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 mb-3">
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    <span className="font-semibold">예시:</span> {capability.example}
                  </p>
                </div>
                <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                  <p className="text-sm text-violet-900 dark:text-violet-100">
                    <span className="font-semibold">성능:</span> {capability.metric}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* DALL-E 아키텍처 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <ImageIcon className="w-6 h-6 text-violet-600" />
            DALL-E: Text-to-Image Generation
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              DALL-E는 텍스트 설명으로부터 이미지를 생성하는 생성 모델입니다.
              DALL-E 1은 Transformer 기반, DALL-E 2와 3는 Diffusion Model 기반으로 발전했습니다.
            </p>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
                <h3 className="text-lg font-bold text-blue-900 dark:text-blue-100 mb-3">
                  DALL-E 1 (2021)
                </h3>
                <ul className="space-y-2 text-blue-800 dark:text-blue-200">
                  <li className="flex gap-2">
                    <span>•</span>
                    <span><strong>아키텍처:</strong> 12B 파라미터 Transformer (GPT-3 변형)</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span><strong>토크나이징:</strong> VQ-VAE로 이미지를 8192개 토큰으로 변환</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span><strong>학습:</strong> 2억 5천만 개 이미지-텍스트 쌍</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span><strong>해상도:</strong> 256×256 픽셀</span>
                  </li>
                </ul>
              </div>

              <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800">
                <h3 className="text-lg font-bold text-purple-900 dark:text-purple-100 mb-3">
                  DALL-E 2 & 3 (2022-2023)
                </h3>
                <ul className="space-y-2 text-purple-800 dark:text-purple-200">
                  <li className="flex gap-2">
                    <span>•</span>
                    <span><strong>아키텍처:</strong> CLIP + Diffusion Model (unCLIP)</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span><strong>Prior:</strong> 텍스트에서 CLIP 이미지 임베딩 생성</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span><strong>Decoder:</strong> Diffusion model로 이미지 생성</span>
                  </li>
                  <li className="flex gap-2">
                    <span>•</span>
                    <span><strong>해상도:</strong> 1024×1024 픽셀 (DALL-E 3)</span>
                  </li>
                </ul>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h4 className="font-bold text-gray-900 dark:text-white mb-4">
                DALL-E 2/3 생성 파이프라인
              </h4>
              <div className="space-y-3">
                {[
                  { step: '1', title: 'Text Encoding', desc: 'CLIP Text Encoder로 텍스트를 임베딩으로 변환' },
                  { step: '2', title: 'Prior Network', desc: '텍스트 임베딩에서 CLIP 이미지 임베딩 예측 (Diffusion Prior)' },
                  { step: '3', title: 'Diffusion Decoder', desc: '이미지 임베딩을 조건으로 노이즈에서 이미지 생성' },
                  { step: '4', title: 'Upsampling', desc: '저해상도 이미지를 고해상도로 업샘플링' }
                ].map((stage, idx) => (
                  <div key={idx} className="flex gap-4 items-center">
                    <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white font-bold">
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
          </div>
        </section>

        {/* Flamingo 아키텍처 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Brain className="w-6 h-6 text-violet-600" />
            Flamingo: Few-shot Visual Language Model
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              Flamingo는 DeepMind가 2022년 발표한 Few-shot 학습이 가능한 비전-언어 모델입니다.
              단 몇 개의 예시만으로 새로운 태스크를 수행할 수 있는 강력한 In-Context Learning 능력을 갖췄습니다.
            </p>

            <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6 border border-orange-200 dark:border-orange-800">
              <h3 className="text-lg font-bold text-orange-900 dark:text-orange-100 mb-4">
                🔥 Flamingo의 핵심 혁신
              </h3>
              <div className="space-y-4">
                {[
                  {
                    title: 'Perceiver Resampler',
                    desc: '가변 길이 이미지/비디오 특징을 고정 개수의 토큰으로 압축 (예: 2048개 특징 → 64개 토큰)'
                  },
                  {
                    title: 'Cross-Attention Layers',
                    desc: '사전 학습된 LLM에 크로스 어텐션 레이어를 삽입하여 시각 정보와 텍스트를 융합'
                  },
                  {
                    title: 'Interleaved Image-Text',
                    desc: '이미지와 텍스트가 섞인 시퀀스를 처리 (예: [이미지1] 질문1 답변1 [이미지2] 질문2 ?)'
                  },
                  {
                    title: 'Frozen LLM + Vision Adapter',
                    desc: '대규모 LLM(Chinchilla 70B)은 고정하고 Vision Adapter만 학습하여 효율성 극대화'
                  }
                ].map((innovation, idx) => (
                  <div key={idx} className="flex gap-3">
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-orange-500 text-white flex items-center justify-center font-bold">
                      {idx + 1}
                    </div>
                    <div>
                      <h4 className="font-bold text-orange-900 dark:text-orange-100">{innovation.title}</h4>
                      <p className="text-sm text-orange-800 dark:text-orange-200">{innovation.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h4 className="font-bold text-gray-900 dark:text-white mb-4">
                Flamingo 아키텍처 구성
              </h4>
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                  <p className="font-semibold text-gray-900 dark:text-white mb-2">Vision Encoder</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Normalizer-Free ResNet (NFNet) - 이미지/비디오 프레임을 고차원 특징으로 인코딩
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                  <p className="font-semibold text-gray-900 dark:text-white mb-2">Perceiver Resampler</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Transformer 기반 압축 모듈 - 가변 길이 시각 특징을 고정 개수 토큰으로 변환
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                  <p className="font-semibold text-gray-900 dark:text-white mb-2">Language Model (Frozen)</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    Chinchilla 70B - 사전 학습된 대규모 언어 모델 (가중치 고정)
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                  <p className="font-semibold text-gray-900 dark:text-white mb-2">Cross-Attention Layers (Trainable)</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    LLM의 각 레이어에 삽입된 크로스 어텐션 - 시각 토큰과 텍스트 토큰을 융합
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-violet-50 dark:bg-violet-900/20 rounded-xl p-6 border-l-4 border-violet-500">
              <p className="font-semibold text-violet-900 dark:text-violet-100 mb-2">
                🎯 Few-shot 성능
              </p>
              <p className="text-violet-800 dark:text-violet-200">
                Flamingo-80B는 16개 멀티모달 벤치마크에서 단 4개의 예시만으로 기존 SOTA를 능가했습니다.
                VQAv2에서 82.0% (이전 SOTA 80.6%), OK-VQA에서 65.1% (이전 54.9%) 달성.
              </p>
            </div>
          </div>
        </section>

        {/* Attention Mechanisms 비교 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🔗 Attention Mechanisms 비교
          </h2>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                type: 'Self-Attention',
                description: '같은 모달리티 내에서 토큰 간 관계 학습',
                example: '텍스트 토큰 간의 의존성, 이미지 패치 간의 공간 관계',
                usage: 'Transformer 인코더 내부',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                type: 'Cross-Attention',
                description: '서로 다른 모달리티 간의 관계 학습',
                example: '텍스트 토큰이 이미지 특징에 어텐션하여 시각 정보 선택',
                usage: 'Flamingo, DALL-E 2 Decoder',
                color: 'from-purple-500 to-pink-500'
              },
              {
                type: 'Contrastive',
                description: '매칭 쌍은 가깝게, 비매칭 쌍은 멀게',
                example: 'CLIP의 이미지-텍스트 코사인 유사도 최대화/최소화',
                usage: 'CLIP, ALIGN',
                color: 'from-green-500 to-emerald-500'
              }
            ].map((attention, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${attention.color} text-white mb-4`}>
                  <Zap className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {attention.type}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-3">
                  {attention.description}
                </p>
                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 mb-3">
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    <span className="font-semibold">예시:</span> {attention.example}
                  </p>
                </div>
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                  <p className="text-sm text-blue-900 dark:text-blue-100">
                    <span className="font-semibold">사용:</span> {attention.usage}
                  </p>
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
              'CLIP의 Contrastive Learning 방식과 Dual Encoder 아키텍처',
              'CLIP의 제로샷 분류, 크로스모달 검색, 프롬프트 엔지니어링 능력',
              'DALL-E 1 (Transformer)과 DALL-E 2/3 (Diffusion Model)의 차이점',
              'DALL-E의 Text → CLIP Embedding → Image 생성 파이프라인',
              'Flamingo의 Few-shot 학습과 Perceiver Resampler + Cross-Attention 구조',
              'Self-Attention, Cross-Attention, Contrastive Learning의 차이와 활용'
            ].map((item, idx) => (
              <li key={idx} className="flex items-start gap-3">
                <span className="text-violet-200 mt-1">✓</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>

          <div className="mt-8 pt-6 border-t border-violet-400">
            <p className="text-violet-100">
              <span className="font-semibold">다음 챕터:</span> 멀티모달 아키텍처의 퓨전 기법
              (Early Fusion, Late Fusion, Cross-Attention)과 최신 Transformer 기반 구조를 살펴봅니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
