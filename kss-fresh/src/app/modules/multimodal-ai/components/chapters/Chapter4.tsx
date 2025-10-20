'use client'

import React from 'react'
import { Mic, Video, Headphones, Waveform, BookOpen, Zap, Film, Music } from 'lucide-react'

export default function Chapter4() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-purple-50 dark:from-gray-900 dark:to-purple-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl">
              <Headphones className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent">
                오디오-비주얼 AI
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                Whisper, Wav2Vec2와 비디오 통합
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-6 h-6 text-violet-600" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              오디오-비주얼 멀티모달 AI란?
            </h2>
          </div>

          <div className="prose dark:prose-invert max-w-none">
            <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
              오디오-비주얼 AI는 음성(Audio)과 시각(Visual) 정보를 동시에 처리하여 더 풍부한 이해를 달성하는
              멀티모달 시스템입니다. 영화를 볼 때 우리는 배우의 대사(음성)와 표정/동작(비디오)을 함께 보며
              맥락을 이해합니다. 오디오-비주얼 AI는 이러한 인간의 인지 방식을 모방합니다.
            </p>

            <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6 border border-violet-200 dark:border-violet-800">
              <p className="text-violet-900 dark:text-violet-100 font-semibold mb-2">
                💡 왜 오디오와 비디오를 함께?
              </p>
              <p className="text-violet-800 dark:text-violet-200">
                음성만으로는 화자의 감정이나 상황을 완전히 파악하기 어렵습니다.
                예를 들어, "괜찮아"라는 말도 표정과 함께 보면 진심인지 위로인지 구분할 수 있습니다.
                오디오-비주얼 통합은 <strong>상호 보완적 정보</strong>를 제공하여 더 정확한 이해를 가능하게 합니다.
              </p>
            </div>
          </div>
        </section>

        {/* 오디오 AI 모델 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Mic className="w-6 h-6 text-violet-600" />
            대표적인 오디오 AI 모델
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            {[
              {
                icon: <Waveform className="w-8 h-8" />,
                name: 'Whisper (OpenAI, 2022)',
                description: '대규모 약지도 학습 기반 음성 인식',
                features: [
                  '680,000시간 다국어 오디오 데이터 학습',
                  '99개 언어 지원, 높은 제로샷 성능',
                  'Timestamp 예측으로 자막 생성 가능',
                  '노이즈 환경에서도 견고한 성능'
                ],
                architecture: 'Encoder-Decoder Transformer (Sequence-to-Sequence)',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                icon: <Music className="w-8 h-8" />,
                name: 'Wav2Vec2 (Meta AI, 2020)',
                description: 'Self-supervised 학습으로 음성 표현 학습',
                features: [
                  '비지도 학습으로 대규모 무라벨 오디오 활용',
                  'Contrastive Learning으로 음성 특징 학습',
                  'Fine-tuning으로 적은 데이터로도 높은 성능',
                  '저자원 언어에서도 효과적'
                ],
                architecture: 'CNN Feature Extractor + Transformer Encoder',
                color: 'from-purple-500 to-pink-500'
              }
            ].map((model, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${model.color} text-white mb-4`}>
                  {model.icon}
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {model.name}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  {model.description}
                </p>
                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 mb-4">
                  <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">주요 특징</p>
                  <ul className="space-y-2">
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
                    <span className="font-semibold">아키텍처:</span> {model.architecture}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Whisper 상세 분석 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🎙️ Whisper 상세 분석
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              Whisper는 OpenAI가 개발한 다목적 음성 인식 모델로, 음성 인식(ASR), 번역, 언어 식별을
              단일 모델에서 수행합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                Whisper 처리 파이프라인
              </h3>
              <div className="space-y-4">
                {[
                  {
                    step: '1',
                    title: '오디오 전처리',
                    desc: '16kHz 샘플링, 30초 청크로 분할, 80차원 Mel Spectrogram 변환'
                  },
                  {
                    step: '2',
                    title: 'Encoder',
                    desc: 'Mel Spectrogram을 Transformer Encoder로 처리하여 오디오 특징 추출'
                  },
                  {
                    step: '3',
                    title: 'Special Tokens',
                    desc: '<|startoftranscript|> <|language|> <|task|> <|notimestamps|> 등 특수 토큰으로 태스크 지정'
                  },
                  {
                    step: '4',
                    title: 'Decoder',
                    desc: 'Transformer Decoder가 자동회귀 방식으로 텍스트 생성'
                  },
                  {
                    step: '5',
                    title: 'Output',
                    desc: '텍스트 전사(Transcription) 또는 번역(Translation) + 타임스탬프'
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
                💻 Whisper 사용 예시 코드
              </h4>
              <div className="bg-gray-900 rounded-lg p-6 overflow-x-auto">
                <pre className="text-sm text-gray-100">
{`import whisper

# 모델 로드 (tiny, base, small, medium, large 중 선택)
model = whisper.load_model("base")

# 오디오 파일 전사
result = model.transcribe("audio.mp3", language="ko")

print(result["text"])  # 전사된 텍스트
print(result["segments"])  # 타임스탬프 포함 세그먼트

# 번역 (다른 언어 → 영어)
result = model.transcribe("korean_audio.mp3", task="translate")
print(result["text"])  # 영어로 번역된 텍스트`}
                </pre>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              {[
                { size: 'Tiny', params: '39M', speed: '~32x', accuracy: '낮음' },
                { size: 'Base', params: '74M', speed: '~16x', accuracy: '중간' },
                { size: 'Small', params: '244M', speed: '~6x', accuracy: '중간-높음' },
                { size: 'Medium', params: '769M', speed: '~2x', accuracy: '높음' },
                { size: 'Large', params: '1550M', speed: '~1x', accuracy: '최고' }
              ].slice(0, 3).map((variant, idx) => (
                <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                  <p className="font-bold text-gray-900 dark:text-white mb-2">{variant.size}</p>
                  <div className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                    <p>파라미터: {variant.params}</p>
                    <p>속도: {variant.speed}</p>
                    <p>정확도: {variant.accuracy}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Wav2Vec2 상세 분석 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🎵 Wav2Vec2 상세 분석
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              Wav2Vec2는 Self-supervised Learning으로 라벨 없는 대규모 오디오 데이터에서
              음성 표현을 학습하는 모델입니다. BERT의 Masked Language Modeling과 유사한 방식을 오디오에 적용합니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                Wav2Vec2 학습 방식
              </h3>
              <div className="space-y-4">
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border-l-4 border-purple-500">
                  <p className="font-semibold text-purple-900 dark:text-purple-100 mb-2">
                    1. Feature Extraction (CNN)
                  </p>
                  <p className="text-sm text-purple-800 dark:text-purple-200">
                    Raw waveform을 7개의 CNN 레이어로 처리하여 시간 축을 압축 (16kHz → 50Hz).
                    각 타임스텝은 25ms의 오디오를 표현.
                  </p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border-l-4 border-purple-500">
                  <p className="font-semibold text-purple-900 dark:text-purple-100 mb-2">
                    2. Masking
                  </p>
                  <p className="text-sm text-purple-800 dark:text-purple-200">
                    연속된 타임스텝의 일부를 마스킹 (BERT의 [MASK]와 유사).
                    마스킹 비율은 약 49% (연속 10 타임스텝을 마스킹).
                  </p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border-l-4 border-purple-500">
                  <p className="font-semibold text-purple-900 dark:text-purple-100 mb-2">
                    3. Transformer Encoding
                  </p>
                  <p className="text-sm text-purple-800 dark:text-purple-200">
                    마스킹된 특징을 12-layer Transformer로 처리하여 컨텍스트 표현 생성.
                  </p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border-l-4 border-purple-500">
                  <p className="font-semibold text-purple-900 dark:text-purple-100 mb-2">
                    4. Quantization (Vector Quantization)
                  </p>
                  <p className="text-sm text-purple-800 dark:text-purple-200">
                    오디오 특징을 이산적 코드북으로 양자화 (320개 코드북, 각 2개 엔트리).
                  </p>
                </div>
                <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border-l-4 border-purple-500">
                  <p className="font-semibold text-purple-900 dark:text-purple-100 mb-2">
                    5. Contrastive Loss
                  </p>
                  <p className="text-sm text-purple-800 dark:text-purple-200">
                    마스킹된 타임스텝의 실제 양자화 코드를 예측하는 대조 학습.
                    100개 negative sample 중에서 정답 코드를 찾도록 학습.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-violet-50 dark:bg-violet-900/20 rounded-xl p-6 border border-violet-200 dark:border-violet-800">
              <h4 className="font-bold text-violet-900 dark:text-violet-100 mb-3">
                🎯 Wav2Vec2의 강점
              </h4>
              <div className="space-y-3 text-violet-800 dark:text-violet-200">
                <p className="flex gap-2">
                  <span className="font-bold">•</span>
                  <span><strong>저자원 언어:</strong> 라벨 없는 데이터만으로 사전 학습 후 적은 라벨 데이터로 Fine-tuning</span>
                </p>
                <p className="flex gap-2">
                  <span className="font-bold">•</span>
                  <span><strong>높은 정확도:</strong> Librispeech 벤치마크에서 1.8% WER 달성 (단 10분 라벨 데이터)</span>
                </p>
                <p className="flex gap-2">
                  <span className="font-bold">•</span>
                  <span><strong>Transfer Learning:</strong> 한 언어에서 학습한 표현을 다른 언어로 전이 가능</span>
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 오디오-비디오 통합 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Video className="w-6 h-6 text-violet-600" />
            오디오-비디오 통합 방법
          </h2>

          <div className="grid md:grid-cols-2 gap-6 mb-6">
            {[
              {
                icon: <Film className="w-8 h-8" />,
                title: 'Audio-Visual Speech Recognition (AVSR)',
                description: '음성과 입 모양(lip movement)을 함께 사용하여 음성 인식',
                benefits: [
                  '노이즈 환경에서 정확도 향상',
                  '칵테일 파티 문제 해결 (화자 분리)',
                  '저품질 오디오 보완'
                ],
                example: 'LRS3 데이터셋 (Lip Reading Sentences 3)',
                color: 'from-orange-500 to-red-500'
              },
              {
                icon: <Zap className="w-8 h-8" />,
                title: 'Audio-Visual Synchronization',
                description: '오디오와 비디오의 시간적 동기화 학습',
                benefits: [
                  '립싱크 품질 검증',
                  '딥페이크 탐지',
                  '비디오 편집 자동화'
                ],
                example: 'SyncNet (입 모양과 음성의 일치 판단)',
                color: 'from-green-500 to-emerald-500'
              }
            ].map((method, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${method.color} text-white mb-4`}>
                  {method.icon}
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {method.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  {method.description}
                </p>
                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 mb-3">
                  <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">장점</p>
                  <ul className="space-y-1">
                    {method.benefits.map((benefit, i) => (
                      <li key={i} className="text-sm text-gray-600 dark:text-gray-400 flex gap-2">
                        <span className="text-green-600">✓</span>
                        <span>{benefit}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                  <p className="text-sm text-blue-900 dark:text-blue-100">
                    <span className="font-semibold">예시:</span> {method.example}
                  </p>
                </div>
              </div>
            ))}
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="font-bold text-gray-900 dark:text-white mb-4">
              Audio-Visual Fusion Architecture
            </h3>
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <div className="space-y-4">
                {[
                  {
                    component: 'Audio Encoder',
                    desc: 'Wav2Vec2 또는 Whisper로 오디오 특징 추출 (예: [T_audio, D_audio])'
                  },
                  {
                    component: 'Video Encoder',
                    desc: '3D CNN 또는 TimeSformer로 비디오 프레임 시퀀스 인코딩 (예: [T_video, D_video])'
                  },
                  {
                    component: 'Temporal Alignment',
                    desc: '오디오와 비디오의 시간 축을 정렬 (샘플링 레이트 다르면 보간)'
                  },
                  {
                    component: 'Fusion Layer',
                    desc: 'Concatenation, Weighted Sum, Cross-Attention 중 선택하여 융합'
                  },
                  {
                    component: 'Output Head',
                    desc: '태스크별 헤드 (ASR: CTC/Seq2Seq, Synchronization: Binary Classifier)'
                  }
                ].map((comp, idx) => (
                  <div key={idx} className="flex gap-3 items-start">
                    <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white font-bold">
                      {idx + 1}
                    </div>
                    <div>
                      <p className="font-semibold text-gray-900 dark:text-white">{comp.component}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">{comp.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* 실전 응용 사례 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🚀 실전 응용 사례
          </h2>

          <div className="grid gap-6">
            {[
              {
                application: '비디오 자막 생성',
                description: 'Whisper로 음성을 텍스트로 변환하고 타임스탬프를 추출하여 자동 자막 생성',
                tech: 'Whisper + timestamp alignment',
                impact: 'YouTube, Netflix 등 스트리밍 서비스에서 다국어 자막 자동 생성',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                application: '회의록 자동 생성',
                description: '여러 화자의 음성을 구분하고 전사하여 회의록 자동 작성',
                tech: 'Whisper + Speaker Diarization (pyannote.audio)',
                impact: 'Zoom, Teams의 회의록 자동 생성 기능',
                color: 'from-purple-500 to-pink-500'
              },
              {
                application: '딥페이크 탐지',
                description: '오디오-비디오 동기화 불일치를 감지하여 조작된 영상 탐지',
                tech: 'SyncNet + Cross-modal consistency check',
                impact: '뉴스 미디어 검증, 법정 증거 검증',
                color: 'from-green-500 to-emerald-500'
              },
              {
                application: '소음 환경 음성 인식',
                description: '입 모양과 음성을 함께 사용하여 시끄러운 환경에서도 정확한 인식',
                tech: 'AVSR (Audio-Visual Speech Recognition)',
                impact: '산업 현장, 공항, 군사 통신',
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
                      <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                        <p className="text-sm text-violet-900 dark:text-violet-100">
                          <span className="font-semibold">영향:</span> {app.impact}
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
              'Whisper의 Encoder-Decoder 구조와 다국어 음성 인식',
              'Wav2Vec2의 Self-supervised Learning과 Contrastive Loss',
              'Audio-Visual Speech Recognition (AVSR)의 원리와 장점',
              'Audio-Visual Synchronization을 통한 딥페이크 탐지',
              'Fusion Architecture: Audio Encoder + Video Encoder + Cross-Attention',
              '실전 응용: 자막 생성, 회의록 작성, 소음 환경 인식'
            ].map((item, idx) => (
              <li key={idx} className="flex items-start gap-3">
                <span className="text-violet-200 mt-1">✓</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>

          <div className="mt-8 pt-6 border-t border-violet-400">
            <p className="text-violet-100">
              <span className="font-semibold">다음 챕터:</span> Text-to-Everything을 살펴봅니다.
              텍스트에서 이미지, 음성, 비디오를 생성하는 최신 생성 모델(DALL-E 3, Stable Diffusion, Sora)을 학습합니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
