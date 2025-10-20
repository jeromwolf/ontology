'use client'

import React from 'react'
import { Sparkles, Image as ImageIcon, Mic, Video, BookOpen, Wand2, Palette, Zap } from 'lucide-react'

export default function Chapter5() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-purple-50 dark:from-gray-900 dark:to-purple-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl">
              <Wand2 className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent">
                Text-to-Everything
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                텍스트에서 이미지, 음성, 비디오 생성
              </p>
            </div>
          </div>
        </div>

        {/* Introduction */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-6 h-6 text-violet-600" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Text-to-Everything이란?
            </h2>
          </div>

          <div className="prose dark:prose-invert max-w-none">
            <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
              Text-to-Everything은 자연어 텍스트로부터 다양한 모달리티의 콘텐츠를 생성하는 생성형 AI 기술입니다.
              텍스트 프롬프트 하나로 이미지, 음성, 비디오, 3D 모델 등을 만들어낼 수 있어,
              창작, 디자인, 마케팅, 교육 등 다양한 분야에서 혁신을 일으키고 있습니다.
            </p>

            <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6 border border-violet-200 dark:border-violet-800">
              <p className="text-violet-900 dark:text-violet-100 font-semibold mb-2">
                💡 생성형 AI의 핵심 패러다임
              </p>
              <p className="text-violet-800 dark:text-violet-200">
                과거에는 "고양이 사진 1000장"이 필요했다면, 이제는 "귀여운 주황색 고양이가 햇빛 아래에서 낮잠 자는 모습"이라는
                텍스트 하나로 원하는 이미지를 즉시 생성할 수 있습니다. 이는 <strong>인간의 의도를 자연어로 표현</strong>하고,
                AI가 이를 <strong>창조적으로 해석</strong>하여 결과물을 만들어내는 새로운 창작 방식입니다.
              </p>
            </div>
          </div>
        </section>

        {/* Text-to-Image 모델들 */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <ImageIcon className="w-6 h-6 text-violet-600" />
            Text-to-Image 생성 모델
          </h2>

          <div className="grid gap-6">
            {[
              {
                name: 'DALL-E 3 (OpenAI, 2023)',
                description: 'GPT-4 수준의 자연어 이해와 고품질 이미지 생성',
                architecture: 'CLIP Text Encoder + Diffusion Prior + Diffusion Decoder',
                features: [
                  '복잡한 프롬프트를 GPT-4가 재작성하여 정확도 향상',
                  '1024×1024 고해상도, 일관된 스타일 유지',
                  '텍스트 렌더링 능력 대폭 개선 (이미지 내 글자 생성)',
                  '부적절한 콘텐츠 생성 방지 안전 시스템'
                ],
                prompt: 'An expressive oil painting of a basketball player dunking, depicted as an explosion of a nebula',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                name: 'Stable Diffusion (Stability AI, 2022)',
                description: '오픈소스 Diffusion 모델로 로컬 실행 가능',
                architecture: 'CLIP Text Encoder + VAE Latent Space + U-Net Diffusion',
                features: [
                  'Latent Diffusion으로 계산 효율성 극대화 (픽셀 대신 압축된 latent space에서 생성)',
                  '512×512 기본, 최대 1024×1024 지원',
                  'LoRA, ControlNet 등 커뮤니티 확장 도구 풍부',
                  '완전 오픈소스로 커스터마이징과 Fine-tuning 가능'
                ],
                prompt: 'A dream of a distant galaxy, by Caspar David Friedrich, matte painting, trending on artstation HQ',
                color: 'from-purple-500 to-pink-500'
              },
              {
                name: 'Midjourney v6 (2024)',
                description: '예술적 품질과 사실성이 뛰어난 상용 서비스',
                architecture: '비공개 (Diffusion 기반 추정)',
                features: [
                  '포토리얼리즘과 예술적 표현의 탁월한 균형',
                  '자연어 프롬프트를 섬세하게 해석',
                  '인물, 손, 얼굴 등 디테일 표현 우수',
                  'Discord 기반 커뮤니티와 협업 환경'
                ],
                prompt: 'cinematic photo of a cyberpunk city at night, neon lights reflecting on wet streets, ultra detailed',
                color: 'from-green-500 to-emerald-500'
              }
            ].map((model, idx) => (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow"
              >
                <div className="flex items-start gap-4 mb-4">
                  <div className={`flex-shrink-0 w-12 h-12 rounded-lg bg-gradient-to-br ${model.color} flex items-center justify-center text-white font-bold text-xl`}>
                    {idx + 1}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-1">
                      {model.name}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 mb-3">
                      {model.description}
                    </p>
                  </div>
                </div>

                <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 mb-4">
                  <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1">아키텍처</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{model.architecture}</p>
                </div>

                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 mb-4">
                  <p className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">주요 특징</p>
                  <ul className="space-y-1">
                    {model.features.map((feature, i) => (
                      <li key={i} className="text-sm text-blue-800 dark:text-blue-200 flex gap-2">
                        <span className="text-blue-600">•</span>
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-4 border-l-4 border-violet-500">
                  <p className="text-xs text-violet-900 dark:text-violet-100 mb-1 font-semibold">예시 프롬프트</p>
                  <p className="text-sm text-violet-800 dark:text-violet-200 italic">
                    "{model.prompt}"
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Diffusion Model 원리 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🌀 Diffusion Model 작동 원리
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              현재 대부분의 Text-to-Image 모델은 Diffusion Model을 기반으로 합니다.
              Diffusion Model은 점진적으로 노이즈를 제거하여 이미지를 생성하는 방식입니다.
            </p>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                Forward Process (노이즈 추가)
              </h3>
              <div className="space-y-3">
                {[
                  { step: 'T=0', desc: '원본 이미지 x₀ (깨끗한 상태)' },
                  { step: 'T=100', desc: '약간의 가우시안 노이즈 추가' },
                  { step: 'T=500', desc: '중간 수준 노이즈 (이미지 형태 흐릿)' },
                  { step: 'T=1000', desc: '완전한 노이즈 (순수 가우시안 분포)' }
                ].map((phase, idx) => (
                  <div key={idx} className="flex items-center gap-4">
                    <div className="flex-shrink-0 w-16 h-16 rounded-lg bg-gradient-to-br from-red-400 to-orange-500 flex items-center justify-center text-white font-bold">
                      {phase.step}
                    </div>
                    <p className="text-gray-700 dark:text-gray-300">{phase.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
              <h3 className="font-bold text-gray-900 dark:text-white mb-4">
                Reverse Process (이미지 생성)
              </h3>
              <div className="space-y-3">
                {[
                  { step: 'T=1000', desc: '순수 노이즈에서 시작', color: 'from-gray-400 to-gray-600' },
                  { step: 'T=750', desc: 'U-Net이 노이즈 예측 및 제거 (텍스트 조건 사용)', color: 'from-blue-400 to-blue-600' },
                  { step: 'T=500', desc: '대략적인 형태와 색상 나타남', color: 'from-green-400 to-green-600' },
                  { step: 'T=250', desc: '디테일과 텍스처 추가', color: 'from-yellow-400 to-yellow-600' },
                  { step: 'T=0', desc: '최종 고품질 이미지 완성', color: 'from-violet-500 to-purple-600' }
                ].map((phase, idx) => (
                  <div key={idx} className="flex items-center gap-4">
                    <div className={`flex-shrink-0 w-16 h-16 rounded-lg bg-gradient-to-br ${phase.color} flex items-center justify-center text-white font-bold`}>
                      {phase.step}
                    </div>
                    <p className="text-gray-700 dark:text-gray-300">{phase.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-violet-50 dark:bg-violet-900/20 rounded-xl p-6 border border-violet-200 dark:border-violet-800">
              <h4 className="font-bold text-violet-900 dark:text-violet-100 mb-3">
                🎯 텍스트 조건화 (Text Conditioning)
              </h4>
              <p className="text-violet-800 dark:text-violet-200 mb-3">
                U-Net에 텍스트 임베딩을 Cross-Attention으로 주입하여 텍스트 프롬프트에 맞는 이미지를 생성합니다.
              </p>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <pre className="text-sm text-gray-700 dark:text-gray-300">
{`# CLIP 텍스트 임베딩
text_embedding = clip_text_encoder(prompt)  # [1, 77, 768]

# Diffusion 단계마다 U-Net에 조건 주입
for t in reversed(range(1000)):
    noise_pred = unet(noisy_image, t, text_embedding)
    noisy_image = denoise(noisy_image, noise_pred)`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Text-to-Speech */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Mic className="w-6 h-6 text-violet-600" />
            Text-to-Speech (TTS)
          </h2>

          <div className="space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              Text-to-Speech는 텍스트를 자연스러운 음성으로 변환하는 기술입니다.
              최신 TTS 모델은 감정, 억양, 화자 스타일까지 제어할 수 있습니다.
            </p>

            <div className="grid md:grid-cols-2 gap-6">
              {[
                {
                  name: 'ElevenLabs',
                  description: '초현실적인 음성 복제와 감정 표현',
                  features: [
                    '29개 언어 지원, 음성 복제 (Voice Cloning)',
                    '감정 제어 (기쁨, 슬픔, 분노 등)',
                    '실시간 스트리밍 TTS',
                    '오디오북, 게임, 광고 등 상업 활용'
                  ],
                  color: 'from-blue-500 to-cyan-500'
                },
                {
                  name: 'Tortoise TTS',
                  description: '오픈소스 고품질 TTS',
                  features: [
                    '6초 샘플로 음성 복제 가능',
                    'GPT 기반 autoregressive 생성',
                    'CLVP + Diffusion Decoder 구조',
                    '느리지만 최고 품질의 음성 생성'
                  ],
                  color: 'from-green-500 to-emerald-500'
                }
              ].map((tts, idx) => (
                <div key={idx} className="bg-gray-50 dark:bg-gray-700/50 rounded-xl p-6">
                  <div className={`inline-flex p-3 rounded-lg bg-gradient-to-br ${tts.color} text-white mb-4`}>
                    <Mic className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
                    {tts.name}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    {tts.description}
                  </p>
                  <ul className="space-y-2">
                    {tts.features.map((feature, i) => (
                      <li key={i} className="text-sm text-gray-700 dark:text-gray-300 flex gap-2">
                        <span className="text-violet-600">•</span>
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800">
              <h4 className="font-bold text-blue-900 dark:text-blue-100 mb-3">
                TTS 아키텍처 발전
              </h4>
              <div className="space-y-3">
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center text-sm font-bold">1</div>
                  <div>
                    <p className="font-semibold text-blue-900 dark:text-blue-100">WaveNet (2016)</p>
                    <p className="text-sm text-blue-800 dark:text-blue-200">픽셀 단위 생성, 매우 느림</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center text-sm font-bold">2</div>
                  <div>
                    <p className="font-semibold text-blue-900 dark:text-blue-100">Tacotron 2 (2017)</p>
                    <p className="text-sm text-blue-800 dark:text-blue-200">Seq2Seq + Attention, Mel Spectrogram 생성</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center text-sm font-bold">3</div>
                  <div>
                    <p className="font-semibold text-blue-900 dark:text-blue-100">FastSpeech (2019)</p>
                    <p className="text-sm text-blue-800 dark:text-blue-200">Non-autoregressive, 실시간 생성 가능</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center text-sm font-bold">4</div>
                  <div>
                    <p className="font-semibold text-blue-900 dark:text-blue-100">VITS (2021)</p>
                    <p className="text-sm text-blue-800 dark:text-blue-200">End-to-end, Variational Inference + GAN</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-violet-500 text-white flex items-center justify-center text-sm font-bold">5</div>
                  <div>
                    <p className="font-semibold text-blue-900 dark:text-blue-100">Diffusion TTS (2023+)</p>
                    <p className="text-sm text-blue-800 dark:text-blue-200">Diffusion 기반, 최고 품질</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Text-to-Video */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Video className="w-6 h-6 text-violet-600" />
            Text-to-Video 생성
          </h2>

          <div className="grid gap-6">
            {[
              {
                name: 'Sora (OpenAI, 2024)',
                description: '최대 1분 길이의 고품질 비디오 생성',
                capabilities: [
                  '복잡한 장면, 다수 캐릭터, 정확한 물리 법칙',
                  '카메라 움직임과 시간적 일관성 유지',
                  '1080p 해상도, 다양한 종횡비 지원',
                  '프롬프트: "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage..."'
                ],
                tech: 'Diffusion Transformer (DiT) on Video Patches',
                color: 'from-purple-500 to-pink-500'
              },
              {
                name: 'Runway Gen-2',
                description: '크리에이터를 위한 Text-to-Video 툴',
                capabilities: [
                  '4초 클립 생성, 이미지→비디오 변환',
                  '스타일 프리셋 (Cinematic, Anime, Watercolor 등)',
                  '모션 제어 및 카메라 설정',
                  '영화, 광고, 뮤직비디오 제작에 활용'
                ],
                tech: 'Latent Diffusion on Video Latents',
                color: 'from-blue-500 to-cyan-500'
              },
              {
                name: 'Pika Labs',
                description: '커뮤니티 기반 비디오 생성',
                capabilities: [
                  '3초 클립, 텍스트 또는 이미지 프롬프트',
                  '애니메이션 스타일, 립싱크 기능',
                  'Discord 기반 접근성',
                  '소셜 미디어 콘텐츠 제작'
                ],
                tech: 'Diffusion Model + Motion Prior',
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
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      {model.description}
                    </p>
                    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 mb-3">
                      <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">주요 기능</p>
                      <ul className="space-y-1">
                        {model.capabilities.map((cap, i) => (
                          <li key={i} className="text-sm text-gray-600 dark:text-gray-400 flex gap-2">
                            <span className="text-violet-600">•</span>
                            <span>{cap}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div className="bg-violet-50 dark:bg-violet-900/20 rounded-lg p-3 border-l-4 border-violet-500">
                      <p className="text-sm text-violet-900 dark:text-violet-100">
                        <span className="font-semibold">기술:</span> {model.tech}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 프롬프트 엔지니어링 */}
        <section className="mb-12 bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-2xl p-8 border border-amber-200 dark:border-amber-800">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Palette className="w-6 h-6 text-amber-600" />
            프롬프트 엔지니어링 팁
          </h2>

          <div className="space-y-4">
            {[
              {
                tip: '구체적인 디테일 추가',
                bad: '"고양이"',
                good: '"햇살 가득한 창가에서 낮잠 자는 주황색 털의 페르시안 고양이, 부드러운 빛, 따뜻한 색조"',
                why: '모호한 프롬프트는 일관성 없는 결과를 낳습니다. 색상, 조명, 분위기를 명시하세요.'
              },
              {
                tip: '스타일 키워드 활용',
                bad: '"도시 풍경"',
                good: '"cyberpunk 도시 풍경, neon lights, cinematic lighting, 4k, octane render"',
                why: '예술적 스타일과 렌더링 키워드는 품질을 크게 향상시킵니다.'
              },
              {
                tip: '부정 프롬프트 (Negative Prompt)',
                example: 'Negative: "blurry, low quality, deformed, watermark, text"',
                why: '원하지 않는 요소를 명시하여 결과물을 정제합니다.'
              },
              {
                tip: '가중치 조절',
                example: '"(cat:1.5), (sunlight:1.2), background:0.8"',
                why: '중요한 요소에 가중치를 부여하여 강조할 수 있습니다.'
              }
            ].map((item, idx) => (
              <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg p-5 shadow">
                <h4 className="font-bold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                  <span className="text-amber-600 dark:text-amber-400">💡</span>
                  {item.tip}
                </h4>
                {item.bad && (
                  <div className="mb-2">
                    <p className="text-sm text-red-600 dark:text-red-400 font-semibold">❌ 나쁜 예:</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 italic">{item.bad}</p>
                  </div>
                )}
                {item.good && (
                  <div className="mb-2">
                    <p className="text-sm text-green-600 dark:text-green-400 font-semibold">✅ 좋은 예:</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 italic">{item.good}</p>
                  </div>
                )}
                {item.example && (
                  <div className="mb-2">
                    <p className="text-sm text-blue-600 dark:text-blue-400 font-semibold">📝 예시:</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 italic">{item.example}</p>
                  </div>
                )}
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  <span className="font-semibold">이유:</span> {item.why}
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
              'Text-to-Image: DALL-E 3, Stable Diffusion, Midjourney 비교',
              'Diffusion Model 작동 원리 (Forward/Reverse Process, Text Conditioning)',
              'Text-to-Speech: ElevenLabs, Tortoise TTS 아키텍처 발전',
              'Text-to-Video: Sora, Runway Gen-2, Pika Labs 기능과 기술',
              '프롬프트 엔지니어링 (구체적 디테일, 스타일 키워드, 부정 프롬프트)',
              'Latent Diffusion으로 계산 효율성 향상'
            ].map((item, idx) => (
              <li key={idx} className="flex items-start gap-3">
                <span className="text-violet-200 mt-1">✓</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>

          <div className="mt-8 pt-6 border-t border-violet-400">
            <p className="text-violet-100">
              <span className="font-semibold">다음 챕터:</span> 멀티모달 임베딩과 공통 임베딩 공간을 학습합니다.
              크로스모달 검색, 메트릭 러닝, 제로샷 기능의 원리를 살펴봅니다.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
