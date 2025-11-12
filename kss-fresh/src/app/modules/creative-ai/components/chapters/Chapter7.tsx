'use client'

import React from 'react'
import { Workflow, Zap, Image, Music, Video, Sparkles } from 'lucide-react'

export default function Chapter7() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      {/* Hero Section */}
      <div className="mb-12">
        <div className="inline-block px-4 py-2 bg-purple-500/10 border border-purple-500/20 rounded-full mb-4">
          <span className="text-purple-400 text-sm font-medium">Chapter 7</span>
        </div>
        <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-purple-400 via-pink-400 to-rose-400 bg-clip-text text-transparent">
          크리에이티브 워크플로우
        </h1>
        <p className="text-xl text-gray-300 leading-relaxed">
          AI 도구를 조합하여 아이디어부터 완성된 콘텐츠까지 전체 제작 파이프라인을 구축합니다.
          이미지, 음악, 비디오를 통합한 종합 크리에이티브 워크플로우를 배웁니다.
        </p>
      </div>

      {/* 1. 통합 워크플로우 개요 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Workflow className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">통합 워크플로우 개요</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-purple-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            각 AI 도구는 강력하지만, 진정한 마법은 이들을 조합할 때 일어납니다.
            아이디어 → 스크립트 → 이미지 → 음악 → 비디오 → 편집까지 전체 파이프라인을 구축해봅시다.
          </p>

          <div className="bg-gray-800/50 border border-pink-500/30 rounded-lg p-6">
            <h3 className="text-xl font-bold text-pink-400 mb-4">5단계 크리에이티브 파이프라인</h3>
            <div className="space-y-4">
              {[
                {
                  step: 1,
                  title: '컨셉 & 스크립트',
                  tools: 'ChatGPT, Claude',
                  desc: '아이디어 브레인스토밍, 스토리텔링, 시나리오 작성',
                  color: 'purple'
                },
                {
                  step: 2,
                  title: '비주얼 생성',
                  tools: 'Midjourney, DALL-E, Stable Diffusion',
                  desc: '키 프레임 이미지, 캐릭터 디자인, 배경 아트',
                  color: 'pink'
                },
                {
                  step: 3,
                  title: '오디오 제작',
                  tools: 'Suno, Udio, ElevenLabs',
                  desc: '배경음악, 보컬, 내레이션, 효과음',
                  color: 'rose'
                },
                {
                  step: 4,
                  title: '비디오 합성',
                  tools: 'Runway ML, Pika Labs',
                  desc: '이미지에 움직임 추가, 장면 전환, 애니메이션',
                  color: 'orange'
                },
                {
                  step: 5,
                  title: '편집 & 후처리',
                  tools: 'Premiere Pro, DaVinci Resolve',
                  desc: '최종 편집, 색보정, 자막, 음향 믹싱',
                  color: 'yellow'
                }
              ].map((phase, idx) => (
                <div key={idx} className="bg-gray-900/50 border border-gray-700 rounded-lg p-4">
                  <div className="flex items-start gap-4">
                    <div className={`w-10 h-10 rounded-full bg-${phase.color}-500/20 border border-${phase.color}-500/50 flex items-center justify-center flex-shrink-0`}>
                      <span className={`text-${phase.color}-400 font-bold`}>{phase.step}</span>
                    </div>
                    <div className="flex-1">
                      <h4 className={`font-bold text-${phase.color}-400 mb-1`}>{phase.title}</h4>
                      <p className="text-sm text-gray-400 mb-2">
                        <strong className="text-white">도구:</strong> {phase.tools}
                      </p>
                      <p className="text-sm text-gray-300">{phase.desc}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* 2. 실전 프로젝트: YouTube 숏폼 제작 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-pink-500 to-rose-500 rounded-lg flex items-center justify-center">
            <Video className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">실전 프로젝트: YouTube 숏폼 제작</h2>
        </div>

        <div className="bg-gradient-to-br from-pink-900/20 to-rose-900/20 border border-pink-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            <strong className="text-white">프로젝트 목표:</strong> AI 도구만으로 60초 YouTube Shorts 제작<br/>
            <strong className="text-white">주제 예시:</strong> "미래 도시의 하루" 컨셉 영상
          </p>

          <div className="space-y-6">
            {/* Step 1: 스크립트 */}
            <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-purple-400 mb-4">Step 1: ChatGPT로 스크립트 작성</h3>
              <div className="bg-gray-900/50 rounded-lg p-4 mb-4">
                <p className="text-sm text-purple-400 mb-2">프롬프트:</p>
                <p className="text-sm text-gray-300 mb-3">
                  "2050년 미래 도시의 아침 풍경을 담은 60초 YouTube Shorts 스크립트를 작성해줘.
                  4개 장면으로 구성하고, 각 장면마다 비주얼 설명과 내레이션을 포함해줘."
                </p>
              </div>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-sm text-pink-400 mb-2">ChatGPT 출력 (예시):</p>
                <div className="text-xs text-gray-300 space-y-3 font-mono">
                  <div>
                    <p className="text-purple-400 mb-1">[Scene 1: 0-15s] 해돋이 풍경</p>
                    <p>비주얼: 네온 빛나는 초고층 빌딩 사이로 떠오르는 태양</p>
                    <p>내레이션: "2050년, 도시가 깨어나는 순간..."</p>
                  </div>
                  <div>
                    <p className="text-pink-400 mb-1">[Scene 2: 15-30s] 하늘을 나는 자동차</p>
                    <p>비주얼: Flying cars gliding between skyscrapers</p>
                    <p>내레이션: "출근길은 이제 3차원입니다."</p>
                  </div>
                  <div>
                    <p className="text-rose-400 mb-1">[Scene 3: 30-45s] 홀로그램 광고판</p>
                    <p>비주얼: 3D holographic advertisements floating in air</p>
                    <p>내레이션: "광고도 진화했습니다."</p>
                  </div>
                  <div>
                    <p className="text-orange-400 mb-1">[Scene 4: 45-60s] 로봇 바리스타</p>
                    <p>비주얼: AI robot making coffee in a futuristic cafe</p>
                    <p>내레이션: "완벽한 커피 한 잔으로 하루를 시작하세요."</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Step 2: 이미지 생성 */}
            <div className="bg-gray-800/50 border border-pink-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-pink-400 mb-4">Step 2: Midjourney로 4개 키프레임 생성</h3>
              <div className="space-y-3">
                {[
                  {
                    scene: 'Scene 1',
                    prompt: 'futuristic city skyline at sunrise, neon lights on skyscrapers, orange and purple sky, cinematic wide shot, ultra detailed --ar 9:16 --v 6',
                    color: 'purple'
                  },
                  {
                    scene: 'Scene 2',
                    prompt: 'flying cars gliding between futuristic skyscrapers, aerial view, motion blur, sci-fi concept art, blue hour lighting --ar 9:16 --v 6',
                    color: 'pink'
                  },
                  {
                    scene: 'Scene 3',
                    prompt: '3D holographic advertisements floating in futuristic city street, neon colors, people looking up, cyberpunk style --ar 9:16 --v 6',
                    color: 'rose'
                  },
                  {
                    scene: 'Scene 4',
                    prompt: 'futuristic robot barista making coffee, sleek metallic design, modern cafe interior, close-up shot, warm lighting --ar 9:16 --v 6',
                    color: 'orange'
                  }
                ].map((item, idx) => (
                  <div key={idx} className="bg-gray-900/50 rounded-lg p-3">
                    <p className={`text-sm text-${item.color}-400 mb-1 font-semibold`}>{item.scene}:</p>
                    <p className="text-xs font-mono text-gray-300">{item.prompt}</p>
                  </div>
                ))}
              </div>
              <div className="mt-4 bg-blue-900/20 border border-blue-500/30 rounded-lg p-3">
                <p className="text-blue-300 text-sm">
                  💡 <strong>--ar 9:16</strong>으로 세로 영상 최적화 (YouTube Shorts, TikTok, Reels)
                </p>
              </div>
            </div>

            {/* Step 3: 음악 */}
            <div className="bg-gray-800/50 border border-rose-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-rose-400 mb-4">Step 3: Suno로 배경음악 생성</h3>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-sm text-purple-400 mb-2">Suno 프롬프트:</p>
                <div className="bg-black/30 rounded p-3 mb-3">
                  <p className="text-xs text-gray-300 mb-2"><strong className="text-white">Style:</strong></p>
                  <p className="text-xs font-mono text-gray-300 mb-3">
                    ambient electronic, futuristic, cinematic, synthesizers, uplifting
                  </p>
                  <p className="text-xs text-gray-300 mb-2"><strong className="text-white">Duration:</strong></p>
                  <p className="text-xs font-mono text-gray-300">
                    60초 (Shorts 길이 맞춤)
                  </p>
                </div>
              </div>
              <div className="mt-4 bg-gray-900/50 rounded-lg p-4">
                <p className="text-sm text-pink-400 mb-2">또는 ElevenLabs로 내레이션 생성:</p>
                <p className="text-xs text-gray-300">
                  • Voice: Professional Male / Female<br/>
                  • Stability: 0.7 (안정적)<br/>
                  • Clarity: 0.8 (명확한 발음)
                </p>
              </div>
            </div>

            {/* Step 4: 비디오 합성 */}
            <div className="bg-gray-800/50 border border-orange-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-orange-400 mb-4">Step 4: Runway ML로 이미지 → 비디오</h3>
              <div className="space-y-3">
                <p className="text-sm text-gray-300">
                  각 키프레임(이미지)을 Runway ML Image-to-Video로 15초 비디오로 변환
                </p>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-sm text-purple-400 mb-2">각 Scene 설정:</p>
                  <ul className="text-xs text-gray-300 space-y-2">
                    <li>• <strong className="text-white">Duration:</strong> 15초</li>
                    <li>• <strong className="text-white">Motion:</strong> Medium (자연스러운 움직임)</li>
                    <li>• <strong className="text-white">Camera:</strong> Slow zoom in / Pan left to right</li>
                    <li>• <strong className="text-white">Quality:</strong> 720p (빠른 생성), 최종 4K 업스케일</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Step 5: 최종 편집 */}
            <div className="bg-gray-800/50 border border-yellow-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-yellow-400 mb-4">Step 5: Premiere Pro / CapCut 최종 편집</h3>
              <div className="space-y-3">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-sm text-purple-400 mb-2">편집 순서:</p>
                  <ol className="text-xs text-gray-300 space-y-2 list-decimal list-inside">
                    <li>4개 비디오 클립 타임라인에 배치 (각 15초)</li>
                    <li>장면 전환 효과 추가 (Crossfade, Zoom transition)</li>
                    <li>배경음악 추가 및 볼륨 조정</li>
                    <li>내레이션 레이어 추가 (ElevenLabs 음성)</li>
                    <li>자막 자동 생성 (CapCut AI)</li>
                    <li>색보정 (일관된 색감 유지)</li>
                    <li>9:16 세로 영상으로 내보내기</li>
                  </ol>
                </div>

                <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
                  <p className="text-blue-300 text-sm">
                    💡 <strong>CapCut Pro Tip:</strong> "Auto Captions" 기능으로 1분 만에 자막 완성<br/>
                    음성 인식 정확도 95% 이상 (한국어/영어)
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* 예상 소요 시간 */}
          <div className="mt-6 bg-green-900/20 border border-green-500/30 rounded-lg p-6">
            <h3 className="text-lg font-bold text-green-400 mb-4">📊 예상 소요 시간 & 비용</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-white mb-2"><strong>시간:</strong></p>
                <ul className="text-xs text-gray-300 space-y-1">
                  <li>• ChatGPT 스크립트: 5분</li>
                  <li>• Midjourney 이미지 4개: 10분</li>
                  <li>• Suno 음악: 2분</li>
                  <li>• Runway ML 비디오 4개: 30분</li>
                  <li>• 최종 편집: 20분</li>
                  <li className="text-green-400 font-bold mt-2">• <strong>총: ~1시간</strong></li>
                </ul>
              </div>
              <div>
                <p className="text-sm text-white mb-2"><strong>비용 (Pro 플랜 기준):</strong></p>
                <ul className="text-xs text-gray-300 space-y-1">
                  <li>• ChatGPT Plus: $20/월 (무제한)</li>
                  <li>• Midjourney Pro: $60/월 (무제한)</li>
                  <li>• Suno Pro: $10/월 (500곡)</li>
                  <li>• Runway ML Pro: $35/월 (2250 sec)</li>
                  <li>• CapCut: 무료</li>
                  <li className="text-green-400 font-bold mt-2">• <strong>총: $125/월</strong></li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. 실전 프로젝트 2: 광고 영상 제작 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-rose-500 to-orange-500 rounded-lg flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">실전 프로젝트 2: 제품 광고 영상</h2>
        </div>

        <div className="bg-gradient-to-br from-rose-900/20 to-orange-900/20 border border-rose-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            <strong className="text-white">프로젝트 목표:</strong> 가상 제품(스마트워치)의 30초 광고 영상<br/>
            <strong className="text-white">타겟:</strong> Instagram Reels, TikTok Ads
          </p>

          <div className="space-y-6">
            {/* 컨셉 기획 */}
            <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-purple-400 mb-4">1. Claude로 광고 컨셉 기획</h3>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-sm text-purple-400 mb-2">프롬프트:</p>
                <p className="text-sm text-gray-300 mb-3">
                  "AI 건강 트래킹 기능이 있는 프리미엄 스마트워치를 위한 30초 광고 컨셉을 기획해줘.
                  3개 핵심 메시지와 비주얼 아이디어를 포함해줘."
                </p>
              </div>
              <div className="mt-4 bg-gray-900/50 rounded-lg p-4">
                <p className="text-sm text-pink-400 mb-2">Claude 출력 (예시):</p>
                <div className="text-xs text-gray-300 space-y-2">
                  <p><strong className="text-purple-400">메시지 1:</strong> "Your health, predicted" - AI가 건강 상태 예측</p>
                  <p><strong className="text-pink-400">메시지 2:</strong> "24/7 monitoring" - 심박수, 수면, 스트레스 분석</p>
                  <p><strong className="text-rose-400">메시지 3:</strong> "Premium design" - 티타늄 케이스, 사파이어 글래스</p>
                </div>
              </div>
            </div>

            {/* 제품 이미지 */}
            <div className="bg-gray-800/50 border border-pink-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-pink-400 mb-4">2. DALL-E 3로 제품 렌더링</h3>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-sm text-purple-400 mb-2">프롬프트:</p>
                <div className="bg-black/30 rounded p-3">
                  <p className="text-xs font-mono text-gray-300 leading-relaxed">
                    A premium smartwatch with titanium case and circular OLED display,
                    showing health metrics on screen, floating on white marble surface,
                    studio product photography, dramatic side lighting with soft reflections,
                    ultra high quality, shot on Canon EOS R5, 8K
                  </p>
                </div>
              </div>
              <p className="text-xs text-gray-400 mt-3">
                💡 여러 각도 생성: 정면, 측면, 3/4 각도, 클로즈업
              </p>
            </div>

            {/* 라이프스타일 장면 */}
            <div className="bg-gray-800/50 border border-rose-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-rose-400 mb-4">3. Midjourney로 라이프스타일 장면</h3>
              <div className="space-y-3">
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <p className="text-xs text-purple-400 mb-1">Scene A: 러닝</p>
                  <p className="text-xs font-mono text-gray-300">
                    athletic young person running in urban park at sunrise, wearing smartwatch
                    on wrist, motion blur background, healthy lifestyle, professional sports
                    photography --ar 9:16 --v 6
                  </p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <p className="text-xs text-pink-400 mb-1">Scene B: 요가/명상</p>
                  <p className="text-xs font-mono text-gray-300">
                    person meditating in modern minimalist room, smartwatch visible,
                    calm atmosphere, soft natural lighting, zen aesthetic --ar 9:16 --v 6
                  </p>
                </div>
              </div>
            </div>

            {/* 음악 & 내레이션 */}
            <div className="bg-gray-800/50 border border-orange-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-orange-400 mb-4">4. Udio + ElevenLabs 오디오 제작</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-sm text-purple-400 mb-2">Udio (배경음악):</p>
                  <p className="text-xs text-gray-300 mb-2">
                    uplifting corporate electronic music, motivational, clean synthesizers,
                    light percussion, 128 BPM, modern tech commercial style
                  </p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-sm text-pink-400 mb-2">ElevenLabs (내레이션):</p>
                  <p className="text-xs text-gray-300">
                    "Your health, predicted. 24/7 monitoring. Premium design.
                    The future of wellness, on your wrist."
                  </p>
                </div>
              </div>
            </div>

            {/* 최종 조립 */}
            <div className="bg-gray-800/50 border border-yellow-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-yellow-400 mb-4">5. Runway ML + Premiere Pro 조립</h3>
              <ol className="text-sm text-gray-300 space-y-2 list-decimal list-inside">
                <li>Runway ML로 제품 이미지에 회전 애니메이션 추가 (360도)</li>
                <li>러닝/요가 장면을 Image-to-Video로 동적 영상화</li>
                <li>Premiere Pro에서 3개 클립 연결 (제품 → 러닝 → 요가)</li>
                <li>텍스트 애니메이션 추가 ("Your health, predicted" 등)</li>
                <li>배경음악 + 내레이션 동기화</li>
                <li>색보정 (프리미엄 느낌의 차가운 톤)</li>
                <li>30초 Instagram Reels 형식 (9:16) 내보내기</li>
              </ol>
            </div>
          </div>

          {/* ROI 계산 */}
          <div className="mt-6 bg-green-900/20 border border-green-500/30 rounded-lg p-6">
            <h3 className="text-lg font-bold text-green-400 mb-4">💰 비용 대비 효과 (ROI)</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-white mb-2"><strong>전통적 광고 제작:</strong></p>
                <ul className="text-xs text-gray-300 space-y-1">
                  <li>• 제작사 섭외 & 미팅: 1주</li>
                  <li>• 촬영 준비 (장소, 배우, 장비): 1주</li>
                  <li>• 촬영 & 후반 작업: 1주</li>
                  <li>• 수정 작업: 3-5일</li>
                  <li className="text-red-400 font-bold mt-2">• <strong>총 비용: $5,000-$10,000</strong></li>
                  <li className="text-red-400 font-bold">• <strong>소요 시간: 3-4주</strong></li>
                </ul>
              </div>
              <div>
                <p className="text-sm text-white mb-2"><strong>AI 도구 활용:</strong></p>
                <ul className="text-xs text-gray-300 space-y-1">
                  <li>• 컨셉 기획: 30분</li>
                  <li>• 이미지 생성: 1시간</li>
                  <li>• 오디오 제작: 30분</li>
                  <li>• 비디오 합성: 1시간</li>
                  <li>• 최종 편집: 1시간</li>
                  <li className="text-green-400 font-bold mt-2">• <strong>총 비용: $125 (도구 구독료)</strong></li>
                  <li className="text-green-400 font-bold">• <strong>소요 시간: 4시간</strong></li>
                </ul>
              </div>
            </div>
            <div className="mt-4 bg-green-900/30 rounded-lg p-4 text-center">
              <p className="text-green-300 text-lg font-bold">
                💰 비용 절감: 98% | ⏱️ 시간 절감: 95%
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 4. 워크플로우 자동화 팁 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-yellow-500 rounded-lg flex items-center justify-center">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">워크플로우 자동화 & 최적화</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* 배치 처리 */}
          <div className="bg-gray-800/50 border border-purple-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-purple-400 mb-4">1. 배치 처리</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• <strong className="text-white">Midjourney:</strong> /imagine 명령어 10개 동시 실행</li>
              <li>• <strong className="text-white">Runway ML:</strong> 여러 이미지 한 번에 큐잉</li>
              <li>• <strong className="text-white">Suno:</strong> 한 번에 5곡 생성 후 선택</li>
              <li>• <strong className="text-white">ChatGPT:</strong> GPT-4 API로 대량 스크립트 생성</li>
            </ul>
          </div>

          {/* 템플릿 활용 */}
          <div className="bg-gray-800/50 border border-pink-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-pink-400 mb-4">2. 재사용 가능한 템플릿</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• <strong className="text-white">Premiere Pro:</strong> 광고 템플릿 저장 (.prproj)</li>
              <li>• <strong className="text-white">프롬프트 라이브러리:</strong> Notion에 검증된 프롬프트 저장</li>
              <li>• <strong className="text-white">스타일 가이드:</strong> 일관된 색감/폰트 문서화</li>
              <li>• <strong className="text-white">LoRA 모델:</strong> 브랜드 스타일 학습 후 재사용</li>
            </ul>
          </div>

          {/* API 활용 */}
          <div className="bg-gray-800/50 border border-rose-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-rose-400 mb-4">3. API 통합 자동화</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• <strong className="text-white">OpenAI API:</strong> Python으로 대량 이미지 생성</li>
              <li>• <strong className="text-white">Runway ML API:</strong> 비디오 생성 파이프라인</li>
              <li>• <strong className="text-white">Zapier/Make:</strong> 노코드 자동화 (Notion → Midjourney)</li>
              <li>• <strong className="text-white">GitHub Actions:</strong> 정기적인 콘텐츠 생성</li>
            </ul>
          </div>

          {/* 품질 관리 */}
          <div className="bg-gray-800/50 border border-orange-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-orange-400 mb-4">4. 품질 관리 체크리스트</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• <strong className="text-white">해상도:</strong> 최소 1080p, 권장 4K</li>
              <li>• <strong className="text-white">색감:</strong> 일관된 LUT 적용</li>
              <li>• <strong className="text-white">오디오:</strong> -3dB to -6dB 범위 유지</li>
              <li>• <strong className="text-white">자막:</strong> 읽기 쉬운 폰트 (최소 48pt)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            📚
          </div>
          References
        </h2>

        <div className="space-y-4">
          <div className="bg-gray-800/50 border border-purple-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-purple-400 mb-4">🎬 워크플로우 튜토리얼</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://www.youtube.com/results?search_query=ai+video+workflow+tutorial"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  AI Video Workflow Tutorials - YouTube
                </a>
                <p className="text-sm text-gray-400 mt-1">전체 파이프라인 영상 튜토리얼</p>
              </li>
              <li>
                <a
                  href="https://www.reddit.com/r/CreativeAI/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  r/CreativeAI - Reddit Community
                </a>
                <p className="text-sm text-gray-400 mt-1">크리에이티브 AI 워크플로우 공유</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-pink-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-pink-400 mb-4">🛠️ 자동화 도구</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://zapier.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Zapier - No-Code Automation
                </a>
                <p className="text-sm text-gray-400 mt-1">AI 도구 간 노코드 자동화</p>
              </li>
              <li>
                <a
                  href="https://www.make.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Make (Integromat) - Visual Automation
                </a>
                <p className="text-sm text-gray-400 mt-1">복잡한 워크플로우 구축</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-rose-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-rose-400 mb-4">📖 사례 연구</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://www.awwwards.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  Awwwards - Creative AI Projects
                </a>
                <p className="text-sm text-gray-400 mt-1">수상작 및 베스트 프랙티스</p>
              </li>
              <li>
                <a
                  href="https://aiartists.org/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  AI Artists Community
                </a>
                <p className="text-sm text-gray-400 mt-1">전문 AI 아티스트 네트워크</p>
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}
