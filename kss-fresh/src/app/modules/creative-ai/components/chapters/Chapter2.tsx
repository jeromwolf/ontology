'use client'

import React from 'react'
import { Wand2, Palette, Settings, Sparkles, Image, AlertCircle } from 'lucide-react'

export default function Chapter2() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      {/* Hero Section */}
      <div className="mb-12">
        <div className="inline-block px-4 py-2 bg-purple-500/10 border border-purple-500/20 rounded-full mb-4">
          <span className="text-purple-400 text-sm font-medium">Chapter 2</span>
        </div>
        <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-purple-400 via-pink-400 to-rose-400 bg-clip-text text-transparent">
          이미지 생성 기초
        </h1>
        <p className="text-xl text-gray-300 leading-relaxed">
          Midjourney, DALL-E 3, Stable Diffusion 등 주요 AI 이미지 생성 모델의 프롬프트 엔지니어링 기술을 배웁니다.
          효과적인 프롬프트 작성으로 원하는 이미지를 정확하게 생성하는 방법을 습득합니다.
        </p>
      </div>

      {/* 1. 주요 AI 이미지 생성 모델 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Image className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">주요 AI 이미지 생성 모델</h2>
        </div>

        <div className="space-y-6">
          {/* Midjourney */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-purple-400 mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5" />
              Midjourney
            </h3>
            <div className="space-y-4 text-gray-300">
              <p>
                <strong className="text-white">특징:</strong> Discord 기반 서비스, 뛰어난 미적 감각과 예술적 표현
              </p>
              <div>
                <strong className="text-white">강점:</strong>
                <ul className="list-disc list-inside ml-4 mt-2 space-y-1">
                  <li>사진처럼 사실적인 인물/풍경 생성</li>
                  <li>예술 스타일 재현 능력 (고흐, 모네 등)</li>
                  <li>컨셉 아트, 판타지 이미지에 특화</li>
                  <li>커뮤니티 갤러리를 통한 학습</li>
                </ul>
              </div>
              <div className="bg-gray-900/50 border border-purple-500/30 rounded-lg p-4 mt-4">
                <p className="text-sm font-mono text-purple-300">
                  /imagine prompt: a serene Japanese garden with cherry blossoms, koi pond,
                  stone lanterns, golden hour lighting, ultra detailed, 8k --ar 16:9 --v 6
                </p>
              </div>
              <p>
                <strong className="text-white">가격:</strong> Basic $10/월 (200회), Standard $30/월 (무제한), Pro $60/월
              </p>
            </div>
          </div>

          {/* DALL-E 3 */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-pink-400 mb-4">DALL-E 3 (OpenAI)</h3>
            <div className="space-y-4 text-gray-300">
              <p>
                <strong className="text-white">특징:</strong> ChatGPT Plus 통합, 자연어 이해력 최고
              </p>
              <div>
                <strong className="text-white">강점:</strong>
                <ul className="list-disc list-inside ml-4 mt-2 space-y-1">
                  <li>복잡한 프롬프트 정확히 이해</li>
                  <li>텍스트 렌더링 능력 (간판, 포스터)</li>
                  <li>ChatGPT로 프롬프트 자동 개선</li>
                  <li>안전성 필터 (윤리적 콘텐츠)</li>
                </ul>
              </div>
              <div className="bg-gray-900/50 border border-pink-500/30 rounded-lg p-4 mt-4">
                <p className="text-sm font-mono text-pink-300">
                  Create a vintage 1950s diner scene with neon signs that say "AI CAFE",
                  chrome furniture, checkered floor, and a robot waitress serving milkshakes
                </p>
              </div>
              <p>
                <strong className="text-white">접근:</strong> ChatGPT Plus ($20/월) 또는 API ($0.04/이미지)
              </p>
            </div>
          </div>

          {/* Stable Diffusion */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-rose-400 mb-4">Stable Diffusion</h3>
            <div className="space-y-4 text-gray-300">
              <p>
                <strong className="text-white">특징:</strong> 오픈소스, 로컬 실행 가능, 완전한 제어권
              </p>
              <div>
                <strong className="text-white">강점:</strong>
                <ul className="list-disc list-inside ml-4 mt-2 space-y-1">
                  <li>무료 사용 (GPU만 있으면 됨)</li>
                  <li>LoRA, DreamBooth로 커스텀 가능</li>
                  <li>ControlNet으로 정밀 제어</li>
                  <li>Automatic1111, ComfyUI 등 강력한 UI</li>
                </ul>
              </div>
              <div className="bg-gray-900/50 border border-rose-500/30 rounded-lg p-4 mt-4">
                <p className="text-sm font-mono text-rose-300">
                  masterpiece, best quality, 1girl, flowing hair, fantasy armor,
                  dramatic lighting, cinematic composition, detailed face
                  Negative: ugly, blurry, low quality, deformed
                </p>
              </div>
              <p>
                <strong className="text-white">비용:</strong> 무료 (GPU 필요: RTX 3060 12GB 이상 권장)
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 2. 프롬프트 엔지니어링 기본 원칙 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-pink-500 to-rose-500 rounded-lg flex items-center justify-center">
            <Wand2 className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">프롬프트 엔지니어링 기본 원칙</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-purple-500/30 rounded-xl p-8">
          <h3 className="text-2xl font-bold text-purple-400 mb-6">효과적인 프롬프트 구조</h3>

          <div className="space-y-6">
            {/* 1. 주제 */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <h4 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                <span className="text-purple-400">1.</span> 주제 (Subject)
              </h4>
              <p className="text-gray-300 mb-3">무엇을 그릴 것인가?</p>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-green-400 font-mono text-sm mb-2">✓ 좋은 예시:</p>
                <p className="text-gray-300 mb-3">- "a majestic lion" (구체적 명사)</p>
                <p className="text-gray-300 mb-3">- "a young woman with long red hair" (세부 묘사)</p>
                <p className="text-red-400 font-mono text-sm mb-2">✗ 나쁜 예시:</p>
                <p className="text-gray-300">- "something cool" (모호함)</p>
              </div>
            </div>

            {/* 2. 스타일 */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <h4 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                <span className="text-pink-400">2.</span> 스타일 (Style)
              </h4>
              <p className="text-gray-300 mb-3">어떤 스타일로 그릴 것인가?</p>
              <div className="grid md:grid-cols-2 gap-4 mt-3">
                <div>
                  <p className="text-purple-400 font-semibold mb-2">예술 스타일:</p>
                  <ul className="list-disc list-inside text-gray-300 space-y-1 text-sm">
                    <li>realistic, photorealistic</li>
                    <li>oil painting, watercolor</li>
                    <li>anime, manga, cartoon</li>
                    <li>cyberpunk, steampunk</li>
                  </ul>
                </div>
                <div>
                  <p className="text-pink-400 font-semibold mb-2">유명 아티스트:</p>
                  <ul className="list-disc list-inside text-gray-300 space-y-1 text-sm">
                    <li>by Greg Rutkowski</li>
                    <li>by Artgerm</li>
                    <li>by Makoto Shinkai</li>
                    <li>in the style of Studio Ghibli</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* 3. 조명 */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <h4 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                <span className="text-rose-400">3.</span> 조명 (Lighting)
              </h4>
              <p className="text-gray-300 mb-3">빛의 방향과 분위기</p>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <ul className="list-disc list-inside text-gray-300 space-y-2">
                  <li><strong className="text-white">golden hour lighting</strong> - 황금빛 노을</li>
                  <li><strong className="text-white">dramatic lighting</strong> - 강렬한 명암</li>
                  <li><strong className="text-white">volumetric lighting</strong> - 빛줄기 효과</li>
                  <li><strong className="text-white">soft ambient lighting</strong> - 부드러운 전체 조명</li>
                  <li><strong className="text-white">cinematic lighting</strong> - 영화 같은 조명</li>
                </ul>
              </div>
            </div>

            {/* 4. 구도 */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <h4 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                <span className="text-purple-400">4.</span> 구도 (Composition)
              </h4>
              <p className="text-gray-300 mb-3">카메라 각도와 프레이밍</p>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <ul className="list-disc list-inside text-gray-300 space-y-2">
                  <li><strong className="text-white">close-up</strong> - 클로즈업 (얼굴)</li>
                  <li><strong className="text-white">portrait</strong> - 인물 중심</li>
                  <li><strong className="text-white">full body shot</strong> - 전신</li>
                  <li><strong className="text-white">aerial view</strong> - 공중에서 본 시점</li>
                  <li><strong className="text-white">wide angle</strong> - 광각 (넓은 풍경)</li>
                </ul>
              </div>
            </div>

            {/* 5. 품질 키워드 */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <h4 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                <span className="text-pink-400">5.</span> 품질 키워드 (Quality Tags)
              </h4>
              <p className="text-gray-300 mb-3">고품질 결과를 위한 필수 키워드</p>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-purple-400 font-mono mb-2">항상 추가:</p>
                <p className="text-gray-300">
                  masterpiece, best quality, ultra detailed, 8k, highly detailed,
                  professional, sharp focus, intricate details
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. Midjourney 파라미터 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-rose-500 to-orange-500 rounded-lg flex items-center justify-center">
            <Settings className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">Midjourney 파라미터</h2>
        </div>

        <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-8">
          <div className="space-y-6">
            {/* --ar (Aspect Ratio) */}
            <div className="border-b border-gray-700 pb-6">
              <h3 className="text-xl font-bold text-purple-400 mb-3">--ar (Aspect Ratio)</h3>
              <p className="text-gray-300 mb-4">이미지 비율 설정</p>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-pink-400 font-semibold mb-2">--ar 1:1</p>
                  <p className="text-sm text-gray-400">정사각형 (Instagram)</p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-pink-400 font-semibold mb-2">--ar 16:9</p>
                  <p className="text-sm text-gray-400">와이드스크린 (YouTube)</p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-pink-400 font-semibold mb-2">--ar 9:16</p>
                  <p className="text-sm text-gray-400">세로 (모바일, Reels)</p>
                </div>
              </div>
            </div>

            {/* --v (Version) */}
            <div className="border-b border-gray-700 pb-6">
              <h3 className="text-xl font-bold text-purple-400 mb-3">--v (Version)</h3>
              <p className="text-gray-300 mb-4">Midjourney 버전 선택</p>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <ul className="space-y-2 text-gray-300">
                  <li><strong className="text-white">--v 6</strong> - 최신 버전 (2024년 기준, 가장 높은 품질)</li>
                  <li><strong className="text-white">--v 5.2</strong> - 이전 안정 버전</li>
                  <li><strong className="text-white">--niji 6</strong> - 애니메이션 특화 버전</li>
                </ul>
              </div>
            </div>

            {/* --stylize */}
            <div className="border-b border-gray-700 pb-6">
              <h3 className="text-xl font-bold text-purple-400 mb-3">--stylize (--s)</h3>
              <p className="text-gray-300 mb-4">예술적 스타일 강도 (0-1000)</p>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <ul className="space-y-2 text-gray-300">
                  <li><strong className="text-white">--s 0</strong> - 프롬프트에 정확히 일치 (사실적)</li>
                  <li><strong className="text-white">--s 100</strong> - 기본값 (균형)</li>
                  <li><strong className="text-white">--s 1000</strong> - 매우 예술적 (창의적)</li>
                </ul>
              </div>
            </div>

            {/* --chaos */}
            <div>
              <h3 className="text-xl font-bold text-purple-400 mb-3">--chaos (--c)</h3>
              <p className="text-gray-300 mb-4">다양성 정도 (0-100)</p>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <ul className="space-y-2 text-gray-300">
                  <li><strong className="text-white">--c 0</strong> - 일관된 결과 (4개 이미지가 비슷)</li>
                  <li><strong className="text-white">--c 50</strong> - 적당한 다양성</li>
                  <li><strong className="text-white">--c 100</strong> - 매우 다양한 결과</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. 실전 프롬프트 예시 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-yellow-500 rounded-lg flex items-center justify-center">
            <Palette className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">실전 프롬프트 예시</h2>
        </div>

        <div className="space-y-6">
          {/* 인물 사진 */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-purple-400 mb-4">1. 인물 사진 (Portrait)</h3>
            <div className="bg-gray-900/50 border border-purple-500/30 rounded-lg p-4 mb-4">
              <p className="text-purple-300 font-mono text-sm leading-relaxed">
                a professional headshot of a confident business woman in her 30s,
                wearing a navy blue blazer, natural smile, studio lighting,
                soft bokeh background, shot on Canon EOS R5, 85mm f/1.4,
                sharp focus on eyes, professional photography, high resolution --ar 2:3 --v 6
              </p>
            </div>
            <p className="text-gray-400 text-sm">
              💡 <strong>핵심 요소:</strong> 나이/성별, 복장, 표정, 조명, 카메라 정보, 초점
            </p>
          </div>

          {/* 판타지 풍경 */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-pink-400 mb-4">2. 판타지 풍경 (Fantasy Landscape)</h3>
            <div className="bg-gray-900/50 border border-pink-500/30 rounded-lg p-4 mb-4">
              <p className="text-pink-300 font-mono text-sm leading-relaxed">
                a mystical floating island in the sky, waterfalls cascading down,
                glowing purple crystals, ancient ruins covered in vines,
                magical aurora in the background, volumetric god rays,
                epic fantasy scene, by Greg Rutkowski and Makoto Shinkai,
                trending on artstation, ultra detailed, 8k --ar 16:9 --v 6 --s 250
              </p>
            </div>
            <p className="text-gray-400 text-sm">
              💡 <strong>핵심 요소:</strong> 주요 오브젝트, 대기 효과, 아티스트 참조, 스타일 강도
            </p>
          </div>

          {/* 제품 사진 */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-rose-400 mb-4">3. 제품 사진 (Product Photography)</h3>
            <div className="bg-gray-900/50 border border-rose-500/30 rounded-lg p-4 mb-4">
              <p className="text-rose-300 font-mono text-sm leading-relaxed">
                premium wireless headphones on a minimalist white marble surface,
                soft reflections, studio product photography, dramatic side lighting,
                sleek modern design, black matte finish with gold accents,
                professional commercial photography, ultra sharp, 8k, Canon EOS R5 --ar 1:1 --v 6
              </p>
            </div>
            <p className="text-gray-400 text-sm">
              💡 <strong>핵심 요소:</strong> 제품 특징, 배경, 조명 방향, 재질 묘사
            </p>
          </div>
        </div>
      </section>

      {/* 5. 네거티브 프롬프트 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-red-500 to-pink-500 rounded-lg flex items-center justify-center">
            <AlertCircle className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">네거티브 프롬프트 (원하지 않는 요소)</h2>
        </div>

        <div className="bg-gradient-to-br from-red-900/20 to-pink-900/20 border border-red-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            Stable Diffusion에서는 "Negative prompt"로, Midjourney에서는 "--no" 파라미터로
            원하지 않는 요소를 명시할 수 있습니다.
          </p>

          <div className="space-y-6">
            {/* 일반적인 네거티브 프롬프트 */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <h3 className="text-lg font-bold text-red-400 mb-4">일반적으로 피해야 할 요소</h3>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-red-300 font-mono text-sm leading-relaxed">
                  ugly, blurry, low quality, distorted, deformed, bad anatomy,
                  extra limbs, missing limbs, watermark, signature, text,
                  low resolution, pixelated, artifacts, duplicate, cropped
                </p>
              </div>
            </div>

            {/* 인물 사진용 */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <h3 className="text-lg font-bold text-pink-400 mb-4">인물 사진 특화</h3>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-pink-300 font-mono text-sm leading-relaxed">
                  bad hands, mutated hands, extra fingers, missing fingers,
                  bad eyes, asymmetric eyes, long neck, bad proportions,
                  facial distortion, multiple heads, double image
                </p>
              </div>
            </div>

            {/* Midjourney --no 예시 */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <h3 className="text-lg font-bold text-purple-400 mb-4">Midjourney --no 파라미터</h3>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-purple-300 font-mono text-sm leading-relaxed">
                  /imagine prompt: beautiful sunset landscape --no people, buildings, text, watermark
                </p>
              </div>
              <p className="text-gray-400 text-sm mt-3">
                💡 여러 요소를 쉼표로 구분하여 한 번에 제외 가능
              </p>
            </div>
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
          {/* 공식 문서 */}
          <div className="bg-gray-800/50 border border-purple-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-purple-400 mb-4">📖 공식 문서 & 가이드</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://docs.midjourney.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Midjourney Documentation
                </a>
                <p className="text-sm text-gray-400 mt-1">Midjourney 공식 가이드 및 파라미터 설명</p>
              </li>
              <li>
                <a
                  href="https://platform.openai.com/docs/guides/images"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  DALL-E 3 API Documentation
                </a>
                <p className="text-sm text-gray-400 mt-1">OpenAI DALL-E 3 이미지 생성 가이드</p>
              </li>
              <li>
                <a
                  href="https://github.com/AUTOMATIC1111/stable-diffusion-webui"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Stable Diffusion Web UI (Automatic1111)
                </a>
                <p className="text-sm text-gray-400 mt-1">가장 인기 있는 Stable Diffusion 인터페이스</p>
              </li>
            </ul>
          </div>

          {/* 학습 리소스 */}
          <div className="bg-gray-800/50 border border-pink-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-pink-400 mb-4">🎓 학습 리소스</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://prompthero.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  PromptHero - AI 프롬프트 검색 엔진
                </a>
                <p className="text-sm text-gray-400 mt-1">100만 개 이상의 AI 생성 이미지와 프롬프트</p>
              </li>
              <li>
                <a
                  href="https://civitai.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Civitai - Stable Diffusion 모델 허브
                </a>
                <p className="text-sm text-gray-400 mt-1">커스텀 모델, LoRA, Embedding 공유 커뮤니티</p>
              </li>
              <li>
                <a
                  href="https://lexica.art/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Lexica - Stable Diffusion 프롬프트 데이터베이스
                </a>
                <p className="text-sm text-gray-400 mt-1">1천만 개 이상의 Stable Diffusion 이미지 검색</p>
              </li>
            </ul>
          </div>

          {/* 커뮤니티 & 도구 */}
          <div className="bg-gray-800/50 border border-rose-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-rose-400 mb-4">🛠️ 커뮤니티 & 도구</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://www.reddit.com/r/StableDiffusion/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  r/StableDiffusion - Reddit 커뮤니티
                </a>
                <p className="text-sm text-gray-400 mt-1">70만 멤버의 Stable Diffusion 커뮤니티</p>
              </li>
              <li>
                <a
                  href="https://www.reddit.com/r/midjourney/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  r/midjourney - Midjourney 커뮤니티
                </a>
                <p className="text-sm text-gray-400 mt-1">Midjourney 사용자들의 팁과 작품 공유</p>
              </li>
              <li>
                <a
                  href="https://huggingface.co/spaces/stabilityai/stable-diffusion"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  Hugging Face Stable Diffusion Demo
                </a>
                <p className="text-sm text-gray-400 mt-1">무료 온라인 Stable Diffusion 데모 (설치 불필요)</p>
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}
