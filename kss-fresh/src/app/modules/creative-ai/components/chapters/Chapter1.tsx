'use client';

import React from 'react';
import { Sparkles, Image, Music, Video, Palette } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      {/* 크리에이티브 AI 소개 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          크리에이티브 AI의 혁명
        </h2>
        <div className="prose prose-lg dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            생성형 AI(Generative AI)는 <strong>인간의 창의성을 증폭시키는 도구</strong>로
            디자인, 예술, 음악, 영상 제작 분야에 혁명을 일으키고 있습니다.
            2022년 Midjourney, DALL-E, Stable Diffusion의 등장 이후 누구나 텍스트 입력만으로
            전문가 수준의 시각 콘텐츠를 생성할 수 있게 되었습니다.
          </p>
        </div>
      </section>

      {/* 주요 크리에이티브 AI 도구 */}
      <section className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-purple-600 dark:text-purple-400" />
          주요 크리에이티브 AI 플랫폼
        </h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Image className="w-8 h-8 text-blue-600 dark:text-blue-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">이미지 생성</h4>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• Midjourney v6 (Discord)</li>
              <li>• DALL-E 3 (OpenAI)</li>
              <li>• Stable Diffusion XL</li>
              <li>• Adobe Firefly</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Music className="w-8 h-8 text-green-600 dark:text-green-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">음악 생성</h4>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• Suno AI (가사 + 음악)</li>
              <li>• Udio (스타일 변환)</li>
              <li>• MusicGen (Meta)</li>
              <li>• Stable Audio</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <Video className="w-8 h-8 text-red-600 dark:text-red-400 mb-2" />
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">비디오 생성</h4>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• Runway Gen-2</li>
              <li>• Pika Labs</li>
              <li>• Stable Video Diffusion</li>
              <li>• Synthesia (아바타)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 생성형 AI의 핵심 기술 */}
      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          생성형 AI의 작동 원리
        </h3>
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">1. Diffusion Models (확산 모델)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              노이즈를 점진적으로 제거하여 이미지를 생성하는 방식. Stable Diffusion, DALL-E 3의 핵심 기술.
            </p>
          </div>
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">2. Transformer Architecture</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              텍스트 프롬프트를 이해하고 이미지로 변환하는 인코더-디코더 구조. GPT와 동일한 아키텍처 활용.
            </p>
          </div>
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">3. CLIP (Contrastive Learning)</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              텍스트와 이미지를 동일한 임베딩 공간에 매핑하여 의미적 유사성을 계산. 프롬프트 이해의 핵심.
            </p>
          </div>
        </div>
      </section>

      {/* 실전 활용 사례 */}
      <section className="bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
          <Palette className="w-6 h-6 text-purple-600 dark:text-purple-400" />
          💡 실전 활용 사례
        </h3>
        <div className="space-y-3">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">🎨 Midjourney로 브랜드 디자인</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              스타트업 로고, 제품 패키징, 마케팅 비주얼을 30분만에 수백 개의 시안 생성
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">🎵 Suno AI로 광고 음악 제작</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              "신나는 여름 광고 음악" 프롬프트만으로 15초 만에 완성곡 생성
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">🎬 Runway로 프로덕트 광고 영상</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              정적 이미지를 4초 동영상으로 자동 변환, 후반 작업 시간 90% 단축
            </p>
          </div>
        </div>
      </section>

      {/* 크리에이티브 AI의 한계와 미래 */}
      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          한계와 극복 방법
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">🚨 현재 한계</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 손, 발 등 세부 해부학 오류</li>
              <li>• 텍스트 렌더링 품질 부족</li>
              <li>• 일관성 있는 캐릭터 생성 어려움</li>
              <li>• 저작권 및 윤리 문제</li>
            </ul>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">✅ 극복 전략</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• Inpainting으로 세부 수정</li>
              <li>• ControlNet으로 포즈 제어</li>
              <li>• LoRA 파인튜닝으로 일관성 확보</li>
              <li>• Adobe Firefly 등 상업용 라이선스 모델 사용</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 핵심 플랫폼 문서',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Midjourney Documentation',
                authors: 'Midjourney Inc.',
                year: '2025',
                description: '가장 인기 있는 이미지 생성 AI 완전 가이드',
                link: 'https://docs.midjourney.com/'
              },
              {
                title: 'DALL-E 3 Guide',
                authors: 'OpenAI',
                year: '2024',
                description: 'ChatGPT 통합 이미지 생성 플랫폼',
                link: 'https://platform.openai.com/docs/guides/images'
              },
              {
                title: 'Stable Diffusion Documentation',
                authors: 'Stability AI',
                year: '2024',
                description: '오픈소스 이미지 생성 모델 완전 정복',
                link: 'https://stability.ai/stable-diffusion'
              },
              {
                title: 'Suno AI Documentation',
                authors: 'Suno',
                year: '2024',
                description: 'AI 음악 생성 플랫폼 사용 가이드',
                link: 'https://suno.com/docs'
              }
            ]
          },
          {
            title: '📖 핵심 연구 논문',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'High-Resolution Image Synthesis with Latent Diffusion Models',
                authors: 'Rombach, R., Blattmann, A., et al.',
                year: '2022',
                description: 'Stable Diffusion의 원조 논문',
                link: 'https://arxiv.org/abs/2112.10752'
              },
              {
                title: 'DALL-E: Creating Images from Text',
                authors: 'Ramesh, A., Dhariwal, P., et al.',
                year: '2021',
                description: 'OpenAI의 텍스트-이미지 변환 혁신',
                link: 'https://arxiv.org/abs/2102.12092'
              },
              {
                title: 'Learning Transferable Visual Models From Natural Language Supervision',
                authors: 'Radford, A., Kim, J. W., et al.',
                year: '2021',
                description: 'CLIP 모델 - 텍스트와 이미지 이해의 핵심',
                link: 'https://arxiv.org/abs/2103.00020'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 커뮤니티',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Civitai - AI Art Community',
                authors: 'Civitai',
                year: '2024',
                description: '30만+ 커스텀 모델 & 프롬프트 공유',
                link: 'https://civitai.com/'
              },
              {
                title: 'Hugging Face Spaces',
                authors: 'Hugging Face',
                year: '2024',
                description: '무료 AI 모델 체험 공간',
                link: 'https://huggingface.co/spaces'
              },
              {
                title: 'Automatic1111 WebUI',
                authors: 'AUTOMATIC1111',
                year: '2024',
                description: 'Stable Diffusion 로컬 실행 GUI',
                link: 'https://github.com/AUTOMATIC1111/stable-diffusion-webui'
              },
              {
                title: 'Prompt Engineering Guide',
                authors: 'DAIR.AI',
                year: '2024',
                description: '프롬프트 작성 베스트 프랙티스',
                link: 'https://www.promptingguide.ai/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
