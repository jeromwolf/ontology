'use client'

import React from 'react'
import { Film, Video, Zap, Sparkles, Wand2, PlayCircle } from 'lucide-react'

export default function Chapter6() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      {/* Hero Section */}
      <div className="mb-12">
        <div className="inline-block px-4 py-2 bg-purple-500/10 border border-purple-500/20 rounded-full mb-4">
          <span className="text-purple-400 text-sm font-medium">Chapter 6</span>
        </div>
        <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-purple-400 via-pink-400 to-rose-400 bg-clip-text text-transparent">
          AI 비디오 제작
        </h1>
        <p className="text-xl text-gray-300 leading-relaxed">
          Runway ML, Pika Labs, Stable Video Diffusion 등으로 텍스트나 이미지를 비디오로 변환합니다.
          AI 비디오 생성의 현재와 미래, 그리고 실전 활용 방법을 다룹니다.
        </p>
      </div>

      {/* 1. AI 비디오 생성 도구 비교 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Video className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">AI 비디오 생성 도구 비교</h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
            <thead className="bg-gradient-to-r from-purple-500/20 to-pink-500/20">
              <tr>
                <th className="px-6 py-4 text-left text-white font-bold">도구</th>
                <th className="px-6 py-4 text-left text-white font-bold">입력</th>
                <th className="px-6 py-4 text-left text-white font-bold">출력</th>
                <th className="px-6 py-4 text-left text-white font-bold">가격</th>
                <th className="px-6 py-4 text-left text-white font-bold">특징</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">🎬</span>
                    <span className="font-bold text-purple-400">Runway ML<br/>Gen-2/Gen-3</span>
                  </div>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  텍스트, 이미지<br/>
                  비디오-비디오
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  최대 18초<br/>
                  720p, 4K
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  무료: 125 크레딧<br/>
                  Standard: $15/월<br/>
                  Pro: $35/월
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  가장 높은 품질<br/>
                  전문가급 도구
                </td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">⚡</span>
                    <span className="font-bold text-pink-400">Pika Labs<br/>Pika 1.0</span>
                  </div>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  텍스트, 이미지<br/>
                  비디오 확장
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  최대 3초<br/>
                  (무제한 확장)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  무료: 제한적<br/>
                  Standard: $10/월<br/>
                  Pro: $35/월
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  빠른 생성<br/>
                  간편한 UI
                </td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">🎥</span>
                    <span className="font-bold text-rose-400">Stable Video<br/>Diffusion</span>
                  </div>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  이미지만<br/>
                  (텍스트 ×)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  25 프레임<br/>
                  576×1024
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  무료 (오픈소스)<br/>
                  로컬 실행
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  오픈소스<br/>
                  커스터마이징 가능
                </td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">🤖</span>
                    <span className="font-bold text-orange-400">OpenAI<br/>Sora</span>
                  </div>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  텍스트<br/>
                  (비공개)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  최대 1분<br/>
                  1080p
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  미출시<br/>
                  (2024 Q4 예정)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  최고 품질<br/>
                  긴 길이
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-6 bg-blue-900/20 border border-blue-500/30 rounded-xl p-6">
          <h3 className="text-lg font-bold text-blue-400 mb-3 flex items-center gap-2">
            💡 선택 가이드
          </h3>
          <ul className="space-y-2 text-gray-300">
            <li>• <strong className="text-purple-400">Runway ML:</strong> 최고 품질, 전문가용, 광고/영화 제작</li>
            <li>• <strong className="text-pink-400">Pika Labs:</strong> 빠른 생성, 소셜 미디어 콘텐츠</li>
            <li>• <strong className="text-rose-400">Stable Video:</strong> 무료, 실험용, 개인 프로젝트</li>
            <li>• <strong className="text-orange-400">Sora:</strong> 아직 출시 안됨 (가장 기대되는 도구)</li>
          </ul>
        </div>
      </section>

      {/* 2. Runway ML Gen-3 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-pink-500 to-rose-500 rounded-lg flex items-center justify-center">
            <Film className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">Runway ML Gen-3 - 프로페셔널 비디오 생성</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-pink-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            Runway ML의 Gen-3는 현재 시장에서 가장 높은 품질의 AI 비디오를 생성합니다.
            할리우드 스튜디오에서도 사용하는 전문가급 도구입니다.
          </p>

          <div className="space-y-6">
            {/* 주요 기능 */}
            <div className="bg-gray-800/50 border border-pink-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-pink-400 mb-4">주요 기능</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-purple-400 mb-2">1. Text to Video</h4>
                  <p className="text-gray-300 text-sm">
                    텍스트 프롬프트만으로 비디오 생성 (최대 18초)
                  </p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-pink-400 mb-2">2. Image to Video</h4>
                  <p className="text-gray-300 text-sm">
                    정적 이미지에 움직임 추가 (첫 프레임 고정)
                  </p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-rose-400 mb-2">3. Video to Video</h4>
                  <p className="text-gray-300 text-sm">
                    기존 비디오의 스타일 변환 (실사 → 애니메이션)
                  </p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-orange-400 mb-2">4. Motion Brush</h4>
                  <p className="text-gray-300 text-sm">
                    특정 영역에만 움직임 지정 (예: 머리카락만 날림)
                  </p>
                </div>
              </div>
            </div>

            {/* 프롬프트 작성법 */}
            <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-purple-400 mb-4">효과적인 프롬프트 작성법</h3>

              <div className="space-y-4">
                <div>
                  <h4 className="font-bold text-white mb-2">프롬프트 구조</h4>
                  <div className="bg-gray-900/50 rounded-lg p-4">
                    <p className="text-sm text-gray-300 mb-3">
                      [주제] + [동작] + [카메라 움직임] + [조명/분위기] + [스타일]
                    </p>
                    <div className="bg-black/30 rounded p-3">
                      <p className="text-xs text-purple-400 mb-2"># 좋은 예시:</p>
                      <p className="text-xs font-mono text-gray-300 leading-relaxed">
                        A majestic eagle soaring through a stormy sky,
                        dynamic camera following the bird,
                        dramatic lightning in the background,
                        cinematic lighting with volumetric fog,
                        shot on RED camera, 4K, ultra detailed
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-bold text-white mb-2">카메라 움직임 키워드</h4>
                  <div className="bg-gray-900/50 rounded-lg p-4">
                    <div className="grid md:grid-cols-2 gap-3">
                      <ul className="text-gray-300 text-sm space-y-1">
                        <li>• <strong className="text-purple-400">Static shot:</strong> 고정</li>
                        <li>• <strong className="text-pink-400">Slow zoom in:</strong> 천천히 줌인</li>
                        <li>• <strong className="text-rose-400">Panning left to right:</strong> 좌우 패닝</li>
                      </ul>
                      <ul className="text-gray-300 text-sm space-y-1">
                        <li>• <strong className="text-orange-400">Orbit around subject:</strong> 피사체 주위 회전</li>
                        <li>• <strong className="text-purple-400">Dolly forward:</strong> 전진 이동</li>
                        <li>• <strong className="text-pink-400">Crane shot up:</strong> 크레인 상승</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* 실전 예시 */}
            <div className="bg-gray-800/50 border border-rose-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-rose-400 mb-4">실전 프롬프트 예시</h3>
              <div className="space-y-4">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-purple-400 mb-2">1. 제품 광고 (Product Demo)</h4>
                  <div className="bg-black/30 rounded p-3">
                    <p className="text-xs font-mono text-gray-300 leading-relaxed">
                      A sleek smartphone rotating on a minimalist white background,
                      slow 360-degree rotation,
                      studio lighting with soft reflections,
                      premium commercial photography style,
                      shot on Canon EOS R5, 8K resolution
                    </p>
                  </div>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-pink-400 mb-2">2. 풍경 (Landscape)</h4>
                  <div className="bg-black/30 rounded p-3">
                    <p className="text-xs font-mono text-gray-300 leading-relaxed">
                      A serene mountain lake at sunrise,
                      mist gently rolling across the water surface,
                      slow drone shot ascending from ground level,
                      golden hour lighting with god rays,
                      cinematic nature documentary style
                    </p>
                  </div>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-rose-400 mb-2">3. 추상/아트 (Abstract Art)</h4>
                  <div className="bg-black/30 rounded p-3">
                    <p className="text-xs font-mono text-gray-300 leading-relaxed">
                      Colorful ink drops dispersing in water,
                      slow motion capture at 240fps,
                      macro lens close-up,
                      black background with dramatic side lighting,
                      abstract fluid art style
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. Pika Labs */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-rose-500 to-orange-500 rounded-lg flex items-center justify-center">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">Pika Labs - 빠른 비디오 생성</h2>
        </div>

        <div className="bg-gradient-to-br from-rose-900/20 to-orange-900/20 border border-rose-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            Pika Labs는 3초 짧은 클립을 빠르게 생성하고, 무제한 확장할 수 있습니다.
            소셜 미디어 콘텐츠, 밈, 짧은 광고에 최적화되어 있습니다.
          </p>

          <div className="space-y-6">
            {/* 주요 기능 */}
            <div className="bg-gray-800/50 border border-rose-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-rose-400 mb-4">Pika 1.0 주요 기능</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-purple-400 mb-2">1. 3초 클립 생성</h4>
                  <p className="text-gray-300 text-sm">
                    빠른 생성 (30초-1분), 무제한 Extend 가능
                  </p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-pink-400 mb-2">2. Modify 기능</h4>
                  <p className="text-gray-300 text-sm">
                    생성된 비디오의 특정 요소 수정 (색상, 스타일 등)
                  </p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-rose-400 mb-2">3. Expand Canvas</h4>
                  <p className="text-gray-300 text-sm">
                    16:9 → 1:1 → 9:16 비율 변환 가능
                  </p>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-orange-400 mb-2">4. Camera Controls</h4>
                  <p className="text-gray-300 text-sm">
                    Zoom, Pan, Rotate 등 카메라 움직임 직접 제어
                  </p>
                </div>
              </div>
            </div>

            {/* 사용법 */}
            <div className="bg-gray-800/50 border border-orange-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-orange-400 mb-4">Discord 명령어</h3>
              <p className="text-gray-300 text-sm mb-4">
                Pika는 Discord 기반 서비스입니다. 명령어로 모든 기능을 제어합니다.
              </p>

              <div className="space-y-3">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-sm font-mono text-purple-400 mb-2">/create</p>
                  <p className="text-xs text-gray-300">
                    새 비디오 생성. 프롬프트 입력 또는 이미지 업로드
                  </p>
                  <div className="bg-black/30 rounded p-2 mt-2">
                    <p className="text-xs font-mono text-gray-300">
                      /create prompt: a cat dancing on a disco floor
                    </p>
                  </div>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-sm font-mono text-pink-400 mb-2">-camera</p>
                  <p className="text-xs text-gray-300">
                    카메라 움직임 지정 (zoom, pan, rotate)
                  </p>
                  <div className="bg-black/30 rounded p-2 mt-2">
                    <p className="text-xs font-mono text-gray-300">
                      -camera zoom in
                    </p>
                  </div>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-sm font-mono text-rose-400 mb-2">-motion</p>
                  <p className="text-xs text-gray-300">
                    움직임 강도 (1-4, 높을수록 빠른 움직임)
                  </p>
                  <div className="bg-black/30 rounded p-2 mt-2">
                    <p className="text-xs font-mono text-gray-300">
                      -motion 3
                    </p>
                  </div>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-sm font-mono text-orange-400 mb-2">-fps</p>
                  <p className="text-xs text-gray-300">
                    프레임레이트 (8, 16, 24)
                  </p>
                  <div className="bg-black/30 rounded p-2 mt-2">
                    <p className="text-xs font-mono text-gray-300">
                      -fps 24
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Stable Video Diffusion */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-yellow-500 rounded-lg flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">Stable Video Diffusion - 오픈소스</h2>
        </div>

        <div className="bg-gradient-to-br from-orange-900/20 to-yellow-900/20 border border-orange-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            Stability AI의 오픈소스 비디오 생성 모델. 이미지에서 25프레임 짧은 비디오를 생성합니다.
            완전 무료이지만, 현재 텍스트-비디오는 미지원입니다.
          </p>

          <div className="space-y-6">
            {/* Hugging Face 데모 */}
            <div className="bg-gray-800/50 border border-orange-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-orange-400 mb-4">1. Hugging Face 데모 (무료)</h3>
              <p className="text-gray-300 text-sm mb-4">
                설치 없이 브라우저에서 바로 사용 가능
              </p>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-xs text-purple-400 mb-2">접속 URL:</p>
                <a
                  href="https://huggingface.co/spaces/stabilityai/stable-video-diffusion"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-green-400 hover:text-green-300 text-sm underline"
                >
                  https://huggingface.co/spaces/stabilityai/stable-video-diffusion
                </a>

                <div className="mt-4 space-y-2 text-xs text-gray-300">
                  <p><strong className="text-white">사용법:</strong></p>
                  <ol className="list-decimal list-inside space-y-1 ml-2">
                    <li>정적 이미지 업로드 (512×512 권장)</li>
                    <li>Motion Bucket ID: 127 (움직임 강도, 1-255)</li>
                    <li>"Generate Video" 클릭</li>
                    <li>25프레임 MP4 다운로드</li>
                  </ol>
                </div>
              </div>
            </div>

            {/* 로컬 실행 */}
            <div className="bg-gray-800/50 border border-yellow-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-yellow-400 mb-4">2. 로컬 실행 (ComfyUI)</h3>
              <p className="text-gray-300 text-sm mb-4">
                ComfyUI에서 Stable Video Diffusion 노드 사용
              </p>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-xs text-purple-400 mb-2"># 설치 단계:</p>
                <ol className="text-xs text-gray-300 space-y-2">
                  <li>
                    <strong className="text-white">1.</strong> ComfyUI 설치 (Python 3.10+)
                    <div className="bg-black/30 rounded p-2 mt-1">
                      <p className="font-mono">
                        git clone https://github.com/comfyanonymous/ComfyUI.git
                      </p>
                    </div>
                  </li>
                  <li>
                    <strong className="text-white">2.</strong> SVD 모델 다운로드 (Hugging Face)
                    <div className="bg-black/30 rounded p-2 mt-1">
                      <p className="font-mono">
                        models/checkpoints/svd.safetensors (3.8GB)
                      </p>
                    </div>
                  </li>
                  <li>
                    <strong className="text-white">3.</strong> ComfyUI 실행 후 SVD 워크플로우 로드
                  </li>
                </ol>
              </div>

              <div className="mt-4 bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
                <p className="text-blue-300 text-sm">
                  💡 <strong>장점:</strong> 완전한 제어, 무제한 생성, 프라이버시<br/>
                  <strong>단점:</strong> 고성능 GPU 필요 (RTX 4090 권장)
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. 비디오 편집 & 후처리 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-yellow-500 to-green-500 rounded-lg flex items-center justify-center">
            <Wand2 className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">AI 비디오 편집 & 후처리</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Runway ML 편집 도구 */}
          <div className="bg-gray-800/50 border border-purple-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-purple-400 mb-4">Runway ML 편집 도구</h3>
            <ul className="space-y-3 text-gray-300 text-sm">
              <li>
                <strong className="text-white">Inpainting:</strong> 비디오 내 특정 오브젝트 제거/교체
              </li>
              <li>
                <strong className="text-white">Green Screen:</strong> 자동 배경 제거
              </li>
              <li>
                <strong className="text-white">Super-Slow Motion:</strong> 프레임 보간으로 슬로우 모션
              </li>
              <li>
                <strong className="text-white">Frame Interpolation:</strong> 프레임 2배, 4배 증가 (24fps → 96fps)
              </li>
              <li>
                <strong className="text-white">Upscaling:</strong> 720p → 4K AI 업스케일
              </li>
            </ul>
          </div>

          {/* 무료 도구 */}
          <div className="bg-gray-800/50 border border-pink-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-pink-400 mb-4">무료 편집 도구</h3>
            <ul className="space-y-3 text-gray-300 text-sm">
              <li>
                <strong className="text-white">Topaz Video AI:</strong> AI 업스케일 & 노이즈 제거 (유료 $299)
              </li>
              <li>
                <strong className="text-white">DaVinci Resolve:</strong> 전문가급 무료 편집 (AI 색보정)
              </li>
              <li>
                <strong className="text-white">CapCut:</strong> 무료 모바일 편집 (AI 자막, 음성 제거)
              </li>
              <li>
                <strong className="text-white">Descript:</strong> 텍스트 기반 비디오 편집 (AI 음성 클론)
              </li>
            </ul>
          </div>

          {/* 최적화 팁 */}
          <div className="bg-gray-800/50 border border-rose-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-rose-400 mb-4">AI 비디오 품질 개선 팁</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• <strong className="text-white">첫 프레임 고정:</strong> Image-to-Video로 일관성 확보</li>
              <li>• <strong className="text-white">짧게 생성 후 연결:</strong> 3초씩 생성 후 Premiere Pro 연결</li>
              <li>• <strong className="text-white">프레임 보간:</strong> 24fps → 60fps로 부드럽게</li>
              <li>• <strong className="text-white">색보정:</strong> DaVinci Resolve로 일관된 색감</li>
            </ul>
          </div>

          {/* 워크플로우 */}
          <div className="bg-gray-800/50 border border-orange-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-orange-400 mb-4">완성된 비디오 워크플로우</h3>
            <ol className="space-y-2 text-gray-300 text-sm list-decimal list-inside">
              <li>Midjourney로 키프레임 이미지 생성</li>
              <li>Runway ML Image-to-Video (각 3초)</li>
              <li>Premiere Pro / DaVinci Resolve 연결 편집</li>
              <li>Suno AI로 배경음악 생성</li>
              <li>CapCut으로 자막 자동 생성</li>
              <li>Topaz Video AI로 4K 업스케일</li>
            </ol>
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
            <h3 className="text-lg font-bold text-purple-400 mb-4">🎬 AI 비디오 생성 플랫폼</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://runwayml.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Runway ML - AI Video Generation
                </a>
                <p className="text-sm text-gray-400 mt-1">가장 높은 품질의 AI 비디오 생성</p>
              </li>
              <li>
                <a
                  href="https://pika.art/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Pika Labs - Fast Video Creation
                </a>
                <p className="text-sm text-gray-400 mt-1">빠른 3초 클립 생성 및 확장</p>
              </li>
              <li>
                <a
                  href="https://huggingface.co/spaces/stabilityai/stable-video-diffusion"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Stable Video Diffusion Demo
                </a>
                <p className="text-sm text-gray-400 mt-1">무료 오픈소스 비디오 생성</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-pink-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-pink-400 mb-4">✂️ 비디오 편집 도구</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://www.blackmagicdesign.com/products/davinciresolve"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  DaVinci Resolve - Free Professional Editing
                </a>
                <p className="text-sm text-gray-400 mt-1">무료 전문가급 편집 프로그램</p>
              </li>
              <li>
                <a
                  href="https://www.topazlabs.com/topaz-video-ai"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Topaz Video AI - AI Upscaling
                </a>
                <p className="text-sm text-gray-400 mt-1">AI 기반 비디오 업스케일 및 개선</p>
              </li>
              <li>
                <a
                  href="https://www.capcut.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  CapCut - Free Mobile/Desktop Editing
                </a>
                <p className="text-sm text-gray-400 mt-1">무료 AI 자막 및 간편 편집</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-rose-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-rose-400 mb-4">📖 학습 리소스</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://www.youtube.com/results?search_query=runway+ml+tutorial"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  Runway ML Tutorials - YouTube
                </a>
                <p className="text-sm text-gray-400 mt-1">Runway ML 사용법 영상 튜토리얼</p>
              </li>
              <li>
                <a
                  href="https://www.reddit.com/r/RunwayML/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  r/RunwayML - Reddit Community
                </a>
                <p className="text-sm text-gray-400 mt-1">Runway ML 사용자 커뮤니티</p>
              </li>
              <li>
                <a
                  href="https://github.com/Stability-AI/generative-models"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  Stable Video Diffusion GitHub
                </a>
                <p className="text-sm text-gray-400 mt-1">공식 오픈소스 저장소 및 문서</p>
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}
