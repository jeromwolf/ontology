'use client'

import React from 'react'
import { Layers, Sliders, Image, Zap, Download, Settings } from 'lucide-react'

export default function Chapter3() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      {/* Hero Section */}
      <div className="mb-12">
        <div className="inline-block px-4 py-2 bg-purple-500/10 border border-purple-500/20 rounded-full mb-4">
          <span className="text-purple-400 text-sm font-medium">Chapter 3</span>
        </div>
        <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-purple-400 via-pink-400 to-rose-400 bg-clip-text text-transparent">
          Stable Diffusion 마스터
        </h1>
        <p className="text-xl text-gray-300 leading-relaxed">
          로컬 환경에서 Stable Diffusion을 활용하는 완전 가이드. Automatic1111 Web UI 설치부터
          ControlNet, LoRA, DreamBooth까지 고급 기능을 모두 다룹니다.
        </p>
      </div>

      {/* 1. Stable Diffusion 로컬 설치 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Download className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">Stable Diffusion 로컬 설치</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-purple-500/30 rounded-xl p-8">
          {/* 시스템 요구사항 */}
          <div className="mb-8">
            <h3 className="text-2xl font-bold text-purple-400 mb-4">시스템 요구사항</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                <h4 className="text-lg font-bold text-white mb-3">최소 사양</h4>
                <ul className="space-y-2 text-gray-300">
                  <li><strong className="text-purple-400">GPU:</strong> NVIDIA RTX 3060 (12GB VRAM)</li>
                  <li><strong className="text-purple-400">RAM:</strong> 16GB</li>
                  <li><strong className="text-purple-400">저장공간:</strong> 50GB+ (모델 용량)</li>
                  <li><strong className="text-purple-400">OS:</strong> Windows 10/11, Linux</li>
                </ul>
              </div>
              <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
                <h4 className="text-lg font-bold text-white mb-3">권장 사양</h4>
                <ul className="space-y-2 text-gray-300">
                  <li><strong className="text-pink-400">GPU:</strong> NVIDIA RTX 4080 (16GB VRAM)</li>
                  <li><strong className="text-pink-400">RAM:</strong> 32GB</li>
                  <li><strong className="text-pink-400">저장공간:</strong> 200GB+ SSD</li>
                  <li><strong className="text-pink-400">OS:</strong> Windows 11, Ubuntu 22.04</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Automatic1111 설치 */}
          <div className="mb-8">
            <h3 className="text-2xl font-bold text-pink-400 mb-4">Automatic1111 Web UI 설치</h3>
            <p className="text-gray-300 mb-4">
              가장 인기 있는 Stable Diffusion 인터페이스. 풍부한 기능과 확장성이 특징입니다.
            </p>

            <div className="space-y-4">
              {/* Windows */}
              <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-6">
                <h4 className="text-lg font-bold text-purple-400 mb-3 flex items-center gap-2">
                  🪟 Windows 설치
                </h4>
                <div className="space-y-3">
                  <div className="bg-gray-900/50 rounded-lg p-4">
                    <p className="text-sm font-mono text-purple-300 mb-2"># 1. Git 설치 (git-scm.com)</p>
                    <p className="text-sm font-mono text-purple-300 mb-2"># 2. Python 3.10.6 설치 (python.org)</p>
                    <p className="text-sm font-mono text-purple-300 mb-2"># 3. CUDA Toolkit 설치 (nvidia.com)</p>
                  </div>
                  <div className="bg-gray-900/50 rounded-lg p-4">
                    <p className="text-sm font-mono text-green-400 mb-2"># 레포지토리 클론</p>
                    <p className="text-sm font-mono text-gray-300 mb-3">
                      git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
                    </p>
                    <p className="text-sm font-mono text-green-400 mb-2"># 실행</p>
                    <p className="text-sm font-mono text-gray-300">
                      cd stable-diffusion-webui<br/>
                      webui-user.bat
                    </p>
                  </div>
                </div>
              </div>

              {/* Linux */}
              <div className="bg-gray-800/50 border border-pink-500/30 rounded-lg p-6">
                <h4 className="text-lg font-bold text-pink-400 mb-3 flex items-center gap-2">
                  🐧 Linux 설치 (Ubuntu/Debian)
                </h4>
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <p className="text-sm font-mono text-gray-300">
                    sudo apt update<br/>
                    sudo apt install wget git python3 python3-venv<br/>
                    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git<br/>
                    cd stable-diffusion-webui<br/>
                    ./webui.sh
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-6 bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
              <p className="text-blue-300 flex items-start gap-2">
                <span className="text-xl">💡</span>
                <span>
                  첫 실행 시 필요한 패키지와 모델이 자동으로 다운로드됩니다 (10-20분 소요).
                  완료 후 브라우저에서 http://127.0.0.1:7860 접속
                </span>
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 2. Stable Diffusion 모델 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-pink-500 to-rose-500 rounded-lg flex items-center justify-center">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">Stable Diffusion 모델</h2>
        </div>

        <div className="space-y-6">
          {/* Base Models */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-purple-400 mb-4">Base Models</h3>
            <div className="space-y-4">
              <div className="bg-gray-900/50 border border-purple-500/30 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">SD 1.5 (Stable Diffusion v1.5)</h4>
                <ul className="text-gray-300 space-y-1 text-sm">
                  <li>• 가장 널리 사용되는 모델 (512×512 최적화)</li>
                  <li>• 풍부한 커뮤니티 리소스 (LoRA, Embedding)</li>
                  <li>• 빠른 생성 속도</li>
                  <li>• 다운로드: Hugging Face (4GB)</li>
                </ul>
              </div>

              <div className="bg-gray-900/50 border border-pink-500/30 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">SD 2.1 (Stable Diffusion v2.1)</h4>
                <ul className="text-gray-300 space-y-1 text-sm">
                  <li>• 768×768 해상도 지원</li>
                  <li>• 향상된 텍스트 이해력</li>
                  <li>• 더 엄격한 콘텐츠 필터</li>
                  <li>• 다운로드: Hugging Face (5GB)</li>
                </ul>
              </div>

              <div className="bg-gray-900/50 border border-rose-500/30 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">SDXL 1.0 (Stable Diffusion XL)</h4>
                <ul className="text-gray-300 space-y-1 text-sm">
                  <li>• 최신 모델 (1024×1024 네이티브)</li>
                  <li>• 가장 높은 품질</li>
                  <li>• 더 많은 VRAM 필요 (12GB+)</li>
                  <li>• 다운로드: Hugging Face (6.9GB)</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Community Models */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-pink-400 mb-4">Community Fine-tuned Models (Civitai)</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-900/50 border border-purple-500/30 rounded-lg p-4">
                <h4 className="font-bold text-purple-400 mb-2">Realistic Vision (사실적 인물)</h4>
                <p className="text-gray-300 text-sm">
                  초사실적 인물 사진에 최적화. 인물의 피부, 눈, 머리카락 디테일이 뛰어남
                </p>
              </div>
              <div className="bg-gray-900/50 border border-pink-500/30 rounded-lg p-4">
                <h4 className="font-bold text-pink-400 mb-2">DreamShaper (범용)</h4>
                <p className="text-gray-300 text-sm">
                  예술적이면서도 사실적. 다양한 스타일에 안정적으로 작동
                </p>
              </div>
              <div className="bg-gray-900/50 border border-rose-500/30 rounded-lg p-4">
                <h4 className="font-bold text-rose-400 mb-2">Anything V5 (애니메이션)</h4>
                <p className="text-gray-300 text-sm">
                  애니메이션/만화 스타일에 특화. 일본 애니메이션 스타일 최적화
                </p>
              </div>
              <div className="bg-gray-900/50 border border-orange-500/30 rounded-lg p-4">
                <h4 className="font-bold text-orange-400 mb-2">ChilloutMix (혼합)</h4>
                <p className="text-gray-300 text-sm">
                  사실적 + 일러스트 스타일 혼합. 아시아 인물에 강점
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. ControlNet */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-rose-500 to-orange-500 rounded-lg flex items-center justify-center">
            <Sliders className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">ControlNet - 정밀 제어</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-rose-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            ControlNet은 입력 이미지의 구도, 포즈, 선 등을 유지하면서 AI가 새로운 이미지를 생성하도록 합니다.
            "스케치를 사진으로", "포즈를 복사" 등이 가능합니다.
          </p>

          <div className="space-y-6">
            {/* ControlNet 모델 종류 */}
            <div>
              <h3 className="text-2xl font-bold text-rose-400 mb-4">주요 ControlNet 모델</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-800/50 border border-rose-500/30 rounded-lg p-4">
                  <h4 className="font-bold text-white mb-2">1. Canny (엣지 감지)</h4>
                  <p className="text-gray-300 text-sm mb-3">
                    입력 이미지의 윤곽선을 추출하여 구도 유지
                  </p>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <p className="text-xs text-gray-400">사용 사례:</p>
                    <p className="text-xs text-purple-300">건축물 스케치 → 실제 건물 렌더링</p>
                  </div>
                </div>

                <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-4">
                  <h4 className="font-bold text-white mb-2">2. OpenPose (포즈 감지)</h4>
                  <p className="text-gray-300 text-sm mb-3">
                    인물의 포즈(관절 위치)를 추출하여 동일 포즈로 재생성
                  </p>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <p className="text-xs text-gray-400">사용 사례:</p>
                    <p className="text-xs text-pink-300">댄스 포즈 복사, 캐릭터 포즈 변환</p>
                  </div>
                </div>

                <div className="bg-gray-800/50 border border-pink-500/30 rounded-lg p-4">
                  <h4 className="font-bold text-white mb-2">3. Depth (깊이 맵)</h4>
                  <p className="text-gray-300 text-sm mb-3">
                    이미지의 깊이 정보를 추출하여 3D 구조 유지
                  </p>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <p className="text-xs text-gray-400">사용 사례:</p>
                    <p className="text-xs text-rose-300">낮 풍경 → 밤 풍경 (구조 유지)</p>
                  </div>
                </div>

                <div className="bg-gray-800/50 border border-orange-500/30 rounded-lg p-4">
                  <h4 className="font-bold text-white mb-2">4. Scribble (스케치)</h4>
                  <p className="text-gray-300 text-sm mb-3">
                    간단한 낙서/스케치를 완성된 그림으로 변환
                  </p>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <p className="text-xs text-gray-400">사용 사례:</p>
                    <p className="text-xs text-orange-300">막대 인간 → 사실적 인물</p>
                  </div>
                </div>
              </div>
            </div>

            {/* ControlNet 설치 */}
            <div>
              <h3 className="text-2xl font-bold text-purple-400 mb-4">ControlNet 설치 (Automatic1111)</h3>
              <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-6">
                <ol className="space-y-3 text-gray-300">
                  <li>
                    <strong className="text-white">1.</strong> Web UI 실행 → "Extensions" 탭
                  </li>
                  <li>
                    <strong className="text-white">2.</strong> "Available" 탭 → "Load from:" 클릭
                  </li>
                  <li>
                    <strong className="text-white">3.</strong> "sd-webui-controlnet" 검색 → Install
                  </li>
                  <li>
                    <strong className="text-white">4.</strong> "Installed" 탭 → "Apply and restart UI"
                  </li>
                  <li>
                    <strong className="text-white">5.</strong> Hugging Face에서 ControlNet 모델 다운로드:
                    <div className="bg-gray-900/50 rounded-lg p-3 mt-2">
                      <p className="text-sm font-mono text-purple-300">
                        extensions/sd-webui-controlnet/models/
                      </p>
                      <p className="text-xs text-gray-400 mt-2">
                        • control_canny.pth (1.45GB)<br/>
                        • control_openpose.pth (1.45GB)<br/>
                        • control_depth.pth (1.45GB)
                      </p>
                    </div>
                  </li>
                </ol>
              </div>
            </div>

            {/* 실전 예시 */}
            <div>
              <h3 className="text-2xl font-bold text-pink-400 mb-4">실전 사용 예시</h3>
              <div className="bg-gray-800/50 border border-pink-500/30 rounded-lg p-6">
                <h4 className="font-bold text-white mb-3">시나리오: 스케치를 애니메이션 캐릭터로 변환</h4>
                <ol className="space-y-3 text-gray-300 text-sm">
                  <li>
                    <strong className="text-purple-400">1.</strong> txt2img 탭에서 ControlNet 섹션 펼치기
                  </li>
                  <li>
                    <strong className="text-pink-400">2.</strong> 입력 이미지 업로드 (간단한 스케치)
                  </li>
                  <li>
                    <strong className="text-rose-400">3.</strong> Preprocessor: "canny" 선택
                  </li>
                  <li>
                    <strong className="text-orange-400">4.</strong> Model: "control_canny" 선택
                  </li>
                  <li>
                    <strong className="text-purple-400">5.</strong> 프롬프트 입력:
                    <div className="bg-gray-900/50 rounded-lg p-3 mt-2">
                      <p className="text-purple-300 font-mono">
                        anime character, colorful hair, detailed face, studio lighting,
                        high quality, by Makoto Shinkai
                      </p>
                    </div>
                  </li>
                  <li>
                    <strong className="text-pink-400">6.</strong> Generate 클릭 → 스케치 구도를 유지한 완성작 생성!
                  </li>
                </ol>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Web UI 주요 설정 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-yellow-500 rounded-lg flex items-center justify-center">
            <Settings className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">Web UI 주요 설정</h2>
        </div>

        <div className="space-y-6">
          {/* Sampling Settings */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-purple-400 mb-4">Sampling 설정</h3>
            <div className="space-y-4">
              <div className="bg-gray-900/50 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">Sampling method (샘플링 방법)</h4>
                <p className="text-gray-300 text-sm mb-3">
                  이미지 생성 알고리즘 선택. 속도와 품질의 트레이드오프
                </p>
                <div className="grid md:grid-cols-2 gap-3">
                  <div>
                    <p className="text-purple-400 font-semibold text-sm">빠른 생성:</p>
                    <ul className="text-gray-400 text-xs space-y-1 mt-1">
                      <li>• Euler a (20-30 steps)</li>
                      <li>• DPM++ 2M Karras (20-25 steps)</li>
                    </ul>
                  </div>
                  <div>
                    <p className="text-pink-400 font-semibold text-sm">고품질:</p>
                    <ul className="text-gray-400 text-xs space-y-1 mt-1">
                      <li>• DPM++ SDE Karras (25-35 steps)</li>
                      <li>• UniPC (20-30 steps, 빠르면서 고품질)</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">Sampling steps (샘플링 스텝)</h4>
                <p className="text-gray-300 text-sm">
                  노이즈를 제거하는 반복 횟수. 높을수록 디테일 증가하지만 시간도 증가
                </p>
                <ul className="text-gray-400 text-sm space-y-1 mt-3">
                  <li>• 20-25: 빠른 테스트</li>
                  <li>• 30-40: 일반 사용 (권장)</li>
                  <li>• 50+: 최고 품질 (시간 오래 걸림)</li>
                </ul>
              </div>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">CFG Scale (Classifier Free Guidance)</h4>
                <p className="text-gray-300 text-sm">
                  프롬프트를 얼마나 강하게 따를지 결정 (1-30)
                </p>
                <ul className="text-gray-400 text-sm space-y-1 mt-3">
                  <li>• 5-7: 창의적, 프롬프트를 느슨하게 해석</li>
                  <li>• 7-12: 균형 (권장, 대부분 7-9 사용)</li>
                  <li>• 12+: 프롬프트에 엄격히 일치, 과포화 위험</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Hires Fix */}
          <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6">
            <h3 className="text-xl font-bold text-pink-400 mb-4">Hires. fix (고해상도 보정)</h3>
            <p className="text-gray-300 mb-4">
              512×512로 생성 후 업스케일하여 고해상도(1024×1024 등) 이미지를 깨끗하게 생성
            </p>
            <div className="bg-gray-900/50 rounded-lg p-4">
              <h4 className="font-bold text-white mb-2">사용법</h4>
              <ol className="text-gray-300 text-sm space-y-2">
                <li>1. "Hires. fix" 체크박스 활성화</li>
                <li>2. Upscaler: "Latent" 또는 "R-ESRGAN 4x+" 선택</li>
                <li>3. Upscale by: 2.0 (512 → 1024)</li>
                <li>4. Denoising strength: 0.5-0.7 (높을수록 디테일 추가)</li>
              </ol>
            </div>
          </div>
        </div>
      </section>

      {/* 5. 최적화 팁 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-yellow-500 to-green-500 rounded-lg flex items-center justify-center">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">성능 최적화 팁</h2>
        </div>

        <div className="space-y-6">
          <div className="bg-gray-800/50 border border-yellow-500/30 rounded-xl p-6">
            <h3 className="text-xl font-bold text-yellow-400 mb-4">VRAM 절약 방법</h3>
            <div className="space-y-3 text-gray-300">
              <div className="bg-gray-900/50 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">1. xformers 활성화</h4>
                <p className="text-sm mb-2">Settings → Optimizations → "Use xformers" 체크</p>
                <p className="text-xs text-gray-400">VRAM 사용량 30-50% 감소, 속도도 빨라짐</p>
              </div>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">2. --medvram / --lowvram 플래그</h4>
                <p className="text-sm mb-2">webui-user.bat 파일에 추가:</p>
                <div className="bg-black/30 rounded p-2 mt-2">
                  <p className="text-xs font-mono text-green-400">
                    set COMMANDLINE_ARGS=--medvram --xformers
                  </p>
                </div>
                <p className="text-xs text-gray-400 mt-2">
                  8GB VRAM: --medvram / 6GB VRAM: --lowvram
                </p>
              </div>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">3. 배치 크기 줄이기</h4>
                <p className="text-sm">
                  Batch size: 1 (한 번에 1장), Batch count: 4 (4번 반복)
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-800/50 border border-green-500/30 rounded-xl p-6">
            <h3 className="text-xl font-bold text-green-400 mb-4">속도 향상 방법</h3>
            <div className="space-y-3 text-gray-300">
              <div className="bg-gray-900/50 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">1. TensorRT 사용 (고급)</h4>
                <p className="text-sm">
                  NVIDIA GPU 전용 최적화. 2-3배 속도 향상 (별도 설치 필요)
                </p>
              </div>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">2. VAE 캐싱</h4>
                <p className="text-sm">
                  Settings → Stable Diffusion → "Cache VAE" 체크
                </p>
              </div>

              <div className="bg-gray-900/50 rounded-lg p-4">
                <h4 className="font-bold text-white mb-2">3. 해상도 낮추기</h4>
                <p className="text-sm">
                  512×512로 생성 후 Hires. fix로 업스케일 (처음부터 1024보다 빠름)
                </p>
              </div>
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
          <div className="bg-gray-800/50 border border-purple-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-purple-400 mb-4">📖 공식 문서 & 설치 가이드</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://github.com/AUTOMATIC1111/stable-diffusion-webui"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Automatic1111 Web UI GitHub
                </a>
                <p className="text-sm text-gray-400 mt-1">가장 인기 있는 SD Web UI 공식 저장소</p>
              </li>
              <li>
                <a
                  href="https://github.com/lllyasviel/ControlNet"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  ControlNet Official Repository
                </a>
                <p className="text-sm text-gray-400 mt-1">ControlNet 원본 논문 및 모델</p>
              </li>
              <li>
                <a
                  href="https://github.com/Mikubill/sd-webui-controlnet"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  ControlNet Extension for Automatic1111
                </a>
                <p className="text-sm text-gray-400 mt-1">Web UI용 ControlNet 확장 프로그램</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-pink-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-pink-400 mb-4">🎨 모델 & 리소스</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://civitai.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Civitai - Stable Diffusion 모델 허브
                </a>
                <p className="text-sm text-gray-400 mt-1">커뮤니티 파인튜닝 모델, LoRA, Embedding</p>
              </li>
              <li>
                <a
                  href="https://huggingface.co/models?pipeline_tag=text-to-image"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Hugging Face - Text-to-Image Models
                </a>
                <p className="text-sm text-gray-400 mt-1">공식 Stable Diffusion 모델 다운로드</p>
              </li>
              <li>
                <a
                  href="https://openmodeldb.info/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  OpenModelDB - Upscaler Models
                </a>
                <p className="text-sm text-gray-400 mt-1">고해상도 업스케일 모델 데이터베이스</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-rose-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-rose-400 mb-4">🛠️ 유틸리티 & 튜토리얼</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://github.com/comfyanonymous/ComfyUI"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  ComfyUI - 노드 기반 Stable Diffusion
                </a>
                <p className="text-sm text-gray-400 mt-1">고급 사용자를 위한 노드 기반 인터페이스</p>
              </li>
              <li>
                <a
                  href="https://www.reddit.com/r/StableDiffusion/wiki/index/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  r/StableDiffusion Wiki
                </a>
                <p className="text-sm text-gray-400 mt-1">커뮤니티 가이드, 튜토리얼, FAQ</p>
              </li>
              <li>
                <a
                  href="https://invoke-ai.github.io/InvokeAI/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  InvokeAI - 프로페셔널 SD UI
                </a>
                <p className="text-sm text-gray-400 mt-1">전문가를 위한 고급 인터페이스</p>
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}
