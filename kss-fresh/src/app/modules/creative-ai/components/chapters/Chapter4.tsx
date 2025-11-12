'use client'

import React from 'react'
import { Wand2, Users, Upload, Zap, Image, Brain } from 'lucide-react'

export default function Chapter4() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      {/* Hero Section */}
      <div className="mb-12">
        <div className="inline-block px-4 py-2 bg-purple-500/10 border border-purple-500/20 rounded-full mb-4">
          <span className="text-purple-400 text-sm font-medium">Chapter 4</span>
        </div>
        <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-purple-400 via-pink-400 to-rose-400 bg-clip-text text-transparent">
          스타일 전이와 파인튜닝
        </h1>
        <p className="text-xl text-gray-300 leading-relaxed">
          LoRA, DreamBooth, Textual Inversion 등으로 AI 모델을 커스터마이징합니다.
          특정 스타일, 인물, 오브젝트를 학습시켜 나만의 AI 모델을 만드는 방법을 배웁니다.
        </p>
      </div>

      {/* 1. 파인튜닝 기법 비교 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">파인튜닝 기법 비교</h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
            <thead className="bg-gradient-to-r from-purple-500/20 to-pink-500/20">
              <tr>
                <th className="px-6 py-4 text-left text-white font-bold">기법</th>
                <th className="px-6 py-4 text-left text-white font-bold">파일 크기</th>
                <th className="px-6 py-4 text-left text-white font-bold">학습 시간</th>
                <th className="px-6 py-4 text-left text-white font-bold">이미지 수</th>
                <th className="px-6 py-4 text-left text-white font-bold">용도</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <span className="font-bold text-purple-400">LoRA</span>
                </td>
                <td className="px-6 py-4 text-gray-300">10-200MB</td>
                <td className="px-6 py-4 text-gray-300">30분-2시간</td>
                <td className="px-6 py-4 text-gray-300">15-50장</td>
                <td className="px-6 py-4 text-gray-300">스타일, 캐릭터, 의상</td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <span className="font-bold text-pink-400">DreamBooth</span>
                </td>
                <td className="px-6 py-4 text-gray-300">2-7GB (전체 모델)</td>
                <td className="px-6 py-4 text-gray-300">1-4시간</td>
                <td className="px-6 py-4 text-gray-300">20-100장</td>
                <td className="px-6 py-4 text-gray-300">인물, 반려동물, 제품</td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <span className="font-bold text-rose-400">Textual Inversion</span>
                </td>
                <td className="px-6 py-4 text-gray-300">~100KB</td>
                <td className="px-6 py-4 text-gray-300">30분-1시간</td>
                <td className="px-6 py-4 text-gray-300">5-20장</td>
                <td className="px-6 py-4 text-gray-300">간단한 컨셉, 오브젝트</td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <span className="font-bold text-orange-400">Hypernetworks</span>
                </td>
                <td className="px-6 py-4 text-gray-300">70-200MB</td>
                <td className="px-6 py-4 text-gray-300">1-3시간</td>
                <td className="px-6 py-4 text-gray-300">30-100장</td>
                <td className="px-6 py-4 text-gray-300">예술 스타일, 화풍</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-6 bg-blue-900/20 border border-blue-500/30 rounded-xl p-6">
          <h3 className="text-lg font-bold text-blue-400 mb-3 flex items-center gap-2">
            💡 선택 가이드
          </h3>
          <ul className="space-y-2 text-gray-300">
            <li>• <strong className="text-purple-400">LoRA:</strong> 가장 인기, 작은 파일 크기, 빠른 학습, 다른 LoRA와 조합 가능</li>
            <li>• <strong className="text-pink-400">DreamBooth:</strong> 높은 정확도, 인물/얼굴 특화, GPU 많이 필요</li>
            <li>• <strong className="text-rose-400">Textual Inversion:</strong> 가장 간단, 새로운 단어(토큰) 추가, 낮은 학습 비용</li>
            <li>• <strong className="text-orange-400">Hypernetworks:</strong> 예술 스타일 학습에 특화, 복잡한 스타일 표현</li>
          </ul>
        </div>
      </section>

      {/* 2. LoRA 학습 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-pink-500 to-rose-500 rounded-lg flex items-center justify-center">
            <Wand2 className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">LoRA (Low-Rank Adaptation) 학습</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-pink-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            LoRA는 모델 전체를 재학습하지 않고, 작은 어댑터를 학습시켜 특정 스타일이나 캐릭터를 재현합니다.
            작은 파일 크기(10-200MB)와 빠른 학습 시간으로 가장 인기 있는 방법입니다.
          </p>

          <div className="space-y-6">
            {/* LoRA 학습 준비 */}
            <div>
              <h3 className="text-2xl font-bold text-pink-400 mb-4">1. 데이터셋 준비</h3>
              <div className="bg-gray-800/50 border border-pink-500/30 rounded-lg p-6">
                <h4 className="font-bold text-white mb-3">이미지 수집</h4>
                <ul className="space-y-2 text-gray-300 text-sm">
                  <li>• <strong className="text-purple-400">최소 15장,</strong> 권장 30-50장</li>
                  <li>• <strong className="text-pink-400">다양한 각도:</strong> 정면, 측면, 뒤쪽, 3/4 각도</li>
                  <li>• <strong className="text-rose-400">다양한 조명:</strong> 밝은 조명, 어두운 조명, 실내/실외</li>
                  <li>• <strong className="text-orange-400">다양한 표정/포즈:</strong> (인물의 경우) 웃는 얼굴, 정면 응시 등</li>
                  <li>• <strong className="text-purple-400">해상도:</strong> 512×512 이상 (1024×1024 권장)</li>
                  <li>• <strong className="text-pink-400">배경:</strong> 깔끔한 배경 선호 (주제가 명확히 보이도록)</li>
                </ul>

                <div className="mt-4 bg-red-900/20 border border-red-500/30 rounded-lg p-4">
                  <p className="text-red-300 text-sm">
                    ⚠️ <strong>피해야 할 것:</strong> 흐린 이미지, 워터마크, 텍스트, 여러 인물이 함께 있는 사진
                  </p>
                </div>
              </div>
            </div>

            {/* Kohya GUI */}
            <div>
              <h3 className="text-2xl font-bold text-purple-400 mb-4">2. Kohya LoRA GUI 사용</h3>
              <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-6">
                <p className="text-gray-300 mb-4">
                  Kohya_ss는 LoRA 학습을 위한 가장 인기 있는 도구입니다. GUI 제공으로 사용이 쉽습니다.
                </p>

                <div className="space-y-4">
                  {/* 설치 */}
                  <div>
                    <h4 className="font-bold text-white mb-2">설치 (Windows)</h4>
                    <div className="bg-gray-900/50 rounded-lg p-4">
                      <p className="text-sm font-mono text-purple-300 mb-2">
                        # 1. Kohya GUI 다운로드
                      </p>
                      <p className="text-sm font-mono text-gray-300 mb-3">
                        https://github.com/bmaltais/kohya_ss
                      </p>
                      <p className="text-sm font-mono text-purple-300 mb-2">
                        # 2. 압축 해제 후 gui.bat 실행
                      </p>
                      <p className="text-sm font-mono text-gray-300">
                        gui.bat
                      </p>
                    </div>
                  </div>

                  {/* 학습 단계 */}
                  <div>
                    <h4 className="font-bold text-white mb-2">학습 단계</h4>
                    <ol className="space-y-3 text-gray-300 text-sm">
                      <li>
                        <strong className="text-purple-400">Step 1: 이미지 폴더 설정</strong>
                        <div className="bg-gray-900/50 rounded-lg p-3 mt-2">
                          <p className="text-xs">
                            Image folder: <span className="text-green-400">C:\lora\images\mycharacter</span>
                          </p>
                        </div>
                      </li>
                      <li>
                        <strong className="text-pink-400">Step 2: 베이스 모델 선택</strong>
                        <p className="text-xs text-gray-400 mt-1">
                          SD 1.5 또는 SDXL 기반 모델 (.safetensors 파일)
                        </p>
                      </li>
                      <li>
                        <strong className="text-rose-400">Step 3: 출력 설정</strong>
                        <div className="bg-gray-900/50 rounded-lg p-3 mt-2">
                          <p className="text-xs">
                            Output folder: <span className="text-green-400">C:\lora\output</span><br/>
                            Output name: <span className="text-green-400">mycharacter_v1</span>
                          </p>
                        </div>
                      </li>
                      <li>
                        <strong className="text-orange-400">Step 4: 하이퍼파라미터 설정</strong>
                        <div className="bg-gray-900/50 rounded-lg p-3 mt-2">
                          <ul className="text-xs space-y-1">
                            <li>• Network Rank (Dimension): <strong className="text-white">32</strong> (권장)</li>
                            <li>• Network Alpha: <strong className="text-white">16</strong> (Rank의 절반)</li>
                            <li>• Learning Rate: <strong className="text-white">1e-4</strong> (0.0001)</li>
                            <li>• Batch Size: <strong className="text-white">2-4</strong> (VRAM에 따라)</li>
                            <li>• Epochs: <strong className="text-white">10-20</strong></li>
                          </ul>
                        </div>
                      </li>
                      <li>
                        <strong className="text-purple-400">Step 5: 학습 시작</strong>
                        <p className="text-xs text-gray-400 mt-1">
                          "Start training" 버튼 클릭. RTX 4090 기준 30분-1시간 소요
                        </p>
                      </li>
                    </ol>
                  </div>
                </div>
              </div>
            </div>

            {/* LoRA 사용 */}
            <div>
              <h3 className="text-2xl font-bold text-rose-400 mb-4">3. 학습된 LoRA 사용</h3>
              <div className="bg-gray-800/50 border border-rose-500/30 rounded-lg p-6">
                <h4 className="font-bold text-white mb-3">Automatic1111에서 사용</h4>
                <ol className="space-y-2 text-gray-300 text-sm">
                  <li>
                    1. 학습된 .safetensors 파일을 <span className="text-purple-400 font-mono">models/Lora/</span> 폴더로 이동
                  </li>
                  <li>
                    2. Web UI 재시작 (또는 "Refresh" 버튼)
                  </li>
                  <li>
                    3. 프롬프트에 LoRA 트리거 추가:
                    <div className="bg-gray-900/50 rounded-lg p-3 mt-2">
                      <p className="text-purple-300 font-mono text-xs">
                        &lt;lora:mycharacter_v1:0.8&gt; 1girl, red hair, blue eyes
                      </p>
                      <p className="text-gray-400 text-xs mt-2">
                        • :0.8 = LoRA 강도 (0.5-1.0 권장)<br/>
                        • 여러 LoRA 동시 사용 가능
                      </p>
                    </div>
                  </li>
                  <li>
                    4. Generate 클릭!
                  </li>
                </ol>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. DreamBooth */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-rose-500 to-orange-500 rounded-lg flex items-center justify-center">
            <Users className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">DreamBooth - 인물 학습</h2>
        </div>

        <div className="bg-gradient-to-br from-rose-900/20 to-orange-900/20 border border-rose-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            DreamBooth는 Google이 개발한 기법으로, 소량의 이미지(20-100장)로 특정 인물, 반려동물, 제품을
            모델에 학습시킵니다. LoRA보다 정확도가 높지만, 전체 모델을 재학습하므로 시간과 VRAM이 많이 필요합니다.
          </p>

          <div className="space-y-6">
            {/* 데이터셋 준비 */}
            <div className="bg-gray-800/50 border border-rose-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-rose-400 mb-4">1. Instance Images (대상 이미지)</h3>
              <ul className="space-y-2 text-gray-300 text-sm">
                <li>• <strong className="text-white">수량:</strong> 20-100장 (30-50장 권장)</li>
                <li>• <strong className="text-white">품질:</strong> 고해상도 (768×768 이상), 깨끗한 배경</li>
                <li>• <strong className="text-white">다양성:</strong> 다양한 각도, 조명, 표정, 의상</li>
                <li>• <strong className="text-white">일관성:</strong> 학습하려는 대상(얼굴, 머리색 등)이 일관되게 보여야 함</li>
              </ul>

              <div className="mt-4 bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
                <p className="text-blue-300 text-sm">
                  💡 <strong>예시:</strong> "나"를 학습시키려면 다양한 각도의 셀카 30장 + 실외 사진 20장
                </p>
              </div>
            </div>

            <div className="bg-gray-800/50 border border-orange-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-orange-400 mb-4">2. Class Images (정규화 이미지)</h3>
              <p className="text-gray-300 text-sm mb-3">
                과적합 방지를 위한 추가 이미지. DreamBooth가 "일반적인 여성"과 "특정 인물"을 구별하도록 돕습니다.
              </p>
              <ul className="space-y-2 text-gray-300 text-sm">
                <li>• <strong className="text-white">수량:</strong> Instance의 2-10배 (100-500장)</li>
                <li>• <strong className="text-white">생성 방법:</strong> Stable Diffusion으로 자동 생성 가능</li>
                <li>• <strong className="text-white">프롬프트 예시:</strong> "photo of a person" (학습 대상과 같은 카테고리)</li>
              </ul>
            </div>

            {/* 학습 도구 */}
            <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-purple-400 mb-4">3. 학습 도구</h3>

              <div className="space-y-4">
                {/* Google Colab */}
                <div>
                  <h4 className="font-bold text-pink-400 mb-2">🌐 Google Colab (추천 - GPU 무료)</h4>
                  <p className="text-gray-300 text-sm mb-2">
                    GPU가 없어도 클라우드에서 학습 가능 (T4 GPU 무료 제공)
                  </p>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <p className="text-xs text-gray-400">TheLastBen의 Fast-DreamBooth Colab:</p>
                    <p className="text-xs font-mono text-green-400 mt-1">
                      https://github.com/TheLastBen/fast-stable-diffusion
                    </p>
                  </div>
                </div>

                {/* 로컬 학습 */}
                <div>
                  <h4 className="font-bold text-rose-400 mb-2">💻 로컬 학습 (고성능 GPU 필요)</h4>
                  <p className="text-gray-300 text-sm mb-2">
                    RTX 3090 (24GB) 이상 권장. Kohya GUI로 DreamBooth도 학습 가능
                  </p>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <p className="text-xs">
                      • <strong className="text-white">장점:</strong> 완전한 제어, 무제한 학습<br/>
                      • <strong className="text-white">단점:</strong> 고성능 GPU 필요, 긴 학습 시간 (2-4시간)
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Unique Identifier */}
            <div className="bg-gray-800/50 border border-pink-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-pink-400 mb-4">4. Unique Identifier (고유 식별자)</h3>
              <p className="text-gray-300 text-sm mb-3">
                학습 대상을 구분하기 위한 특별한 단어. 프롬프트에서 이 단어를 사용하면 학습된 인물이 생성됩니다.
              </p>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-sm mb-3">
                  <strong className="text-purple-400">형식:</strong> [unique_token] [class]
                </p>
                <ul className="text-xs text-gray-400 space-y-2">
                  <li>• unique_token: 희귀한 단어 조합 (예: sks, xyz, abc123)</li>
                  <li>• class: 일반 범주 (예: person, man, woman, dog, cat)</li>
                </ul>
                <div className="mt-3 bg-black/30 rounded p-3">
                  <p className="text-xs text-green-400 mb-1">✓ 좋은 예시:</p>
                  <p className="text-xs text-gray-300 mb-3">
                    • "sks person" (가장 흔히 사용)<br/>
                    • "xyz woman" (여성 인물)<br/>
                    • "abc123 dog" (반려견)
                  </p>
                  <p className="text-xs text-red-400 mb-1">✗ 나쁜 예시:</p>
                  <p className="text-xs text-gray-300">
                    • "john person" (흔한 이름, 혼동 가능)<br/>
                    • "beautiful woman" (이미 존재하는 단어 조합)
                  </p>
                </div>
              </div>
            </div>

            {/* 사용 예시 */}
            <div className="bg-gray-800/50 border border-rose-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-rose-400 mb-4">5. 학습된 모델 사용</h3>
              <div className="bg-gray-900/50 rounded-lg p-4">
                <p className="text-sm text-gray-300 mb-3">
                  학습된 모델을 Automatic1111의 <span className="text-purple-400 font-mono">models/Stable-diffusion/</span> 폴더에 넣고 사용
                </p>
                <div className="bg-black/30 rounded p-3">
                  <p className="text-xs text-purple-400 mb-2"># 프롬프트 예시:</p>
                  <p className="text-xs font-mono text-gray-300 mb-3">
                    portrait photo of sks person wearing a suit, professional lighting,
                    studio background, high quality, sharp focus
                  </p>
                  <p className="text-xs text-pink-400 mb-2"># 다양한 시나리오:</p>
                  <p className="text-xs font-mono text-gray-300">
                    • sks person as a superhero<br/>
                    • sks person in the style of Van Gogh<br/>
                    • sks person wearing medieval armor
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Textual Inversion */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-yellow-500 rounded-lg flex items-center justify-center">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">Textual Inversion - 간단한 컨셉 학습</h2>
        </div>

        <div className="bg-gradient-to-br from-orange-900/20 to-yellow-900/20 border border-orange-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            Textual Inversion은 새로운 단어(토큰)를 모델의 사전에 추가하는 방법입니다.
            가장 간단하고 빠르며(30분-1시간), 파일 크기도 가장 작습니다(~100KB).
          </p>

          <div className="space-y-6">
            <div className="bg-gray-800/50 border border-orange-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-orange-400 mb-4">장점 & 단점</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-bold text-green-400 mb-2">✓ 장점</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• 매우 작은 파일 크기 (~100KB)</li>
                    <li>• 빠른 학습 (30분-1시간)</li>
                    <li>• 적은 이미지 필요 (5-20장)</li>
                    <li>• 낮은 VRAM 요구량</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-bold text-red-400 mb-2">✗ 단점</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• 낮은 정확도 (간단한 컨셉만 가능)</li>
                    <li>• 복잡한 스타일/인물 표현 어려움</li>
                    <li>• LoRA/DreamBooth보다 제한적</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-gray-800/50 border border-yellow-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-yellow-400 mb-4">Automatic1111에서 학습</h3>
              <ol className="space-y-3 text-gray-300 text-sm">
                <li>
                  <strong className="text-white">1.</strong> "Train" 탭 → "Create embedding" 클릭
                </li>
                <li>
                  <strong className="text-white">2.</strong> Name: 임베딩 이름 (예: my-style)
                </li>
                <li>
                  <strong className="text-white">3.</strong> Initialization text: 유사한 기존 단어 (예: "girl", "portrait")
                </li>
                <li>
                  <strong className="text-white">4.</strong> Number of vectors: 4-16 (높을수록 정확하지만 느림)
                </li>
                <li>
                  <strong className="text-white">5.</strong> "Train" 탭 → 이미지 폴더 설정 → "Train Embedding"
                </li>
              </ol>

              <div className="mt-4 bg-gray-900/50 rounded-lg p-4">
                <p className="text-xs text-purple-400 mb-2"># 사용 예시:</p>
                <p className="text-xs font-mono text-gray-300">
                  portrait of a woman in &lt;my-style&gt;, detailed face, high quality
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. 최적화 팁 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-yellow-500 to-green-500 rounded-lg flex items-center justify-center">
            <Upload className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">파인튜닝 최적화 팁</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-gray-800/50 border border-purple-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-purple-400 mb-4">이미지 전처리</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• <strong>크롭:</strong> 주제가 중앙에 오도록 자르기</li>
              <li>• <strong>정사각형:</strong> 512×512 또는 768×768로 리사이즈</li>
              <li>• <strong>배경 제거:</strong> Remove.bg 등 도구 활용</li>
              <li>• <strong>밝기 조정:</strong> 너무 어둡거나 밝지 않게</li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-pink-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-pink-400 mb-4">캡셔닝 (프롬프트 작성)</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• <strong>정확한 묘사:</strong> 이미지 내용을 정확히 설명</li>
              <li>• <strong>일관된 형식:</strong> 모든 이미지에 동일한 스타일</li>
              <li>• <strong>자동 캡셔닝:</strong> BLIP, WD14 Tagger 사용</li>
              <li>• <strong>수동 수정:</strong> 자동 캡션 검토 및 보정</li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-rose-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-rose-400 mb-4">학습률 조정</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• <strong>너무 높으면:</strong> 과적합, 이상한 아티팩트</li>
              <li>• <strong>너무 낮으면:</strong> 학습 안됨, 시간만 소모</li>
              <li>• <strong>권장값:</strong> LoRA 1e-4, DreamBooth 5e-6</li>
              <li>• <strong>조정 방법:</strong> 5-10 epoch마다 결과 확인</li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-orange-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-orange-400 mb-4">정기적인 체크포인트</h3>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>• <strong>저장 간격:</strong> 500-1000 steps마다 저장</li>
              <li>• <strong>중간 테스트:</strong> 학습 중간에 이미지 생성 테스트</li>
              <li>• <strong>과적합 방지:</strong> 너무 오래 학습하지 않기</li>
              <li>• <strong>최적 모델 선택:</strong> 여러 epoch 결과 비교</li>
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
            <h3 className="text-lg font-bold text-purple-400 mb-4">📖 공식 논문 & 문서</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://arxiv.org/abs/2106.09685"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  LoRA: Low-Rank Adaptation (Microsoft 2021)
                </a>
                <p className="text-sm text-gray-400 mt-1">LoRA 원본 논문</p>
              </li>
              <li>
                <a
                  href="https://arxiv.org/abs/2208.12242"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  DreamBooth: Fine Tuning (Google 2022)
                </a>
                <p className="text-sm text-gray-400 mt-1">DreamBooth 원본 논문</p>
              </li>
              <li>
                <a
                  href="https://textual-inversion.github.io/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  Textual Inversion Official Page
                </a>
                <p className="text-sm text-gray-400 mt-1">Textual Inversion 프로젝트 페이지</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-pink-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-pink-400 mb-4">🛠️ 학습 도구</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://github.com/bmaltais/kohya_ss"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Kohya_ss - LoRA/DreamBooth GUI
                </a>
                <p className="text-sm text-gray-400 mt-1">가장 인기 있는 학습 도구 (GUI 제공)</p>
              </li>
              <li>
                <a
                  href="https://github.com/TheLastBen/fast-stable-diffusion"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Fast-DreamBooth Google Colab
                </a>
                <p className="text-sm text-gray-400 mt-1">GPU 없이 클라우드에서 DreamBooth 학습</p>
              </li>
              <li>
                <a
                  href="https://huggingface.co/spaces/sd-dreambooth-library"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Hugging Face DreamBooth Library
                </a>
                <p className="text-sm text-gray-400 mt-1">커뮤니티 학습 모델 공유</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-rose-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-rose-400 mb-4">🎓 튜토리얼 & 가이드</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://www.youtube.com/watch?v=70H03cv57-o"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  Sebastian Kamph - LoRA Training Guide
                </a>
                <p className="text-sm text-gray-400 mt-1">가장 인기 있는 LoRA 학습 영상 튜토리얼</p>
              </li>
              <li>
                <a
                  href="https://rentry.org/lora_train"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  Community LoRA Training Guide
                </a>
                <p className="text-sm text-gray-400 mt-1">커뮤니티 작성 완전 가이드</p>
              </li>
              <li>
                <a
                  href="https://www.reddit.com/r/StableDiffusion/wiki/guides/lora"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  r/StableDiffusion - LoRA Wiki
                </a>
                <p className="text-sm text-gray-400 mt-1">Reddit 커뮤니티 LoRA 가이드</p>
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}
