'use client'

import React from 'react'
import { Scale, AlertTriangle, FileCheck, DollarSign, Shield, BookOpen } from 'lucide-react'

export default function Chapter8() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8">
      {/* Hero Section */}
      <div className="mb-12">
        <div className="inline-block px-4 py-2 bg-purple-500/10 border border-purple-500/20 rounded-full mb-4">
          <span className="text-purple-400 text-sm font-medium">Chapter 8</span>
        </div>
        <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-purple-400 via-pink-400 to-rose-400 bg-clip-text text-transparent">
          상업적 활용과 저작권
        </h1>
        <p className="text-xl text-gray-300 leading-relaxed">
          AI 생성 콘텐츠의 법적 문제, 저작권, 라이선스, 윤리적 고려사항을 다룹니다.
          상업적으로 안전하게 AI 도구를 활용하는 방법을 배웁니다.
        </p>
      </div>

      {/* 1. AI 생성물의 저작권 현황 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Scale className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">AI 생성물의 저작권 현황</h2>
        </div>

        <div className="bg-gradient-to-br from-gray-800 to-gray-900 border border-purple-500/30 rounded-xl p-8">
          <div className="space-y-6">
            {/* 국가별 현황 */}
            <div className="bg-gray-800/50 border border-purple-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-purple-400 mb-4">국가별 법적 현황 (2024년 기준)</h3>
              <div className="space-y-4">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-pink-400 mb-2">🇺🇸 미국</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• <strong className="text-white">저작권청 입장:</strong> AI 단독 생성물은 저작권 보호 불가</li>
                    <li>• <strong className="text-white">인간의 창의적 기여:</strong> 프롬프트 작성 + 편집 시 저작권 인정 가능</li>
                    <li>• <strong className="text-white">판례:</strong> Zarya of the Dawn 사건 (2023) - 부분 저작권 인정</li>
                    <li>• <strong className="text-white">트렌드:</strong> "AI-assisted" 표기 권장</li>
                  </ul>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-rose-400 mb-2">🇪🇺 유럽연합</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• <strong className="text-white">AI Act (2024):</strong> AI 생성물 투명성 의무화</li>
                    <li>• <strong className="text-white">저작권:</strong> 국가마다 상이, 대부분 인간 개입 필요</li>
                    <li>• <strong className="text-white">GDPR:</strong> 개인정보 학습 데이터 규제</li>
                  </ul>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-orange-400 mb-2">🇰🇷 한국</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• <strong className="text-white">저작권법:</strong> AI 생성물 명시적 조항 없음 (법 개정 논의 중)</li>
                    <li>• <strong className="text-white">문화체육관광부:</strong> AI 창작물 가이드라인 검토 중</li>
                    <li>• <strong className="text-white">실무:</strong> 인간의 창작적 기여 입증 시 보호 가능</li>
                  </ul>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-purple-400 mb-2">🇯🇵 일본</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• <strong className="text-white">상대적 자유:</strong> AI 학습용 저작물 사용 폭넓게 허용</li>
                    <li>• <strong className="text-white">생성물 저작권:</strong> 창작성 입증 시 보호</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* 핵심 원칙 */}
            <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-6">
              <h3 className="text-lg font-bold text-blue-400 mb-4">💡 글로벌 공통 원칙</h3>
              <ul className="space-y-2 text-gray-300 text-sm">
                <li>
                  <strong className="text-white">1. 인간의 창의적 기여 필수:</strong> 단순 프롬프트만으로는 저작권 보호 어려움
                </li>
                <li>
                  <strong className="text-white">2. 편집 & 후처리 중요:</strong> Photoshop 편집, 색보정, 조합 등은 인간 창작으로 인정
                </li>
                <li>
                  <strong className="text-white">3. 투명성 원칙:</strong> AI 사용 사실 공개 권장 (특히 상업적 사용)
                </li>
                <li>
                  <strong className="text-white">4. 플랫폼 ToS 준수:</strong> 각 AI 도구의 이용 약관 반드시 확인
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 2. 플랫폼별 상업적 사용 라이선스 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-pink-500 to-rose-500 rounded-lg flex items-center justify-center">
            <FileCheck className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">플랫폼별 상업적 사용 라이선스</h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
            <thead className="bg-gradient-to-r from-purple-500/20 to-pink-500/20">
              <tr>
                <th className="px-6 py-4 text-left text-white font-bold">플랫폼</th>
                <th className="px-6 py-4 text-left text-white font-bold">무료 플랜</th>
                <th className="px-6 py-4 text-left text-white font-bold">유료 플랜</th>
                <th className="px-6 py-4 text-left text-white font-bold">저작권 귀속</th>
                <th className="px-6 py-4 text-left text-white font-bold">주의사항</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <span className="font-bold text-purple-400">Midjourney</span>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  ❌ 무료 플랜 폐지<br/>
                  (2023년 3월부터)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  ✅ 상업적 사용 가능<br/>
                  (Basic $10/월 이상)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  <strong className="text-green-400">사용자 소유</strong><br/>
                  (유료 플랜)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  • 연 매출 $1M 이상 기업은 Pro 플랜 필요
                </td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <span className="font-bold text-pink-400">DALL-E 3</span>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  ❌ 무료 플랜 없음
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  ✅ 상업적 사용 가능<br/>
                  (ChatGPT Plus $20/월)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  <strong className="text-green-400">사용자 소유</strong><br/>
                  (전체 권리)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  • API: $0.04/이미지<br/>
                  • Content Policy 준수 필수
                </td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <span className="font-bold text-rose-400">Stable<br/>Diffusion</span>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  ✅ 완전 무료<br/>
                  (오픈소스)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  ✅ 제한 없음<br/>
                  (CreativeML License)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  <strong className="text-green-400">사용자 소유</strong><br/>
                  (완전 자유)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  • 유해 콘텐츠 생성 금지<br/>
                  • 일부 커스텀 모델은 별도 라이선스
                </td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <span className="font-bold text-orange-400">Runway ML</span>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  ⚠️ 비상업적 용도만<br/>
                  (125 크레딧)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  ✅ 상업적 사용 가능<br/>
                  (Standard $15/월 이상)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  <strong className="text-green-400">사용자 소유</strong><br/>
                  (유료 플랜)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  • 무료 플랜 워터마크 있음
                </td>
              </tr>
              <tr className="hover:bg-gray-700/30 transition-colors">
                <td className="px-6 py-4">
                  <span className="font-bold text-purple-400">Suno</span>
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  ⚠️ 비상업적 용도만<br/>
                  (50곡/월)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  ✅ 상업적 사용 가능<br/>
                  (Pro $10/월 이상)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  <strong className="text-green-400">사용자 소유</strong><br/>
                  (유료 플랜)
                </td>
                <td className="px-6 py-4 text-gray-300 text-sm">
                  • 음원 스트리밍 수익 100% 사용자
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-6 bg-red-900/20 border border-red-500/30 rounded-xl p-6">
          <h3 className="text-lg font-bold text-red-400 mb-4 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5" />
            ⚠️ 반드시 확인해야 할 사항
          </h3>
          <ul className="space-y-2 text-gray-300 text-sm">
            <li>• <strong className="text-white">무료 플랜 → 유료 전환 필수:</strong> 대부분 무료는 비상업적 용도만</li>
            <li>• <strong className="text-white">ToS 정기 확인:</strong> 라이선스 정책이 자주 변경됨</li>
            <li>• <strong className="text-white">기업 사용:</strong> Enterprise 플랜 별도 문의 필요 (매출 $1M+ 기준)</li>
            <li>• <strong className="text-white">워터마크 제거:</strong> 유료 플랜에서도 일부 도구는 워터마크 존재 확인</li>
          </ul>
        </div>
      </section>

      {/* 3. 저작권 침해 리스크 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-rose-500 to-orange-500 rounded-lg flex items-center justify-center">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">저작권 침해 리스크 & 대응</h2>
        </div>

        <div className="bg-gradient-to-br from-rose-900/20 to-orange-900/20 border border-rose-500/30 rounded-xl p-8">
          <div className="space-y-6">
            {/* 주요 리스크 */}
            <div className="bg-gray-800/50 border border-rose-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-rose-400 mb-4">주요 저작권 리스크</h3>
              <div className="space-y-4">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-purple-400 mb-2">1. 학습 데이터 저작권 논란</h4>
                  <p className="text-gray-300 text-sm mb-2">
                    <strong className="text-white">이슈:</strong> AI 모델이 저작권 있는 작품으로 학습됨
                  </p>
                  <ul className="text-gray-400 text-xs space-y-1">
                    <li>• Getty Images vs Stability AI (2023) - 계류 중</li>
                    <li>• 수많은 아티스트들이 Midjourney/Stable Diffusion 고소</li>
                    <li>• GitHub Copilot 코드 저작권 소송</li>
                  </ul>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-pink-400 mb-2">2. 스타일 모방 (Style Mimicry)</h4>
                  <p className="text-gray-300 text-sm mb-2">
                    <strong className="text-white">위험:</strong> "in the style of [artist name]" 프롬프트 사용
                  </p>
                  <div className="bg-red-900/20 border border-red-500/30 rounded p-3 mt-2">
                    <p className="text-red-300 text-xs">
                      <strong>⚠️ 피해야 할 프롬프트:</strong><br/>
                      • "by Greg Rutkowski" (아티스트가 공개 반대)<br/>
                      • "Disney style" (상표권 문제)<br/>
                      • "in the style of Studio Ghibli" (법적 리스크)
                    </p>
                  </div>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-rose-400 mb-2">3. 캐릭터 유사성</h4>
                  <p className="text-gray-300 text-sm mb-2">
                    <strong className="text-white">위험:</strong> 유명 캐릭터와 유사한 이미지 생성
                  </p>
                  <ul className="text-gray-400 text-xs space-y-1">
                    <li>• Marvel 히어로, Disney 캐릭터 → 상표권 침해 위험</li>
                    <li>• "Mickey Mouse" 직접 생성 → 고소 위험 높음</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* 안전한 사용법 */}
            <div className="bg-gray-800/50 border border-green-500/30 rounded-lg p-6">
              <h3 className="text-xl font-bold text-green-400 mb-4">✅ 안전한 사용법 (Best Practices)</h3>
              <div className="space-y-3">
                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-purple-400 mb-2">1. 일반적인 스타일 용어 사용</h4>
                  <div className="grid md:grid-cols-2 gap-3">
                    <div>
                      <p className="text-xs text-red-400 mb-1">❌ 위험:</p>
                      <p className="text-xs text-gray-400">"by [artist name]"</p>
                    </div>
                    <div>
                      <p className="text-xs text-green-400 mb-1">✓ 안전:</p>
                      <p className="text-xs text-gray-400">"impressionist style", "cyberpunk aesthetic"</p>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-pink-400 mb-2">2. 충분한 변형 (Transformative Use)</h4>
                  <ul className="text-gray-300 text-xs space-y-1">
                    <li>• AI 생성 후 Photoshop으로 50% 이상 편집</li>
                    <li>• 여러 이미지 조합 (collage)</li>
                    <li>• 색감, 구도 대폭 변경</li>
                  </ul>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-rose-400 mb-2">3. 오리지널 컨셉 우선</h4>
                  <p className="text-gray-300 text-xs">
                    기존 작품 재현보다, 완전히 새로운 컨셉 창작 권장
                  </p>
                </div>

                <div className="bg-gray-900/50 rounded-lg p-4">
                  <h4 className="font-bold text-orange-400 mb-2">4. 출처 명시 (Attribution)</h4>
                  <p className="text-gray-300 text-xs">
                    "AI-generated with [tool name]" 표기 권장 (투명성)
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. 윤리적 고려사항 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-yellow-500 rounded-lg flex items-center justify-center">
            <BookOpen className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">윤리적 고려사항</h2>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* 딥페이크 위험 */}
          <div className="bg-gray-800/50 border border-purple-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-purple-400 mb-4">1. 딥페이크 & 허위정보</h3>
            <div className="space-y-3">
              <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
                <p className="text-red-300 text-sm mb-2"><strong>⚠️ 금지 행위:</strong></p>
                <ul className="text-gray-400 text-xs space-y-1">
                  <li>• 유명인 얼굴 무단 사용 (초상권 침해)</li>
                  <li>• 가짜 뉴스 이미지/비디오 생성</li>
                  <li>• 정치인 허위 발언 조작</li>
                  <li>• 타인 신분증/문서 위조</li>
                </ul>
              </div>
              <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-4">
                <p className="text-green-300 text-sm mb-2"><strong>✓ 합법적 사용:</strong></p>
                <ul className="text-gray-400 text-xs space-y-1">
                  <li>• 패러디 (명백히 가짜임을 표시)</li>
                  <li>• 교육용 데모 (워터마크 필수)</li>
                  <li>• 본인 얼굴 사용 (동의 필요)</li>
                </ul>
              </div>
            </div>
          </div>

          {/* 아티스트 권리 존중 */}
          <div className="bg-gray-800/50 border border-pink-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-pink-400 mb-4">2. 아티스트 권리 존중</h3>
            <div className="space-y-3 text-gray-300 text-sm">
              <div>
                <p className="text-white font-semibold mb-2">Opt-Out 요청 존중:</p>
                <p className="text-xs text-gray-400">
                  일부 아티스트가 자신의 스타일 학습을 반대. Glaze, Nightshade 등 도구로 방어
                </p>
              </div>
              <div>
                <p className="text-white font-semibold mb-2">크레딧 표기:</p>
                <p className="text-xs text-gray-400">
                  AI 도구 사용 사실 공개 (특히 상업 작품)
                </p>
              </div>
              <div>
                <p className="text-white font-semibold mb-2">공정한 대가:</p>
                <p className="text-xs text-gray-400">
                  AI로 대체하되, 아티스트 고용도 병행 (균형)
                </p>
              </div>
            </div>
          </div>

          {/* 편향성 문제 */}
          <div className="bg-gray-800/50 border border-rose-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-rose-400 mb-4">3. 편향성 (Bias) 문제</h3>
            <div className="space-y-2 text-gray-300 text-sm">
              <p className="text-white font-semibold mb-2">AI 모델의 내재 편향:</p>
              <ul className="text-xs text-gray-400 space-y-1">
                <li>• <strong>인종/성별 편향:</strong> 특정 인종/성별 과대 또는 과소 표현</li>
                <li>• <strong>문화 편향:</strong> 서구 중심 학습 데이터</li>
                <li>• <strong>미적 편향:</strong> "beauty" = 젊은 백인 여성 경향</li>
              </ul>
              <div className="mt-3 bg-blue-900/20 border border-blue-500/30 rounded-lg p-3">
                <p className="text-blue-300 text-xs">
                  💡 <strong>대응:</strong> 프롬프트에 다양성 명시 (예: "diverse group of people")
                </p>
              </div>
            </div>
          </div>

          {/* 환경 영향 */}
          <div className="bg-gray-800/50 border border-orange-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-orange-400 mb-4">4. 환경 영향 (Carbon Footprint)</h3>
            <div className="space-y-2 text-gray-300 text-sm">
              <p className="text-xs text-gray-400 mb-2">
                AI 모델 학습 & 추론은 막대한 전력 소비
              </p>
              <ul className="text-xs text-gray-400 space-y-1">
                <li>• GPT-3 학습: 약 552톤 CO₂ 배출 (자동차 5년 운행량)</li>
                <li>• 이미지 생성 1회: 약 2.9g CO₂</li>
                <li>• 비디오 생성: 이미지의 10배 이상</li>
              </ul>
              <div className="mt-3 bg-green-900/20 border border-green-500/30 rounded-lg p-3">
                <p className="text-green-300 text-xs">
                  💡 <strong>친환경 실천:</strong> 로컬 SD 사용, 배치 처리, 불필요한 재생성 자제
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. 상업적 사용 체크리스트 */}
      <section className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-yellow-500 to-green-500 rounded-lg flex items-center justify-center">
            <DollarSign className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white">상업적 사용 체크리스트</h2>
        </div>

        <div className="bg-gradient-to-br from-yellow-900/20 to-green-900/20 border border-yellow-500/30 rounded-xl p-8">
          <p className="text-gray-300 mb-6">
            AI 생성 콘텐츠를 상업적으로 사용하기 전, 아래 체크리스트를 반드시 확인하세요.
          </p>

          <div className="space-y-4">
            {[
              {
                category: '라이선스 확인',
                items: [
                  '사용한 AI 도구의 유료 플랜 구독 여부',
                  '각 도구의 ToS에서 상업적 사용 허용 명시 확인',
                  '기업 매출 $1M+ 시 Enterprise 플랜 필요 여부',
                  '생성물 저작권이 사용자에게 귀속되는지 확인'
                ],
                color: 'purple'
              },
              {
                category: '법적 리스크 최소화',
                items: [
                  '유명 아티스트 이름을 프롬프트에 사용하지 않음',
                  '유명 캐릭터/상표 재현 시도하지 않음',
                  '타인의 얼굴/초상권 무단 사용 금지',
                  '충분한 변형 작업 (50% 이상 편집)'
                ],
                color: 'pink'
              },
              {
                category: '투명성 & 윤리',
                items: [
                  'AI 사용 사실 공개 (크레딧 또는 설명에)',
                  '딥페이크/허위정보 생성 금지',
                  '편향성 검토 (인종/성별/문화)',
                  '아티스트 커뮤니티 가이드라인 준수'
                ],
                color: 'rose'
              },
              {
                category: '보험 & 대비',
                items: [
                  '상업용 보험 (E&O Insurance) 가입 검토',
                  '법률 자문 (고액 프로젝트 시)',
                  '저작권 분쟁 대응 계획 수립',
                  '정기적인 라이선스 정책 업데이트 확인'
                ],
                color: 'orange'
              }
            ].map((section, idx) => (
              <div key={idx} className={`bg-gray-800/50 border border-${section.color}-500/30 rounded-lg p-6`}>
                <h3 className={`text-lg font-bold text-${section.color}-400 mb-4`}>
                  {idx + 1}. {section.category}
                </h3>
                <div className="space-y-2">
                  {section.items.map((item, itemIdx) => (
                    <label key={itemIdx} className="flex items-start gap-3 cursor-pointer group">
                      <input
                        type="checkbox"
                        className="mt-1 w-4 h-4 rounded border-gray-600 text-purple-500 focus:ring-purple-500"
                      />
                      <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
                        {item}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 bg-green-900/30 border border-green-500/30 rounded-lg p-6 text-center">
            <p className="text-green-300 text-lg font-bold">
              ✅ 모든 항목 체크 완료 → 안전하게 상업적 사용 가능!
            </p>
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
            <h3 className="text-lg font-bold text-purple-400 mb-4">⚖️ 법률 & 라이선스</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://www.copyright.gov/ai/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  U.S. Copyright Office - AI and Copyright
                </a>
                <p className="text-sm text-gray-400 mt-1">미국 저작권청 AI 정책 공식 페이지</p>
              </li>
              <li>
                <a
                  href="https://www.europarl.europa.eu/topics/en/article/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  EU AI Act - European Parliament
                </a>
                <p className="text-sm text-gray-400 mt-1">유럽연합 AI 규제 법안</p>
              </li>
              <li>
                <a
                  href="https://www.copyright.or.kr/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-purple-400 hover:text-purple-300 transition-colors"
                >
                  한국저작권위원회 - AI 저작권 가이드
                </a>
                <p className="text-sm text-gray-400 mt-1">한국 저작권 관련 정보 및 상담</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-pink-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-pink-400 mb-4">📖 윤리 & 가이드라인</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://partnershiponai.org/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Partnership on AI - Responsible AI
                </a>
                <p className="text-sm text-gray-400 mt-1">AI 윤리 가이드라인 (Google, Meta, OpenAI 참여)</p>
              </li>
              <li>
                <a
                  href="https://aiartonline.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  AI Art - Ethics & Best Practices
                </a>
                <p className="text-sm text-gray-400 mt-1">AI 아트 윤리 커뮤니티</p>
              </li>
              <li>
                <a
                  href="https://www.spawning.ai/have-i-been-trained"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-pink-400 hover:text-pink-300 transition-colors"
                >
                  Have I Been Trained? - Spawning AI
                </a>
                <p className="text-sm text-gray-400 mt-1">본인 작품이 AI 학습에 사용되었는지 확인 도구</p>
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 border border-rose-500/30 rounded-xl p-6">
            <h3 className="text-lg font-bold text-rose-400 mb-4">🛡️ 보호 도구 (아티스트용)</h3>
            <ul className="space-y-3">
              <li>
                <a
                  href="https://glaze.cs.uchicago.edu/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  Glaze - Style Protection Tool
                </a>
                <p className="text-sm text-gray-400 mt-1">AI 학습으로부터 스타일 보호 도구</p>
              </li>
              <li>
                <a
                  href="https://nightshade.cs.uchicago.edu/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-rose-400 hover:text-rose-300 transition-colors"
                >
                  Nightshade - Data Poisoning Tool
                </a>
                <p className="text-sm text-gray-400 mt-1">AI 모델 학습 방해 도구 (아티스트 권리 보호)</p>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 최종 요약 */}
      <section className="mb-16">
        <div className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border border-purple-500/30 rounded-xl p-8">
          <h2 className="text-2xl font-bold text-white mb-6 text-center">
            🎓 Creative AI 모듈 완료를 축하합니다!
          </h2>
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-gray-800/50 rounded-lg p-6">
              <h3 className="text-lg font-bold text-purple-400 mb-3">배운 내용:</h3>
              <ul className="text-gray-300 text-sm space-y-2">
                <li>✓ AI 이미지 생성 (Midjourney, DALL-E, SD)</li>
                <li>✓ LoRA, DreamBooth 파인튜닝</li>
                <li>✓ AI 음악 생성 (Suno, Udio)</li>
                <li>✓ AI 비디오 제작 (Runway ML, Pika Labs)</li>
                <li>✓ 통합 크리에이티브 워크플로우</li>
                <li>✓ 저작권 & 윤리</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-6">
              <h3 className="text-lg font-bold text-pink-400 mb-3">다음 단계:</h3>
              <ul className="text-gray-300 text-sm space-y-2">
                <li>• 실전 프로젝트 시작 (YouTube, 광고 등)</li>
                <li>• 포트폴리오 구축 (Behance, Dribbble)</li>
                <li>• 클라이언트 작업 수주</li>
                <li>• 커뮤니티 참여 (Discord, Reddit)</li>
                <li>• 지속적인 학습 (새 도구 모니터링)</li>
              </ul>
            </div>
          </div>
          <div className="text-center">
            <p className="text-gray-300 text-lg">
              AI는 도구입니다. 진정한 창의성은 여러분에게서 나옵니다. 🚀
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
