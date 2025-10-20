'use client'

import References from '@/components/common/References';

export default function Chapter4() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Chapter 4: 반도체 제조 공정
      </h1>

      {/* 웨이퍼 제조 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          4.1 웨이퍼(Wafer) 제조
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            웨이퍼는 반도체 칩의 기판이 되는 얇은 실리콘 원판입니다.
            현대 반도체는 주로 300mm(12인치) 웨이퍼를 사용하며, 차세대 450mm도 연구 중입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                Czochralski (CZ) 공정
              </h3>
              <svg className="w-full h-56" viewBox="0 0 300 240">
                {/* 도가니 */}
                <path d="M 80 180 L 100 220 L 200 220 L 220 180 Z"
                      fill="#EF4444" stroke="#DC2626" strokeWidth="2" />
                <text x="130" y="235" fontSize="10" fill="#DC2626">용융 실리콘 (1414°C)</text>

                {/* 시드 크리스탈 */}
                <rect x="145" y="40" width="10" height="30" fill="#9CA3AF" stroke="#4B5563" strokeWidth="2" />
                <circle cx="150" cy="70" r="8" fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />
                <text x="160" y="55" fontSize="9" fill="#6B7280">시드</text>

                {/* 단결정 잉곳 */}
                <path d="M 145 70 L 140 90 L 140 150 L 160 150 L 160 90 L 155 70"
                      fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />
                <text x="165" y="115" fontSize="9" fill="#2563EB">단결정 잉곳</text>

                {/* 회전 화살표 */}
                <path d="M 150 25 Q 170 25, 170 40" stroke="#10B981" strokeWidth="2" fill="none"
                      markerEnd="url(#arrow3)" />
                <text x="175" y="35" fontSize="8" fill="#10B981">회전</text>

                {/* 인상 화살표 */}
                <line x1="130" y1="50" x2="130" y2="30" stroke="#F59E0B" strokeWidth="2"
                      markerEnd="url(#arrow3)" />
                <text x="100" y="45" fontSize="8" fill="#F59E0B">인상 (1mm/min)</text>

                <defs>
                  <marker id="arrow3" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#10B981" />
                  </marker>
                </defs>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded-lg text-sm">
                <h4 className="font-semibold mb-2">공정 단계:</h4>
                <pre className="text-xs"><code>{`1. 시드 투입
   → 단결정 방향 설정 (100)

2. 네킹 (Necking)
   → 지름 3mm까지 감소
   → 전위 결함 제거

3. 숄더링 (Shouldering)
   → 목표 지름까지 확대

4. 바디 성장 (Body)
   → 일정 지름 유지 (300mm)
   → 길이: 2~3m

5. 테일링 (Tailing)
   → 서서히 분리

잉곳 무게: 200~300kg`}</code></pre>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded-lg">
                <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">
                  웨이퍼 크기 진화
                </h4>
                <div className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                  <div className="flex justify-between">
                    <span>50mm (2")</span>
                    <span className="text-gray-500">1960년대</span>
                  </div>
                  <div className="flex justify-between">
                    <span>150mm (6")</span>
                    <span className="text-gray-500">1980년대</span>
                  </div>
                  <div className="flex justify-between">
                    <span>200mm (8")</span>
                    <span className="text-gray-500">1990년대</span>
                  </div>
                  <div className="flex justify-between font-semibold">
                    <span className="text-blue-600">300mm (12")</span>
                    <span className="text-blue-600">2000년대~현재</span>
                  </div>
                  <div className="flex justify-between text-gray-500">
                    <span>450mm (18")</span>
                    <span>연구 단계</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-2">
              웨이퍼 가공 공정
            </h3>
            <div className="grid grid-cols-4 gap-2 text-xs text-center">
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">1. 슬라이싱</div>
                <div className="text-gray-600 dark:text-gray-400">다이아몬드 와이어</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">2. 래핑</div>
                <div className="text-gray-600 dark:text-gray-400">평탄화</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">3. CMP</div>
                <div className="text-gray-600 dark:text-gray-400">경면 연마</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">4. 세정/검사</div>
                <div className="text-gray-600 dark:text-gray-400">결함 검출</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 증착 공정 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          4.2 박막 증착 (Thin Film Deposition)
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          {/* CVD */}
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-green-800 dark:text-green-300 mb-3">
              CVD (Chemical Vapor Deposition)
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <svg className="w-full h-40" viewBox="0 0 280 160">
                {/* 챔버 */}
                <rect x="20" y="30" width="240" height="110" rx="10" fill="none"
                      stroke="#6B7280" strokeWidth="3" />

                {/* 가스 유입 */}
                <line x1="20" y1="60" x2="50" y2="60" stroke="#3B82F6" strokeWidth="3"
                      markerEnd="url(#arrow4)" />
                <text x="25" y="55" fontSize="9" fill="#3B82F6">SiH₄</text>

                <line x1="20" y1="80" x2="50" y2="80" stroke="#10B981" strokeWidth="3"
                      markerEnd="url(#arrow4)" />
                <text x="25" y="95" fontSize="9" fill="#10B981">NH₃</text>

                {/* 웨이퍼 */}
                <rect x="100" y="110" width="80" height="10" fill="#9CA3AF"
                      stroke="#4B5563" strokeWidth="2" />
                <text x="115" y="135" fontSize="9" fill="#6B7280">웨이퍼 (300~600°C)</text>

                {/* 증착층 */}
                <rect x="100" y="105" width="80" height="5" fill="#60A5FA" />
                <text x="190" y="110" fontSize="8" fill="#2563EB">Si₃N₄</text>

                {/* 배기 */}
                <line x1="230" y1="100" x2="260" y2="100" stroke="#EF4444" strokeWidth="3"
                      markerEnd="url(#arrow4)" />
                <text x="220" y="95" fontSize="9" fill="#EF4444">부산물</text>

                <defs>
                  <marker id="arrow4" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#6B7280" />
                  </marker>
                </defs>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>{`SiH₄ + 4NH₃ → Si₃N₄ + 12H₂

장점:
- 우수한 단차 피복성
- 균일한 두께
- 대면적 처리 가능

용도: 절연막, 배리어막`}</code>
            </div>
          </div>

          {/* ALD */}
          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-purple-800 dark:text-purple-300 mb-3">
              ALD (Atomic Layer Deposition)
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <svg className="w-full h-40" viewBox="0 0 280 160">
                {/* 사이클 표시 */}
                <text x="20" y="25" fontSize="11" fontWeight="bold" fill="#7C3AED">
                  자기제한 반응 (Self-limiting)
                </text>

                {/* 4단계 프로세스 */}
                <rect x="20" y="40" width="50" height="25" fill="#DDD6FE" stroke="#7C3AED" />
                <text x="25" y="50" fontSize="8" fill="#7C3AED">1. 전구체A</text>
                <text x="30" y="60" fontSize="7" fill="#6B7280">흡착</text>

                <rect x="85" y="40" width="50" height="25" fill="#FEE2E2" stroke="#EF4444" />
                <text x="92" y="50" fontSize="8" fill="#EF4444">2. 퍼지</text>
                <text x="95" y="60" fontSize="7" fill="#6B7280">잔류물 제거</text>

                <rect x="150" y="40" width="50" height="25" fill="#DBEAFE" stroke="#3B82F6" />
                <text x="155" y="50" fontSize="8" fill="#3B82F6">3. 전구체B</text>
                <text x="160" y="60" fontSize="7" fill="#6B7280">반응</text>

                <rect x="215" y="40" width="50" height="25" fill="#FEE2E2" stroke="#EF4444" />
                <text x="222" y="50" fontSize="8" fill="#EF4444">4. 퍼지</text>
                <text x="225" y="60" fontSize="7" fill="#6B7280">완료</text>

                {/* 화살표 */}
                <path d="M 70 52 L 80 52" stroke="#6B7280" strokeWidth="2" markerEnd="url(#arrow5)" />
                <path d="M 135 52 L 145 52" stroke="#6B7280" strokeWidth="2" markerEnd="url(#arrow5)" />
                <path d="M 200 52 L 210 52" stroke="#6B7280" strokeWidth="2" markerEnd="url(#arrow5)" />

                {/* 증착 결과 */}
                <rect x="50" y="90" width="180" height="15" fill="#E5E7EB" />
                <rect x="50" y="90" width="180" height="3" fill="#7C3AED" />
                <rect x="50" y="93" width="180" height="3" fill="#3B82F6" />
                <rect x="50" y="96" width="180" height="3" fill="#10B981" />
                <rect x="50" y="99" width="180" height="3" fill="#F59E0B" />
                <rect x="50" y="102" width="180" height="3" fill="#EF4444" />

                <text x="85" y="125" fontSize="9" fill="#6B7280">원자층 단위 제어</text>
                <text x="95" y="138" fontSize="8" fill="#7C3AED">1 사이클 = 0.1~0.3nm</text>

                <defs>
                  <marker id="arrow5" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="#6B7280" />
                  </marker>
                </defs>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>{`장점:
- 원자층 수준 정밀도
- 완벽한 단차 피복성
- 균일도 > 99%

용도: High-k 유전체,
      3D NAND 홀 라이너`}</code>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <h3 className="font-semibold text-blue-800 dark:text-blue-300 mb-3">
            PVD (Physical Vapor Deposition)
          </h3>
          <div className="grid md:grid-cols-2 gap-3">
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <h4 className="font-semibold text-sm mb-2">스퍼터링 (Sputtering)</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                아르곤 이온으로 타겟을 충격하여 원자를 방출
              </p>
              <div className="text-xs bg-gray-100 dark:bg-gray-700 p-2 rounded">
                용도: 금속 배선 (Cu, W, Al)
              </div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <h4 className="font-semibold text-sm mb-2">증발 (Evaporation)</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                고온으로 물질을 증발시켜 증착
              </p>
              <div className="text-xs bg-gray-100 dark:bg-gray-700 p-2 rounded">
                용도: 금속 콘택 (Al, Au)
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 에칭 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          4.3 에칭 (Etching)
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-red-800 dark:text-red-300 mb-3">
              건식 에칭 (Dry Etching / RIE)
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <svg className="w-full h-48" viewBox="0 0 250 200">
                {/* 플라즈마 챔버 */}
                <rect x="40" y="20" width="170" height="140" rx="5" fill="none"
                      stroke="#6B7280" strokeWidth="2" />

                {/* 상부 전극 */}
                <rect x="50" y="30" width="150" height="15" fill="#60A5FA" />
                <text x="95" y="24" fontSize="9" fill="#2563EB">RF 전극 (+)</text>

                {/* 플라즈마 */}
                {[...Array(20)].map((_, i) => (
                  <circle key={i}
                    cx={60 + Math.random() * 130}
                    cy={60 + Math.random() * 60}
                    r="2" fill="#A855F7" opacity="0.7" />
                ))}
                <text x="105" y="95" fontSize="10" fill="#7C3AED" fontWeight="bold">Plasma</text>

                {/* 이온 충격 */}
                <line x1="80" y1="60" x2="80" y2="130" stroke="#EF4444" strokeWidth="2"
                      markerEnd="url(#arrow6)" />
                <line x1="125" y1="60" x2="125" y2="130" stroke="#EF4444" strokeWidth="2"
                      markerEnd="url(#arrow6)" />
                <line x1="170" y1="60" x2="170" y2="130" stroke="#EF4444" strokeWidth="2"
                      markerEnd="url(#arrow6)" />
                <text x="75" y="75" fontSize="7" fill="#EF4444">CF₄⁺</text>

                {/* 웨이퍼 */}
                <rect x="60" y="130" width="130" height="10" fill="#9CA3AF" />
                <rect x="80" y="125" width="30" height="5" fill="#FCD34D" />
                <rect x="140" y="125" width="30" height="5" fill="#FCD34D" />
                <text x="95" y="155" fontSize="8" fill="#6B7280">웨이퍼</text>

                {/* 하부 전극 */}
                <rect x="50" y="145" width="150" height="10" fill="#3B82F6" />
                <text x="105" y="172" fontSize="9" fill="#2563EB">척 (-)</text>

                <defs>
                  <marker id="arrow6" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="#EF4444" />
                  </marker>
                </defs>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>{`장점:
- 이방성 (수직) 에칭
- 고종횡비 (AR > 50:1)
- 우수한 선택비

가스: CF₄, SF₆, Cl₂, BCl₃`}</code>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">
              습식 에칭 (Wet Etching)
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <svg className="w-full h-48" viewBox="0 0 250 200">
                {/* 에칭 전 */}
                <text x="50" y="25" fontSize="10" fontWeight="bold" fill="#374151">에칭 전</text>
                <rect x="40" y="40" width="80" height="40" fill="#E5E7EB" />
                <rect x="50" y="35" width="20" height="5" fill="#FCD34D" />
                <rect x="90" y="35" width="20" height="5" fill="#FCD34D" />

                {/* 에칭 중 */}
                <text x="130" y="95" fontSize="10" fontWeight="bold" fill="#374151">에칭 중</text>
                <rect x="130" y="110" width="80" height="40" fill="#DBEAFE" />
                <text x="155" y="135" fontSize="9" fill="#3B82F6">HF 용액</text>

                {/* 등방성 에칭 */}
                <path d="M 145 105 Q 145 115, 135 115 L 135 110"
                      fill="none" stroke="#EF4444" strokeWidth="2" />
                <path d="M 195 105 Q 195 115, 205 115 L 205 110"
                      fill="none" stroke="#EF4444" strokeWidth="2" />

                {/* 에칭 후 */}
                <text x="50" y="115" fontSize="10" fontWeight="bold" fill="#374151">에칭 후</text>
                <rect x="40" y="130" width="80" height="40" fill="#E5E7EB" />
                <path d="M 50 130 Q 55 125, 60 130"
                      fill="#E5E7EB" stroke="#374151" strokeWidth="1" />
                <path d="M 90 130 Q 95 125, 100 130"
                      fill="#E5E7EB" stroke="#374151" strokeWidth="1" />

                <text x="75" y="180" fontSize="8" fill="#EF4444">언더컷</text>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>{`특징:
- 등방성 에칭
- 저비용, 고처리량
- 언더컷 발생

용액:
- SiO₂: HF
- Si: KOH, TMAH
- Al: H₃PO₄`}</code>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 text-white p-4 rounded-lg">
          <h3 className="font-semibold mb-2">에칭 선택비 (Selectivity):</h3>
          <pre><code>{`선택비 = 목표막 에칭 속도 / 마스크막 에칭 속도

예시: Si 에칭 (Cl₂ 플라즈마)
- Si 에칭 속도: 500 nm/min
- SiO₂ 마스크 에칭: 25 nm/min
→ 선택비 = 500/25 = 20:1

목표: 선택비 > 10:1 (통상)
      선택비 > 50:1 (고종횡비 구조)`}</code></pre>
        </div>
      </section>

      {/* 이온주입과 CMP */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          4.4 이온주입 (Ion Implantation)과 CMP
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-indigo-800 dark:text-indigo-300 mb-3">
              이온주입
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <svg className="w-full h-40" viewBox="0 0 280 150">
                {/* 이온 소스 */}
                <circle cx="30" cy="40" r="15" fill="#FCD34D" stroke="#D97706" strokeWidth="2" />
                <text x="15" y="70" fontSize="8" fill="#D97706">이온 소스</text>

                {/* 질량 분석기 */}
                <path d="M 50 40 Q 80 20, 110 40" stroke="#3B82F6" strokeWidth="3" fill="none" />
                <text x="60" y="25" fontSize="8" fill="#3B82F6">질량 분석</text>

                {/* 가속기 */}
                <rect x="120" y="30" width="60" height="20" fill="#E0E7FF" stroke="#4F46E5" strokeWidth="2" />
                <text x="125" y="25" fontSize="8" fill="#4F46E5">가속 (10~500keV)</text>

                {/* 이온 빔 */}
                <line x1="180" y1="40" x2="220" y2="40" stroke="#EF4444" strokeWidth="3"
                      markerEnd="url(#arrow7)" />
                <line x1="180" y1="45" x2="220" y2="60" stroke="#EF4444" strokeWidth="2" opacity="0.6"
                      markerEnd="url(#arrow7)" />
                <line x1="180" y1="35" x2="220" y2="20" stroke="#EF4444" strokeWidth="2" opacity="0.6"
                      markerEnd="url(#arrow7)" />

                {/* 웨이퍼 */}
                <rect x="220" y="30" width="40" height="50" fill="#9CA3AF" stroke="#4B5563" strokeWidth="2" />

                {/* 농도 프로파일 */}
                <path d="M 225 100 L 225 130 L 255 130" stroke="#374151" strokeWidth="1" />
                <path d="M 225 115 Q 235 105, 245 115 L 255 120"
                      stroke="#10B981" strokeWidth="2" fill="none" />
                <text x="220" y="145" fontSize="7" fill="#6B7280">깊이 프로파일</text>

                <defs>
                  <marker id="arrow7" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="#EF4444" />
                  </marker>
                </defs>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>{`도펀트: B, P, As, Sb

주입 깊이 (Rp):
Rp ∝ √(E/M)

장점: 정밀한 도즈/깊이 제어
후공정: 어닐링 (활성화)`}</code>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-orange-800 dark:text-orange-300 mb-3">
              CMP (Chemical Mechanical Polishing)
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <svg className="w-full h-40" viewBox="0 0 280 150">
                {/* 연마 패드 */}
                <ellipse cx="140" cy="30" rx="80" ry="15" fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />
                <text x="110" y="20" fontSize="9" fill="#2563EB">연마 패드</text>

                {/* 회전 화살표 */}
                <path d="M 190 25 Q 210 25, 210 35" stroke="#10B981" strokeWidth="2" fill="none"
                      markerEnd="url(#arrow8)" />
                <text x="215" y="33" fontSize="7" fill="#10B981">회전</text>

                {/* 슬러리 */}
                <circle cx="100" cy="30" r="3" fill="#F59E0B" />
                <circle cx="130" cy="35" r="3" fill="#F59E0B" />
                <circle cx="160" cy="32" r="3" fill="#F59E0B" />
                <text x="85" y="55" fontSize="8" fill="#D97706">슬러리 (연마제 + 화학액)</text>

                {/* 웨이퍼 (연마 전) */}
                <rect x="80" y="80" width="120" height="10" fill="#9CA3AF" />
                <rect x="90" y="75" width="30" height="5" fill="#60A5FA" />
                <rect x="135" y="72" width="25" height="8" fill="#60A5FA" />
                <rect x="170" y="74" width="20" height="6" fill="#60A5FA" />
                <text x="50" y="87" fontSize="9" fill="#6B7280">연마 전</text>

                {/* 웨이퍼 (연마 후) */}
                <rect x="80" y="110" width="120" height="10" fill="#9CA3AF" />
                <rect x="80" y="105" width="120" height="5" fill="#60A5FA" />
                <text x="50" y="117" fontSize="9" fill="#6B7280">연마 후</text>

                {/* 평탄화 화살표 */}
                <line x1="210" y1="85" x2="210" y2="110" stroke="#10B981" strokeWidth="2"
                      markerEnd="url(#arrow8)" />
                <text x="215" y="100" fontSize="8" fill="#10B981">평탄화</text>

                <defs>
                  <marker id="arrow8" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="#10B981" />
                  </marker>
                </defs>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>{`슬러리 성분:
- SiO₂: 콜로이달 실리카
- Cu: H₂O₂ + 글라이신

제거율: 100~500nm/min
균일도: < 2% (3σ)

용도: 층간 평탄화, Cu Damascene`}</code>
            </div>
          </div>
        </div>
      </section>

      {/* 요약 */}
      <section className="mb-8">
        <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            핵심 요약
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>웨이퍼는 CZ 공정으로 단결정 잉곳을 성장시켜 제조하며, 현재 300mm가 주류입니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>증착 공정은 CVD, ALD, PVD 등이 있으며, 각각 용도와 특성이 다릅니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>건식 에칭은 이방성으로 고종횡비 구조 형성에 필수적입니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>이온주입과 CMP는 도핑 제어와 표면 평탄화의 핵심 공정입니다</span>
            </li>
          </ul>
        </div>
      </section>

      <References
        sections={[
          {
            title: '📚 공식 문서 & 표준',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'SEMI Equipment Standards',
                link: 'https://www.semi.org/en/products-services/standards',
                description: 'SEMI 반도체 장비 및 공정 국제 표준 (E-series, F-series)'
              },
              {
                title: 'Applied Materials - Process Technology',
                link: 'https://www.appliedmaterials.com/semiconductor/semiconductor-fabrication',
                description: 'Applied Materials 증착/에칭 장비 기술 문서 (1위 업체)'
              },
              {
                title: 'Lam Research - Etch and Deposition Systems',
                link: 'https://www.lamresearch.com/',
                description: 'Lam Research 에칭/증착 시스템 공식 백서'
              },
              {
                title: 'Tokyo Electron (TEL) - Semiconductor Production Equipment',
                link: 'https://www.tel.com/products/',
                description: 'Tokyo Electron 반도체 생산 장비 기술 자료'
              }
            ]
          },
          {
            title: '📖 핵심 교재 & 논문',
            icon: 'book' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Semiconductor Manufacturing Technology',
                authors: 'Quirk, M., Serda, J.',
                year: '2001',
                description: 'Prentice Hall - 반도체 제조 공정 표준 교재'
              },
              {
                title: 'Silicon VLSI Technology: Fundamentals, Practice, and Modeling',
                authors: 'Plummer, J. D., Deal, M. D., Griffin, P. B.',
                year: '2000',
                description: 'Prentice Hall - Stanford 대학 VLSI 제조 교재'
              },
              {
                title: 'Chemical Mechanical Polishing: Theory and Practice',
                authors: 'Oliver, M. R.',
                year: '2004',
                description: 'Elsevier - CMP 공정 완전 가이드'
              },
              {
                title: 'Atomic Layer Deposition: Principles, Characteristics, and Nanotechnology',
                authors: 'George, S. M.',
                year: '2010',
                description: 'Chemical Reviews - ALD 기술 리뷰 논문 (고인용 논문)'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 장비',
            icon: 'web' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Synopsys Sentaurus TCAD - Process Simulation',
                link: 'https://www.synopsys.com/silicon/tcad.html',
                description: 'Synopsys 반도체 공정 시뮬레이션 툴 (TCAD)'
              },
              {
                title: 'Silvaco Athena - Process Simulator',
                link: 'https://silvaco.com/tcad/athena/',
                description: 'Silvaco 증착/에칭/확산 공정 시뮬레이터'
              },
              {
                title: 'KLA-Tencor - Process Control & Metrology',
                link: 'https://www.kla.com/products/process-control',
                description: 'KLA 공정 제어 및 계측 시스템 (업계 1위)'
              },
              {
                title: 'ASML - Wafer Processing Integration',
                link: 'https://www.asml.com/',
                description: 'ASML 웨이퍼 처리 및 리소그래피 통합 솔루션'
              },
              {
                title: 'Virtual Wafer Fab - Online Simulator',
                link: 'https://www.virtualwafer.com/',
                description: '온라인 반도체 공정 시뮬레이터 (교육용 무료)'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
