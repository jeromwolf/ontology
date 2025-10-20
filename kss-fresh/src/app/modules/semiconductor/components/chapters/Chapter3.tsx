'use client'

import References from '@/components/common/References';

export default function Chapter3() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Chapter 3: 포토리소그래피 (Photolithography)
      </h1>

      {/* 포토리소그래피 개요 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          3.1 포토리소그래피 개요
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            포토리소그래피는 반도체 제조의 핵심 공정으로, 빛을 이용하여 웨이퍼 위에
            나노미터 수준의 패턴을 전사하는 기술입니다. 반도체 칩의 모든 레이어는
            이 공정을 통해 형성됩니다.
          </p>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <svg className="w-full h-64" viewBox="0 0 600 280">
              {/* 광원 */}
              <circle cx="50" cy="40" r="20" fill="#FCD34D" stroke="#F59E0B" strokeWidth="2" />
              <text x="30" y="75" fontSize="11" fill="#D97706">광원</text>

              {/* 광선 */}
              <line x1="70" y1="40" x2="140" y2="40" stroke="#FCD34D" strokeWidth="3" />
              <line x1="70" y1="35" x2="140" y2="35" stroke="#FCD34D" strokeWidth="2" />
              <line x1="70" y1="45" x2="140" y2="45" stroke="#FCD34D" strokeWidth="2" />

              {/* 마스크 */}
              <rect x="140" y="20" width="100" height="5" fill="#374151" />
              <rect x="140" y="55" width="100" height="5" fill="#374151" />
              <rect x="165" y="25" width="25" height="30" fill="none" />
              <rect x="215" y="25" width="25" height="30" fill="none" />
              <text x="160" y="15" fontSize="11" fill="#6B7280">마스크</text>

              {/* 투과 광선 */}
              <line x1="177" y1="60" x2="177" y2="120" stroke="#FCD34D" strokeWidth="3" />
              <line x1="227" y1="60" x2="227" y2="120" stroke="#FCD34D" strokeWidth="3" />

              {/* 렌즈 시스템 */}
              <ellipse cx="202" cy="100" rx="40" ry="15" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
              <text x="170" y="95" fontSize="11" fill="#2563EB">렌즈</text>

              {/* 축소된 광선 */}
              <line x1="177" y1="115" x2="192" y2="150" stroke="#FCD34D" strokeWidth="2" />
              <line x1="227" y1="115" x2="212" y2="150" stroke="#FCD34D" strokeWidth="2" />

              {/* 웨이퍼 구조 */}
              <rect x="150" y="170" width="120" height="10" fill="#A78BFA" />
              <text x="280" y="178" fontSize="10" fill="#7C3AED">포토레지스트</text>

              <rect x="150" y="180" width="120" height="15" fill="#60A5FA" />
              <text x="280" y="190" fontSize="10" fill="#2563EB">하부층</text>

              <rect x="150" y="195" width="120" height="30" fill="#9CA3AF" />
              <text x="280" y="213" fontSize="10" fill="#4B5563">실리콘 웨이퍼</text>

              {/* 노광 후 */}
              <text x="350" y="30" fontSize="12" fontWeight="bold" fill="#374151">현상 후</text>

              {/* 패턴 형성 */}
              <rect x="420" y="170" width="25" height="10" fill="#A78BFA" />
              <rect x="465" y="170" width="25" height="10" fill="#A78BFA" />

              <rect x="420" y="180" width="70" height="15" fill="#60A5FA" />
              <rect x="420" y="195" width="70" height="30" fill="#9CA3AF" />

              <text x="495" y="178" fontSize="9" fill="#7C3AED">레지스트 패턴</text>

              {/* 화살표 */}
              <defs>
                <marker id="arrowhead2" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="#10B981" />
                </marker>
              </defs>
              <line x1="305" y1="190" x2="400" y2="190" stroke="#10B981" strokeWidth="2" markerEnd="url(#arrowhead2)" />
              <text x="330" y="185" fontSize="10" fill="#10B981">현상</text>
            </svg>
            <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-3">
              포토리소그래피 공정 흐름
            </p>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-2">1. 노광</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              UV 광원으로 마스크 패턴을 포토레지스트에 전사
            </p>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-green-800 dark:text-green-300 mb-2">2. 현상</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              노광된 영역을 화학적으로 제거하여 패턴 형성
            </p>
          </div>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">3. 에칭</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              레지스트 패턴을 마스크로 하부층 식각
            </p>
          </div>
        </div>
      </section>

      {/* EUV */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          3.2 EUV (Extreme Ultraviolet) 리소그래피
        </h2>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            EUV는 13.5nm 파장의 극자외선을 사용하는 차세대 리소그래피 기술로,
            7nm 이하 공정에서 필수적입니다. ASML이 독점 공급하는 EUV 노광기는
            반도체 산업의 가장 중요한 장비입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                광원 비교
              </h3>
              <div className="space-y-3">
                <div className="border-l-4 border-blue-500 pl-3">
                  <div className="font-semibold text-blue-700 dark:text-blue-400">ArF (DUV)</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">파장: 193nm</div>
                  <div className="text-xs text-gray-500">~22nm 공정까지</div>
                </div>
                <div className="border-l-4 border-purple-500 pl-3">
                  <div className="font-semibold text-purple-700 dark:text-purple-400">EUV</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">파장: 13.5nm</div>
                  <div className="text-xs text-gray-500">7nm 이하 공정</div>
                </div>
                <div className="border-l-4 border-green-500 pl-3">
                  <div className="font-semibold text-green-700 dark:text-green-400">High-NA EUV</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">NA: 0.55</div>
                  <div className="text-xs text-gray-500">2nm 이하 공정</div>
                </div>
              </div>
            </div>

            <div className="bg-gray-800 text-white p-4 rounded-lg">
              <h3 className="font-semibold mb-3 text-sm">해상도 공식:</h3>
              <pre className="text-xs"><code>{`R = k₁ × λ / NA

R: 최소 해상도
k₁: 공정 상수 (0.25~0.30)
λ: 파장
NA: 개구수 (Numerical Aperture)

DUV (193nm, NA=1.35):
R = 0.25 × 193 / 1.35 ≈ 36nm

EUV (13.5nm, NA=0.33):
R = 0.25 × 13.5 / 0.33 ≈ 10nm

High-NA EUV (13.5nm, NA=0.55):
R = 0.25 × 13.5 / 0.55 ≈ 6nm`}</code></pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
              EUV 시스템 구조
            </h3>
            <svg className="w-full h-48" viewBox="0 0 700 200">
              {/* 플라즈마 광원 */}
              <circle cx="50" cy="100" r="25" fill="#FCD34D" stroke="#D97706" strokeWidth="2" />
              <text x="30" y="140" fontSize="10" fill="#D97706">Sn Plasma</text>
              <text x="35" y="152" fontSize="8" fill="#6B7280">13.5nm</text>

              {/* 집광 미러 */}
              <path d="M 100 60 Q 150 80, 150 100 Q 150 120, 100 140"
                    fill="#E0E7FF" stroke="#4F46E5" strokeWidth="2" />
              <text x="110" y="155" fontSize="9" fill="#4F46E5">Collector</text>

              {/* 중간 미러들 */}
              <rect x="220" y="85" width="30" height="30" rx="5" fill="#E0E7FF" stroke="#4F46E5" strokeWidth="2" />
              <text x="215" y="130" fontSize="9" fill="#4F46E5">Mirror 1</text>

              <rect x="320" y="85" width="30" height="30" rx="5" fill="#E0E7FF" stroke="#4F46E5" strokeWidth="2" />
              <text x="315" y="130" fontSize="9" fill="#4F46E5">Mirror 2</text>

              {/* 마스크 (반사형) */}
              <rect x="420" y="75" width="50" height="50" fill="#F3F4F6" stroke="#374151" strokeWidth="2" />
              <rect x="430" y="85" width="15" height="30" fill="#1F2937" />
              <text x="420" y="140" fontSize="9" fill="#374151">Reflective</text>
              <text x="430" y="152" fontSize="9" fill="#374151">Mask</text>

              {/* 투사 광학계 */}
              <ellipse cx="540" cy="100" rx="35" ry="20" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
              <text x="515" y="140" fontSize="9" fill="#2563EB">Projection</text>
              <text x="525" y="152" fontSize="9" fill="#2563EB">Optics</text>

              {/* 웨이퍼 */}
              <rect x="630" y="90" width="50" height="20" fill="#9CA3AF" stroke="#4B5563" strokeWidth="2" />
              <text x="640" y="125" fontSize="9" fill="#4B5563">Wafer</text>

              {/* 광선 경로 */}
              <line x1="75" y1="100" x2="100" y2="100" stroke="#FCD34D" strokeWidth="2" strokeDasharray="3,3" />
              <line x1="150" y1="100" x2="220" y2="100" stroke="#FCD34D" strokeWidth="2" strokeDasharray="3,3" />
              <line x1="250" y1="100" x2="320" y2="100" stroke="#FCD34D" strokeWidth="2" strokeDasharray="3,3" />
              <line x1="350" y1="100" x2="420" y2="100" stroke="#FCD34D" strokeWidth="2" strokeDasharray="3,3" />
              <line x1="470" y1="100" x2="505" y2="100" stroke="#FCD34D" strokeWidth="2" strokeDasharray="3,3" />
              <line x1="575" y1="100" x2="630" y2="100" stroke="#FCD34D" strokeWidth="2" strokeDasharray="3,3" />

              {/* 진공 챔버 표시 */}
              <rect x="10" y="10" width="670" height="180" rx="10" fill="none" stroke="#6B7280" strokeWidth="2" strokeDasharray="5,5" />
              <text x="300" y="25" fontSize="11" fill="#6B7280">진공 환경 (10⁻⁶ Pa)</text>
            </svg>
            <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-2">
              EUV는 모든 광학계가 반사형 미러로 구성 (EUV는 대부분 물질을 투과하지 못함)
            </p>
          </div>
        </div>

        <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
          <h3 className="font-semibold text-red-800 dark:text-red-300 mb-3">EUV의 기술적 과제</h3>
          <div className="grid md:grid-cols-2 gap-3 text-sm text-gray-700 dark:text-gray-300">
            <div>
              <strong>광원 출력:</strong> 250W 이상 필요 (현재 ~350W)
            </div>
            <div>
              <strong>마스크 결함:</strong> 나노 단위 결함도 치명적
            </div>
            <div>
              <strong>레지스트 감도:</strong> 높은 감도와 해상도 양립 어려움
            </div>
            <div>
              <strong>장비 가격:</strong> 1대당 2,000억원 이상
            </div>
          </div>
        </div>
      </section>

      {/* 마스크 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          3.3 마스크(Mask) 및 레티클(Reticle)
        </h2>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            마스크는 회로 패턴이 그려진 원판으로, 수십억원의 가치를 가진 정밀 부품입니다.
            하나의 칩을 만들기 위해 40~60개의 마스크가 필요합니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                바이너리 마스크 (Binary Mask)
              </h3>
              <svg className="w-full h-40" viewBox="0 0 300 160">
                {/* 석영 기판 */}
                <rect x="50" y="40" width="200" height="15" fill="#E0E7FF" stroke="#4F46E5" strokeWidth="1" />
                <text x="120" y="35" fontSize="10" fill="#4F46E5">석영 기판</text>

                {/* 크롬 패턴 */}
                <rect x="70" y="55" width="40" height="8" fill="#374151" />
                <rect x="140" y="55" width="30" height="8" fill="#374151" />
                <rect x="200" y="55" width="35" height="8" fill="#374151" />
                <text x="130" y="75" fontSize="9" fill="#374151">크롬 (Cr) 패턴</text>

                {/* 입사광 */}
                {[80, 100, 120, 160, 180, 210, 230].map((x, i) => (
                  <line key={i} x1={x} y1="20" x2={x} y2="40" stroke="#FCD34D" strokeWidth="2" />
                ))}
                <text x="130" y="15" fontSize="10" fill="#D97706">입사광</text>

                {/* 투과/차단 */}
                {[100, 180].map((x, i) => (
                  <line key={i} x1={x} y1="63" x2={x} y2="90" stroke="#10B981" strokeWidth="2" />
                ))}
                <text x="85" y="105" fontSize="9" fill="#10B981">투과</text>

                {[80, 150, 217].map((x, i) => (
                  <text key={i} x={x} y="80" fontSize="14" fill="#EF4444">✕</text>
                ))}
                <text x="140" y="105" fontSize="9" fill="#EF4444">차단</text>

                {/* 패턴 결과 */}
                <rect x="95" y="115" width="15" height="30" fill="#A78BFA" />
                <rect x="175" y="115" width="15" height="30" fill="#A78BFA" />
                <text x="110" y="155" fontSize="9" fill="#7C3AED">레지스트 패턴</text>
              </svg>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                위상 시프트 마스크 (PSM)
              </h3>
              <svg className="w-full h-40" viewBox="0 0 300 160">
                {/* 석영 기판 */}
                <rect x="50" y="40" width="200" height="15" fill="#E0E7FF" stroke="#4F46E5" strokeWidth="1" />

                {/* 위상 시프터 */}
                <rect x="120" y="30" width="60" height="10" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="1" />
                <text x="125" y="28" fontSize="9" fill="#3B82F6">위상 시프터</text>

                {/* 입사광 */}
                {[80, 100, 120, 140, 160, 180, 200, 220].map((x, i) => (
                  <line key={i} x1={x} y1="15" x2={x} y2="30" stroke="#FCD34D" strokeWidth="2" />
                ))}

                {/* 위상 변화 */}
                <path d="M 80 70 Q 100 60, 120 70" stroke="#10B981" strokeWidth="2" fill="none" />
                <path d="M 120 70 Q 140 80, 160 70 Q 180 60, 200 70" stroke="#EF4444" strokeWidth="2" fill="none" />
                <path d="M 200 70 Q 220 80, 240 70" stroke="#10B981" strokeWidth="2" fill="none" />

                <text x="85" y="90" fontSize="8" fill="#10B981">0°</text>
                <text x="145" y="95" fontSize="8" fill="#EF4444">180°</text>
                <text x="215" y="90" fontSize="8" fill="#10B981">0°</text>

                {/* 간섭 효과 */}
                <rect x="125" y="110" width="50" height="35" fill="#A78BFA" opacity="0.7" />
                <text x="100" y="155" fontSize="9" fill="#7C3AED">상쇄간섭으로 향상된 해상도</text>
              </svg>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 text-white p-4 rounded-lg mb-4">
          <h3 className="font-semibold mb-2">마스크 제작 공정:</h3>
          <pre><code>{`1. 석영 기판 준비 (152mm × 152mm)
   → 표면 평탄도: λ/4 이하

2. 크롬 증착 (100nm)
   → 광학 밀도(OD) > 3.0

3. 레지스트 도포 및 전자빔(E-beam) 노광
   → 해상도: ~10nm
   → 노광 시간: 수 시간

4. 현상 및 크롬 에칭

5. 검사 및 수리
   → 허용 결함: 0개 목표
   → 펠리클(Pellicle) 부착

한 세트 제작 비용: 수억~수십억원
제작 기간: 4~8주`}</code></pre>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
          <h3 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">OPC (Optical Proximity Correction)</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
            광학 근접 효과를 보정하기 위해 마스크 패턴을 의도적으로 왜곡시키는 기술:
          </p>
          <div className="grid grid-cols-3 gap-3 text-xs text-gray-700 dark:text-gray-300">
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong>Serif:</strong> 모서리에 작은 돌기 추가
            </div>
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong>Hammerhead:</strong> 라인 끝단 확장
            </div>
            <div className="bg-white dark:bg-gray-800 p-2 rounded">
              <strong>SRAF:</strong> 보조 패턴 삽입
            </div>
          </div>
        </div>
      </section>

      {/* 다중 패터닝 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          3.4 다중 패터닝 (Multi-Patterning)
        </h2>

        <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            단일 노광으로 달성할 수 없는 미세 패턴을 여러 번의 노광으로 구현하는 기술입니다.
            EUV 이전 시대(FinFET 10nm~14nm)의 핵심 기술이었습니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                LELE (Litho-Etch-Litho-Etch)
              </h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded flex items-center justify-center font-bold text-blue-700 dark:text-blue-300">
                    1
                  </div>
                  <div className="flex-1">
                    <div className="font-semibold text-sm">1차 노광/에칭</div>
                    <svg className="w-full h-8 mt-1" viewBox="0 0 150 30">
                      <rect x="20" y="10" width="30" height="10" fill="#3B82F6" />
                      <rect x="80" y="10" width="30" height="10" fill="#3B82F6" />
                    </svg>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-green-100 dark:bg-green-900 rounded flex items-center justify-center font-bold text-green-700 dark:text-green-300">
                    2
                  </div>
                  <div className="flex-1">
                    <div className="font-semibold text-sm">2차 노광/에칭</div>
                    <svg className="w-full h-8 mt-1" viewBox="0 0 150 30">
                      <rect x="20" y="10" width="30" height="10" fill="#3B82F6" />
                      <rect x="55" y="10" width="20" height="10" fill="#10B981" />
                      <rect x="80" y="10" width="30" height="10" fill="#3B82F6" />
                      <rect x="115" y="10" width="20" height="10" fill="#10B981" />
                    </svg>
                  </div>
                </div>

                <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
                  최종 피치: 원래의 1/2
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                SAQP (Self-Aligned Quadruple Patterning)
              </h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded flex items-center justify-center font-bold text-purple-700 dark:text-purple-300">
                    1
                  </div>
                  <div className="flex-1">
                    <div className="font-semibold text-sm">코어 형성</div>
                    <svg className="w-full h-8 mt-1" viewBox="0 0 150 30">
                      <rect x="50" y="12" width="15" height="6" fill="#A855F7" />
                      <rect x="85" y="12" width="15" height="6" fill="#A855F7" />
                    </svg>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900 rounded flex items-center justify-center font-bold text-orange-700 dark:text-orange-300">
                    2
                  </div>
                  <div className="flex-1">
                    <div className="font-semibold text-sm">스페이서 증착 2회</div>
                    <svg className="w-full h-8 mt-1" viewBox="0 0 150 30">
                      <rect x="40" y="12" width="8" height="6" fill="#F59E0B" />
                      <rect x="50" y="12" width="15" height="6" fill="#A855F7" />
                      <rect x="67" y="12" width="8" height="6" fill="#F59E0B" />
                      <rect x="75" y="12" width="8" height="6" fill="#F59E0B" />
                      <rect x="85" y="12" width="15" height="6" fill="#A855F7" />
                      <rect x="102" y="12" width="8" height="6" fill="#F59E0B" />
                    </svg>
                  </div>
                </div>

                <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
                  최종 피치: 원래의 1/4 (자기정렬)
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 text-white p-4 rounded-lg">
            <h3 className="font-semibold mb-2">다중 패터닝 기술 비교:</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-600">
                    <th className="px-3 py-2 text-left">기법</th>
                    <th className="px-3 py-2 text-left">피치 감소</th>
                    <th className="px-3 py-2 text-left">공정 단계</th>
                    <th className="px-3 py-2 text-left">정렬 오차</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-700">
                    <td className="px-3 py-2">LELE</td>
                    <td className="px-3 py-2">1/2</td>
                    <td className="px-3 py-2 text-yellow-400">중간</td>
                    <td className="px-3 py-2 text-red-400">높음</td>
                  </tr>
                  <tr className="border-b border-gray-700">
                    <td className="px-3 py-2">SADP</td>
                    <td className="px-3 py-2">1/2</td>
                    <td className="px-3 py-2 text-yellow-400">중간</td>
                    <td className="px-3 py-2 text-green-400">낮음</td>
                  </tr>
                  <tr>
                    <td className="px-3 py-2">SAQP</td>
                    <td className="px-3 py-2">1/4</td>
                    <td className="px-3 py-2 text-red-400">많음</td>
                    <td className="px-3 py-2 text-green-400">매우낮음</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* 요약 */}
      <section className="mb-8">
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            핵심 요약
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>포토리소그래피는 빛을 이용해 나노미터 수준의 패턴을 웨이퍼에 전사하는 핵심 공정입니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>EUV는 13.5nm 파장으로 7nm 이하 공정을 가능하게 하는 차세대 기술입니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>마스크는 수십억원 가치의 정밀 부품으로, OPC 등의 보정 기술이 필수적입니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>다중 패터닝은 단일 노광의 한계를 극복하여 더 미세한 패턴을 구현합니다</span>
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
                title: 'ASML - EUV Technology Leadership',
                link: 'https://www.asml.com/en/products/euv-lithography-systems',
                description: 'ASML EUV 리소그래피 시스템 공식 문서 (세계 유일 공급업체)'
              },
              {
                title: 'SPIE - Advanced Lithography Conference',
                link: 'https://spie.org/conferences-and-exhibitions/advanced-lithography',
                description: 'SPIE 국제 리소그래피 컨퍼런스 논문집 (연례 행사)'
              },
              {
                title: 'SEMI P37 - Specifications for Photomasks',
                link: 'https://www.semi.org/',
                description: 'SEMI 포토마스크 규격 및 품질 표준'
              },
              {
                title: 'ITRS Lithography Roadmap',
                link: 'https://www.irds.ieee.org/',
                description: 'ITRS 리소그래피 기술 로드맵 (EUV, High-NA EUV)'
              }
            ]
          },
          {
            title: '📖 핵심 논문 & 교재',
            icon: 'book' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Microlithography: Science and Technology (2nd Edition)',
                authors: 'Mack, C. A.',
                year: '2007',
                description: 'CRC Press - 포토리소그래피 표준 교재'
              },
              {
                title: 'EUV Lithography: Enabling Moore\'s Law',
                authors: 'Bakshi, V.',
                year: '2018',
                description: 'SPIE Press - EUV 리소그래피 완전 가이드'
              },
              {
                title: 'OPC and Multi-Patterning: Critical Issues and Solutions',
                authors: 'Wong, A. K. K.',
                year: '2012',
                description: 'SPIE - OPC 및 다중 패터닝 기술 백서'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 리소스',
            icon: 'web' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Mentor Calibre - OPC and Mask Synthesis',
                link: 'https://eda.sw.siemens.com/en-US/ic/calibre-design/',
                description: 'Siemens EDA OPC 및 마스크 합성 툴 (업계 표준)'
              },
              {
                title: 'Synopsys Proteus - Mask Data Preparation',
                link: 'https://www.synopsys.com/silicon/mask-synthesis.html',
                description: 'Synopsys 마스크 데이터 준비 및 검증 툴'
              },
              {
                title: 'KLA - Mask Inspection Systems',
                link: 'https://www.kla.com/products/mask-inspection',
                description: 'KLA 마스크 검사 시스템 및 결함 분석 솔루션'
              },
              {
                title: 'Prolith - Lithography Simulator',
                link: 'https://www.kla.com/products/computational-lithography',
                description: 'KLA 리소그래피 시뮬레이션 소프트웨어'
              },
              {
                title: 'Nikon/Canon - DUV Scanner Documentation',
                link: 'https://www.nikonusa.com/',
                description: 'Nikon/Canon DUV 노광 장비 공식 문서'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
