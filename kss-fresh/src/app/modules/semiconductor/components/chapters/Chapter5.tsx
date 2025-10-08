'use client'

import References from '@/components/common/References';

export default function Chapter5() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Chapter 5: 첨단 반도체 기술
      </h1>

      {/* FinFET */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          5.1 FinFET (Fin Field-Effect Transistor)
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            FinFET은 22nm 공정부터 도입된 3D 트랜지스터 구조로, 게이트가 채널을
            3면에서 감싸 우수한 전류 제어와 낮은 누설전류를 실현합니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                평면 MOSFET vs FinFET
              </h3>
              <svg className="w-full h-48" viewBox="0 0 300 200">
                {/* 평면 MOSFET */}
                <text x="30" y="20" fontSize="11" fontWeight="bold" fill="#6B7280">
                  평면 MOSFET
                </text>
                <rect x="30" y="60" width="100" height="30" fill="#FEE2E2" />
                <rect x="40" y="55" width="20" height="5" fill="#3B82F6" />
                <rect x="100" y="55" width="20" height="5" fill="#3B82F6" />
                <rect x="50" y="40" width="60" height="8" fill="#9CA3AF" />
                <line x1="80" y1="30" x2="80" y2="40" stroke="#374151" strokeWidth="2" />
                <text x="75" y="28" fontSize="8" fill="#374151">G</text>
                <text x="45" y="68" fontSize="7" fill="#EF4444">누설전류 ↑</text>

                {/* FinFET */}
                <text x="170" y="20" fontSize="11" fontWeight="bold" fill="#6B7280">
                  FinFET
                </text>

                {/* Fin 구조 (3D) */}
                <path d="M 200 60 L 200 90 L 210 85 L 210 55 Z"
                      fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
                <path d="M 240 60 L 240 90 L 250 85 L 250 55 Z"
                      fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />

                {/* Gate wrapping */}
                <path d="M 195 65 L 195 75 L 205 70 L 215 75 L 215 65 L 205 60 Z"
                      fill="#9CA3AF" opacity="0.7" stroke="#4B5563" strokeWidth="1" />
                <path d="M 235 65 L 235 75 L 245 70 L 255 75 L 255 65 L 245 60 Z"
                      fill="#9CA3AF" opacity="0.7" stroke="#4B5563" strokeWidth="1" />

                <text x="205" y="105" fontSize="7" fill="#10B981">누설전류 ↓</text>

                {/* 비교 화살표 */}
                <line x1="140" y1="75" x2="160" y2="75" stroke="#F59E0B" strokeWidth="2"
                      markerEnd="url(#arrow9)" />
                <text x="142" y="70" fontSize="8" fill="#F59E0B">진화</text>

                <defs>
                  <marker id="arrow9" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="#F59E0B" />
                  </marker>
                </defs>
              </svg>
            </div>

            <div className="bg-gray-800 text-white p-4 rounded-lg">
              <h3 className="font-semibold mb-3 text-sm">FinFET의 장점:</h3>
              <pre className="text-xs"><code>{`1. 게이트 제어 향상
   - 3면 게이트 → 전류 제어 3배
   - SCE (Short Channel Effect) 억제

2. 누설전류 감소
   - DIBL 개선: 60mV/V → 30mV/V
   - 서브스레시홀드 스윙: 80→65mV/dec

3. 성능 향상
   - 구동전류 37% 증가
   - 스위칭 속도 25% 향상

4. 전력 효율
   - 동일 성능에서 50% 저전력
   - 또는 동일 전력에서 37% 고성능

공정 노드:
22nm (Intel 2012)
14nm (Intel/TSMC 2014)
10nm, 7nm, 5nm...`}</code></pre>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">
              FinFET 설계 파라미터
            </h3>
            <div className="grid md:grid-cols-3 gap-3 text-sm text-gray-700 dark:text-gray-300">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-semibold mb-1">Fin Height</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">30~50nm</div>
                <div className="text-xs text-blue-600">높을수록 구동전류↑</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-semibold mb-1">Fin Width</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">5~10nm</div>
                <div className="text-xs text-blue-600">좁을수록 SCE↓</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-semibold mb-1">Fin Pitch</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">20~40nm</div>
                <div className="text-xs text-blue-600">집적도 결정</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* GAA */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          5.2 GAA (Gate-All-Around) / Nanosheet
        </h2>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            GAA는 FinFET의 다음 세대로, 게이트가 채널을 완전히 감싸 최고의 전기적
            제어를 제공합니다. Samsung 3nm, Intel 20A에서 채택되었습니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                구조 진화
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* FinFET */}
                <text x="20" y="25" fontSize="10" fontWeight="bold" fill="#3B82F6">
                  FinFET (3-side)
                </text>
                <rect x="40" y="40" width="60" height="8" fill="#9CA3AF" opacity="0.5" />
                <rect x="55" y="48" width="30" height="35" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
                <text x="50" y="95" fontSize="8" fill="#6B7280">Gate 3면</text>

                {/* Nanowire */}
                <text x="140" y="25" fontSize="10" fontWeight="bold" fill="#7C3AED">
                  Nanowire (4-side)
                </text>
                <circle cx="190" cy="65" r="20" fill="#9CA3AF" opacity="0.3" />
                <circle cx="190" cy="65" r="12" fill="#DDD6FE" stroke="#7C3AED" strokeWidth="2" />
                <text x="165" y="95" fontSize="8" fill="#6B7280">Gate 4면 (원형)</text>

                {/* Nanosheet */}
                <text x="50" y="130" fontSize="10" fontWeight="bold" fill="#10B981">
                  Nanosheet (4-side)
                </text>
                <rect x="40" y="145" width="80" height="70" fill="#9CA3AF" opacity="0.2" />

                {/* 3개의 나노시트 스택 */}
                <rect x="50" y="155" width="60" height="8" fill="#D1FAE5" stroke="#10B981" strokeWidth="2" />
                <rect x="50" y="175" width="60" height="8" fill="#D1FAE5" stroke="#10B981" strokeWidth="2" />
                <rect x="50" y="195" width="60" height="8" fill="#D1FAE5" stroke="#10B981" strokeWidth="2" />

                <text x="45" y="225" fontSize="8" fill="#6B7280">Gate 4면 (사각)</text>
                <text x="40" y="237" fontSize="7" fill="#10B981">폭 조절 가능</text>

                {/* CFET 개념 */}
                <text x="150" y="130" fontSize="10" fontWeight="bold" fill="#EF4444">
                  CFET (미래)
                </text>
                <rect x="160" y="155" width="60" height="8" fill="#FEE2E2" stroke="#EF4444" strokeWidth="2" />
                <text x="180" y="161" fontSize="6" fill="#DC2626">PMOS</text>

                <rect x="160" y="175" width="60" height="8" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
                <text x="180" y="181" fontSize="6" fill="#2563EB">NMOS</text>

                <text x="155" y="200" fontSize="7" fill="#6B7280">수직 적층</text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">GAA 장점:</h4>
                <code>{`1. 최고의 게이트 제어
   - 4면 완전 감싸기
   - DIBL < 20mV/V

2. 채널 폭 조절
   - Nanosheet 폭 변경 가능
   - NMOS/PMOS 최적화

3. 집적도 향상
   - 수직 적층 (3~4단)
   - 면적 20% 감소

4. 성능 향상
   - FinFET 대비 15% 고성능
   - 또는 25% 저전력`}</code>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded">
                <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">
                  채택 로드맵
                </h4>
                <div className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                  <div className="flex justify-between">
                    <span>Samsung 3nm (2022)</span>
                    <span className="text-green-600">✓ GAA</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Intel 20A (2024)</span>
                    <span className="text-green-600">✓ RibbonFET</span>
                  </div>
                  <div className="flex justify-between">
                    <span>TSMC 2nm (2025)</span>
                    <span className="text-green-600">✓ Nanosheet</span>
                  </div>
                  <div className="flex justify-between">
                    <span>CFET (2028~)</span>
                    <span className="text-gray-500">연구 중</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3D NAND */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          5.3 3D NAND Flash
        </h2>

        <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            3D NAND는 메모리 셀을 수직으로 적층하여 집적도를 획기적으로 향상시킨
            기술입니다. 현재 200단 이상까지 발전했습니다.
          </p>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
            <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
              3D NAND 구조
            </h3>
            <svg className="w-full h-64" viewBox="0 0 600 280">
              {/* 워드라인 스택 */}
              <text x="20" y="25" fontSize="11" fontWeight="bold" fill="#374151">
                워드라인 적층 구조
              </text>

              {/* 다층 워드라인 */}
              {[...Array(16)].map((_, i) => (
                <g key={i}>
                  <rect x="50" y={50 + i * 12} width="150" height="8"
                        fill={i % 2 === 0 ? "#60A5FA" : "#9CA3AF"}
                        stroke="#374151" strokeWidth="0.5" />
                  {i === 0 && <text x="205" y={57 + i * 12} fontSize="7" fill="#6B7280">WL{i}</text>}
                  {i === 15 && <text x="205" y={57 + i * 12} fontSize="7" fill="#6B7280">WL{i}</text>}
                </g>
              ))}

              <text x="70" y="255" fontSize="9" fill="#2563EB">워드라인 (W/SiO₂)</text>
              <text x="90" y="268" fontSize="8" fill="#6B7280">128~200+ 층</text>

              {/* 비트라인 홀 */}
              <ellipse cx="120" cy="50" rx="8" ry="5" fill="#FCD34D" stroke="#D97706" strokeWidth="1" />
              <ellipse cx="120" cy="242" rx="8" ry="5" fill="#FCD34D" stroke="#D97706" strokeWidth="1" />
              <rect x="112" y="50" width="16" height="192" fill="#FEF3C7" stroke="#D97706" strokeWidth="1" />

              <line x1="120" y1="35" x2="120" y2="45" stroke="#F59E0B" strokeWidth="2"
                    markerEnd="url(#arrow10)" />
              <text x="95" y="32" fontSize="8" fill="#D97706">Bit Line</text>

              {/* 채널 홀 상세 */}
              <text x="280" y="25" fontSize="11" fontWeight="bold" fill="#374151">
                채널 홀 상세
              </text>

              <circle cx="380" cy="140" r="60" fill="none" stroke="#6B7280" strokeWidth="2" />

              {/* 내부 구조 (동심원) */}
              <circle cx="380" cy="140" r="50" fill="#E5E7EB" />
              <circle cx="380" cy="140" r="40" fill="#DBEAFE" />
              <circle cx="380" cy="140" r="30" fill="#DDD6FE" />
              <circle cx="380" cy="140" r="20" fill="#FEF3C7" />
              <circle cx="380" cy="140" r="10" fill="white" />

              {/* 레이블 */}
              <line x1="440" y1="140" x2="480" y2="120" stroke="#6B7280" strokeWidth="1" />
              <text x="485" y="115" fontSize="7" fill="#6B7280">워드라인 (W)</text>

              <line x1="420" y1="140" x2="480" y2="140" stroke="#6B7280" strokeWidth="1" />
              <text x="485" y="143" fontSize="7" fill="#6B7280">블로킹 옥사이드</text>

              <line x1="410" y1="160" x2="480" y2="165" stroke="#6B7280" strokeWidth="1" />
              <text x="485" y="168" fontSize="7" fill="#6B7280">전하트랩층 (SiN)</text>

              <line x1="400" y1="175" x2="480" y2="185" stroke="#6B7280" strokeWidth="1" />
              <text x="485" y="188" fontSize="7" fill="#6B7280">터널 옥사이드</text>

              <line x1="380" y1="190" x2="480" y2="210" stroke="#6B7280" strokeWidth="1" />
              <text x="485" y="213" fontSize="7" fill="#6B7280">폴리실리콘 채널</text>

              {/* 홀 직경 */}
              <line x1="340" y1="230" x2="420" y2="230" stroke="#EF4444" strokeWidth="1" />
              <text x="360" y="245" fontSize="8" fill="#EF4444">Ø ~600nm</text>

              <defs>
                <marker id="arrow10" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                  <polygon points="0 0, 8 3, 0 6" fill="#F59E0B" />
                </marker>
              </defs>
            </svg>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-800 text-white p-3 rounded text-xs">
              <h4 className="font-semibold mb-2">제조 공정 (String Stacking):</h4>
              <code>{`1. 워드라인/옥사이드 교대 증착
   → 128~200+ 층

2. 채널 홀 에칭
   → AR > 60:1 (고종횡비)
   → Ø ~600nm

3. ONO 증착 (ALD)
   → Oxide-Nitride-Oxide
   → 각 층 수 nm

4. 폴리실리콘 채널 증착

5. 슬릿 에칭 및 교체
   → 옥사이드 → 텅스텐 (WL)

6. 컨택 형성
   → 비트라인, 소스라인`}</code>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
                세대별 진화
              </h4>
              <div className="space-y-2 text-xs text-gray-700 dark:text-gray-300">
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <div className="flex justify-between mb-1">
                    <span className="font-semibold">1세대 (2013)</span>
                    <span className="text-blue-600">24~32층</span>
                  </div>
                  <div className="text-gray-500">Toshiba, Samsung</div>
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <div className="flex justify-between mb-1">
                    <span className="font-semibold">2세대 (2016)</span>
                    <span className="text-blue-600">48~64층</span>
                  </div>
                  <div className="text-gray-500">TLC 본격화</div>
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <div className="flex justify-between mb-1">
                    <span className="font-semibold">3세대 (2018)</span>
                    <span className="text-blue-600">96~128층</span>
                  </div>
                  <div className="text-gray-500">QLC 도입</div>
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <div className="flex justify-between mb-1">
                    <span className="font-semibold">4세대 (2020~)</span>
                    <span className="text-blue-600">176~232층</span>
                  </div>
                  <div className="text-gray-500">PLC 연구</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Chiplet */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          5.4 Chiplet 및 3D 패키징
        </h2>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Chiplet은 작은 칩들을 패키지 레벨에서 통합하여 거대한 칩을 구현하는
            기술입니다. 수율 향상과 비용 절감에 효과적입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                2.5D 패키징 (인터포저)
              </h3>
              <svg className="w-full h-48" viewBox="0 0 280 200">
                {/* 칩렛들 */}
                <rect x="40" y="40" width="60" height="50" fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />
                <text x="55" y="68" fontSize="10" fill="white" fontWeight="bold">CPU</text>

                <rect x="110" y="40" width="60" height="50" fill="#A78BFA" stroke="#7C3AED" strokeWidth="2" />
                <text x="125" y="68" fontSize="10" fill="white" fontWeight="bold">GPU</text>

                <rect x="180" y="40" width="60" height="50" fill="#34D399" stroke="#10B981" strokeWidth="2" />
                <text x="192" y="68" fontSize="10" fill="white" fontWeight="bold">HBM</text>

                {/* 인터포저 */}
                <rect x="30" y="100" width="220" height="20" fill="#F59E0B" stroke="#D97706" strokeWidth="2" />
                <text x="100" y="113" fontSize="10" fill="white" fontWeight="bold">Interposer (Si)</text>

                {/* 배선 표시 */}
                {[50, 80, 120, 150, 190, 220].map((x, i) => (
                  <line key={i} x1={x} y1="90" x2={x} y2="100" stroke="#374151" strokeWidth="2" />
                ))}

                {/* TSV */}
                {[60, 100, 140, 180, 220].map((x, i) => (
                  <line key={i} x1={x} y1="120" x2={x} y2="140" stroke="#EF4444" strokeWidth="3" />
                ))}
                <text x="110" y="155" fontSize="8" fill="#EF4444">TSV (관통전극)</text>

                {/* 기판 */}
                <rect x="20" y="140" width="240" height="30" fill="#9CA3AF" stroke="#6B7280" strokeWidth="2" />
                <text x="100" y="158" fontSize="10" fill="white">패키지 기판</text>

                {/* 범프 */}
                {[40, 70, 100, 130, 160, 190, 220, 250].map((x, i) => (
                  <circle key={i} cx={x} cy="175" r="4" fill="#FCD34D" stroke="#D97706" />
                ))}
                <text x="110" y="188" fontSize="7" fill="#D97706">솔더 범프</text>
              </svg>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                3D 패키징 (적층)
              </h3>
              <svg className="w-full h-48" viewBox="0 0 280 200">
                {/* 최상층 칩 */}
                <rect x="80" y="30" width="120" height="25" fill="#A78BFA" stroke="#7C3AED" strokeWidth="2" />
                <text x="115" y="46" fontSize="10" fill="white" fontWeight="bold">SRAM</text>

                {/* TSV */}
                {[100, 130, 160].map((x, i) => (
                  <line key={i} x1={x} y1="55" x2={x} y2="70" stroke="#EF4444" strokeWidth="4" />
                ))}

                {/* 중간층 칩 */}
                <rect x="70" y="70" width="140" height="30" fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />
                <text x="115" y="88" fontSize="10" fill="white" fontWeight="bold">Logic</text>

                {/* TSV */}
                {[90, 120, 150, 180].map((x, i) => (
                  <line key={i} x1={x} y1="100" x2={x} y2="120" stroke="#EF4444" strokeWidth="4" />
                ))}

                {/* 하층 칩 */}
                <rect x="60" y="120" width="160" height="35" fill="#34D399" stroke="#10B981" strokeWidth="2" />
                <text x="110" y="141" fontSize="10" fill="white" fontWeight="bold">Memory</text>

                {/* 기판 */}
                <rect x="50" y="160" width="180" height="25" fill="#9CA3AF" stroke="#6B7280" strokeWidth="2" />
                <text x="110" y="175" fontSize="9" fill="white">기판</text>

                {/* 범프 */}
                {[70, 100, 130, 160, 190, 220].map((x, i) => (
                  <circle key={i} cx={x} cy="190" r="4" fill="#FCD34D" stroke="#D97706" />
                ))}

                {/* 레이블 */}
                <line x1="220" y1="45" x2="250" y2="45" stroke="#EF4444" strokeWidth="2" />
                <text x="255" y="48" fontSize="8" fill="#EF4444">TSV</text>

                <text x="240" y="110" fontSize="8" fill="#6B7280">수직 적층</text>
              </svg>
            </div>
          </div>

          <div className="bg-gray-800 text-white p-4 rounded-lg mt-4">
            <h3 className="font-semibold mb-3">Chiplet 장점 및 사례:</h3>
            <div className="grid md:grid-cols-2 gap-4 text-xs">
              <div>
                <h4 className="font-semibold mb-2 text-yellow-400">장점:</h4>
                <code>{`1. 수율 향상
   - 작은 칩 → 결함 확률↓
   - 거대 모놀리식 대비 2~3배

2. 비용 절감
   - 공정 최적화 (노드 mix)
   - 재사용 가능한 IP

3. 성능 향상
   - 초고대역폭 연결
   - 이종 통합 가능

4. 유연성
   - 맞춤형 구성
   - 빠른 제품화`}</code>
              </div>
              <div>
                <h4 className="font-semibold mb-2 text-green-400">사례:</h4>
                <code>{`AMD EPYC (2017~)
- 최대 8개 Chiplet
- 64코어 (Zen4)

Intel Ponte Vecchio (2022)
- 47개 타일
- 5개 공정 노드

Apple M1 Ultra (2022)
- 2개 M1 Max 연결
- UltraFusion 2.5TB/s

UCIe 표준 (2022)
- Universal Chiplet
- Intel, AMD, TSMC 참여`}</code>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 요약 */}
      <section className="mb-8">
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            핵심 요약
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>FinFET은 3면 게이트로 SCE를 억제하여 22nm 이하 공정을 가능하게 했습니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>GAA/Nanosheet는 4면 게이트로 최고의 전기적 제어를 제공하는 차세대 트랜지스터입니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>3D NAND는 수직 적층으로 200단 이상까지 발전하여 고용량 메모리를 실현합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>Chiplet과 3D 패키징은 수율 향상과 이종 통합을 가능하게 하는 핵심 기술입니다</span>
            </li>
          </ul>
        </div>
      </section>

      <References
        sections={[
          {
            title: '원본 논문 (Original Papers)',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'FinFET: A Self-Aligned Double-Gate MOSFET Scalable to 20 nm',
                authors: 'Hisamoto Digh, Wen-Chin Lee, Jakub Kedzierski, Hideki Takeuchi, Kazuya Asano, Charles Kuo, Erik Anderson, Tsu-Jae King, Jeffrey Bokor, Chenming Hu',
                year: '2000',
                description: 'FinFET을 최초로 제안한 UC Berkeley의 역사적 논문',
                link: 'https://ieeexplore.ieee.org/document/914303'
              },
              {
                title: 'Multi-Gate MOSFETs for 22nm Technology and Beyond',
                authors: 'C. Auth et al.',
                year: '2012',
                description: 'Intel 22nm Tri-Gate (FinFET) 양산 기술 발표 (VLSI 2012)',
                link: 'https://ieeexplore.ieee.org/document/6243796'
              },
              {
                title: 'Gate-All-Around (GAA) FET',
                authors: 'J. P. Colinge',
                year: '1990',
                description: 'GAA 트랜지스터 개념을 최초로 제안한 논문',
                link: 'https://ieeexplore.ieee.org/document/106776'
              },
              {
                title: 'First Demonstration of GAA Silicon Nanowire FET as a Reliable 3-nm Node Device Technology',
                authors: 'Naoki Loubet et al.',
                year: '2017',
                description: 'IBM의 3nm GAA 나노와이어 FET 구현 (IEDM 2017)',
                link: 'https://ieeexplore.ieee.org/document/8268337'
              },
              {
                title: 'Bit Cost Scalable Technology with Punch and Plug Process for Ultra High Density Flash Memory',
                authors: 'Yohji Komori et al. (Toshiba)',
                year: '2008',
                description: '세계 최초 3D NAND Flash 기술 발표 (VLSI 2008)',
                link: 'https://ieeexplore.ieee.org/document/4585427'
              },
              {
                title: '128Gb 3b/Cell V-NAND Flash Memory With 1Gb/s I/O Rate',
                authors: 'Kang-Deog Suh et al. (Samsung)',
                year: '2015',
                description: 'Samsung 3D V-NAND 양산 기술 (ISSCC 2015)',
                link: 'https://ieeexplore.ieee.org/document/7062958'
              }
            ]
          },
          {
            title: '산업 표준 및 백서 (Industry Standards & Whitepapers)',
            icon: 'book',
            color: 'border-indigo-500',
            items: [
              {
                title: 'UCIe (Universal Chiplet Interconnect Express) Specification',
                authors: 'UCIe Consortium',
                year: '2022',
                description: 'Intel, AMD, TSMC 등이 참여한 Chiplet 표준 규격',
                link: 'https://www.uciexpress.org/'
              },
              {
                title: 'Intel RibbonFET: The Next Generation of Transistor Technology',
                authors: 'Intel Corporation',
                year: '2021',
                description: 'Intel 20A 공정의 GAA RibbonFET 기술 소개',
                link: 'https://www.intel.com/content/www/us/en/newsroom/opinion/ribbonfet-powervia-20a.html'
              },
              {
                title: 'Samsung 3nm GAA Technology',
                authors: 'Samsung Electronics',
                year: '2022',
                description: 'Samsung 3nm GAA (MBCFET) 공정 기술 백서',
                link: 'https://semiconductor.samsung.com/us/foundry/process-technology/'
              },
              {
                title: 'TSMC N2 Technology: Nanosheet Transistors',
                authors: 'TSMC',
                year: '2023',
                description: 'TSMC 2nm 공정의 Nanosheet 기술 로드맵',
                link: 'https://www.tsmc.com/english/dedicatedFoundry/technology/logic/l_2nm'
              },
              {
                title: 'AMD Chiplet Architecture: EPYC and Ryzen',
                authors: 'AMD',
                year: '2017-2023',
                description: 'AMD Zen 아키텍처의 Chiplet 기반 설계',
                link: 'https://www.amd.com/en/technologies/chiplet'
              }
            ]
          },
          {
            title: '기술 컨퍼런스 자료 (Technical Conference Papers)',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'IEDM (International Electron Devices Meeting)',
                description: '반도체 소자 기술의 최고 권위 학회 - FinFET, GAA, 3D NAND 등 최신 논문',
                link: 'https://ieee-iedm.org/'
              },
              {
                title: 'ISSCC (International Solid-State Circuits Conference)',
                description: '반도체 회로 설계 최고 학회 - 양산 기술 발표 중심',
                link: 'https://www.isscc.org/'
              },
              {
                title: 'VLSI Technology Symposium',
                description: '반도체 공정 기술 전문 학회 - 양산 공정 상세 발표',
                link: 'https://vlsitechnologysymposium.org/'
              },
              {
                title: 'DAC (Design Automation Conference)',
                description: 'Chiplet 및 3D IC 설계 자동화 기술 학회',
                link: 'https://www.dac.com/'
              }
            ]
          },
          {
            title: '산업 자료 (Industry Resources)',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'IEEE Electron Device Society',
                description: '반도체 소자 기술 전문 학회 - 무료 논문 및 튜토리얼',
                link: 'https://eds.ieee.org/'
              },
              {
                title: 'SemiWiki: Advanced Node Technologies',
                description: '반도체 산업 전문 미디어 - FinFET, GAA, Chiplet 심층 분석',
                link: 'https://semiwiki.com/'
              },
              {
                title: 'WikiChip: Microarchitecture Database',
                description: '반도체 아키텍처 데이터베이스 - 실제 칩 설계 상세 정보',
                link: 'https://en.wikichip.org/'
              },
              {
                title: 'TechInsights: Technology Teardowns',
                description: '반도체 역설계 분석 전문 - 실제 칩 구조 분석 리포트',
                link: 'https://www.techinsights.com/'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
