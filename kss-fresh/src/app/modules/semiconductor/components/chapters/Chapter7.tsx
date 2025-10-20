'use client'

import References from '@/components/common/References';

export default function Chapter7() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Chapter 7: 메모리 반도체
      </h1>

      {/* DRAM */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          7.1 DRAM (Dynamic RAM)
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            DRAM은 1T1C (1 Transistor 1 Capacitor) 구조로 고집적도를 실현한 휘발성
            메모리입니다. PC, 서버, 모바일의 주기억장치로 사용됩니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                DRAM 셀 구조
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* Word Line */}
                <line x1="20" y1="60" x2="120" y2="60" stroke="#3B82F6" strokeWidth="3" />
                <text x="25" y="55" fontSize="10" fill="#2563EB" fontWeight="bold">
                  Word Line
                </text>

                {/* Access Transistor */}
                <rect x="80" y="65" width="30" height="40" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
                <circle cx="95" cy="85" r="8" fill="none" stroke="#2563EB" strokeWidth="1.5" />
                <text x="118" y="88" fontSize="9" fill="#2563EB">Access Tr</text>

                {/* Bit Line */}
                <line x1="95" y1="30" x2="95" y2="65" stroke="#10B981" strokeWidth="3" />
                <text x="100" y="45" fontSize="10" fill="#059669" fontWeight="bold">
                  Bit Line
                </text>

                {/* Capacitor */}
                <line x1="95" y1="105" x2="95" y2="140" stroke="#374151" strokeWidth="2" />
                <line x1="75" y1="140" x2="115" y2="140" stroke="#EF4444" strokeWidth="4" />
                <line x1="75" y1="148" x2="115" y2="148" stroke="#6B7280" strokeWidth="4" />
                <text x="118" y="145" fontSize="9" fill="#EF4444">Capacitor</text>
                <text x="125" y="156" fontSize="8" fill="#6B7280">(25~30 fF)</text>

                {/* GND */}
                <line x1="85" y1="165" x2="105" y2="165" stroke="#6B7280" strokeWidth="2" />
                <line x1="90" y1="170" x2="100" y2="170" stroke="#6B7280" strokeWidth="1.5" />

                {/* 동작 설명 */}
                <rect x="160" y="30" width="110" height="180" fill="#FEF3C7" stroke="#F59E0B" strokeWidth="2" />
                <text x="175" y="25" fontSize="10" fontWeight="bold" fill="#D97706">
                  동작 원리
                </text>

                <text x="170" y="55" fontSize="9" fontWeight="bold" fill="#374151">
                  WRITE:
                </text>
                <text x="170" y="70" fontSize="8" fill="#6B7280">
                  1. WL = HIGH
                </text>
                <text x="170" y="82" fontSize="8" fill="#6B7280">
                  2. BL에 데이터 인가
                </text>
                <text x="170" y="94" fontSize="8" fill="#6B7280">
                  3. 커패시터 충/방전
                </text>

                <text x="170" y="115" fontSize="9" fontWeight="bold" fill="#374151">
                  READ:
                </text>
                <text x="170" y="130" fontSize="8" fill="#6B7280">
                  1. WL = HIGH
                </text>
                <text x="170" y="142" fontSize="8" fill="#6B7280">
                  2. 전하 공유
                </text>
                <text x="170" y="154" fontSize="8" fill="#6B7280">
                  3. Sense Amp 증폭
                </text>

                <text x="170" y="175" fontSize="9" fontWeight="bold" fill="#EF4444">
                  REFRESH:
                </text>
                <text x="170" y="190" fontSize="8" fill="#DC2626">
                  누설로 인한 전하 손실
                </text>
                <text x="170" y="202" fontSize="8" fill="#DC2626">
                  주기적 재충전 필요
                </text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">DRAM 주요 파라미터:</h4>
                <code>{`셀 크기: 6F² (F = 최소 피처)
  - DDR4: 10~20nm 노드
  - DDR5: 10nm 이하

커패시터 용량:
  - 25~30 fF (펨토패럿)
  - 저장 전하: ~50,000 전자

리프레시 주기:
  - DDR4: 64ms (8K rows)
  - DDR5: 32ms (더 잦음)

동작 전압:
  - DDR4: 1.2V
  - DDR5: 1.1V
  - LPDDR5: 0.5V (VDD2)

속도:
  - DDR4: 3200 MT/s
  - DDR5: 6400+ MT/s`}</code>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded">
                <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">
                  DDR5 vs DDR4
                </h4>
                <div className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                  <div className="flex justify-between">
                    <span>데이터 속도</span>
                    <span className="text-green-600">2배 향상</span>
                  </div>
                  <div className="flex justify-between">
                    <span>뱅크 그룹</span>
                    <span>4 → 8개</span>
                  </div>
                  <div className="flex justify-between">
                    <span>버스트 길이</span>
                    <span>8 → 16</span>
                  </div>
                  <div className="flex justify-between">
                    <span>온다이 ECC</span>
                    <span className="text-blue-600">추가</span>
                  </div>
                  <div className="flex justify-between">
                    <span>전력 관리</span>
                    <span>PMIC 통합</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">
              DRAM 커패시터 구조 진화
            </h3>
            <div className="grid grid-cols-4 gap-2 text-xs text-center">
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">Trench</div>
                <div className="text-gray-600 dark:text-gray-400">웨이퍼 내부 파기</div>
                <div className="text-gray-500 mt-1">~100nm</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">Stack</div>
                <div className="text-gray-600 dark:text-gray-400">기둥 형태 적층</div>
                <div className="text-gray-500 mt-1">~60nm</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">Cylinder</div>
                <div className="text-gray-600 dark:text-gray-400">원통 형태</div>
                <div className="text-gray-500 mt-1">~30nm</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">PNP</div>
                <div className="text-gray-600 dark:text-gray-400">펀칭 구조</div>
                <div className="text-gray-500 mt-1">~10nm</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* NAND Flash */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          7.2 NAND Flash (SSD)
        </h2>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            NAND Flash는 부동 게이트에 전하를 저장하여 데이터를 보존하는 비휘발성
            메모리입니다. SSD, USB, 메모리카드의 핵심 기술입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                NAND 셀 구조
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* Control Gate */}
                <rect x="80" y="40" width="120" height="15" fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />
                <text x="110" y="32" fontSize="10" fill="#2563EB" fontWeight="bold">
                  Control Gate
                </text>

                {/* Oxide */}
                <rect x="80" y="55" width="120" height="8" fill="#E5E7EB" stroke="#9CA3AF" strokeWidth="1" />
                <text x="205" y="62" fontSize="8" fill="#6B7280">Oxide</text>

                {/* Floating Gate */}
                <rect x="90" y="63" width="100" height="20" fill="#FCD34D" stroke="#F59E0B" strokeWidth="2" />
                <text x="105" y="76" fontSize="10" fill="#D97706" fontWeight="bold">
                  Floating Gate
                </text>

                {/* 전자 표시 (프로그램 상태) */}
                {[...Array(8)].map((_, i) => (
                  <circle key={i} cx={100 + i * 12} cy="73" r="3" fill="#EF4444" />
                ))}
                <text x="205" y="76" fontSize="8" fill="#EF4444">e⁻ (Data=0)</text>

                {/* Tunnel Oxide */}
                <rect x="80" y="83" width="120" height="5" fill="#D1FAE5" stroke="#10B981" strokeWidth="1" />
                <text x="205" y="88" fontSize="8" fill="#059669">Tunnel Oxide (~10nm)</text>

                {/* Channel (Silicon) */}
                <rect x="80" y="88" width="120" height="30" fill="#FEE2E2" stroke="#DC2626" strokeWidth="2" />
                <text x="120" y="106" fontSize="10" fill="#DC2626">P-Si Channel</text>

                {/* Source/Drain */}
                <rect x="60" y="95" width="15" height="16" fill="#3B82F6" />
                <rect x="205" y="95" width="15" height="16" fill="#3B82F6" />
                <text x="50" y="130" fontSize="9" fill="#2563EB">S</text>
                <text x="215" y="130" fontSize="9" fill="#2563EB">D</text>

                {/* 기판 */}
                <rect x="50" y="118" width="180" height="20" fill="#9CA3AF" stroke="#6B7280" strokeWidth="2" />
                <text x="120" y="132" fontSize="9" fill="white">P-Substrate</text>

                {/* 동작 설명 */}
                <text x="80" y="160" fontSize="10" fontWeight="bold" fill="#374151">
                  Program (Write):
                </text>
                <text x="80" y="175" fontSize="8" fill="#6B7280">
                  고전압(~20V) 인가 → 전자 터널링 → FG에 전하 주입
                </text>

                <text x="80" y="195" fontSize="10" fontWeight="bold" fill="#374151">
                  Erase:
                </text>
                <text x="80" y="210" fontSize="8" fill="#6B7280">
                  기판에 고전압 → 전자 방출 → FG 전하 제거
                </text>

                <text x="80" y="230" fontSize="8" fill="#EF4444">
                  * 블록 단위 소거만 가능 (512KB~4MB)
                </text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">비트/셀 기술:</h4>
                <code>{`SLC (Single-Level Cell):
  - 1 bit/cell
  - 속도: 매우 빠름
  - 수명: 100,000 P/E
  - 용도: 엔터프라이즈, 캐시

MLC (Multi-Level Cell):
  - 2 bit/cell
  - 속도: 빠름
  - 수명: 10,000 P/E
  - 용도: 컨슈머 SSD

TLC (Triple-Level Cell):
  - 3 bit/cell
  - 속도: 중간
  - 수명: 3,000 P/E
  - 용도: 일반 SSD (주류)

QLC (Quad-Level Cell):
  - 4 bit/cell
  - 속도: 느림
  - 수명: 1,000 P/E
  - 용도: 대용량 저장

PLC (Penta-Level Cell):
  - 5 bit/cell (연구중)
  - 수명: ~500 P/E`}</code>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
                <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
                  전압 레벨 구분
                </h4>
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <svg className="w-full h-24" viewBox="0 0 220 100">
                    {/* TLC 전압 레벨 */}
                    {[...Array(8)].map((_, i) => (
                      <g key={i}>
                        <rect x={10 + i * 26} y={70 - i * 8} width="22" height="25"
                              fill={`hsl(${i * 45}, 70%, 70%)`}
                              stroke="#374151" strokeWidth="1" />
                        <text x={14 + i * 26} y={85 - i * 8} fontSize="9" fill="#374151">
                          {i.toString(2).padStart(3, '0')}
                        </text>
                      </g>
                    ))}
                    <text x="60" y="15" fontSize="9" fill="#6B7280">
                      TLC: 8개 전압 레벨 (2³)
                    </text>
                  </svg>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-red-800 dark:text-red-300 mb-3">
              NAND Flash 과제와 해결책
            </h3>
            <div className="grid md:grid-cols-2 gap-3 text-sm text-gray-700 dark:text-gray-300">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">Wear Leveling</h4>
                <p className="text-xs">
                  P/E 사이클 제한 극복: 쓰기를 모든 블록에 균등 분산
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">ECC (Error Correction)</h4>
                <p className="text-xs">
                  비트 에러 보정: BCH, LDPC 코드로 신뢰성 향상
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">Garbage Collection</h4>
                <p className="text-xs">
                  무효 페이지 정리: 백그라운드에서 블록 재활용
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">Over-Provisioning</h4>
                <p className="text-xs">
                  여유 공간 확보: 7~28% 추가 용량으로 수명 연장
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SRAM */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          7.3 SRAM (Static RAM)
        </h2>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            SRAM은 6T (6 Transistor) 구조로 빠르고 안정적인 캐시 메모리를 구현합니다.
            CPU/GPU의 L1/L2/L3 캐시에 필수적입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                6T SRAM 셀
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* Word Line */}
                <line x1="20" y1="60" x2="260" y2="60" stroke="#3B82F6" strokeWidth="3" />
                <text x="25" y="55" fontSize="10" fill="#2563EB" fontWeight="bold">
                  Word Line
                </text>

                {/* Cross-coupled Inverters */}
                <circle cx="100" cy="120" r="40" fill="none" stroke="#7C3AED" strokeWidth="2" strokeDasharray="5,5" />
                <circle cx="180" cy="120" r="40" fill="none" stroke="#7C3AED" strokeWidth="2" strokeDasharray="5,5" />

                {/* M1, M2 (PMOS) */}
                <rect x="85" y="90" width="30" height="20" fill="#FEE2E2" stroke="#DC2626" strokeWidth="1.5" />
                <rect x="165" y="90" width="30" height="20" fill="#FEE2E2" stroke="#DC2626" strokeWidth="1.5" />
                <text x="92" y="103" fontSize="8" fill="#DC2626">M1</text>
                <text x="172" y="103" fontSize="8" fill="#DC2626">M2</text>

                {/* M3, M4 (NMOS) */}
                <rect x="85" y="130" width="30" height="20" fill="#DBEAFE" stroke="#2563EB" strokeWidth="1.5" />
                <rect x="165" y="130" width="30" height="20" fill="#DBEAFE" stroke="#2563EB" strokeWidth="1.5" />
                <text x="92" y="143" fontSize="8" fill="#2563EB">M3</text>
                <text x="172" y="143" fontSize="8" fill="#2563EB">M4</text>

                {/* M5, M6 (Access Transistors) */}
                <rect x="50" y="115" width="25" height="20" fill="#D1FAE5" stroke="#10B981" strokeWidth="1.5" />
                <rect x="205" y="115" width="25" height="20" fill="#D1FAE5" stroke="#10B981" strokeWidth="1.5" />
                <text x="56" y="128" fontSize="8" fill="#059669">M5</text>
                <text x="211" y="128" fontSize="8" fill="#059669">M6</text>

                {/* VDD */}
                <line x1="80" y1="30" x2="200" y2="30" stroke="#EF4444" strokeWidth="3" />
                <text x="120" y="25" fontSize="10" fill="#DC2626" fontWeight="bold">VDD</text>
                <line x1="100" y1="30" x2="100" y2="90" stroke="#EF4444" strokeWidth="2" />
                <line x1="180" y1="30" x2="180" y2="90" stroke="#EF4444" strokeWidth="2" />

                {/* GND */}
                <line x1="100" y1="150" x2="100" y2="180" stroke="#6B7280" strokeWidth="2" />
                <line x1="180" y1="150" x2="180" y2="180" stroke="#6B7280" strokeWidth="2" />
                <line x1="80" y1="180" x2="200" y2="180" stroke="#6B7280" strokeWidth="3" />

                {/* Cross-couple connections */}
                <line x1="100" y1="120" x2="165" y2="100" stroke="#7C3AED" strokeWidth="2" />
                <line x1="180" y1="120" x2="115" y2="100" stroke="#7C3AED" strokeWidth="2" />

                {/* Bit Lines */}
                <line x1="40" y1="20" x2="40" y2="200" stroke="#10B981" strokeWidth="3" />
                <line x1="240" y1="20" x2="240" y2="200" stroke="#10B981" strokeWidth="3" />
                <text x="25" y="215" fontSize="9" fill="#059669">BL</text>
                <text x="230" y="215" fontSize="9" fill="#059669">/BL</text>

                {/* Connections */}
                <line x1="40" y1="125" x2="50" y2="125" stroke="#10B981" strokeWidth="2" />
                <line x1="230" y1="125" x2="240" y2="125" stroke="#10B981" strokeWidth="2" />
                <line x1="75" y1="125" x2="85" y2="125" stroke="#374151" strokeWidth="2" />
                <line x1="195" y1="125" x2="205" y2="125" stroke="#374151" strokeWidth="2" />

                {/* Storage nodes */}
                <circle cx="100" cy="120" r="4" fill="#FCD34D" stroke="#D97706" strokeWidth="2" />
                <circle cx="180" cy="120" r="4" fill="#FCD34D" stroke="#D97706" strokeWidth="2" />
                <text x="85" y="165" fontSize="8" fill="#D97706">Q</text>
                <text x="185" y="165" fontSize="8" fill="#D97706">/Q</text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">SRAM 특성:</h4>
                <code>{`장점:
- 빠른 속도 (< 1ns)
- 리프레시 불필요
- 낮은 레이턴시
- 단순한 제어

단점:
- 큰 셀 크기 (120~160 F²)
- 높은 비용 (DRAM의 30배)
- 제한적 용량

셀 크기:
- 7nm: 0.027 μm²
- 5nm: 0.021 μm²
- 3nm: 0.016 μm²

동작 마진:
- Read SNM > 30% VDD
- Write Margin > 20% VDD`}</code>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                <h4 className="font-semibold text-orange-800 dark:text-orange-300 mb-2">
                  CPU 캐시 계층
                </h4>
                <div className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="flex justify-between mb-1">
                      <span className="font-semibold">L1 캐시</span>
                      <span>32~64KB/코어</span>
                    </div>
                    <div className="text-gray-500">지연: 4사이클, 속도: ~1TB/s</div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="flex justify-between mb-1">
                      <span className="font-semibold">L2 캐시</span>
                      <span>256KB~1MB/코어</span>
                    </div>
                    <div className="text-gray-500">지연: 12사이클, 속도: ~500GB/s</div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="flex justify-between mb-1">
                      <span className="font-semibold">L3 캐시</span>
                      <span>32~128MB (공유)</span>
                    </div>
                    <div className="text-gray-500">지연: 40사이클, 속도: ~200GB/s</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 차세대 메모리 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          7.4 차세대 메모리 기술
        </h2>

        <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-lg mb-4">
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            {/* MRAM */}
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-indigo-800 dark:text-indigo-300 mb-3">
                MRAM (Magnetic RAM)
              </h3>
              <div className="bg-indigo-50 dark:bg-indigo-900/30 p-3 rounded mb-2">
                <svg className="w-full h-32" viewBox="0 0 240 140">
                  {/* MTJ 구조 */}
                  <rect x="80" y="30" width="80" height="15" fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />
                  <text x="95" y="42" fontSize="9" fill="white">Free Layer</text>

                  <rect x="80" y="45" width="80" height="10" fill="#E5E7EB" stroke="#9CA3AF" strokeWidth="1" />
                  <text x="165" y="52" fontSize="8" fill="#6B7280">MgO Barrier</text>

                  <rect x="80" y="55" width="80" height="15" fill="#EF4444" stroke="#DC2626" strokeWidth="2" />
                  <text x="92" y="65" fontSize="9" fill="white">Fixed Layer</text>

                  {/* 스핀 방향 */}
                  <text x="50" y="40" fontSize="12" fill="#3B82F6">↑</text>
                  <text x="50" y="65" fontSize="12" fill="#EF4444">↑</text>
                  <text x="40" y="85" fontSize="8" fill="#10B981">Parallel (0)</text>

                  <text x="180" y="40" fontSize="12" fill="#3B82F6">↓</text>
                  <text x="180" y="65" fontSize="12" fill="#EF4444">↑</text>
                  <text x="170" y="85" fontSize="8" fill="#EF4444">Anti-parallel (1)</text>
                </svg>
              </div>
              <div className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                <div><strong>장점:</strong> 비휘발성, 무한 수명, 빠른 속도</div>
                <div><strong>용도:</strong> IoT, 임베디드, 캐시</div>
                <div><strong>기업:</strong> Everspin, Samsung (eMRAM)</div>
              </div>
            </div>

            {/* RRAM */}
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-purple-800 dark:text-purple-300 mb-3">
                RRAM (Resistive RAM)
              </h3>
              <div className="bg-purple-50 dark:bg-purple-900/30 p-3 rounded mb-2">
                <svg className="w-full h-32" viewBox="0 0 240 140">
                  {/* 필라멘트 */}
                  <rect x="80" y="30" width="80" height="10" fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />
                  <text x="105" y="40" fontSize="9" fill="white">Top Electrode</text>

                  <rect x="80" y="40" width="80" height="40" fill="#DDD6FE" stroke="#7C3AED" strokeWidth="2" />
                  <text x="165" y="63" fontSize="8" fill="#6B21A8">Oxide (HfO₂)</text>

                  {/* LRS 상태 */}
                  <line x1="120" y1="40" x2="120" y2="80" stroke="#EF4444" strokeWidth="4" />
                  <text x="100" y="95" fontSize="8" fill="#EF4444">Filament (LRS)</text>

                  <rect x="80" y="80" width="80" height="10" fill="#9CA3AF" stroke="#6B7280" strokeWidth="2" />
                  <text x="100" y="90" fontSize="9" fill="white">Bottom Electrode</text>

                  <text x="85" y="110" fontSize="9" fill="#6B7280">SET: 필라멘트 형성 (저항↓)</text>
                  <text x="85" y="122" fontSize="9" fill="#6B7280">RESET: 필라멘트 파괴 (저항↑)</text>
                </svg>
              </div>
              <div className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                <div><strong>장점:</strong> 초소형, 저전력, 3D 적층</div>
                <div><strong>용도:</strong> AI 가중치, 뉴로모픽</div>
                <div><strong>기업:</strong> Crossbar, Weebit Nano</div>
              </div>
            </div>

            {/* PCM */}
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-orange-800 dark:text-orange-300 mb-3">
                PCM (Phase Change Memory)
              </h3>
              <div className="bg-orange-50 dark:bg-orange-900/30 p-3 rounded mb-2">
                <svg className="w-full h-32" viewBox="0 0 240 140">
                  {/* GST 상태 변화 */}
                  <rect x="50" y="40" width="60" height="50" fill="#FEE2E2" stroke="#DC2626" strokeWidth="2" />
                  <text x="55" y="30" fontSize="9" fontWeight="bold" fill="#DC2626">
                    결정질 (SET)
                  </text>
                  <text x="60" y="68" fontSize="8" fill="#6B7280">낮은 저항</text>
                  <text x="65" y="80" fontSize="8" fill="#6B7280">데이터: 1</text>

                  <rect x="130" y="40" width="60" height="50" fill="#DBEAFE" stroke="#2563EB" strokeWidth="2" />
                  <text x="135" y="30" fontSize="9" fontWeight="bold" fill="#2563EB">
                    비정질 (RESET)
                  </text>
                  <text x="140" y="68" fontSize="8" fill="#6B7280">높은 저항</text>
                  <text x="145" y="80" fontSize="8" fill="#6B7280">데이터: 0</text>

                  {/* 화살표 */}
                  <path d="M 110 55 L 125 55" stroke="#F59E0B" strokeWidth="2" markerEnd="url(#arrow12)" />
                  <text x="105" y="50" fontSize="7" fill="#F59E0B">가열</text>
                  <path d="M 130 75 L 115 75" stroke="#10B981" strokeWidth="2" markerEnd="url(#arrow12)" />
                  <text x="110" y="90" fontSize="7" fill="#10B981">서냉</text>

                  <defs>
                    <marker id="arrow12" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                      <polygon points="0 0, 8 3, 0 6" fill="currentColor" />
                    </marker>
                  </defs>
                </svg>
              </div>
              <div className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                <div><strong>장점:</strong> 고속, 고내구성, SCM</div>
                <div><strong>용도:</strong> Optane (Intel, 단종)</div>
                <div><strong>물질:</strong> Ge₂Sb₂Te₅ (GST)</div>
              </div>
            </div>

            {/* FeRAM */}
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-green-800 dark:text-green-300 mb-3">
                FeRAM (Ferroelectric RAM)
              </h3>
              <div className="bg-green-50 dark:bg-green-900/30 p-3 rounded mb-2">
                <svg className="w-full h-32" viewBox="0 0 240 140">
                  {/* 강유전체 분극 */}
                  <rect x="60" y="50" width="50" height="40" fill="#D1FAE5" stroke="#10B981" strokeWidth="2" />
                  <text x="67" y="75" fontSize="10" fill="#059669">PZT</text>
                  <text x="50" y="65" fontSize="12" fill="#EF4444">⊕</text>
                  <text x="50" y="85" fontSize="12" fill="#3B82F6">⊖</text>
                  <text x="65" y="105" fontSize="7" fill="#6B7280">분극 ↑ (1)</text>

                  <rect x="130" y="50" width="50" height="40" fill="#D1FAE5" stroke="#10B981" strokeWidth="2" />
                  <text x="137" y="75" fontSize="10" fill="#059669">PZT</text>
                  <text x="120" y="85" fontSize="12" fill="#EF4444">⊕</text>
                  <text x="120" y="65" fontSize="12" fill="#3B82F6">⊖</text>
                  <text x="135" y="105" fontSize="7" fill="#6B7280">분극 ↓ (0)</text>
                </svg>
              </div>
              <div className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                <div><strong>장점:</strong> 저전력, 고속, 고내구성</div>
                <div><strong>용도:</strong> RFID, 스마트카드</div>
                <div><strong>기업:</strong> Fujitsu, TI</div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 text-white p-4 rounded-lg">
            <h3 className="font-semibold mb-3">차세대 메모리 비교:</h3>
            <div className="overflow-x-auto text-xs">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-600">
                    <th className="px-3 py-2 text-left">기술</th>
                    <th className="px-3 py-2 text-left">속도</th>
                    <th className="px-3 py-2 text-left">수명</th>
                    <th className="px-3 py-2 text-left">셀 크기</th>
                    <th className="px-3 py-2 text-left">성숙도</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-700">
                    <td className="px-3 py-2">MRAM</td>
                    <td className="px-3 py-2 text-green-400">~10ns</td>
                    <td className="px-3 py-2 text-green-400">&gt;10¹⁵</td>
                    <td className="px-3 py-2">20~40F²</td>
                    <td className="px-3 py-2 text-green-400">양산</td>
                  </tr>
                  <tr className="border-b border-gray-700">
                    <td className="px-3 py-2">RRAM</td>
                    <td className="px-3 py-2 text-yellow-400">~50ns</td>
                    <td className="px-3 py-2 text-yellow-400">10⁶~10⁹</td>
                    <td className="px-3 py-2 text-green-400">4~8F²</td>
                    <td className="px-3 py-2 text-yellow-400">개발</td>
                  </tr>
                  <tr className="border-b border-gray-700">
                    <td className="px-3 py-2">PCM</td>
                    <td className="px-3 py-2 text-yellow-400">~100ns</td>
                    <td className="px-3 py-2 text-yellow-400">10⁸~10⁹</td>
                    <td className="px-3 py-2">4~20F²</td>
                    <td className="px-3 py-2 text-red-400">일부양산</td>
                  </tr>
                  <tr>
                    <td className="px-3 py-2">FeRAM</td>
                    <td className="px-3 py-2 text-green-400">~20ns</td>
                    <td className="px-3 py-2 text-green-400">10¹⁴</td>
                    <td className="px-3 py-2">25~50F²</td>
                    <td className="px-3 py-2 text-green-400">양산</td>
                  </tr>
                </tbody>
              </table>
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
              <span>DRAM은 1T1C 구조로 고집적도를 실현하며 주기적 리프레시가 필요합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>NAND Flash는 3D 적층과 다중 비트/셀 기술로 대용량 비휘발성 저장장치를 제공합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>SRAM은 6T 구조로 CPU/GPU 캐시에 사용되며 빠르지만 면적이 큽니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>MRAM, RRAM, PCM 등 차세대 메모리는 비휘발성과 고속을 모두 추구합니다</span>
            </li>
          </ul>
        </div>
      </section>

      <References
        sections={[
          {
            title: '📚 공식 문서 & 표준',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'JEDEC DDR5 SDRAM Standard (JESD79-5)',
                url: 'https://www.jedec.org/standards-documents/docs/jesd79-5',
                description: 'DDR5 메모리 공식 표준 규격 - 6400+ MT/s, On-die ECC'
              },
              {
                title: 'Samsung V-NAND Technology',
                url: 'https://semiconductor.samsung.com/consumer-storage/support/tools/',
                description: 'Samsung 3D V-NAND 기술 백서 - 232단 적층 기술'
              },
              {
                title: 'Micron 3D NAND Technology Brief',
                url: 'https://www.micron.com/products/nand-flash',
                description: 'Micron 232단 3D NAND 아키텍처 및 QLC/PLC 기술'
              },
              {
                title: 'SK hynix DDR5 Product Specification',
                url: 'https://www.skhynix.com/products/dram/ddr5/',
                description: 'DDR5 메모리 제품 스펙 - PMIC, On-die ECC 상세'
              },
              {
                title: 'ONFI (Open NAND Flash Interface) Standard',
                url: 'https://www.onfi.org/',
                description: 'NAND Flash 인터페이스 표준 - NVMe, PCIe 통합'
              }
            ]
          },
          {
            title: '🔬 핵심 논문 & 연구',
            icon: 'research' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'DRAM Scaling Challenges and Future Directions',
                url: 'https://ieeexplore.ieee.org/document/8993467',
                description: 'DRAM 스케일링 한계 및 차세대 커패시터 기술 (IEDM 2019)'
              },
              {
                title: 'Vertical NAND Flash Memory Technologies',
                url: 'https://ieeexplore.ieee.org/document/8993513',
                description: '3D NAND 적층 기술 - String Stacking, PUC 구조 (IEDM 2019)'
              },
              {
                title: 'SRAM Bitcell Design Challenges in Sub-3nm Technologies',
                url: 'https://ieeexplore.ieee.org/document/9731622',
                description: '3nm 이하 SRAM 설계 과제 - FinFET to GAA (VLSI 2022)'
              },
              {
                title: 'Emerging Non-Volatile Memory Technologies',
                url: 'https://ieeexplore.ieee.org/document/8993629',
                description: 'MRAM, RRAM, PCM 차세대 메모리 기술 비교 (IEDM 2019)'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 제조사',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Samsung Memory Solutions',
                url: 'https://semiconductor.samsung.com/dram/',
                description: 'DDR5, LPDDR5X, HBM3e 메모리 솔루션 및 기술 문서'
              },
              {
                title: 'SK hynix Memory Technology',
                url: 'https://www.skhynix.com/',
                description: 'HBM3e 1.15TB/s, DDR5 8000+ MT/s 제품 라인업'
              },
              {
                title: 'Micron DRAM & NAND Products',
                url: 'https://www.micron.com/products',
                description: 'Crucial 브랜드 DRAM/SSD - 기술 스펙 및 성능 가이드'
              },
              {
                title: 'Western Digital NAND Solutions',
                url: 'https://www.westerndigital.com/products',
                description: 'BiCS (3D NAND) 기술 - 112단+ TLC/QLC SSD'
              },
              {
                title: 'SNIA (Storage Networking Industry Association)',
                url: 'https://www.snia.org/',
                description: 'SSD 표준, NVMe 스펙, 메모리 기술 백서'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
