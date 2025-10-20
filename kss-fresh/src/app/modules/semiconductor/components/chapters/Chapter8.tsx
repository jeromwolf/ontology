'use client'

import References from '@/components/common/References';

export default function Chapter8() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Chapter 8: 미래 반도체 기술
      </h1>

      {/* 양자 컴퓨팅 칩 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          8.1 양자 컴퓨팅 칩 (Quantum Computing)
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            양자 컴퓨터는 큐비트(qubit)의 중첩과 얽힘 현상을 활용하여 고전 컴퓨터로
            불가능한 계산을 수행합니다. 초전도, 이온 트랩, 광자 방식 등이 연구되고 있습니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                초전도 큐비트 (Superconducting Qubit)
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* Transmon 큐비트 */}
                <text x="90" y="25" fontSize="11" fontWeight="bold" fill="#3B82F6">
                  Transmon 큐비트
                </text>

                {/* Josephson Junction */}
                <rect x="100" y="50" width="80" height="40" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
                <rect x="135" y="60" width="10" height="20" fill="#FEE2E2" stroke="#EF4444" strokeWidth="2" />
                <text x="105" y="90" fontSize="9" fill="#2563EB">Al</text>
                <text x="165" y="90" fontSize="9" fill="#2563EB">Al</text>
                <text x="125" y="75" fontSize="7" fill="#DC2626">AlO_x</text>
                <text x="190" y="72" fontSize="8" fill="#6B7280">Josephson Junction</text>

                {/* 커패시터 */}
                <line x1="120" y1="100" x2="120" y2="120" stroke="#374151" strokeWidth="2" />
                <line x1="160" y1="100" x2="160" y2="120" stroke="#374151" strokeWidth="2" />
                <line x1="100" y1="120" x2="180" y2="120" stroke="#10B981" strokeWidth="4" />
                <line x1="100" y1="128" x2="180" y2="128" stroke="#10B981" strokeWidth="4" />
                <text x="185" y="126" fontSize="8" fill="#059669">Shunt Capacitor</text>

                {/* 양자 상태 */}
                <rect x="40" y="150" width="200" height="70" fill="#F3F4F6" stroke="#9CA3AF" strokeWidth="2" />
                <text x="50" y="145" fontSize="10" fontWeight="bold" fill="#374151">
                  양자 상태
                </text>

                <circle cx="80" cy="180" r="15" fill="#60A5FA" opacity="0.7" />
                <text x="72" y="185" fontSize="10" fill="white">|0⟩</text>

                <circle cx="160" cy="180" r="15" fill="#A78BFA" opacity="0.7" />
                <text x="152" y="185" fontSize="10" fill="white">|1⟩</text>

                <ellipse cx="120" cy="180" rx="50" ry="20" fill="#FCD34D" opacity="0.5" />
                <text x="95" y="185" fontSize="9" fill="#D97706">중첩 상태</text>

                <text x="50" y="210" fontSize="8" fill="#6B7280">
                  α|0⟩ + β|1⟩ (|α|² + |β|² = 1)
                </text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">양자 컴퓨팅 주요 방식:</h4>
                <code>{`1. 초전도 큐비트 (Superconducting)
   - 온도: 15 mK (절대영도 근처)
   - 기업: IBM, Google, Rigetti
   - 큐비트: 433개 (IBM Osprey)

2. 이온 트랩 (Trapped Ion)
   - 온도: 상온
   - 기업: IonQ, Honeywell
   - 큐비트: 32개 (높은 정확도)

3. 광자 (Photonic)
   - 온도: 상온
   - 기업: PsiQuantum, Xanadu
   - 장점: 긴 결맞음 시간

4. 중성 원자 (Neutral Atom)
   - 기업: QuEra, Pasqal
   - 큐비트: 256개

5. 실리콘 스핀 (Silicon Spin)
   - 기업: Intel, SiQure
   - 장점: 기존 CMOS 호환`}</code>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded">
                <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">
                  주요 과제
                </h4>
                <div className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                  <div className="flex items-start gap-2">
                    <span className="text-red-600">●</span>
                    <span>결맞음 시간 (Coherence Time): ~100μs → 초 단위 필요</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-orange-600">●</span>
                    <span>게이트 정확도: 99.9% → 99.99% 이상 필요</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-yellow-600">●</span>
                    <span>확장성: 수천~수백만 큐비트 필요</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-blue-600">●</span>
                    <span>에러 수정: 논리 큐비트당 1000개 물리 큐비트</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-green-800 dark:text-green-300 mb-3">
              양자 우위 (Quantum Advantage) 달성 사례
            </h3>
            <div className="grid md:grid-cols-3 gap-3 text-sm text-gray-700 dark:text-gray-300">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-semibold mb-2 text-blue-700 dark:text-blue-400">
                  Google Sycamore (2019)
                </div>
                <div className="text-xs">53 큐비트</div>
                <div className="text-xs text-gray-500">200초 vs 10,000년 (고전)</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-semibold mb-2 text-purple-700 dark:text-purple-400">
                  IBM Osprey (2022)
                </div>
                <div className="text-xs">433 큐비트</div>
                <div className="text-xs text-gray-500">Qiskit Runtime 최적화</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-semibold mb-2 text-green-700 dark:text-green-400">
                  Atom Computing (2023)
                </div>
                <div className="text-xs">1,225 큐비트</div>
                <div className="text-xs text-gray-500">중성 원자 방식</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 뉴로모픽 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          8.2 뉴로모픽 칩 (Neuromorphic Computing)
        </h2>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            뉴로모픽 칩은 생물학적 뇌의 뉴런과 시냅스를 모방하여 초저전력, 실시간
            학습이 가능한 AI 하드웨어를 구현합니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                인공 뉴런 & 시냅스
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* 뉴런 구조 */}
                <text x="90" y="25" fontSize="11" fontWeight="bold" fill="#7C3AED">
                  Spiking Neural Network
                </text>

                {/* Neuron 1 */}
                <circle cx="60" cy="80" r="25" fill="#DDD6FE" stroke="#7C3AED" strokeWidth="2" />
                <text x="48" y="85" fontSize="10" fill="#6B21A8" fontWeight="bold">N1</text>

                {/* Neuron 2 */}
                <circle cx="60" cy="160" r="25" fill="#DDD6FE" stroke="#7C3AED" strokeWidth="2" />
                <text x="48" y="165" fontSize="10" fill="#6B21A8" fontWeight="bold">N2</text>

                {/* Neuron 3 (출력) */}
                <circle cx="180" cy="120" r="30" fill="#A78BFA" stroke="#7C3AED" strokeWidth="3" />
                <text x="165" y="127" fontSize="11" fill="white" fontWeight="bold">N3</text>

                {/* 시냅스 (선) */}
                <line x1="85" y1="80" x2="150" y2="110" stroke="#10B981" strokeWidth="3" />
                <line x1="85" y1="160" x2="150" y2="130" stroke="#EF4444" strokeWidth="3" />

                {/* 가중치 */}
                <circle cx="120" cy="95" r="12" fill="#D1FAE5" stroke="#10B981" strokeWidth="2" />
                <text x="114" y="100" fontSize="9" fill="#059669">w₁</text>

                <circle cx="120" cy="145" r="12" fill="#FEE2E2" stroke="#EF4444" strokeWidth="2" />
                <text x="114" y="150" fontSize="9" fill="#DC2626">w₂</text>

                {/* 스파이크 */}
                <path d="M 20 80 L 25 80 L 27 65 L 29 80 L 35 80"
                      stroke="#3B82F6" strokeWidth="2" fill="none" />
                <text x="15" y="75" fontSize="7" fill="#2563EB">Spike</text>

                <path d="M 20 160 L 25 160 L 27 145 L 29 160 L 35 160"
                      stroke="#3B82F6" strokeWidth="2" fill="none" />

                {/* 출력 스파이크 */}
                <path d="M 215 120 L 220 120 L 222 105 L 224 120 L 230 120 L 232 105 L 234 120 L 245 120"
                      stroke="#EF4444" strokeWidth="2" fill="none" />
                <text x="250" y="125" fontSize="7" fill="#DC2626">Output</text>

                {/* STDP 설명 */}
                <rect x="40" y="200" width="200" height="30" fill="#FEF3C7" stroke="#F59E0B" strokeWidth="1" />
                <text x="50" y="195" fontSize="9" fontWeight="bold" fill="#D97706">
                  STDP (Spike-Timing-Dependent Plasticity)
                </text>
                <text x="50" y="215" fontSize="8" fill="#6B7280">
                  스파이크 타이밍에 따라 시냅스 가중치 자동 조절
                </text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">주요 뉴로모픽 칩:</h4>
                <code>{`IBM TrueNorth (2014)
- 뉴런: 1M
- 시냅스: 256M
- 전력: 70mW
- 용도: 패턴 인식

Intel Loihi 2 (2021)
- 뉴런: 1M
- 시냅스: 120M
- 공정: Intel 4
- 비동기 스파이킹

BrainScaleS-2 (EU, 2020)
- 아날로그 뉴런
- 실시간 학습
- 1000배 가속

SpiNNaker2 (Manchester)
- 152개 ARM 코어/칩
- 10M 뉴런 시뮬레이션

Akida (BrainChip, 2021)
- 엣지 AI
- 온칩 학습
- 초저전력 (1mW)`}</code>
              </div>

              <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
                <h4 className="font-semibold text-orange-800 dark:text-orange-300 mb-2">
                  뉴로모픽 vs 전통 AI
                </h4>
                <div className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                  <div className="flex justify-between">
                    <span>전력 효율</span>
                    <span className="text-green-600">1000배↑</span>
                  </div>
                  <div className="flex justify-between">
                    <span>실시간 학습</span>
                    <span className="text-green-600">O</span>
                  </div>
                  <div className="flex justify-between">
                    <span>시간 정보 처리</span>
                    <span className="text-green-600">우수</span>
                  </div>
                  <div className="flex justify-between">
                    <span>범용성</span>
                    <span className="text-red-600">제한적</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-blue-800 dark:text-blue-300 mb-3">
              응용 분야
            </h3>
            <div className="grid md:grid-cols-4 gap-2 text-xs text-center">
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">로봇 비전</div>
                <div className="text-gray-600 dark:text-gray-400">실시간 물체 추적</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">음성 인식</div>
                <div className="text-gray-600 dark:text-gray-400">저전력 웨이크워드</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">센서 융합</div>
                <div className="text-gray-600 dark:text-gray-400">IoT 엣지</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <div className="font-semibold mb-1">뇌 인터페이스</div>
                <div className="text-gray-600 dark:text-gray-400">BMI 신호 처리</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 광자 칩 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          8.3 광자 집적회로 (Photonic IC)
        </h2>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            광자 칩은 빛을 이용하여 정보를 전송하고 처리합니다. 초고속, 초저전력
            데이터 통신과 AI 연산이 가능합니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                실리콘 포토닉스
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* 도파로 */}
                <rect x="40" y="80" width="200" height="30" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
                <text x="100" y="100" fontSize="10" fill="#2563EB" fontWeight="bold">
                  Silicon Waveguide
                </text>

                {/* 광원 */}
                <circle cx="30" cy="95" r="12" fill="#FCD34D" stroke="#F59E0B" strokeWidth="2" />
                <text x="20" y="125" fontSize="8" fill="#D97706">Laser</text>

                {/* 광선 */}
                <path d="M 42 95 L 235 95" stroke="#FCD34D" strokeWidth="3" opacity="0.7" />
                <path d="M 42 93 L 235 93" stroke="#FBBF24" strokeWidth="2" opacity="0.5" />
                <path d="M 42 97 L 235 97" stroke="#FBBF24" strokeWidth="2" opacity="0.5" />

                {/* 변조기 */}
                <rect x="80" y="70" width="40" height="50" fill="#DDD6FE" stroke="#7C3AED" strokeWidth="2" />
                <text x="85" y="100" fontSize="9" fill="#6B21A8">MZI</text>
                <text x="82" y="135" fontSize="7" fill="#6B7280">변조기</text>

                {/* 링 공진기 */}
                <circle cx="160" cy="50" r="20" fill="none" stroke="#10B981" strokeWidth="3" />
                <text x="145" y="35" fontSize="7" fill="#059669">Ring Resonator</text>

                {/* 검출기 */}
                <rect x="240" y="80" width="30" height="30" fill="#FEE2E2" stroke="#EF4444" strokeWidth="2" />
                <text x="245" y="100" fontSize="9" fill="#DC2626">PD</text>
                <text x="235" y="125" fontSize="7" fill="#6B7280">검출기</text>

                {/* 전기 신호 */}
                <line x1="255" y1="110" x2="255" y2="130" stroke="#374151" strokeWidth="2" />
                <path d="M 250 135 L 255 125 L 260 135" stroke="#374151" strokeWidth="2" fill="none" />
                <text x="240" y="150" fontSize="8" fill="#6B7280">전기 신호</text>

                {/* 레이어 구조 */}
                <rect x="40" y="170" width="200" height="10" fill="#E5E7EB" />
                <text x="245" y="177" fontSize="7" fill="#9CA3AF">SiO₂</text>
                <rect x="40" y="180" width="200" height="15" fill="#60A5FA" />
                <text x="245" y="190" fontSize="7" fill="#2563EB">Si Core</text>
                <rect x="40" y="195" width="200" height="10" fill="#E5E7EB" />
                <text x="245" y="202" fontSize="7" fill="#9CA3AF">SiO₂</text>
                <rect x="40" y="205" width="200" height="20" fill="#9CA3AF" />
                <text x="245" y="217" fontSize="7" fill="#6B7280">Si Substrate</text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">광자 칩 장점:</h4>
                <code>{`1. 초고속 전송
   - 광속 (3×10⁸ m/s)
   - 대역폭: 수 THz

2. 초저전력
   - 전기 대비 10~100배 ↓
   - fJ/bit 수준

3. 낮은 지연
   - 광학 스위칭: ~ps
   - 전기 대비 1000배 빠름

4. 열 발생 적음
   - 쿨링 비용 감소
   - 고밀도 집적

5. 간섭 없음
   - EMI 면역
   - 크로스토크 최소`}</code>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
                <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
                  주요 기업 및 응용
                </h4>
                <div className="space-y-2 text-xs text-gray-700 dark:text-gray-300">
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="font-semibold mb-1">Lightmatter</div>
                    <div className="text-gray-500">광학 AI 가속기 (Envise)</div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="font-semibold mb-1">Ayar Labs</div>
                    <div className="text-gray-500">광학 I/O (TeraPHY)</div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="font-semibold mb-1">Intel</div>
                    <div className="text-gray-500">실리콘 포토닉스 (800G)</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">
              광학 뉴럴 네트워크 (ONN)
            </h3>
            <div className="grid md:grid-cols-3 gap-3 text-sm text-gray-700 dark:text-gray-300">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2 text-blue-700 dark:text-blue-400">
                  행렬 곱셈
                </h4>
                <p className="text-xs">
                  Mach-Zehnder 간섭계 배열로 광학적 MAC 연산
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2 text-purple-700 dark:text-purple-400">
                  활성화 함수
                </h4>
                <p className="text-xs">
                  비선형 광학 소자로 ReLU, Sigmoid 구현
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2 text-green-700 dark:text-green-400">
                  성능
                </h4>
                <p className="text-xs">
                  POPS (Peta OPs) 급 연산 속도 가능
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 탄소 나노튜브 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          8.4 탄소 나노튜브 트랜지스터 (CNFET)
        </h2>

        <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            탄소 나노튜브(CNT)는 실리콘을 대체할 차세대 반도체 소재로, 뛰어난
            전기적 특성과 원자 수준의 얇은 두께를 가집니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                CNFET 구조
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* Gate */}
                <rect x="80" y="50" width="120" height="15" fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />
                <text x="115" y="43" fontSize="10" fill="#2563EB" fontWeight="bold">
                  Gate
                </text>

                {/* Gate Dielectric */}
                <rect x="80" y="65" width="120" height="10" fill="#E5E7EB" stroke="#9CA3AF" strokeWidth="1" />
                <text x="205" y="72" fontSize="8" fill="#6B7280">HfO₂</text>

                {/* CNT Channel */}
                {[...Array(5)].map((_, i) => (
                  <ellipse key={i}
                    cx={100 + i * 20} cy="85"
                    rx="8" ry="3"
                    fill="none" stroke="#374151" strokeWidth="2" />
                ))}
                <text x="205" y="88" fontSize="8" fill="#374151">CNT (직경 1~2nm)</text>

                {/* Source/Drain */}
                <rect x="60" y="95" width="25" height="20" fill="#FCD34D" stroke="#F59E0B" strokeWidth="2" />
                <rect x="195" y="95" width="25" height="20" fill="#FCD34D" stroke="#F59E0B" strokeWidth="2" />
                <text x="55" y="130" fontSize="9" fill="#D97706">S</text>
                <text x="210" y="130" fontSize="9" fill="#D97706">D</text>

                {/* Substrate */}
                <rect x="50" y="115" width="180" height="30" fill="#9CA3AF" stroke="#6B7280" strokeWidth="2" />
                <text x="110" y="133" fontSize="10" fill="white">Substrate</text>

                {/* CNT 특성 */}
                <rect x="40" y="160" width="200" height="70" fill="#F3F4F6" stroke="#9CA3AF" strokeWidth="2" />
                <text x="50" y="155" fontSize="10" fontWeight="bold" fill="#374151">
                  CNT 전기적 특성
                </text>

                <text x="50" y="180" fontSize="8" fill="#6B7280">
                  • 전자 이동도: ~100,000 cm²/V·s
                </text>
                <text x="50" y="195" fontSize="8" fill="#6B7280">
                  • 전류 밀도: ~10⁹ A/cm² (Cu의 1000배)
                </text>
                <text x="50" y="210" fontSize="8" fill="#6B7280">
                  • 밴드갭: 직경으로 조절 가능 (0~2eV)
                </text>
                <text x="50" y="225" fontSize="8" fill="#EF4444">
                  • 과제: 순도, 정렬, 금속/반도체 분리
                </text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">CNFET vs Si MOSFET:</h4>
                <code>{`성능 비교 (동일 크기):

전자 이동도:
  Si: 1,400 cm²/V·s
  CNT: 100,000 cm²/V·s
  → 70배 향상

스위칭 속도:
  Si: 기준
  CNT: 5~10배 빠름

전력 소비:
  동일 성능시 1/10

집적도:
  CNT 직경: 1~2nm
  Si FinFET: 5~10nm
  → 더 미세화 가능

열전도도:
  Si: 150 W/m·K
  CNT: 3,000 W/m·K
  → 우수한 방열`}</code>
              </div>

              <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded">
                <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">
                  주요 연구 성과
                </h4>
                <div className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="font-semibold">MIT (2019)</div>
                    <div className="text-gray-500">14,000개 CNFET 칩 (RISC-V)</div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="font-semibold">Stanford (2020)</div>
                    <div className="text-gray-500">CNT 웨이퍼 스케일 제조</div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 p-2 rounded">
                    <div className="font-semibold">Berkeley (2022)</div>
                    <div className="text-gray-500">자기정렬 CNT 배열</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-red-800 dark:text-red-300 mb-3">
              상용화 과제
            </h3>
            <div className="grid md:grid-cols-4 gap-3 text-sm text-gray-700 dark:text-gray-300">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">순도 제어</h4>
                <p className="text-xs">
                  금속성 CNT &lt; 0.01% 필요 (현재 1~10%)
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">정렬 기술</h4>
                <p className="text-xs">
                  균일한 배열 및 밀도 제어 필요
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">컨택 저항</h4>
                <p className="text-xs">
                  금속-CNT 계면 저항 최소화
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <h4 className="font-semibold mb-2">대량 생산</h4>
                <p className="text-xs">
                  CMOS 호환 공정 개발 필요
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 미래 전망 */}
      <section className="mb-8">
        <div className="bg-gradient-to-r from-purple-50 via-blue-50 to-green-50 dark:from-purple-900/20 dark:via-blue-900/20 dark:to-green-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            미래 반도체 로드맵
          </h3>

          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-3">
                단기 (2025~2030)
              </h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>GAA/CFET 양산</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>HBM4 (2TB/s)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>EUV High-NA 도입</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>Chiplet 표준화 (UCIe)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600">✓</span>
                  <span>1nm 공정 달성</span>
                </li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-700 dark:text-purple-400 mb-3">
                중기 (2030~2040)
              </h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <span className="text-yellow-600">◐</span>
                  <span>광자 IC 상용화</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-yellow-600">◐</span>
                  <span>뉴로모픽 메인스트림</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-yellow-600">◐</span>
                  <span>양자 컴퓨터 실용화</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-yellow-600">◐</span>
                  <span>CNFET 초기 양산</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-yellow-600">◐</span>
                  <span>차세대 메모리 보편화</span>
                </li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-400 mb-3">
                장기 (2040~2050)
              </h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <span className="text-gray-400">○</span>
                  <span>탄소 기반 전자소자</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-gray-400">○</span>
                  <span>분자/DNA 컴퓨팅</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-gray-400">○</span>
                  <span>실온 양자 컴퓨터</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-gray-400">○</span>
                  <span>뇌-컴퓨터 융합</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-gray-400">○</span>
                  <span>포스트 CMOS 시대</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="bg-gray-800 text-white p-6 rounded-lg">
            <h4 className="text-lg font-semibold mb-4 text-center">
              반도체의 미래: 무어의 법칙을 넘어서
            </h4>
            <div className="grid md:grid-cols-2 gap-6 text-sm">
              <div>
                <h5 className="text-yellow-400 font-semibold mb-2">패러다임 전환:</h5>
                <ul className="space-y-1 text-xs">
                  <li>• More Moore → More than Moore</li>
                  <li>• 2D → 3D 적층 (수직 통합)</li>
                  <li>• 범용 → 도메인 특화 (AI, 양자)</li>
                  <li>• 실리콘 → 다양한 소재 (CNT, 광자)</li>
                  <li>• 폰 노이만 → 새로운 컴퓨팅 (뉴로모픽, 양자)</li>
                </ul>
              </div>
              <div>
                <h5 className="text-green-400 font-semibold mb-2">핵심 과제:</h5>
                <ul className="space-y-1 text-xs">
                  <li>• 전력 효율: 10~1000배 향상 필요</li>
                  <li>• 메모리 병목: HBM, CXL, 새로운 메모리</li>
                  <li>• 이질적 통합: Chiplet, 3D IC</li>
                  <li>• 지속 가능성: 탄소 중립, 재활용</li>
                  <li>• AI 시대 대응: 온디바이스 AI, 엣지 컴퓨팅</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 최종 요약 */}
      <section className="mb-8">
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            핵심 요약
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>양자 컴퓨팅은 큐비트의 중첩과 얽힘으로 고전 컴퓨터 불가능한 계산을 수행합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>뉴로모픽 칩은 생물학적 뇌를 모방하여 초저전력 실시간 학습을 구현합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>광자 집적회로는 빛으로 초고속, 초저전력 데이터 전송과 AI 연산을 수행합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>탄소 나노튜브는 실리콘 대비 70배 높은 이동도로 차세대 소재로 연구되고 있습니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>반도체의 미래는 이질적 통합, 도메인 특화, 새로운 컴퓨팅 패러다임으로 진화합니다</span>
            </li>
          </ul>
        </div>
      </section>

      <References
        sections={[
          {
            title: '📚 공식 문서 & 최신 연구',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'IBM Quantum Computing Roadmap',
                url: 'https://www.ibm.com/quantum/roadmap',
                description: 'IBM 433-qubit Osprey, 1000+ qubit Condor 로드맵 (2023-2025)'
              },
              {
                title: 'Google Quantum AI Research',
                url: 'https://quantumai.google/',
                description: 'Sycamore 양자 우위 달성, Willow 칩 개발 현황'
              },
              {
                title: 'Intel Loihi 2 Neuromorphic System',
                url: 'https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html',
                description: 'Intel 4 공정 뉴로모픽 칩 - 1M 뉴런, 비동기 스파이킹'
              },
              {
                title: 'IMEC 2nm Technology Roadmap',
                url: 'https://www.imec-int.com/en/articles/imec-2nm-roadmap',
                description: 'GAA, CFET, 백사이드 전력 배선 기술 로드맵 (2024-2028)'
              },
              {
                title: 'Nature Electronics: Future Semiconductor Materials',
                url: 'https://www.nature.com/natelectron/',
                description: '2D 소재, CNT, 그래핀 반도체 최신 연구 (2024-2025)'
              }
            ]
          },
          {
            title: '🔬 핵심 논문 & 브레이크스루',
            icon: 'research' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'Quantum Supremacy Using a Programmable Superconducting Processor',
                url: 'https://www.nature.com/articles/s41586-019-1666-5',
                description: 'Google Sycamore 양자 우위 달성 논문 (Nature 2019)'
              },
              {
                title: 'A Million Spiking-Neuron Integrated Circuit with Scalable Communication Network',
                url: 'https://www.science.org/doi/10.1126/science.1254642',
                description: 'IBM TrueNorth 뉴로모픽 칩 아키텍처 (Science 2014)'
              },
              {
                title: 'Deep Learning with Coherent Nanophotonic Circuits',
                url: 'https://www.nature.com/articles/nphoton.2017.93',
                description: 'MIT 광학 뉴럴 네트워크 (ONN) 구현 (Nature Photonics 2017)'
              },
              {
                title: 'Carbon Nanotube Computer Built at Stanford',
                url: 'https://www.nature.com/articles/nature12502',
                description: 'Stanford 14,000 CNFET RISC-V 칩 제작 성공 (Nature 2013)'
              },
              {
                title: 'Roadmap on Emerging Hardware and Technology for Machine Learning',
                url: 'https://iopscience.iop.org/article/10.1088/1361-6528/ac69f8',
                description: 'AI 하드웨어 미래 기술 로드맵 - 뉴로모픽, 광자, 양자 (Nanotechnology 2022)'
              }
            ]
          },
          {
            title: '🛠️ 실전 플랫폼 & 도구',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'IBM Qiskit Quantum Development Kit',
                url: 'https://qiskit.org/',
                description: '양자 컴퓨팅 프로그래밍 프레임워크 - Python SDK, 시뮬레이터'
              },
              {
                title: 'Intel Lava Neuromorphic Framework',
                url: 'https://github.com/lava-nc/lava',
                description: 'Loihi 2 뉴로모픽 칩 프로그래밍 - 스파이킹 뉴럴 네트워크'
              },
              {
                title: 'Lightmatter Envise Photonic AI Platform',
                url: 'https://lightmatter.co/products/envise/',
                description: '광학 AI 가속기 - 실리콘 포토닉스 기반 추론 엔진'
              },
              {
                title: 'PennyLane Quantum Machine Learning',
                url: 'https://pennylane.ai/',
                description: '양자 기계학습 라이브러리 - TensorFlow/PyTorch 통합'
              },
              {
                title: 'Cirq: Google Quantum Programming Framework',
                url: 'https://quantumai.google/cirq',
                description: 'Google Sycamore용 양자 회로 설계 및 시뮬레이션'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
