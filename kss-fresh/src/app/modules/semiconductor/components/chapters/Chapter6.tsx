'use client'

import References from '@/components/common/References';

export default function Chapter6() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Chapter 6: AI 반도체 아키텍처
      </h1>

      {/* NPU 개요 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          6.1 NPU (Neural Processing Unit)
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            NPU는 딥러닝 추론에 특화된 프로세서로, 행렬 연산과 컨볼루션을 효율적으로
            처리합니다. 스마트폰, 엣지 디바이스에 탑재되어 온디바이스 AI를 구현합니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                NPU 아키텍처
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* PE Array */}
                <rect x="40" y="40" width="120" height="100" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
                <text x="70" y="30" fontSize="11" fontWeight="bold" fill="#2563EB">
                  PE Array (Processing Elements)
                </text>

                {/* PE 그리드 */}
                {[...Array(4)].map((_, i) =>
                  [...Array(4)].map((_, j) => (
                    <rect key={`${i}-${j}`}
                      x={50 + j * 27} y={50 + i * 22}
                      width="22" height="17"
                      fill="#60A5FA" stroke="#3B82F6" strokeWidth="1" />
                  ))
                )}
                <text x="75" y="95" fontSize="9" fill="white" fontWeight="bold">MAC</text>
                <text x="65" y="155" fontSize="8" fill="#2563EB">16×16 = 256 PE</text>

                {/* 메모리 */}
                <rect x="180" y="40" width="80" height="40" fill="#FEE2E2" stroke="#EF4444" strokeWidth="2" />
                <text x="195" y="63" fontSize="10" fill="#DC2626" fontWeight="bold">
                  On-chip SRAM
                </text>

                <rect x="180" y="90" width="80" height="40" fill="#FEF3C7" stroke="#F59E0B" strokeWidth="2" />
                <text x="200" y="113" fontSize="10" fill="#D97706" fontWeight="bold">
                  Weight Buffer
                </text>

                {/* 제어 로직 */}
                <rect x="40" y="160" width="220" height="40" fill="#DDD6FE" stroke="#7C3AED" strokeWidth="2" />
                <text x="105" y="183" fontSize="10" fill="#6B21A8" fontWeight="bold">
                  Control & Dataflow Engine
                </text>

                {/* 인터커넥트 */}
                <line x1="160" y1="90" x2="180" y2="60" stroke="#10B981" strokeWidth="2" />
                <line x1="160" y1="110" x2="180" y2="110" stroke="#10B981" strokeWidth="2" />
                <text x="165" y="85" fontSize="7" fill="#10B981">Bus</text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">MAC (Multiply-Accumulate):</h4>
                <code>{`result = Σ(ai × wi) + bias

1 MAC = 2 FLOPs
  - 1 곱셈 (Multiply)
  - 1 덧셈 (Accumulate)

NPU 성능:
16×16 PE Array @ 2GHz
= 256 MACs × 2 GHz
= 512 GFLOPS (FP16)
= 1 TOPS (INT8)`}</code>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded">
                <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">
                  주요 NPU 사례
                </h4>
                <div className="space-y-1 text-xs text-gray-700 dark:text-gray-300">
                  <div className="flex justify-between">
                    <span>Apple A17 Pro</span>
                    <span className="text-green-600">35 TOPS</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Qualcomm 8 Gen 3</span>
                    <span className="text-green-600">45 TOPS</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Google Tensor G4</span>
                    <span className="text-green-600">32 TOPS</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Samsung Exynos 2400</span>
                    <span className="text-green-600">40 TOPS</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">
              Dataflow 아키텍처
            </h3>
            <div className="grid md:grid-cols-3 gap-3 text-sm text-gray-700 dark:text-gray-300">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-semibold mb-2 text-blue-700 dark:text-blue-400">
                  Weight Stationary
                </div>
                <div className="text-xs">가중치 고정, 입력/출력 이동</div>
                <div className="text-xs text-gray-500 mt-1">예: Google TPU</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-semibold mb-2 text-green-700 dark:text-green-400">
                  Output Stationary
                </div>
                <div className="text-xs">출력 고정, 가중치/입력 이동</div>
                <div className="text-xs text-gray-500 mt-1">예: Eyeriss</div>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="font-semibold mb-2 text-purple-700 dark:text-purple-400">
                  Row Stationary
                </div>
                <div className="text-xs">행 단위 재사용</div>
                <div className="text-xs text-gray-500 mt-1">메모리 최적화</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* TPU */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          6.2 TPU (Tensor Processing Unit)
        </h2>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Google이 개발한 TPU는 대규모 학습과 추론에 특화된 ASIC입니다.
            Transformer 모델 학습에 최적화되어 있으며, 현재 v5p까지 발전했습니다.
          </p>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
            <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
              TPU v4/v5 아키텍처
            </h3>
            <svg className="w-full h-64" viewBox="0 0 600 280">
              {/* Systolic Array */}
              <rect x="30" y="40" width="180" height="180" fill="#DDD6FE" stroke="#7C3AED" strokeWidth="3" />
              <text x="70" y="25" fontSize="12" fontWeight="bold" fill="#6B21A8">
                Systolic Array
              </text>

              {/* 256×256 그리드 (단순화) */}
              {[...Array(8)].map((_, i) =>
                [...Array(8)].map((_, j) => (
                  <rect key={`${i}-${j}`}
                    x={40 + j * 21} y={50 + i * 21}
                    width="18" height="18"
                    fill="#A78BFA" stroke="#7C3AED" strokeWidth="0.5" />
                ))
              )}
              <text x="80" y="235" fontSize="10" fill="#6B21A8">
                256×256 = 65,536 MACs
              </text>

              {/* HBM */}
              <rect x="240" y="40" width="100" height="60" fill="#FEE2E2" stroke="#EF4444" strokeWidth="2" />
              <text x="265" y="63" fontSize="11" fill="#DC2626" fontWeight="bold">HBM2e</text>
              <text x="255" y="80" fontSize="9" fill="#6B7280">32GB</text>
              <text x="250" y="92" fontSize="9" fill="#6B7280">1.6TB/s</text>

              {/* Vector Unit */}
              <rect x="240" y="110" width="100" height="50" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
              <text x="255" y="138" fontSize="10" fill="#2563EB" fontWeight="bold">
                Vector Units
              </text>

              {/* Scalar Unit */}
              <rect x="240" y="170" width="100" height="50" fill="#D1FAE5" stroke="#10B981" strokeWidth="2" />
              <text x="260" y="198" fontSize="10" fill="#059669" fontWeight="bold">
                Scalar Units
              </text>

              {/* Interconnect */}
              <rect x="360" y="40" width="220" height="180" fill="#FEF3C7" stroke="#F59E0B" strokeWidth="2" />
              <text x="420" y="25" fontSize="11" fontWeight="bold" fill="#D97706">
                ICI (Inter-Chip Interconnect)
              </text>

              {/* 네트워크 토폴로지 */}
              <circle cx="420" cy="80" r="15" fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />
              <circle cx="480" cy="80" r="15" fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />
              <circle cx="540" cy="80" r="15" fill="#60A5FA" stroke="#3B82F6" strokeWidth="2" />

              <circle cx="420" cy="140" r="15" fill="#A78BFA" stroke="#7C3AED" strokeWidth="2" />
              <circle cx="480" cy="140" r="15" fill="#A78BFA" stroke="#7C3AED" strokeWidth="2" />
              <circle cx="540" cy="140" r="15" fill="#A78BFA" stroke="#7C3AED" strokeWidth="2" />

              <circle cx="420" cy="200" r="15" fill="#34D399" stroke="#10B981" strokeWidth="2" />
              <circle cx="480" cy="200" r="15" fill="#34D399" stroke="#10B981" strokeWidth="2" />
              <circle cx="540" cy="200" r="15" fill="#34D399" stroke="#10B981" strokeWidth="2" />

              {/* 연결선 */}
              <line x1="435" y1="80" x2="465" y2="80" stroke="#374151" strokeWidth="2" />
              <line x1="495" y1="80" x2="525" y2="80" stroke="#374151" strokeWidth="2" />
              <line x1="420" y1="95" x2="420" y2="125" stroke="#374151" strokeWidth="2" />
              <line x1="480" y1="95" x2="480" y2="125" stroke="#374151" strokeWidth="2" />

              <text x="410" y="250" fontSize="9" fill="#D97706">3D Torus Network</text>
              <text x="420" y="262" fontSize="8" fill="#6B7280">4,096 칩 연결</text>
            </svg>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-800 text-white p-4 rounded text-sm">
              <h4 className="font-semibold mb-3">TPU 세대별 진화:</h4>
              <div className="space-y-2 text-xs">
                <div className="border-l-4 border-blue-500 pl-3">
                  <div className="font-semibold">TPU v1 (2016)</div>
                  <div>추론 전용, 92 TOPS (INT8)</div>
                </div>
                <div className="border-l-4 border-purple-500 pl-3">
                  <div className="font-semibold">TPU v2 (2017)</div>
                  <div>학습 지원, 45 TFLOPS (bfloat16)</div>
                </div>
                <div className="border-l-4 border-green-500 pl-3">
                  <div className="font-semibold">TPU v3 (2018)</div>
                  <div>2배 성능, 액체냉각</div>
                </div>
                <div className="border-l-4 border-yellow-500 pl-3">
                  <div className="font-semibold">TPU v4 (2021)</div>
                  <div>275 TFLOPS, ICI 3.6TB/s</div>
                </div>
                <div className="border-l-4 border-red-500 pl-3">
                  <div className="font-semibold">TPU v5p (2023)</div>
                  <div>459 TFLOPS, 8,192개 Pod</div>
                </div>
              </div>
            </div>

            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded">
              <h4 className="font-semibold text-indigo-800 dark:text-indigo-300 mb-3">
                Systolic Array 동작
              </h4>
              <div className="bg-white dark:bg-gray-800 p-3 rounded mb-2">
                <svg className="w-full h-32" viewBox="0 0 240 140">
                  {/* 3x3 Systolic Array */}
                  {[...Array(3)].map((_, i) =>
                    [...Array(3)].map((_, j) => (
                      <g key={`${i}-${j}`}>
                        <rect
                          x={60 + j * 40} y={20 + i * 40}
                          width="35" height="35"
                          fill="#DDD6FE" stroke="#7C3AED" strokeWidth="2" />
                        <text x={73 + j * 40} y={40 + i * 40} fontSize="10" fill="#6B21A8">
                          PE
                        </text>
                      </g>
                    ))
                  )}

                  {/* 입력 화살표 (상단) */}
                  <line x1="77" y1="5" x2="77" y2="18" stroke="#3B82F6" strokeWidth="2" markerEnd="url(#arrow11)" />
                  <line x1="117" y1="5" x2="117" y2="18" stroke="#3B82F6" strokeWidth="2" markerEnd="url(#arrow11)" />
                  <line x1="157" y1="5" x2="157" y2="18" stroke="#3B82F6" strokeWidth="2" markerEnd="url(#arrow11)" />
                  <text x="95" y="12" fontSize="8" fill="#3B82F6">Weights</text>

                  {/* 입력 화살표 (좌측) */}
                  <line x1="5" y1="37" x2="58" y2="37" stroke="#10B981" strokeWidth="2" markerEnd="url(#arrow11)" />
                  <line x1="5" y1="77" x2="58" y2="77" stroke="#10B981" strokeWidth="2" markerEnd="url(#arrow11)" />
                  <line x1="5" y1="117" x2="58" y2="117" stroke="#10B981" strokeWidth="2" markerEnd="url(#arrow11)" />
                  <text x="15" y="33" fontSize="8" fill="#10B981">Activations</text>

                  {/* 출력 화살표 */}
                  <line x1="195" y1="37" x2="230" y2="37" stroke="#EF4444" strokeWidth="2" markerEnd="url(#arrow11)" />
                  <text x="200" y="33" fontSize="8" fill="#EF4444">Out</text>

                  <defs>
                    <marker id="arrow11" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                      <polygon points="0 0, 8 3, 0 6" fill="currentColor" />
                    </marker>
                  </defs>
                </svg>
              </div>
              <p className="text-xs text-gray-700 dark:text-gray-300">
                데이터가 파동처럼 흘러가며 병렬 연산 수행.
                메모리 접근 최소화로 높은 효율 달성.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* GPU AI */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          6.3 GPU 아키텍처 (NVIDIA Hopper/Blackwell)
        </h2>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            NVIDIA GPU는 범용성과 고성능을 모두 갖춘 AI 가속기입니다.
            Transformer 엔진과 FP8 지원으로 대규모 모델 학습을 주도합니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
                H100 (Hopper) 아키텍처
              </h3>
              <svg className="w-full h-56" viewBox="0 0 280 240">
                {/* SM (Streaming Multiprocessor) */}
                <text x="90" y="20" fontSize="11" fontWeight="bold" fill="#10B981">
                  132 SMs (16,896 CUDA Cores)
                </text>

                {/* SM 그리드 */}
                {[...Array(4)].map((_, i) =>
                  [...Array(6)].map((_, j) => (
                    <rect key={`${i}-${j}`}
                      x={30 + j * 40} y={30 + i * 40}
                      width="35" height="35"
                      fill="#D1FAE5" stroke="#10B981" strokeWidth="1" />
                  ))
                )}
                <text x="110" y="185" fontSize="9" fill="#059669">
                  각 SM = 128 FP32 cores
                </text>

                {/* Tensor Core */}
                <rect x="30" y="200" width="110" height="30" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
                <text x="45" y="218" fontSize="10" fill="#2563EB" fontWeight="bold">
                  528 Tensor Cores (4th Gen)
                </text>

                {/* HBM3 */}
                <rect x="150" y="200" width="110" height="30" fill="#FEE2E2" stroke="#EF4444" strokeWidth="2" />
                <text x="165" y="218" fontSize="10" fill="#DC2626" fontWeight="bold">
                  HBM3 80GB (3TB/s)
                </text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded text-xs">
                <h4 className="font-semibold mb-2">성능 지표 (H100 SXM):</h4>
                <code>{`FP64: 60 TFLOPS
FP32: 120 TFLOPS (TF32 사용시)
FP16: 989 TFLOPS (Tensor)
FP8: 1,979 TFLOPS (Transformer)
INT8: 3,958 TOPS

메모리:
- HBM3: 80GB
- 대역폭: 3.35 TB/s
- L2 캐시: 50MB

전력: 700W (SXM), 350W (PCIe)
트랜지스터: 800억개 (4nm)`}</code>
              </div>

              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
                <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
                  Transformer 엔진
                </h4>
                <p className="text-xs text-gray-700 dark:text-gray-300 mb-2">
                  FP8과 FP16을 동적으로 전환하여 정확도를 유지하면서 성능 2배 향상
                </p>
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-xs">
                  <code className="text-blue-600">
                    FP8 (E4M3): 빠른 연산<br/>
                    FP8 (E5M2): 넓은 범위<br/>
                    자동 스케일링
                  </code>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 text-white p-4 rounded-lg">
            <h3 className="font-semibold mb-3">B200 (Blackwell) 혁신:</h3>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <h4 className="text-yellow-400 mb-2">아키텍처:</h4>
                <ul className="space-y-1 text-xs">
                  <li>• 2080억 트랜지스터 (4nm)</li>
                  <li>• 듀얼 다이 (10TB/s NVLink-C2C)</li>
                  <li>• 192GB HBM3e (8TB/s)</li>
                  <li>• 5세대 Tensor Core</li>
                  <li>• 2세대 Transformer 엔진</li>
                </ul>
              </div>
              <div>
                <h4 className="text-green-400 mb-2">성능:</h4>
                <ul className="space-y-1 text-xs">
                  <li>• FP4: 20 PFLOPS (추론)</li>
                  <li>• FP8: 10 PFLOPS (학습)</li>
                  <li>• FP16: 5 PFLOPS</li>
                  <li>• 전력: 1200W</li>
                  <li>• H100 대비 2.5배 (학습), 30배 (추론)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* HBM */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          6.4 HBM (High Bandwidth Memory)
        </h2>

        <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            HBM은 DRAM 칩을 수직으로 적층하고 실리콘 인터포저로 연결하여
            초고대역폭을 제공하는 메모리입니다. AI 칩의 필수 요소입니다.
          </p>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
            <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
              HBM 구조
            </h3>
            <svg className="w-full h-56" viewBox="0 0 500 240">
              {/* DRAM 스택 */}
              <text x="30" y="25" fontSize="11" fontWeight="bold" fill="#EF4444">
                DRAM 스택 (8~16 다이)
              </text>

              {[...Array(8)].map((_, i) => (
                <g key={i}>
                  <rect x="40" y={40 + i * 15} width="100" height="12"
                        fill="#FEE2E2" stroke="#DC2626" strokeWidth="1" />
                  <text x="75" y={49 + i * 15} fontSize="7" fill="#DC2626">
                    DRAM {i}
                  </text>
                </g>
              ))}

              {/* TSV */}
              {[50, 70, 90, 110, 130].map((x, i) => (
                <line key={i} x1={x} y1="40" x2={x} y2="160" stroke="#3B82F6" strokeWidth="2" />
              ))}
              <text x="60" y="175" fontSize="8" fill="#2563EB">TSV (1024개)</text>

              {/* Logic Die */}
              <rect x="40" y="180" width="100" height="20" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
              <text x="60" y="193" fontSize="9" fill="#2563EB" fontWeight="bold">
                Logic Die (PHY)
              </text>

              {/* Microbump */}
              {[...Array(10)].map((_, i) => (
                <circle key={i} cx={45 + i * 10} cy="205" r="2" fill="#FCD34D" stroke="#D97706" />
              ))}
              <text x="60" y="220" fontSize="7" fill="#D97706">Microbumps</text>

              {/* 인터포저 */}
              <rect x="20" y="210" width="140" height="15" fill="#F59E0B" stroke="#D97706" strokeWidth="2" />
              <text x="50" y="220" fontSize="9" fill="white" fontWeight="bold">
                Silicon Interposer
              </text>

              {/* GPU */}
              <rect x="200" y="150" width="120" height="75" fill="#D1FAE5" stroke="#10B981" strokeWidth="3" />
              <text x="230" y="190" fontSize="12" fill="#059669" fontWeight="bold">
                GPU/NPU
              </text>

              {/* 연결 */}
              <line x1="160" y1="217" x2="200" y2="217" stroke="#374151" strokeWidth="4" />
              <text x="165" y="212" fontSize="8" fill="#374151">1024-bit</text>
              <text x="165" y="235" fontSize="7" fill="#6B7280">채널당 8Gb/s × 128채널</text>

              {/* 스택 정보 */}
              <rect x="360" y="40" width="120" height="160" fill="#FEF3C7" stroke="#F59E0B" strokeWidth="2" />
              <text x="375" y="30" fontSize="10" fontWeight="bold" fill="#D97706">
                HBM 세대별 스펙
              </text>

              <text x="370" y="60" fontSize="8" fill="#6B7280">HBM1 (2013)</text>
              <text x="370" y="72" fontSize="7" fill="#6B7280">128 GB/s, 4GB</text>

              <text x="370" y="90" fontSize="8" fill="#6B7280">HBM2 (2016)</text>
              <text x="370" y="102" fontSize="7" fill="#6B7280">256 GB/s, 8GB</text>

              <text x="370" y="120" fontSize="8" fill="#6B7280">HBM2e (2018)</text>
              <text x="370" y="132" fontSize="7" fill="#6B7280">460 GB/s, 24GB</text>

              <text x="370" y="150" fontSize="8" fill="#10B981" fontWeight="bold">HBM3 (2022)</text>
              <text x="370" y="162" fontSize="7" fill="#059669">819 GB/s, 24GB</text>

              <text x="370" y="180" fontSize="8" fill="#EF4444" fontWeight="bold">HBM3e (2024)</text>
              <text x="370" y="192" fontSize="7" fill="#DC2626">1.15 TB/s, 36GB</text>
            </svg>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-800 text-white p-3 rounded text-xs">
              <h4 className="font-semibold mb-2">HBM vs GDDR 비교:</h4>
              <code>{`HBM3e:
- 대역폭: 1.15 TB/s (per stack)
- 버스 폭: 1024-bit
- 전력 효율: 7.5 pJ/bit
- 용량: 최대 36GB (per stack)

GDDR6X:
- 대역폭: 1 TB/s (전체)
- 버스 폭: 384-bit
- 전력 효율: 10 pJ/bit
- 용량: 최대 24GB (전체)

HBM 장점:
- 3D 적층 → 초고대역폭
- 짧은 신호 경로 → 저전력
- 작은 풋프린트`}</code>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded">
              <h4 className="font-semibold text-red-800 dark:text-red-300 mb-2">
                HBM 시장 현황
              </h4>
              <div className="space-y-2 text-xs text-gray-700 dark:text-gray-300">
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <div className="font-semibold mb-1">SK하이닉스</div>
                  <div className="text-gray-500">시장 점유율 50% (HBM3e 선두)</div>
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <div className="font-semibold mb-1">삼성전자</div>
                  <div className="text-gray-500">30% (HBM3 12단 양산)</div>
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <div className="font-semibold mb-1">마이크론</div>
                  <div className="text-gray-500">20% (HBM3e Gen2 개발)</div>
                </div>
                <div className="bg-yellow-100 dark:bg-yellow-900/30 p-2 rounded mt-2">
                  <div className="font-semibold text-yellow-800 dark:text-yellow-400">
                    2024년 시장: $200억 (전년비 200%↑)
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 요약 */}
      <section className="mb-8">
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            핵심 요약
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>NPU는 MAC 연산에 특화된 PE Array로 엣지 디바이스에서 효율적인 AI 추론을 제공합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>TPU는 Systolic Array와 ICI로 대규모 모델 학습에 최적화되어 있습니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>NVIDIA GPU는 Tensor Core와 Transformer 엔진으로 범용성과 고성능을 모두 제공합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>HBM은 3D 적층으로 TB/s급 대역폭을 제공하여 AI 칩의 메모리 병목을 해결합니다</span>
            </li>
          </ul>
        </div>
      </section>

      <References
        sections={[
          {
            title: '📚 공식 문서 & 제품 백서',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'NVIDIA GPU Architecture Documentation',
                url: 'https://docs.nvidia.com/cuda/hopper-architecture/index.html',
                description: 'Hopper H100, Blackwell B200 GPU 아키텍처 상세 문서'
              },
              {
                title: 'Google Cloud TPU System Architecture',
                url: 'https://cloud.google.com/tpu/docs/system-architecture-tpu-vm',
                description: 'TPU v4/v5 아키텍처, Systolic Array, ICI 네트워크 기술'
              },
              {
                title: 'Apple Neural Engine Technical Overview',
                url: 'https://machinelearning.apple.com/research/neural-engine',
                description: 'A17 Pro NPU 아키텍처 및 온디바이스 AI 기술'
              },
              {
                title: 'SK hynix HBM3e Product Specification',
                url: 'https://www.skhynix.com/products/hbm/',
                description: 'HBM3e 1.15TB/s 메모리 스펙 및 AI 가속기 적용 사례'
              },
              {
                title: 'Qualcomm AI Engine Direct SDK',
                url: 'https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk',
                description: 'Hexagon NPU 프로그래밍 가이드 및 최적화 기법'
              }
            ]
          },
          {
            title: '🔬 핵심 논문 & 연구',
            icon: 'research' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'In-Datacenter Performance Analysis of a Tensor Processing Unit',
                url: 'https://arxiv.org/abs/1704.04760',
                description: 'Google TPU v1 아키텍처 및 성능 분석 논문 (ISCA 2017)'
              },
              {
                title: 'Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks',
                url: 'https://ieeexplore.ieee.org/document/7738524',
                description: 'MIT Eyeriss NPU - Row Stationary Dataflow 설계 (ISSCC 2016)'
              },
              {
                title: 'NVIDIA A100 Tensor Core GPU Architecture',
                url: 'https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf',
                description: 'Ampere 아키텍처 백서 - Tensor Core, MIG 기술 상세'
              },
              {
                title: 'HBM3: The Next Generation High Bandwidth Memory',
                url: 'https://ieeexplore.ieee.org/document/9444087',
                description: 'JEDEC HBM3 표준 및 AI 워크로드 최적화 (VLSI 2021)'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 프레임워크',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'NVIDIA CUDA Toolkit & cuDNN',
                url: 'https://developer.nvidia.com/cuda-toolkit',
                description: 'GPU 프로그래밍 툴킷 - Tensor Core 최적화 라이브러리'
              },
              {
                title: 'TensorFlow XLA Compiler',
                url: 'https://www.tensorflow.org/xla',
                description: 'TPU/GPU 최적화 컴파일러 - Accelerated Linear Algebra'
              },
              {
                title: 'Intel OpenVINO Toolkit',
                url: 'https://docs.openvino.ai/',
                description: 'NPU/GPU 통합 추론 엔진 - 모델 최적화 및 배포'
              },
              {
                title: 'AMD ROCm Platform',
                url: 'https://rocm.docs.amd.com/',
                description: 'AMD GPU HIP 프로그래밍 - MI300 시리즈 지원'
              },
              {
                title: 'MLPerf Benchmark Suite',
                url: 'https://mlcommons.org/benchmarks/',
                description: 'AI 칩 성능 벤치마크 - 학습/추론 표준 평가'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
