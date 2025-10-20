'use client'

import References from '@/components/common/References';

export default function Chapter1() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Chapter 1: 반도체 기초
      </h1>

      {/* 실리콘 구조 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          1.1 실리콘의 결정 구조
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            실리콘(Si)은 원자번호 14번 원소로, 4개의 가전자를 가지고 있습니다.
            다이아몬드 결정 구조를 형성하며, 각 실리콘 원자는 4개의 이웃 원자와 공유결합을 이룹니다.
          </p>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <svg className="w-full h-64" viewBox="0 0 400 300">
              {/* 실리콘 결정 구조 시각화 */}
              <circle cx="200" cy="150" r="20" fill="#4F46E5" />
              <circle cx="150" cy="100" r="15" fill="#818CF8" />
              <circle cx="250" cy="100" r="15" fill="#818CF8" />
              <circle cx="150" cy="200" r="15" fill="#818CF8" />
              <circle cx="250" cy="200" r="15" fill="#818CF8" />

              <line x1="200" y1="150" x2="150" y2="100" stroke="#6366F1" strokeWidth="2" />
              <line x1="200" y1="150" x2="250" y2="100" stroke="#6366F1" strokeWidth="2" />
              <line x1="200" y1="150" x2="150" y2="200" stroke="#6366F1" strokeWidth="2" />
              <line x1="200" y1="150" x2="250" y2="200" stroke="#6366F1" strokeWidth="2" />

              <text x="200" y="150" textAnchor="middle" dy="5" fill="white" fontSize="12">Si</text>
              <text x="150" y="100" textAnchor="middle" dy="5" fill="white" fontSize="10">Si</text>
              <text x="250" y="100" textAnchor="middle" dy="5" fill="white" fontSize="10">Si</text>
              <text x="150" y="200" textAnchor="middle" dy="5" fill="white" fontSize="10">Si</text>
              <text x="250" y="200" textAnchor="middle" dy="5" fill="white" fontSize="10">Si</text>
            </svg>
            <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-2">
              실리콘 다이아몬드 결정 구조 (공유결합)
            </p>
          </div>
        </div>

        <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
          <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">핵심 특성:</h3>
          <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
            <li>밴드갭(Band Gap): 1.12 eV (300K)</li>
            <li>격자 상수: 5.43 Å</li>
            <li>전자 이동도: 1,400 cm²/V·s</li>
            <li>정공 이동도: 450 cm²/V·s</li>
          </ul>
        </div>
      </section>

      {/* 도핑 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          1.2 도핑(Doping)과 캐리어
        </h2>
        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-red-800 dark:text-red-300 mb-3">N형 반도체 (전자 과잉)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
              5가 원소(P, As, Sb)를 도핑하여 전자를 제공합니다.
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <svg className="w-full h-32" viewBox="0 0 200 150">
                <circle cx="100" cy="75" r="25" fill="#EF4444" />
                <circle cx="60" cy="45" r="15" fill="#F87171" />
                <circle cx="140" cy="45" r="15" fill="#F87171" />
                <circle cx="60" cy="105" r="15" fill="#F87171" />
                <circle cx="140" cy="105" r="15" fill="#F87171" />
                <circle cx="100" cy="30" r="8" fill="#FEE2E2" stroke="#EF4444" strokeWidth="2" />
                <text x="100" y="78" textAnchor="middle" fill="white" fontSize="14">P</text>
                <text x="100" y="25" textAnchor="middle" fill="#DC2626" fontSize="10">e⁻</text>
              </svg>
              <p className="text-xs text-center text-gray-600 dark:text-gray-400 mt-2">
                인(P) 도핑 → 자유 전자 생성
              </p>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-blue-800 dark:text-blue-300 mb-3">P형 반도체 (정공 과잉)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
              3가 원소(B, Al, Ga)를 도핑하여 정공을 생성합니다.
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <svg className="w-full h-32" viewBox="0 0 200 150">
                <circle cx="100" cy="75" r="25" fill="#3B82F6" />
                <circle cx="60" cy="45" r="15" fill="#60A5FA" />
                <circle cx="140" cy="45" r="15" fill="#60A5FA" />
                <circle cx="60" cy="105" r="15" fill="#60A5FA" />
                <circle cx="140" cy="105" r="15" fill="#60A5FA" />
                <circle cx="100" cy="30" r="8" fill="white" stroke="#3B82F6" strokeWidth="2" />
                <text x="100" y="78" textAnchor="middle" fill="white" fontSize="14">B</text>
                <text x="100" y="25" textAnchor="middle" fill="#2563EB" fontSize="10">h⁺</text>
              </svg>
              <p className="text-xs text-center text-gray-600 dark:text-gray-400 mt-2">
                붕소(B) 도핑 → 정공 생성
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 text-white p-4 rounded-lg">
          <h3 className="font-semibold mb-2">도핑 농도 계산:</h3>
          <pre><code>{`# N형 반도체 전자 농도
n = Nd + ni²/Na  (Nd >> ni인 경우 n ≈ Nd)

# P형 반도체 정공 농도
p = Na + ni²/Nd  (Na >> ni인 경우 p ≈ Na)

# 질량작용의 법칙
n × p = ni²

여기서:
- Nd: 도너 농도 (N형)
- Na: 억셉터 농도 (P형)
- ni: 진성 캐리어 농도 (Si: 1.5×10¹⁰ cm⁻³ @ 300K)`}</code></pre>
        </div>
      </section>

      {/* PN 접합 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          1.3 PN 접합(PN Junction)
        </h2>
        <div className="bg-gradient-to-r from-red-50 to-blue-50 dark:from-red-900/20 dark:to-blue-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            P형과 N형 반도체를 접합하면 접합면에서 전자와 정공이 재결합하여
            공핍층(Depletion Region)이 형성되고, 내부 전기장이 생성됩니다.
          </p>

          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <svg className="w-full h-48" viewBox="0 0 500 200">
              {/* P형 영역 */}
              <rect x="0" y="0" width="200" height="200" fill="#FEE2E2" />
              <text x="100" y="30" textAnchor="middle" fill="#DC2626" fontWeight="bold">P형</text>
              {/* 정공 표시 */}
              {[...Array(8)].map((_, i) => (
                <circle key={`p${i}`} cx={40 + (i % 4) * 40} cy={80 + Math.floor(i / 4) * 40} r="6" fill="#EF4444" />
              ))}

              {/* 공핍층 */}
              <rect x="200" y="0" width="100" height="200" fill="#F3F4F6" />
              <text x="250" y="30" textAnchor="middle" fill="#6B7280" fontSize="12">공핍층</text>
              <line x1="250" y1="50" x2="250" y2="150" stroke="#9CA3AF" strokeWidth="2" strokeDasharray="5,5" />

              {/* 이온 표시 */}
              <text x="220" y="100" fill="#DC2626" fontSize="20">⊖</text>
              <text x="220" y="130" fill="#DC2626" fontSize="20">⊖</text>
              <text x="280" y="100" fill="#3B82F6" fontSize="20">⊕</text>
              <text x="280" y="130" fill="#3B82F6" fontSize="20">⊕</text>

              {/* N형 영역 */}
              <rect x="300" y="0" width="200" height="200" fill="#DBEAFE" />
              <text x="400" y="30" textAnchor="middle" fill="#2563EB" fontWeight="bold">N형</text>
              {/* 전자 표시 */}
              {[...Array(8)].map((_, i) => (
                <circle key={`n${i}`} cx={340 + (i % 4) * 40} cy={80 + Math.floor(i / 4) * 40} r="6" fill="#3B82F6" />
              ))}

              {/* 전기장 화살표 */}
              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="#9CA3AF" />
                </marker>
              </defs>
              <line x1="230" y1="180" x2="270" y2="180" stroke="#9CA3AF" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <text x="250" y="195" textAnchor="middle" fontSize="10" fill="#6B7280">E-field</text>
            </svg>
            <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-2">
              PN 접합의 공핍층 형성과 내부 전기장
            </p>
          </div>
        </div>

        <div className="bg-gray-800 text-white p-4 rounded-lg mb-4">
          <h3 className="font-semibold mb-2">내부 전위(Built-in Potential):</h3>
          <pre><code>{`V_bi = (kT/q) × ln(Na × Nd / ni²)

300K에서:
V_bi ≈ 0.026 × ln(Na × Nd / ni²) [V]

예시 (Na = Nd = 10¹⁶ cm⁻³):
V_bi ≈ 0.026 × ln(10¹⁶ × 10¹⁶ / (1.5×10¹⁰)²)
    ≈ 0.026 × 27.6
    ≈ 0.72 V`}</code></pre>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-green-800 dark:text-green-300 mb-2">순방향 바이어스</h3>
            <p className="text-gray-700 dark:text-gray-300 text-sm">
              P형에 (+), N형에 (-)를 인가하면 공핍층이 좁아지고 전류가 흐릅니다.
            </p>
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-orange-800 dark:text-orange-300 mb-2">역방향 바이어스</h3>
            <p className="text-gray-700 dark:text-gray-300 text-sm">
              P형에 (-), N형에 (+)를 인가하면 공핍층이 넓어지고 전류가 차단됩니다.
            </p>
          </div>
        </div>
      </section>

      {/* 다이오드 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          1.4 다이오드(Diode) 특성
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            PN 접합 다이오드는 전류를 한 방향으로만 흐르게 하는 정류 소자입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">I-V 특성 곡선</h3>
              <svg className="w-full h-48" viewBox="0 0 300 200">
                {/* 좌표축 */}
                <line x1="50" y1="100" x2="280" y2="100" stroke="#6B7280" strokeWidth="2" />
                <line x1="150" y1="20" x2="150" y2="180" stroke="#6B7280" strokeWidth="2" />

                {/* 순방향 곡선 */}
                <path d="M 150 100 Q 160 100, 180 70 Q 200 40, 250 20"
                      stroke="#10B981" strokeWidth="3" fill="none" />

                {/* 역방향 곡선 */}
                <path d="M 150 100 L 50 105" stroke="#EF4444" strokeWidth="3" />

                {/* 레이블 */}
                <text x="270" y="115" fontSize="12" fill="#6B7280">V</text>
                <text x="155" y="20" fontSize="12" fill="#6B7280">I</text>
                <text x="180" y="95" fontSize="10" fill="#10B981">순방향</text>
                <text x="60" y="95" fontSize="10" fill="#EF4444">역방향</text>
                <text x="155" y="95" fontSize="10" fill="#6B7280">0.7V</text>
              </svg>
            </div>

            <div className="bg-gray-800 text-white p-4 rounded-lg">
              <h3 className="font-semibold mb-2">Shockley 다이오드 방정식:</h3>
              <pre><code className="text-sm">{`I = Is × (e^(qV/nkT) - 1)

여기서:
- Is: 역포화 전류
- V: 인가 전압
- n: 이상계수 (1~2)
- q: 전자 전하량
- k: 볼츠만 상수
- T: 절대온도

실리콘 다이오드:
- Vf ≈ 0.7V (순방향)
- Vr > 50V (역방향)`}</code></pre>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
          <h3 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-2">주요 파라미터:</h3>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="text-gray-700 dark:text-gray-300">
              <strong>순방향 전압 강하 (Vf):</strong> 0.6~0.7V (Si)
            </div>
            <div className="text-gray-700 dark:text-gray-300">
              <strong>역방향 항복 전압 (Vbr):</strong> 50~1000V
            </div>
            <div className="text-gray-700 dark:text-gray-300">
              <strong>순방향 전류 (If):</strong> mA ~ 수십 A
            </div>
            <div className="text-gray-700 dark:text-gray-300">
              <strong>역방향 누설전류 (Ir):</strong> nA ~ μA
            </div>
          </div>
        </div>
      </section>

      {/* 트랜지스터 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          1.5 트랜지스터 원리
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          {/* BJT */}
          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-purple-800 dark:text-purple-300 mb-3">
              BJT (Bipolar Junction Transistor)
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <svg className="w-full h-40" viewBox="0 0 200 150">
                {/* NPN 구조 */}
                <rect x="20" y="50" width="50" height="50" fill="#DBEAFE" />
                <rect x="70" y="40" width="30" height="70" fill="#FEE2E2" />
                <rect x="100" y="50" width="50" height="50" fill="#DBEAFE" />

                <text x="45" y="80" textAnchor="middle" fontSize="12" fill="#2563EB">N</text>
                <text x="85" y="80" textAnchor="middle" fontSize="12" fill="#DC2626">P</text>
                <text x="125" y="80" textAnchor="middle" fontSize="12" fill="#2563EB">N</text>

                <text x="45" y="35" fontSize="10" fill="#6B7280">Emitter</text>
                <text x="75" y="35" fontSize="10" fill="#6B7280">Base</text>
                <text x="115" y="35" fontSize="10" fill="#6B7280">Collector</text>

                {/* 전류 화살표 */}
                <line x1="45" y1="30" x2="45" y2="50" stroke="#10B981" strokeWidth="2" markerEnd="url(#arrow)" />
                <line x1="125" y1="50" x2="125" y2="30" stroke="#10B981" strokeWidth="2" markerEnd="url(#arrow)" />
                <text x="30" y="45" fontSize="8" fill="#10B981">Ie</text>
                <text x="130" y="45" fontSize="8" fill="#10B981">Ic</text>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>{`Ic = β × Ib
Ie = Ic + Ib
β = 50~200 (전류증폭률)`}</code>
            </div>
          </div>

          {/* MOSFET */}
          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-indigo-800 dark:text-indigo-300 mb-3">
              MOSFET (Metal-Oxide-Semiconductor FET)
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <svg className="w-full h-40" viewBox="0 0 200 150">
                {/* NMOS 구조 */}
                <rect x="40" y="80" width="120" height="30" fill="#FEE2E2" />
                <rect x="70" y="70" width="20" height="15" fill="#DBEAFE" />
                <rect x="110" y="70" width="20" height="15" fill="#DBEAFE" />
                <rect x="60" y="50" width="80" height="5" fill="#9CA3AF" />
                <line x1="100" y1="40" x2="100" y2="50" stroke="#374151" strokeWidth="2" />

                <text x="80" y="105" fontSize="10" fill="#2563EB">S</text>
                <text x="120" y="105" fontSize="10" fill="#2563EB">D</text>
                <text x="100" y="35" fontSize="10" fill="#6B7280">Gate</text>
                <text x="95" y="125" fontSize="10" fill="#DC2626">P-sub</text>

                <text x="60" y="68" fontSize="8" fill="#2563EB">n+</text>
                <text x="120" y="68" fontSize="8" fill="#2563EB">n+</text>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>{`Id = (W/L) × μ × Cox × (Vgs-Vth)²
Vth: 문턱전압 (0.3~0.7V)
전압제어형 소자`}</code>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <h3 className="font-semibold text-blue-800 dark:text-blue-300 mb-3">트랜지스터 비교</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-blue-200 dark:border-blue-800">
                  <th className="px-3 py-2 text-left">특성</th>
                  <th className="px-3 py-2 text-left">BJT</th>
                  <th className="px-3 py-2 text-left">MOSFET</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300">
                <tr className="border-b border-blue-100 dark:border-blue-900">
                  <td className="px-3 py-2">제어 방식</td>
                  <td className="px-3 py-2">전류 제어</td>
                  <td className="px-3 py-2">전압 제어</td>
                </tr>
                <tr className="border-b border-blue-100 dark:border-blue-900">
                  <td className="px-3 py-2">입력 임피던스</td>
                  <td className="px-3 py-2">낮음 (kΩ)</td>
                  <td className="px-3 py-2">매우 높음 (MΩ)</td>
                </tr>
                <tr className="border-b border-blue-100 dark:border-blue-900">
                  <td className="px-3 py-2">스위칭 속도</td>
                  <td className="px-3 py-2">빠름</td>
                  <td className="px-3 py-2">매우 빠름</td>
                </tr>
                <tr>
                  <td className="px-3 py-2">집적도</td>
                  <td className="px-3 py-2">낮음</td>
                  <td className="px-3 py-2">매우 높음</td>
                </tr>
              </tbody>
            </table>
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
              <span>실리콘은 4가 원소로 다이아몬드 결정구조를 형성하며 반도체의 기본 재료입니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>도핑을 통해 N형(전자 과잉)과 P형(정공 과잉) 반도체를 만들 수 있습니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>PN 접합은 공핍층과 내부 전위를 형성하여 정류 기능을 수행합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>다이오드는 PN 접합의 정류 특성을 이용한 가장 기본적인 반도체 소자입니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-purple-600 dark:text-purple-400 mr-2">▪</span>
              <span>트랜지스터는 BJT(전류제어)와 MOSFET(전압제어) 방식으로 증폭/스위칭을 수행합니다</span>
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
                title: 'ITRS - International Technology Roadmap for Semiconductors',
                link: 'https://www.irds.ieee.org/',
                description: 'IEEE IRDS 반도체 기술 로드맵 (ITRS 후속) - 산업 표준'
              },
              {
                title: 'SEMI Standards - Semiconductor Equipment and Materials',
                link: 'https://www.semi.org/en/products-services/standards',
                description: 'SEMI 국제 반도체 장비 및 재료 표준'
              },
              {
                title: 'SIA - Semiconductor Industry Association Reports',
                link: 'https://www.semiconductors.org/',
                description: '미국 반도체 산업협회 공식 보고서 및 통계'
              },
              {
                title: 'JESD - JEDEC Solid State Technology Standards',
                link: 'https://www.jedec.org/',
                description: 'JEDEC 반도체 소자 표준 및 규격'
              }
            ]
          },
          {
            title: '📖 핵심 교재 & 논문',
            icon: 'book' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Physics of Semiconductor Devices (3rd Edition)',
                authors: 'Sze, S. M., Ng, K. K.',
                year: '2006',
                description: 'Wiley - 반도체 물리 바이블, 전세계 대학 표준 교재'
              },
              {
                title: 'Semiconductor Device Fundamentals',
                authors: 'Pierret, R. F.',
                year: '1996',
                description: 'Addison-Wesley - 반도체 소자 기초 이론 교재'
              },
              {
                title: 'Modern Semiconductor Devices for Integrated Circuits',
                authors: 'Hu, C.',
                year: '2010',
                description: 'Prentice Hall - UC Berkeley 교수 집필, 실무 중심'
              },
              {
                title: 'PN Junction Diode Characteristics',
                authors: 'Shockley, W.',
                year: '1949',
                description: 'Bell Labs - Shockley 다이오드 방정식 원본 논문 (노벨상)'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 시뮬레이터',
            icon: 'web' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'MIT Semiconductor Device Simulations',
                link: 'https://nanohub.org/',
                description: 'NanoHUB - MIT/Purdue 온라인 반도체 소자 시뮬레이터'
              },
              {
                title: 'Silvaco TCAD - Device Simulation Software',
                link: 'https://silvaco.com/',
                description: 'Silvaco TCAD 반도체 소자 시뮬레이션 툴 (학생용 무료)'
              },
              {
                title: 'SPICE Circuit Simulator',
                link: 'https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html',
                description: 'LTspice - Analog Devices 무료 회로 시뮬레이터'
              },
              {
                title: 'PhET Interactive Simulations - Semiconductors',
                link: 'https://phet.colorado.edu/',
                description: '콜로라도대 PhET - 반도체 물리 인터랙티브 시뮬레이션'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
