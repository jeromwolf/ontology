'use client'

export default function Chapter2() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Chapter 2: 디지털 회로 설계
      </h1>

      {/* CMOS 기초 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          2.1 CMOS (Complementary MOS) 로직
        </h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            CMOS는 PMOS와 NMOS를 상보적으로 사용하여 저전력, 고집적 디지털 회로를 구현합니다.
            현대 모든 디지털 칩의 기본 기술입니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">CMOS 인버터</h3>
              <svg className="w-full h-56" viewBox="0 0 200 240">
                {/* VDD */}
                <line x1="0" y1="20" x2="200" y2="20" stroke="#EF4444" strokeWidth="3" />
                <text x="10" y="15" fill="#EF4444" fontWeight="bold">VDD</text>

                {/* PMOS */}
                <rect x="85" y="40" width="30" height="40" fill="#FEE2E2" stroke="#DC2626" strokeWidth="2" />
                <circle cx="100" cy="60" r="3" fill="white" stroke="#DC2626" strokeWidth="1" />
                <line x1="70" y1="60" x2="85" y2="60" stroke="#374151" strokeWidth="2" />
                <line x1="100" y1="40" x2="100" y2="20" stroke="#374151" strokeWidth="2" />
                <line x1="100" y1="80" x2="100" y2="120" stroke="#374151" strokeWidth="2" />
                <text x="120" y="65" fontSize="12" fill="#DC2626">PMOS</text>

                {/* NMOS */}
                <rect x="85" y="140" width="30" height="40" fill="#DBEAFE" stroke="#2563EB" strokeWidth="2" />
                <circle cx="100" cy="160" r="8" fill="none" stroke="#2563EB" strokeWidth="1" />
                <line x1="70" y1="160" x2="85" y2="160" stroke="#374151" strokeWidth="2" />
                <line x1="100" y1="120" x2="100" y2="140" stroke="#374151" strokeWidth="2" />
                <line x1="100" y1="180" x2="100" y2="200" stroke="#374151" strokeWidth="2" />
                <text x="120" y="165" fontSize="12" fill="#2563EB">NMOS</text>

                {/* GND */}
                <line x1="90" y1="200" x2="110" y2="200" stroke="#6B7280" strokeWidth="3" />
                <line x1="95" y1="205" x2="105" y2="205" stroke="#6B7280" strokeWidth="2" />
                <line x1="98" y1="210" x2="102" y2="210" stroke="#6B7280" strokeWidth="1" />

                {/* 입출력 */}
                <line x1="20" y1="110" x2="70" y2="110" stroke="#10B981" strokeWidth="2" />
                <line x1="70" y1="60" x2="70" y2="160" stroke="#10B981" strokeWidth="2" />
                <text x="25" y="105" fontSize="12" fill="#10B981" fontWeight="bold">IN</text>

                <line x1="100" y1="120" x2="150" y2="120" stroke="#F59E0B" strokeWidth="2" />
                <circle cx="100" cy="120" r="3" fill="#F59E0B" />
                <text x="155" y="125" fontSize="12" fill="#F59E0B" fontWeight="bold">OUT</text>
              </svg>
            </div>

            <div className="space-y-3">
              <div className="bg-gray-800 text-white p-3 rounded-lg">
                <h4 className="font-semibold mb-2 text-sm">동작 원리:</h4>
                <pre className="text-xs"><code>{`IN = 0 (LOW):
  PMOS: ON  → VDD 연결
  NMOS: OFF → GND 차단
  OUT = 1 (HIGH)

IN = 1 (HIGH):
  PMOS: OFF → VDD 차단
  NMOS: ON  → GND 연결
  OUT = 0 (LOW)`}</code></pre>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg">
                <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2 text-sm">
                  CMOS의 장점:
                </h4>
                <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                  <li>✓ 정적 전력 소비 거의 0</li>
                  <li>✓ 높은 노이즈 마진</li>
                  <li>✓ 온도 안정성 우수</li>
                  <li>✓ 고집적 가능</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 text-white p-4 rounded-lg">
          <h3 className="font-semibold mb-2">전력 소비 분석:</h3>
          <pre><code>{`P_total = P_static + P_dynamic + P_short-circuit

1. 정적 전력 (누설 전류):
   P_static = V_dd × I_leakage

2. 동적 전력 (스위칭):
   P_dynamic = α × C_load × V_dd² × f
   여기서:
   - α: 활동도 계수 (0~1)
   - C_load: 부하 커패시턴스
   - f: 동작 주파수

3. 단락 전력 (천이 구간):
   P_short = V_dd × I_peak × t_sc × f

현대 칩에서: P_dynamic이 60~70% 차지`}</code></pre>
        </div>
      </section>

      {/* 기본 게이트 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          2.2 기본 로직 게이트 설계
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          {/* NAND */}
          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-purple-800 dark:text-purple-300 mb-3">
              2-Input NAND Gate
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-2">
              <svg className="w-full h-48" viewBox="0 0 200 200">
                {/* VDD */}
                <line x1="0" y1="15" x2="200" y2="15" stroke="#EF4444" strokeWidth="2" />

                {/* PMOS 병렬 */}
                <rect x="50" y="25" width="25" height="30" fill="#FEE2E2" stroke="#DC2626" />
                <rect x="125" y="25" width="25" height="30" fill="#FEE2E2" stroke="#DC2626" />
                <line x1="62" y1="15" x2="62" y2="25" stroke="#374151" strokeWidth="2" />
                <line x1="137" y1="15" x2="137" y2="25" stroke="#374151" strokeWidth="2" />
                <line x1="62" y1="55" x2="62" y2="90" stroke="#374151" strokeWidth="2" />
                <line x1="137" y1="55" x2="137" y2="90" stroke="#374151" strokeWidth="2" />
                <line x1="62" y1="90" x2="100" y2="90" stroke="#374151" strokeWidth="2" />
                <line x1="137" y1="90" x2="100" y2="90" stroke="#374151" strokeWidth="2" />

                {/* NMOS 직렬 */}
                <rect x="87" y="110" width="25" height="30" fill="#DBEAFE" stroke="#2563EB" />
                <rect x="87" y="150" width="25" height="30" fill="#DBEAFE" stroke="#2563EB" />
                <line x1="100" y1="90" x2="100" y2="110" stroke="#374151" strokeWidth="2" />
                <line x1="100" y1="140" x2="100" y2="150" stroke="#374151" strokeWidth="2" />
                <line x1="100" y1="180" x2="100" y2="195" stroke="#374151" strokeWidth="2" />

                {/* GND */}
                <line x1="90" y1="195" x2="110" y2="195" stroke="#6B7280" strokeWidth="2" />

                {/* 입력 */}
                <line x1="20" y1="40" x2="50" y2="40" stroke="#10B981" strokeWidth="2" />
                <line x1="20" y1="40" x2="20" y2="125" stroke="#10B981" strokeWidth="2" />
                <line x1="20" y1="125" x2="87" y2="125" stroke="#10B981" strokeWidth="2" />
                <text x="25" y="38" fontSize="10" fill="#10B981">A</text>

                <line x1="170" y1="40" x2="125" y2="40" stroke="#F59E0B" strokeWidth="2" />
                <line x1="170" y1="40" x2="170" y2="165" stroke="#F59E0B" strokeWidth="2" />
                <line x1="170" y1="165" x2="112" y2="165" stroke="#F59E0B" strokeWidth="2" />
                <text x="175" y="38" fontSize="10" fill="#F59E0B">B</text>

                {/* 출력 */}
                <circle cx="100" cy="90" r="3" fill="#A855F7" />
                <line x1="100" y1="90" x2="180" y2="90" stroke="#A855F7" strokeWidth="2" />
                <text x="185" y="93" fontSize="10" fill="#A855F7">OUT</text>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>OUT = NOT(A AND B)</code>
            </div>
          </div>

          {/* NOR */}
          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-indigo-800 dark:text-indigo-300 mb-3">
              2-Input NOR Gate
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-2">
              <svg className="w-full h-48" viewBox="0 0 200 200">
                {/* VDD */}
                <line x1="0" y1="15" x2="200" y2="15" stroke="#EF4444" strokeWidth="2" />

                {/* PMOS 직렬 */}
                <rect x="87" y="25" width="25" height="30" fill="#FEE2E2" stroke="#DC2626" />
                <rect x="87" y="65" width="25" height="30" fill="#FEE2E2" stroke="#DC2626" />
                <line x1="100" y1="15" x2="100" y2="25" stroke="#374151" strokeWidth="2" />
                <line x1="100" y1="55" x2="100" y2="65" stroke="#374151" strokeWidth="2" />
                <line x1="100" y1="95" x2="100" y2="110" stroke="#374151" strokeWidth="2" />

                {/* NMOS 병렬 */}
                <rect x="50" y="125" width="25" height="30" fill="#DBEAFE" stroke="#2563EB" />
                <rect x="125" y="125" width="25" height="30" fill="#DBEAFE" stroke="#2563EB" />
                <line x1="62" y1="110" x2="62" y2="125" stroke="#374151" strokeWidth="2" />
                <line x1="137" y1="110" x2="137" y2="125" stroke="#374151" strokeWidth="2" />
                <line x1="62" y1="155" x2="62" y2="195" stroke="#374151" strokeWidth="2" />
                <line x1="137" y1="155" x2="137" y2="195" stroke="#374151" strokeWidth="2" />
                <line x1="62" y1="110" x2="100" y2="110" stroke="#374151" strokeWidth="2" />
                <line x1="137" y1="110" x2="100" y2="110" stroke="#374151" strokeWidth="2" />

                {/* GND */}
                <line x1="52" y1="195" x2="72" y2="195" stroke="#6B7280" strokeWidth="2" />
                <line x1="127" y1="195" x2="147" y2="195" stroke="#6B7280" strokeWidth="2" />

                {/* 입력 */}
                <line x1="20" y1="40" x2="87" y2="40" stroke="#10B981" strokeWidth="2" />
                <line x1="20" y1="40" x2="20" y2="140" stroke="#10B981" strokeWidth="2" />
                <line x1="20" y1="140" x2="50" y2="140" stroke="#10B981" strokeWidth="2" />
                <text x="25" y="38" fontSize="10" fill="#10B981">A</text>

                <line x1="170" y1="80" x2="112" y2="80" stroke="#F59E0B" strokeWidth="2" />
                <line x1="170" y1="80" x2="170" y2="140" stroke="#F59E0B" strokeWidth="2" />
                <line x1="170" y1="140" x2="125" y2="140" stroke="#F59E0B" strokeWidth="2" />
                <text x="175" y="78" fontSize="10" fill="#F59E0B">B</text>

                {/* 출력 */}
                <circle cx="100" cy="110" r="3" fill="#A855F7" />
                <line x1="100" y1="110" x2="180" y2="110" stroke="#A855F7" strokeWidth="2" />
                <text x="185" y="113" fontSize="10" fill="#A855F7">OUT</text>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>OUT = NOT(A OR B)</code>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <h3 className="font-semibold text-blue-800 dark:text-blue-300 mb-3">진리표 및 설계 규칙</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="overflow-x-auto">
              <table className="w-full text-sm bg-white dark:bg-gray-800 rounded">
                <thead>
                  <tr className="border-b border-blue-200 dark:border-blue-800">
                    <th className="px-3 py-2">A</th>
                    <th className="px-3 py-2">B</th>
                    <th className="px-3 py-2">NAND</th>
                    <th className="px-3 py-2">NOR</th>
                  </tr>
                </thead>
                <tbody className="text-gray-700 dark:text-gray-300 text-center">
                  <tr className="border-b border-blue-100 dark:border-blue-900">
                    <td className="px-3 py-2">0</td>
                    <td className="px-3 py-2">0</td>
                    <td className="px-3 py-2 text-green-600 font-bold">1</td>
                    <td className="px-3 py-2 text-green-600 font-bold">1</td>
                  </tr>
                  <tr className="border-b border-blue-100 dark:border-blue-900">
                    <td className="px-3 py-2">0</td>
                    <td className="px-3 py-2">1</td>
                    <td className="px-3 py-2 text-green-600 font-bold">1</td>
                    <td className="px-3 py-2 text-red-600 font-bold">0</td>
                  </tr>
                  <tr className="border-b border-blue-100 dark:border-blue-900">
                    <td className="px-3 py-2">1</td>
                    <td className="px-3 py-2">0</td>
                    <td className="px-3 py-2 text-green-600 font-bold">1</td>
                    <td className="px-3 py-2 text-red-600 font-bold">0</td>
                  </tr>
                  <tr>
                    <td className="px-3 py-2">1</td>
                    <td className="px-3 py-2">1</td>
                    <td className="px-3 py-2 text-red-600 font-bold">0</td>
                    <td className="px-3 py-2 text-red-600 font-bold">0</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="bg-gray-800 text-white p-3 rounded text-xs">
              <h4 className="font-semibold mb-2">설계 규칙:</h4>
              <code>{`1. Pull-up Network (PMOS):
   - NAND: 병렬
   - NOR: 직렬

2. Pull-down Network (NMOS):
   - NAND: 직렬
   - NOR: 병렬

3. 이중성 원리:
   AND ↔ OR
   직렬 ↔ 병렬`}</code>
            </div>
          </div>
        </div>
      </section>

      {/* 타이밍 분석 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          2.3 타이밍 분석과 지연
        </h2>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-lg mb-4">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            디지털 회로의 성능은 신호 전파 지연(Propagation Delay)에 의해 결정됩니다.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">전파 지연</h3>
              <svg className="w-full h-40" viewBox="0 0 300 120">
                {/* 입력 신호 */}
                <path d="M 20 80 L 60 80 L 60 30 L 140 30" stroke="#10B981" strokeWidth="2" fill="none" />
                <text x="25" y="95" fontSize="10" fill="#10B981">IN</text>

                {/* 출력 신호 (지연) */}
                <path d="M 20 100 L 80 100 L 80 50 L 160 50" stroke="#F59E0B" strokeWidth="2" fill="none" />
                <text x="25" y="115" fontSize="10" fill="#F59E0B">OUT</text>

                {/* 지연 표시 */}
                <line x1="60" y1="35" x2="80" y2="35" stroke="#EF4444" strokeWidth="1" strokeDasharray="2,2" />
                <line x1="60" y1="30" x2="60" y2="40" stroke="#EF4444" strokeWidth="1" />
                <line x1="80" y1="30" x2="80" y2="40" stroke="#EF4444" strokeWidth="1" />
                <text x="63" y="28" fontSize="8" fill="#EF4444">t_pd</text>

                {/* 천이 시간 */}
                <line x1="60" y1="75" x2="60" y2="35" stroke="#A855F7" strokeWidth="1" strokeDasharray="2,2" />
                <text x="45" y="58" fontSize="8" fill="#A855F7">t_r</text>
              </svg>
            </div>

            <div className="bg-gray-800 text-white p-3 rounded">
              <h4 className="font-semibold mb-2 text-sm">타이밍 파라미터:</h4>
              <pre className="text-xs"><code>{`t_pd: 전파 지연
  - t_pHL: HIGH→LOW 지연
  - t_pLH: LOW→HIGH 지연
  - t_pd = (t_pHL + t_pLH)/2

t_r: 상승 시간 (10%→90%)
t_f: 하강 시간 (90%→10%)

게이트 지연 모델:
t_pd = R_eq × C_load + t_intrinsic`}</code></pre>
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-red-800 dark:text-red-300 mb-3">셋업/홀드 타임</h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-2">
              <svg className="w-full h-32" viewBox="0 0 300 100">
                {/* 클럭 */}
                <path d="M 20 60 L 60 60 L 60 30 L 100 30 L 100 60 L 140 60"
                      stroke="#3B82F6" strokeWidth="2" fill="none" />
                <text x="25" y="75" fontSize="10" fill="#3B82F6">CLK</text>

                {/* 데이터 */}
                <path d="M 20 85 L 45 85 L 50 80 L 120 80 L 125 85 L 140 85"
                      stroke="#10B981" strokeWidth="2" fill="none" />
                <text x="25" y="100" fontSize="10" fill="#10B981">DATA</text>

                {/* 셋업 */}
                <line x1="50" y1="73" x2="60" y2="73" stroke="#EF4444" strokeWidth="1" />
                <text x="52" y="70" fontSize="8" fill="#EF4444">t_su</text>

                {/* 홀드 */}
                <line x1="60" y1="90" x2="125" y2="90" stroke="#F59E0B" strokeWidth="1" />
                <text x="85" y="97" fontSize="8" fill="#F59E0B">t_h</text>
              </svg>
            </div>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              t_su: 클럭 엣지 전 데이터 안정 시간<br/>
              t_h: 클럭 엣지 후 데이터 유지 시간
            </p>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-green-800 dark:text-green-300 mb-3">최대 동작 주파수</h3>
            <div className="bg-gray-800 text-white p-3 rounded">
              <pre className="text-xs"><code>{`T_clk ≥ t_cq + t_logic + t_su

여기서:
- t_cq: 클럭→출력 지연
- t_logic: 조합회로 지연
- t_su: 셋업 타임

f_max = 1 / T_clk_min

예시:
t_cq = 0.5ns
t_logic = 2.0ns
t_su = 0.3ns
→ T_clk ≥ 2.8ns
→ f_max = 357 MHz`}</code></pre>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <h3 className="font-semibold text-blue-800 dark:text-blue-300 mb-3">지연 최적화 기법</h3>
          <div className="grid md:grid-cols-3 gap-3 text-sm text-gray-700 dark:text-gray-300">
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <h4 className="font-semibold mb-2 text-blue-700 dark:text-blue-400">트랜지스터 사이징</h4>
              <p className="text-xs">더 넓은 트랜지스터 사용으로 구동 능력 향상 (면적↑, 전력↑)</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <h4 className="font-semibold mb-2 text-blue-700 dark:text-blue-400">버퍼 삽입</h4>
              <p className="text-xs">긴 배선에 버퍼 체인 삽입으로 RC 지연 감소</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <h4 className="font-semibold mb-2 text-blue-700 dark:text-blue-400">로직 재구성</h4>
              <p className="text-xs">크리티컬 패스의 로직 뎁스 감소</p>
            </div>
          </div>
        </div>
      </section>

      {/* 전력 최적화 */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
          2.4 전력 최적화 설계
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-purple-800 dark:text-purple-300 mb-3">
              클럭 게이팅 (Clock Gating)
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <svg className="w-full h-32" viewBox="0 0 250 120">
                {/* 클럭 입력 */}
                <line x1="20" y1="40" x2="80" y2="40" stroke="#3B82F6" strokeWidth="2" />
                <text x="25" y="35" fontSize="10" fill="#3B82F6">CLK</text>

                {/* Enable */}
                <line x1="20" y1="80" x2="80" y2="80" stroke="#10B981" strokeWidth="2" />
                <text x="25" y="95" fontSize="10" fill="#10B981">EN</text>

                {/* AND 게이트 */}
                <path d="M 80 30 L 80 90 L 120 90 Q 140 60, 120 30 Z"
                      fill="#FEE2E2" stroke="#DC2626" strokeWidth="2" />
                <text x="95" y="65" fontSize="12" fill="#DC2626">&</text>

                {/* Gated CLK */}
                <line x1="120" y1="60" x2="200" y2="60" stroke="#F59E0B" strokeWidth="2" />
                <text x="150" y="55" fontSize="10" fill="#F59E0B">Gated_CLK</text>

                {/* Register */}
                <rect x="200" y="40" width="30" height="40" fill="#DBEAFE" stroke="#2563EB" strokeWidth="2" />
                <text x="210" y="65" fontSize="10" fill="#2563EB">FF</text>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>전력 절감: 20~40%<br/>사용하지 않는 블록의 클럭 차단</code>
            </div>
          </div>

          <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg">
            <h3 className="font-semibold text-indigo-800 dark:text-indigo-300 mb-3">
              멀티 Vdd (Multi-Vdd)
            </h3>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg mb-3">
              <svg className="w-full h-32" viewBox="0 0 250 120">
                {/* 크리티컬 패스 - High Vdd */}
                <rect x="30" y="20" width="80" height="30" fill="#FEE2E2" stroke="#EF4444" strokeWidth="2" />
                <text x="50" y="38" fontSize="10" fill="#EF4444">Critical Path</text>
                <text x="55" y="50" fontSize="8" fill="#DC2626">Vdd = 1.0V</text>

                {/* 비크리티컬 - Low Vdd */}
                <rect x="30" y="70" width="80" height="30" fill="#DBEAFE" stroke="#3B82F6" strokeWidth="2" />
                <text x="35" y="88" fontSize="10" fill="#3B82F6">Non-Critical</text>
                <text x="50" y="98" fontSize="8" fill="#2563EB">Vdd = 0.7V</text>

                {/* 레벨 시프터 */}
                <path d="M 130 50 L 150 35 L 150 65 Z" fill="#F59E0B" stroke="#D97706" strokeWidth="2" />
                <text x="155" y="53" fontSize="8" fill="#D97706">Level Shifter</text>
              </svg>
            </div>
            <div className="bg-gray-800 text-white p-2 rounded text-xs">
              <code>전력 ∝ V²<br/>0.7V 사용시 50% 전력 절감</code>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 text-white p-4 rounded-lg">
          <h3 className="font-semibold mb-3">전력 최적화 전략 비교:</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-600">
                  <th className="px-3 py-2 text-left">기법</th>
                  <th className="px-3 py-2 text-left">절감률</th>
                  <th className="px-3 py-2 text-left">복잡도</th>
                  <th className="px-3 py-2 text-left">적용 시기</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-700">
                  <td className="px-3 py-2">클럭 게이팅</td>
                  <td className="px-3 py-2 text-green-400">20-40%</td>
                  <td className="px-3 py-2 text-yellow-400">중간</td>
                  <td className="px-3 py-2">RTL/게이트</td>
                </tr>
                <tr className="border-b border-gray-700">
                  <td className="px-3 py-2">멀티 Vdd</td>
                  <td className="px-3 py-2 text-green-400">30-50%</td>
                  <td className="px-3 py-2 text-red-400">높음</td>
                  <td className="px-3 py-2">물리설계</td>
                </tr>
                <tr className="border-b border-gray-700">
                  <td className="px-3 py-2">파워 게이팅</td>
                  <td className="px-3 py-2 text-green-400">50-70%</td>
                  <td className="px-3 py-2 text-red-400">높음</td>
                  <td className="px-3 py-2">아키텍처</td>
                </tr>
                <tr>
                  <td className="px-3 py-2">DVFS</td>
                  <td className="px-3 py-2 text-green-400">40-60%</td>
                  <td className="px-3 py-2 text-red-400">매우높음</td>
                  <td className="px-3 py-2">시스템</td>
                </tr>
              </tbody>
            </table>
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
              <span>CMOS는 PMOS와 NMOS의 상보적 구조로 저전력 디지털 회로를 구현합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>NAND/NOR 게이트는 모든 디지털 로직의 기본 빌딩 블록입니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>타이밍 분석은 전파 지연, 셋업/홀드 타임을 고려하여 최대 동작 주파수를 결정합니다</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 dark:text-blue-400 mr-2">▪</span>
              <span>클럭 게이팅, 멀티 Vdd 등의 기법으로 전력 소비를 최적화할 수 있습니다</span>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}
