'use client';

export default function Chapter19() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">산업 분석과 기업 비교</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          같은 업종 내에서도 기업별 성과는 천차만별입니다. 
          산업의 특성을 이해하고 기업을 비교 분석하는 방법을 배워봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 산업(섹터) 이해하기</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3">주요 산업 분류</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">1. IT/기술</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 반도체: 삼성전자, SK하이닉스</li>
                <li>• 소프트웨어: 카카오, 네이버</li>
                <li>• 게임: 엔씨소프트, 넷마블</li>
                <li>• 특징: 높은 성장성, 변동성도 큼</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">2. 제조업</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 자동차: 현대차, 기아</li>
                <li>• 화학: LG화학, 롯데케미칼</li>
                <li>• 철강: 포스코, 현대제철</li>
                <li>• 특징: 경기 민감, 설비투자 중요</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">3. 소비재</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 식음료: CJ제일제당, 오리온</li>
                <li>• 화장품: 아모레퍼시픽, LG생활건강</li>
                <li>• 유통: 이마트, 롯데쇼핑</li>
                <li>• 특징: 안정적 수익, 브랜드 가치 중요</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">4. 금융</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• 은행: KB금융, 신한지주</li>
                <li>• 증권: 미래에셋증권, 한국투자증권</li>
                <li>• 보험: 삼성생명, 한화생명</li>
                <li>• 특징: 금리 영향 큼, 배당 매력</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔍 산업 분석의 핵심 포인트</h2>
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">1. 산업 생명주기 파악</h3>
            <div className="grid md:grid-cols-4 gap-3">
              <div className="bg-white dark:bg-gray-800 rounded p-3 text-center">
                <div className="text-2xl mb-1">🌱</div>
                <div className="font-semibold">도입기</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  신기술/신사업<br/>높은 위험·수익
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-3 text-center">
                <div className="text-2xl mb-1">🚀</div>
                <div className="font-semibold">성장기</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  매출 급증<br/>투자 기회
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-3 text-center">
                <div className="text-2xl mb-1">🏢</div>
                <div className="font-semibold">성숙기</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  안정적 수익<br/>배당 중심
                </div>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-3 text-center">
                <div className="text-2xl mb-1">📉</div>
                <div className="font-semibold">쇠퇴기</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  시장 축소<br/>구조조정
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">2. 진입장벽 분석</h3>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-green-600">✓</span>
                <div>
                  <span className="font-semibold">기술 장벽:</span> 특허, 기술력이 필요한가?
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600">✓</span>
                <div>
                  <span className="font-semibold">자본 장벽:</span> 초기 투자 규모가 큰가?
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600">✓</span>
                <div>
                  <span className="font-semibold">규제 장벽:</span> 인허가나 규제가 있는가?
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600">✓</span>
                <div>
                  <span className="font-semibold">브랜드 장벽:</span> 브랜드 파워가 중요한가?
                </div>
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚖️ 동종업계 기업 비교 방법</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">실전 예시: 대형 마트 3사 비교</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-200 dark:bg-gray-700">
                <tr>
                  <th className="px-4 py-2 text-left">구분</th>
                  <th className="px-4 py-2 text-center">이마트</th>
                  <th className="px-4 py-2 text-center">홈플러스</th>
                  <th className="px-4 py-2 text-center">롯데마트</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-300 dark:divide-gray-600">
                <tr>
                  <td className="px-4 py-2 font-semibold">시장점유율</td>
                  <td className="px-4 py-2 text-center">1위 (35%)</td>
                  <td className="px-4 py-2 text-center">2위 (25%)</td>
                  <td className="px-4 py-2 text-center">3위 (20%)</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 font-semibold">강점</td>
                  <td className="px-4 py-2 text-center">점포 수, PB상품</td>
                  <td className="px-4 py-2 text-center">온라인 강화</td>
                  <td className="px-4 py-2 text-center">그룹 시너지</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 font-semibold">영업이익률</td>
                  <td className="px-4 py-2 text-center">3.5%</td>
                  <td className="px-4 py-2 text-center">2.8%</td>
                  <td className="px-4 py-2 text-center">2.2%</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 font-semibold">PER</td>
                  <td className="px-4 py-2 text-center">15배</td>
                  <td className="px-4 py-2 text-center">18배</td>
                  <td className="px-4 py-2 text-center">22배</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
            💡 같은 업종이라도 시장 지위, 수익성, 밸류에이션이 다름을 확인!
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 산업 트렌드 읽기</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-3">성장 산업 신호</h3>
            <ul className="space-y-2 text-sm">
              <li>🔸 정부 정책 지원 (그린뉴딜, K-반도체 등)</li>
              <li>🔸 글로벌 메가트렌드 (AI, 전기차, ESG)</li>
              <li>🔸 소비 패턴 변화 (언택트, 홈코노미)</li>
              <li>🔸 기술 혁신 (5G, 메타버스, 블록체인)</li>
            </ul>
          </div>
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-3">쇠퇴 산업 신호</h3>
            <ul className="space-y-2 text-sm">
              <li>🔻 대체재 등장 (전통 미디어 → OTT)</li>
              <li>🔻 규제 강화 (환경 규제, 담배 규제)</li>
              <li>🔻 수요 감소 (저출산 → 유아용품)</li>
              <li>🔻 기술 변화 (내연기관 → 전기차)</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 실전 체크리스트</h2>
        <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">산업 분석 시 꼭 확인할 사항</h3>
          <div className="space-y-3">
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>산업의 성장률이 GDP 성장률보다 높은가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>업계 1~3위 기업의 시장점유율 합이 50% 이상인가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>신규 진입자의 위협이 낮은가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>정부 정책이 우호적인가?</span>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <span>글로벌 경쟁력을 갖추고 있는가?</span>
            </label>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 투자 인사이트</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <span className="text-2xl">1️⃣</span>
              <div>
                <h4 className="font-semibold mb-1">산업의 성장성 > 기업의 성장성</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  아무리 좋은 기업도 쇠퇴하는 산업에서는 한계가 있습니다.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">2️⃣</span>
              <div>
                <h4 className="font-semibold mb-1">1등 기업이 항상 좋은 투자처는 아님</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  때로는 2~3등 기업이 더 높은 성장 잠재력을 가질 수 있습니다.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">3️⃣</span>
              <div>
                <h4 className="font-semibold mb-1">산업 싸이클을 활용하라</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  경기 민감주는 불황기에 매수, 호황기에 매도가 기본입니다.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}