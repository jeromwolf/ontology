'use client';

export default function Chapter22() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">고급 차트 패턴과 하모닉 트레이딩</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          기본적인 차트 패턴을 넘어, 프로 트레이더들이 사용하는 고급 기법들을 배워봅시다.
          엘리어트 파동, 하모닉 패턴, 피보나치의 심화 활용법을 마스터하게 됩니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 엘리어트 파동 이론</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">파동의 기본 구조</h3>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 mb-4">
            <h4 className="font-semibold mb-3">충격파동 (Impulse Wave) - 5파 구조</h4>
            <div className="grid md:grid-cols-5 gap-3 mb-4">
              <div className="text-center p-3 bg-green-100 dark:bg-green-900/30 rounded">
                <div className="text-2xl font-bold text-green-600">1</div>
                <div className="text-sm">상승</div>
              </div>
              <div className="text-center p-3 bg-red-100 dark:bg-red-900/30 rounded">
                <div className="text-2xl font-bold text-red-600">2</div>
                <div className="text-sm">조정</div>
              </div>
              <div className="text-center p-3 bg-green-100 dark:bg-green-900/30 rounded">
                <div className="text-2xl font-bold text-green-600">3</div>
                <div className="text-sm">주상승</div>
              </div>
              <div className="text-center p-3 bg-red-100 dark:bg-red-900/30 rounded">
                <div className="text-2xl font-bold text-red-600">4</div>
                <div className="text-sm">조정</div>
              </div>
              <div className="text-center p-3 bg-green-100 dark:bg-green-900/30 rounded">
                <div className="text-2xl font-bold text-green-600">5</div>
                <div className="text-sm">마지막</div>
              </div>
            </div>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• 3파는 절대 가장 짧은 파동이 될 수 없음</li>
              <li>• 2파는 1파의 시작점 아래로 내려갈 수 없음</li>
              <li>• 4파는 1파의 고점과 겹칠 수 없음 (레버리지 제외)</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
            <h4 className="font-semibold mb-3">조정파동 (Corrective Wave) - ABC 구조</h4>
            <div className="grid md:grid-cols-3 gap-3 mb-4">
              <div className="text-center p-3 bg-red-100 dark:bg-red-900/30 rounded">
                <div className="text-2xl font-bold text-red-600">A</div>
                <div className="text-sm">초기 하락</div>
              </div>
              <div className="text-center p-3 bg-green-100 dark:bg-green-900/30 rounded">
                <div className="text-2xl font-bold text-green-600">B</div>
                <div className="text-sm">반등</div>
              </div>
              <div className="text-center p-3 bg-red-100 dark:bg-red-900/30 rounded">
                <div className="text-2xl font-bold text-red-600">C</div>
                <div className="text-sm">최종 하락</div>
              </div>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              지그재그, 플랫, 삼각형 등 다양한 조정 패턴이 존재
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🦋 하모닉 패턴 (Harmonic Patterns)</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">Gartley 패턴</h3>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-3">
              <div className="text-sm space-y-2">
                <p><strong>X → A:</strong> 초기 움직임</p>
                <p><strong>A → B:</strong> XA의 61.8% 되돌림</p>
                <p><strong>B → C:</strong> AB의 38.2-88.6% 되돌림</p>
                <p><strong>C → D:</strong> XA의 78.6% (PRZ)</p>
              </div>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              성공률: 약 70% | 리스크/리워드: 1:2 이상
            </p>
          </div>

          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-3">Butterfly 패턴</h3>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-3">
              <div className="text-sm space-y-2">
                <p><strong>X → A:</strong> 초기 움직임</p>
                <p><strong>A → B:</strong> XA의 78.6% 되돌림</p>
                <p><strong>B → C:</strong> AB의 38.2-88.6% 되돌림</p>
                <p><strong>C → D:</strong> XA의 127.2-161.8% 확장</p>
              </div>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              성공률: 약 65% | 더 큰 수익 잠재력
            </p>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3">Bat 패턴</h3>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-3">
              <div className="text-sm space-y-2">
                <p><strong>X → A:</strong> 초기 움직임</p>
                <p><strong>A → B:</strong> XA의 38.2-50% 되돌림</p>
                <p><strong>B → C:</strong> AB의 38.2-88.6% 되돌림</p>
                <p><strong>C → D:</strong> XA의 88.6% (PRZ)</p>
              </div>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              성공률: 약 75% | 가장 정확한 패턴
            </p>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-3">Crab 패턴</h3>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-3">
              <div className="text-sm space-y-2">
                <p><strong>X → A:</strong> 초기 움직임</p>
                <p><strong>A → B:</strong> XA의 38.2-61.8% 되돌림</p>
                <p><strong>B → C:</strong> AB의 38.2-88.6% 되돌림</p>
                <p><strong>C → D:</strong> XA의 161.8% 확장</p>
              </div>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              극단적 반전 지점 | 높은 리워드
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔢 피보나치 고급 활용법</h2>
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">피보나치 클러스터 (Confluence)</h3>
          
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">시간 피보나치</h4>
              <ul className="text-sm space-y-1">
                <li>• 주요 저점/고점에서 8, 13, 21, 34일</li>
                <li>• 시간대별 변곡점 예측</li>
                <li>• 사이클 분석과 결합</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">가격 피보나치</h4>
              <ul className="text-sm space-y-1">
                <li>• 다중 스윙의 피보나치 레벨 중첩</li>
                <li>• 확장 레벨: 127.2%, 161.8%, 261.8%</li>
                <li>• 내부 되돌림: 23.6%, 78.6%, 88.6%</li>
              </ul>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-2">피보나치 채널</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              추세선에 피보나치 비율을 적용하여 가격 채널 구성
            </p>
            <ul className="text-sm space-y-1">
              <li>• 0% - 기준 추세선</li>
              <li>• 61.8% - 1차 저항/지지</li>
              <li>• 100% - 2차 저항/지지</li>
              <li>• 161.8% - 극단적 레벨</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⏱️ 멀티 타임프레임 분석</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">3-Screen Trading System</h3>
          
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
                Screen 1: 주 추세 (주봉/일봉)
              </h4>
              <ul className="text-sm space-y-1">
                <li>• MACD 히스토그램으로 추세 방향 확인</li>
                <li>• 이동평균선 정렬 상태 점검</li>
                <li>• 주요 지지/저항 레벨 식별</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
                Screen 2: 중간 신호 (일봉/4시간봉)
              </h4>
              <ul className="text-sm space-y-1">
                <li>• 스토캐스틱/RSI로 진입 타이밍</li>
                <li>• 주 추세와 반대 방향 신호 필터링</li>
                <li>• Force Index로 매수/매도 압력 측정</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">
                Screen 3: 정확한 진입 (시간봉/분봉)
              </h4>
              <ul className="text-sm space-y-1">
                <li>• 브레이크아웃 또는 풀백 진입</li>
                <li>• 트레일링 스탑 설정</li>
                <li>• 목표가 및 손절가 미세 조정</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 실전 적용 팁</h2>
        <div className="bg-blue-100 dark:bg-blue-900/30 rounded-lg p-6">
          <h3 className="font-semibold mb-4">고급 패턴 트레이딩 체크리스트</h3>
          
          <div className="space-y-3">
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">패턴 완성도 확인</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  피보나치 비율이 ±5% 이내로 정확한가?
                </p>
              </div>
            </label>
            
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">다중 확인 (Confluence)</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  2개 이상의 기술적 요소가 같은 레벨을 지지하는가?
                </p>
              </div>
            </label>
            
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">시장 환경 적합성</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  현재 시장이 패턴 트레이딩에 적합한 변동성을 보이는가?
                </p>
              </div>
            </label>
            
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">리스크 관리 계획</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  PRZ(Potential Reversal Zone)에서의 손절 및 목표가 설정
                </p>
              </div>
            </label>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 주의사항</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-red-800 dark:text-red-200 mb-3">
            고급 기법의 함정
          </h3>
          <ul className="space-y-2 text-sm">
            <li className="flex items-start gap-2">
              <span className="text-red-600">⚠️</span>
              <div>
                <strong>과도한 복잡성:</strong> 단순한 것이 더 효과적일 때가 많음
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-600">⚠️</span>
              <div>
                <strong>확증 편향:</strong> 원하는 패턴을 억지로 찾으려 하지 말 것
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-600">⚠️</span>
              <div>
                <strong>백테스팅 부족:</strong> 실전 적용 전 충분한 검증 필수
              </div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-600">⚠️</span>
              <div>
                <strong>시장 상황 무시:</strong> 모든 패턴이 모든 시장에서 작동하지 않음
              </div>
            </li>
          </ul>
        </div>
      </section>
    </div>
  )
}