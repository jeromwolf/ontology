'use client';

import { useState } from 'react';

export default function Chapter9() {
  const [selectedCandle, setSelectedCandle] = useState<'bullish' | 'bearish' | null>(null);

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">캔들차트 이해하기 🕯️</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          주식 차트의 기본인 캔들차트를 읽는 방법을 체계적으로 배워봅시다.
          캔들 하나하나가 전하는 시장의 이야기를 이해할 수 있게 됩니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 캔들차트란?</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            캔들차트는 일정 기간 동안의 시가, 고가, 저가, 종가를 하나의 봉(캔들) 모양으로 표현한 차트입니다.
            일본에서 개발되어 전 세계적으로 가장 널리 사용되는 차트 형태입니다.
          </p>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mt-4">
            <h3 className="font-semibold mb-3">캔들의 구성 요소</h3>
            <ul className="space-y-2 text-sm">
              <li><strong>몸통(Body):</strong> 시가와 종가 사이의 영역</li>
              <li><strong>위꼬리(Upper Shadow):</strong> 고가까지의 선</li>
              <li><strong>아래꼬리(Lower Shadow):</strong> 저가까지의 선</li>
              <li><strong>색상:</strong> 상승(빨강/흰색), 하락(파랑/검정)</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔴 양봉(상승 캔들) 이해하기</h2>
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-red-700 dark:text-red-300 mb-3">양봉의 특징</h3>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• 종가가 시가보다 높음</li>
                <li>• 매수세가 매도세보다 강했음을 의미</li>
                <li>• 빨간색 또는 흰색으로 표시</li>
                <li>• 몸통이 클수록 상승 강도가 강함</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">캔들 해석 예시</h4>
              <div className="space-y-2 text-sm">
                <p><strong>시가:</strong> 10,000원</p>
                <p><strong>고가:</strong> 10,500원</p>
                <p><strong>저가:</strong> 9,900원</p>
                <p><strong>종가:</strong> 10,400원</p>
                <p className="text-red-600 dark:text-red-400 font-semibold mt-2">
                  → 4% 상승, 강한 매수세 확인
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔵 음봉(하락 캔들) 이해하기</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-blue-700 dark:text-blue-300 mb-3">음봉의 특징</h3>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• 종가가 시가보다 낮음</li>
                <li>• 매도세가 매수세보다 강했음을 의미</li>
                <li>• 파란색 또는 검은색으로 표시</li>
                <li>• 몸통이 클수록 하락 강도가 강함</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">캔들 해석 예시</h4>
              <div className="space-y-2 text-sm">
                <p><strong>시가:</strong> 10,000원</p>
                <p><strong>고가:</strong> 10,100원</p>
                <p><strong>저가:</strong> 9,500원</p>
                <p><strong>종가:</strong> 9,600원</p>
                <p className="text-blue-600 dark:text-blue-400 font-semibold mt-2">
                  → 4% 하락, 강한 매도세 확인
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 주요 캔들 패턴</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-3">상승 신호 패턴</h3>
            <div className="space-y-3">
              <div>
                <h4 className="font-medium">망치형(Hammer)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  하락 추세 끝에 나타나는 긴 아래꼬리 캔들
                </p>
              </div>
              <div>
                <h4 className="font-medium">역망치형(Inverted Hammer)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  하락 추세 끝에 나타나는 긴 위꼬리 캔들
                </p>
              </div>
              <div>
                <h4 className="font-medium">상승 장악형</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  음봉 다음 큰 양봉이 음봉을 완전히 감싸는 패턴
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-red-600 dark:text-red-400 mb-3">하락 신호 패턴</h3>
            <div className="space-y-3">
              <div>
                <h4 className="font-medium">유성형(Shooting Star)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  상승 추세 끝에 나타나는 긴 위꼬리 캔들
                </p>
              </div>
              <div>
                <h4 className="font-medium">교수형(Hanging Man)</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  상승 추세 끝에 나타나는 긴 아래꼬리 캔들
                </p>
              </div>
              <div>
                <h4 className="font-medium">하락 장악형</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  양봉 다음 큰 음봉이 양봉을 완전히 감싸는 패턴
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📈 캔들과 거래량의 관계</h2>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">
            거래량으로 캔들의 신뢰도 판단하기
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-green-600 dark:text-green-400 mb-2">
                신뢰도 높은 캔들
              </h4>
              <ul className="space-y-1 text-sm">
                <li>✅ 큰 양봉 + 거래량 증가</li>
                <li>✅ 큰 음봉 + 거래량 증가</li>
                <li>✅ 패턴 형성 + 거래량 급증</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">
                신뢰도 낮은 캔들
              </h4>
              <ul className="space-y-1 text-sm">
                <li>❌ 큰 양봉 + 거래량 감소</li>
                <li>❌ 큰 음봉 + 거래량 감소</li>
                <li>❌ 패턴 형성 + 거래량 미미</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔍 인터랙티브 캔들 분석</h2>
        <div className="bg-gray-50 dark:bg-gray-900/20 rounded-lg p-6">
          <p className="mb-4">캔들을 클릭하여 세부 정보를 확인해보세요!</p>
          <div className="grid md:grid-cols-2 gap-4">
            <button
              onClick={() => setSelectedCandle('bullish')}
              className={`p-6 rounded-lg border-2 transition-all ${
                selectedCandle === 'bullish' 
                  ? 'border-red-500 bg-red-50 dark:bg-red-900/20' 
                  : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800'
              }`}
            >
              <div className="text-4xl mb-2">🟥</div>
              <h4 className="font-semibold">양봉 분석</h4>
            </button>
            
            <button
              onClick={() => setSelectedCandle('bearish')}
              className={`p-6 rounded-lg border-2 transition-all ${
                selectedCandle === 'bearish' 
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                  : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800'
              }`}
            >
              <div className="text-4xl mb-2">🟦</div>
              <h4 className="font-semibold">음봉 분석</h4>
            </button>
          </div>
          
          {selectedCandle && (
            <div className="mt-4 p-4 bg-white dark:bg-gray-800 rounded-lg">
              {selectedCandle === 'bullish' ? (
                <div>
                  <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">
                    양봉 상세 분석
                  </h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    • 시가보다 종가가 높아 매수세가 우세했음<br/>
                    • 긴 몸통은 강한 상승 의지를 나타냄<br/>
                    • 위꼬리가 짧으면 상승 압력이 지속될 가능성<br/>
                    • 거래량과 함께 보면 더 정확한 분석 가능
                  </p>
                </div>
              ) : (
                <div>
                  <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
                    음봉 상세 분석
                  </h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    • 시가보다 종가가 낮아 매도세가 우세했음<br/>
                    • 긴 몸통은 강한 하락 압력을 나타냄<br/>
                    • 아래꼬리가 길면 저가 매수세 유입 가능성<br/>
                    • 지지선 근처에서는 반등 가능성 주목
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 실전 활용 팁</h2>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-4">
            캔들차트 분석 시 주의사항
          </h3>
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-2">
              <span>📍</span>
              <span>단일 캔들보다는 연속된 캔들의 흐름을 파악하세요</span>
            </li>
            <li className="flex items-start gap-2">
              <span>📍</span>
              <span>캔들 패턴은 추세 전환점에서 더 의미가 있습니다</span>
            </li>
            <li className="flex items-start gap-2">
              <span>📍</span>
              <span>거래량과 함께 분석하면 신뢰도가 높아집니다</span>
            </li>
            <li className="flex items-start gap-2">
              <span>📍</span>
              <span>일봉, 주봉, 월봉 등 시간대별로 다르게 해석해야 합니다</span>
            </li>
            <li className="flex items-start gap-2">
              <span>📍</span>
              <span>캔들차트만으로 매매 결정을 하지 말고 다른 지표와 병행하세요</span>
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 핵심 정리</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">꼭 기억하세요!</h4>
              <ul className="space-y-1 text-sm">
                <li>• 캔들의 색깔로 상승/하락 구분</li>
                <li>• 몸통의 크기로 강도 파악</li>
                <li>• 꼬리의 길이로 변동성 확인</li>
                <li>• 패턴으로 추세 전환 예측</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">다음 단계</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                캔들차트의 기본을 이해했다면, 다음 챕터에서는 
                거래량 분석과 함께 더 정확한 매매 타이밍을 
                잡는 방법을 배워보겠습니다.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}