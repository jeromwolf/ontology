'use client';

import { useState } from 'react';

export default function Chapter5() {
  const [riskAmount, setRiskAmount] = useState(100000);
  const [stopLoss, setStopLoss] = useState(5);
  
  const calculateRisk = () => {
    const lossAmount = riskAmount * (stopLoss / 100);
    const remaining = riskAmount - lossAmount;
    return { lossAmount, remaining };
  };

  const { lossAmount, remaining } = calculateRisk();

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6 text-center">리스크 관리와 손절매 🛡️</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6 text-center">
          성공적인 투자의 핵심은 수익을 내는 것이 아니라 손실을 제한하는 것입니다.
          체계적인 리스크 관리로 장기적인 투자 성공을 이루어봅시다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 리스크 관리의 황금률</h2>
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-red-600 dark:text-red-400 mb-2">2% 룰</h3>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                한 종목에 전체 자산의 2% 이상 위험 노출 금지
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-orange-600 dark:text-orange-400 mb-2">6% 룰</h3>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                월 최대 손실을 전체 자산의 6%로 제한
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h3 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">손익비 1:2</h3>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                예상 수익이 위험의 2배 이상인 거래만 실행
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💸 손절매 계산기</h2>
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">투자 금액</label>
              <input
                type="number"
                value={riskAmount}
                onChange={(e) => setRiskAmount(Number(e.target.value))}
                className="w-full px-4 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-700"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">손절매 비율 (%)</label>
              <input
                type="range"
                min="1"
                max="20"
                value={stopLoss}
                onChange={(e) => setStopLoss(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                <span>1%</span>
                <span className="font-bold text-lg">{stopLoss}%</span>
                <span>20%</span>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mt-4">
              <div className="grid grid-cols-2 gap-4 text-center">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">예상 손실액</p>
                  <p className="text-2xl font-bold text-red-600">
                    -{lossAmount.toLocaleString()}원
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">남은 자금</p>
                  <p className="text-2xl font-bold text-green-600">
                    {remaining.toLocaleString()}원
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🔍 손절매 설정 방법</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">기술적 분석 기반</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <div>
                  <strong>지지선 하단:</strong> 주요 지지선 아래 2-3% 설정
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <div>
                  <strong>이동평균선:</strong> 20일선 이탈 시 손절
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <div>
                  <strong>전일 저가:</strong> 단기 트레이딩 시 활용
                </div>
              </li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">비율 기반</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <span className="text-purple-500">•</span>
                <div>
                  <strong>고정 비율:</strong> 매수가 대비 -5% 또는 -10%
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-500">•</span>
                <div>
                  <strong>변동성 조정:</strong> ATR × 2배 거리
                </div>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-500">•</span>
                <div>
                  <strong>시간 손절:</strong> 3일 내 무반응 시 청산
                </div>
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 포지션 사이징</h2>
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-4">
            켈리 공식 (Kelly Criterion)
          </h3>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <p className="text-center text-lg font-mono font-bold mb-2">
              f = (p × b - q) / b
            </p>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <p>• f = 투자 비율</p>
              <p>• p = 승률</p>
              <p>• b = 손익비</p>
              <p>• q = 패율 (1-p)</p>
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">예시 계산</h4>
              <p className="text-sm">승률 60%, 손익비 1.5:1</p>
              <p className="text-sm">f = (0.6 × 1.5 - 0.4) / 1.5 = 0.33</p>
              <p className="text-sm font-semibold mt-2">→ 자산의 33% 투자</p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-semibold mb-2">실전 적용</h4>
              <p className="text-sm">• 켈리 값의 25% 사용 권장</p>
              <p className="text-sm">• 최대 투자 비율 20% 제한</p>
              <p className="text-sm">• 분산투자로 리스크 분산</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 심리적 함정과 극복</h2>
        <div className="space-y-4">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-700 dark:text-red-300 mb-3">
              손절매를 못하는 이유
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">심리적 요인</h4>
                <ul className="space-y-1 text-sm">
                  <li>• 손실 회피 성향</li>
                  <li>• 자존심과 고집</li>
                  <li>• 희망적 사고</li>
                  <li>• 매몰비용 오류</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2">극복 방법</h4>
                <ul className="space-y-1 text-sm">
                  <li>• 매수 전 손절가 설정</li>
                  <li>• 시스템 트레이딩 활용</li>
                  <li>• 트레이딩 일지 작성</li>
                  <li>• 멘탈 관리 훈련</li>
                </ul>
              </div>
            </div>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-700 dark:text-blue-300 mb-3">
              트레일링 스톱 (Trailing Stop)
            </h3>
            <p className="text-sm mb-3">
              주가 상승에 따라 손절가를 함께 올려 수익을 보호하는 기법
            </p>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>매수가:</span>
                  <span className="font-medium">100,000원</span>
                </div>
                <div className="flex justify-between">
                  <span>초기 손절가:</span>
                  <span className="font-medium">95,000원 (-5%)</span>
                </div>
                <div className="flex justify-between text-green-600">
                  <span>현재가 상승:</span>
                  <span className="font-medium">110,000원 (+10%)</span>
                </div>
                <div className="flex justify-between text-blue-600">
                  <span>조정된 손절가:</span>
                  <span className="font-medium">104,500원 (현재가 -5%)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💡 리스크 관리 체크리스트</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">매매 전 확인사항</h3>
          <div className="space-y-3">
            <label className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>손절가를 명확히 설정했는가?</span>
            </label>
            <label className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>전체 자산의 2% 이내로 리스크를 제한했는가?</span>
            </label>
            <label className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>손익비가 최소 1:2 이상인가?</span>
            </label>
            <label className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>감정적 판단이 아닌 객관적 분석에 기반했는가?</span>
            </label>
            <label className="flex items-center gap-3">
              <input type="checkbox" className="w-5 h-5" />
              <span>최악의 시나리오를 고려했는가?</span>
            </label>
          </div>
        </div>
      </section>
    </div>
  )
}