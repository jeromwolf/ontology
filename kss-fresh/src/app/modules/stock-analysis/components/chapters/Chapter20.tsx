'use client';

import { useState } from 'react';

export default function Chapter20() {
  const [portfolio, setPortfolio] = useState({
    cash: 1000000,
    stocks: []
  });

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">실전 매매 시뮬레이션</h1>
        <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
          지금까지 배운 모든 지식을 종합하여 실제 매매를 연습해봅시다. 
          모의투자를 통해 실전 감각을 익히고 나만의 매매 스타일을 찾아보세요.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📋 매매 전 체크리스트</h2>
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-4">매수 전 반드시 확인!</h3>
          <div className="space-y-3">
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">1. 매수 이유가 명확한가?</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  "남들이 사니까"가 아닌 나만의 투자 논리가 있어야 합니다.
                </p>
              </div>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">2. 목표 수익률과 손절선을 정했는가?</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  예: +10% 수익 시 매도, -5% 손실 시 손절
                </p>
              </div>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">3. 투자 금액이 적절한가?</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  한 종목에 전체 자산의 20%를 넘지 않도록 합니다.
                </p>
              </div>
            </label>
            <label className="flex items-start gap-3">
              <input type="checkbox" className="mt-1" />
              <div>
                <span className="font-semibold">4. 시장 상황을 확인했는가?</span>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  전체 시장 흐름과 해당 섹터 동향을 파악합니다.
                </p>
              </div>
            </label>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">💰 모의 포트폴리오 구성</h2>
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">초기 자금: 100만원으로 시작</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3">추천 포트폴리오 구성</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center p-3 bg-white dark:bg-gray-800 rounded">
                  <span>안정형 대형주</span>
                  <span className="font-semibold">40%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-white dark:bg-gray-800 rounded">
                  <span>성장형 중형주</span>
                  <span className="font-semibold">30%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-white dark:bg-gray-800 rounded">
                  <span>배당주</span>
                  <span className="font-semibold">20%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-white dark:bg-gray-800 rounded">
                  <span>현금 보유</span>
                  <span className="font-semibold">10%</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold mb-3">섹터별 분산</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center p-3 bg-white dark:bg-gray-800 rounded">
                  <span>IT/기술</span>
                  <span className="font-semibold">30%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-white dark:bg-gray-800 rounded">
                  <span>금융</span>
                  <span className="font-semibold">25%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-white dark:bg-gray-800 rounded">
                  <span>소비재</span>
                  <span className="font-semibold">25%</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-white dark:bg-gray-800 rounded">
                  <span>바이오/헬스케어</span>
                  <span className="font-semibold">20%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📊 실전 매매 시나리오</h2>
        <div className="space-y-4">
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-3">
              시나리오 1: 실적 발표 대응
            </h3>
            <div className="space-y-3 text-sm">
              <p>상황: 보유한 A기업의 분기 실적 발표일이 다가옴</p>
              <div className="pl-4 space-y-2">
                <p>✅ 사전 준비: 컨센서스 확인, 전분기 실적 검토</p>
                <p>✅ 시나리오별 대응:</p>
                <ul className="pl-4 space-y-1">
                  <li>• 어닝 서프라이즈 → 일부 익절 후 홀딩</li>
                  <li>• 컨센서스 부합 → 현상 유지</li>
                  <li>• 어닝 쇼크 → 손절 또는 추가 매수 판단</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3">
              시나리오 2: 급락장 대응
            </h3>
            <div className="space-y-3 text-sm">
              <p>상황: 코스피가 하루에 -3% 이상 급락</p>
              <div className="pl-4 space-y-2">
                <p>✅ 침착하게 상황 파악: 급락 원인 분석</p>
                <p>✅ 포트폴리오 점검:</p>
                <ul className="pl-4 space-y-1">
                  <li>• 우량주 추가 매수 기회로 활용</li>
                  <li>• 약세 종목은 반등 시 정리</li>
                  <li>• 현금 비중 조절 (10% → 20%)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">📝 매매일지 작성법</h2>
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="font-semibold mb-4">매매일지 예시</h3>
          <div className="bg-white dark:bg-gray-700 rounded p-4 font-mono text-sm">
            <p className="text-blue-600 dark:text-blue-400">## 2024년 1월 15일 - 삼성전자 매수</p>
            <p className="mt-2">📍 매수 정보</p>
            <p>- 종목: 삼성전자</p>
            <p>- 매수가: 75,000원</p>
            <p>- 수량: 10주</p>
            <p>- 총 매수금액: 750,000원</p>
            <p className="mt-3">📍 매수 이유</p>
            <p>1. 반도체 업황 회복 기대</p>
            <p>2. PER 15배로 역사적 저점</p>
            <p>3. 이동평균선 골든크로스</p>
            <p className="mt-3">📍 목표 및 손절</p>
            <p>- 목표가: 82,500원 (+10%)</p>
            <p>- 손절가: 71,250원 (-5%)</p>
            <p>- 보유 기간: 3개월 예상</p>
            <p className="mt-3">📍 시장 상황</p>
            <p>- KOSPI: 2,500 (전일 대비 +0.5%)</p>
            <p>- 반도체 섹터: 강세</p>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
            💡 매매일지는 나중에 복기할 때 가장 중요한 학습 자료가 됩니다!
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">⚠️ 초보자가 자주 하는 실수</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-red-800 dark:text-red-200 mb-2">1. 손절을 못함</h4>
            <p className="text-sm">
              "조금만 기다리면 오르겠지"라는 희망적 사고는 금물. 
              정해진 손절선은 반드시 지켜야 합니다.
            </p>
          </div>
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-red-800 dark:text-red-200 mb-2">2. 추격 매수</h4>
            <p className="text-sm">
              급등한 종목을 뒤늦게 따라 사는 것은 위험합니다. 
              차분히 다음 기회를 기다리세요.
            </p>
          </div>
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-red-800 dark:text-red-200 mb-2">3. 물타기의 함정</h4>
            <p className="text-sm">
              하락하는 종목에 계속 추가 매수하는 것은 위험합니다. 
              왜 떨어지는지 먼저 분석하세요.
            </p>
          </div>
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
            <h4 className="font-semibold text-red-800 dark:text-red-200 mb-2">4. 뇌동매매</h4>
            <p className="text-sm">
              남의 추천이나 소문에 의존하지 마세요. 
              스스로 분석하고 판단하는 습관을 기르세요.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🎯 실전 연습 과제</h2>
        <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-lg p-6">
          <h3 className="font-semibold mb-4">4주간 모의투자 챌린지</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm">1</span>
              <div>
                <p className="font-semibold">1주차: 관찰과 분석</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  5개 종목을 선정하여 매일 가격 변동과 뉴스를 기록
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm">2</span>
              <div>
                <p className="font-semibold">2주차: 첫 매수</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  분석한 종목 중 2개를 소액으로 매수하고 일지 작성
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm">3</span>
              <div>
                <p className="font-semibold">3주차: 포트폴리오 구성</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  3-5개 종목으로 분산 투자하고 비중 조절 연습
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="bg-indigo-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm">4</span>
              <div>
                <p className="font-semibold">4주차: 매도와 평가</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  목표가 도달 시 매도, 전체 수익률 계산 및 복기
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}