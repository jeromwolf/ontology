'use client'

import { Coins } from 'lucide-react'

export default function Chapter3() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          DeFi (탈중앙화 금융)의 세계
        </h2>
        
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            DeFi는 전통 금융 서비스를 블록체인 위에 구현한 혁신적인 금융 시스템입니다.
            중개자 없이 대출, 거래, 투자가 가능한 개방형 금융 인프라를 제공합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          💵 스테이블코인 (Stablecoins)
        </h3>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 mb-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3">
            스테이블코인 유형과 메커니즘
          </h4>
          
          <div className="grid md:grid-cols-3 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
                법정화폐 담보형
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                1:1 USD 담보 보유
              </p>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span>USDT (Tether)</span>
                  <span className="text-gray-500">$83B</span>
                </div>
                <div className="flex justify-between">
                  <span>USDC (Circle)</span>
                  <span className="text-gray-500">$32B</span>
                </div>
                <div className="flex justify-between">
                  <span>BUSD (Binance)</span>
                  <span className="text-gray-500">$5B</span>
                </div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-green-600 dark:text-green-400 mb-2">
                암호화폐 담보형
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                초과 담보 (150%+)
              </p>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span>DAI (MakerDAO)</span>
                  <span className="text-gray-500">$5B</span>
                </div>
                <div className="flex justify-between">
                  <span>LUSD (Liquity)</span>
                  <span className="text-gray-500">$300M</span>
                </div>
                <div className="flex justify-between">
                  <span>sUSD (Synthetix)</span>
                  <span className="text-gray-500">$100M</span>
                </div>
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">
                알고리즘형
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                수요-공급 조절
              </p>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span>FRAX (부분담보)</span>
                  <span className="text-gray-500">$1B</span>
                </div>
                <div className="flex justify-between">
                  <span>UST (실패 사례)</span>
                  <span className="text-red-500">Collapsed</span>
                </div>
                <div className="flex justify-between">
                  <span>USDD (TRON)</span>
                  <span className="text-gray-500">$700M</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
            <h5 className="font-semibold text-gray-900 dark:text-white mb-2">
              🔍 DAI 생성 메커니즘 (CDP/Vault)
            </h5>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3 mb-3">
              <pre className="text-xs text-gray-700 dark:text-gray-300">
{`// MakerDAO DAI 생성 프로세스
1. ETH를 Vault에 예치 (예: $1500 상당)
2. 최대 66% DAI 대출 가능 (1000 DAI)
3. Stability Fee 지불 (연 5-10%)
4. 청산 비율: 150% (ETH 가격 하락 시 위험)
5. DAI 상환 후 ETH 회수`}
              </pre>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              ⚠️ 담보 가치가 청산 비율 이하로 떨어지면 13% 페널티와 함께 청산
            </p>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          💱 AMM (Automated Market Maker)
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3">
            Constant Product Formula: x * y = k
          </h4>
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-700 dark:text-gray-300">
{`// Uniswap V2 스왑 로직
function swap(uint amountIn, address tokenIn) returns (uint amountOut) {
    uint reserveIn = getReserve(tokenIn);
    uint reserveOut = getReserve(tokenOut);
    
    // 0.3% 수수료 적용
    uint amountInWithFee = amountIn * 997;
    uint numerator = amountInWithFee * reserveOut;
    uint denominator = (reserveIn * 1000) + amountInWithFee;
    
    amountOut = numerator / denominator;
    
    // 실제 스왑 실행
    executeSwap(amountIn, amountOut);
}`}
            </pre>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 dark:text-white mb-2">장점</h5>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 24/7 거래 가능</li>
                <li>• 무허가형 접근</li>
                <li>• 즉각적인 유동성</li>
              </ul>
            </div>
            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-gray-900 dark:text-white mb-2">위험</h5>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• Impermanent Loss</li>
                <li>• 슬리피지</li>
                <li>• 프론트러닝</li>
              </ul>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🏦 Lending & Borrowing
        </h3>
        
        <div className="space-y-4 mb-8">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              Compound/Aave 모델
            </h4>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h5 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
                  대출자 (Lender)
                </h5>
                <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                  <li>• 자산 예치 → cToken/aToken 수령</li>
                  <li>• 실시간 이자 수익</li>
                  <li>• 언제든 인출 가능</li>
                </ul>
              </div>
              <div>
                <h5 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
                  차입자 (Borrower)
                </h5>
                <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                  <li>• 담보 제공 (초과 담보)</li>
                  <li>• 변동 금리로 차입</li>
                  <li>• 청산 위험 관리</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              ⚠️ 청산 메커니즘
            </h4>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <pre className="text-sm text-gray-700 dark:text-gray-300">
{`Health Factor = (담보 가치 * LTV) / 대출 가치

if (healthFactor < 1) {
    // 청산 트리거
    liquidate(borrower, collateral, debt);
    // 청산자에게 보너스 지급 (일반적으로 5-10%)
}`}
              </pre>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🌾 Yield Farming & 스테이킹
        </h3>
        
        <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6">
          <div className="space-y-4">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">
                유동성 마이닝 전략
              </h4>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                  <h5 className="font-semibold text-sm mb-1">단일 자산</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    USDC → Compound<br/>
                    APY: 3-5%
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                  <h5 className="font-semibold text-sm mb-1">LP 토큰</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    ETH-USDC LP → Farm<br/>
                    APY: 10-30%
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                  <h5 className="font-semibold text-sm mb-1">레버리지</h5>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    순환 대출 전략<br/>
                    APY: 20-50%
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}