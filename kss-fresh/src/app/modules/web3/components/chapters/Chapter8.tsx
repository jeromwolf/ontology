'use client';

export default function Chapter8() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Web3 보안과 감사
        </h2>
        
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            스마트 컨트랙트는 불변성을 가지므로 배포 전 철저한 보안 감사가 필수입니다.
            수십억 달러의 해킹 사례들을 통해 보안의 중요성을 배웁니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          ⚠️ 주요 취약점과 공격 벡터
        </h3>
        
        <div className="space-y-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-red-600 dark:text-red-400 mb-3">
              Reentrancy Attack
            </h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                <h5 className="font-semibold text-sm mb-2">❌ 취약한 코드</h5>
                <pre className="text-xs text-gray-700 dark:text-gray-300">
{`function withdraw(uint amount) external {
    require(balances[msg.sender] >= amount);
    
    // 외부 호출 먼저 (위험!)
    msg.sender.call{value: amount}("");
    
    // 상태 변경 나중에
    balances[msg.sender] -= amount;
}`}
                </pre>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                <h5 className="font-semibold text-sm mb-2">✅ 안전한 코드</h5>
                <pre className="text-xs text-gray-700 dark:text-gray-300">
{`function withdraw(uint amount) external {
    require(balances[msg.sender] >= amount);
    
    // CEI 패턴: Check-Effects-Interactions
    balances[msg.sender] -= amount;
    
    (bool success, ) = msg.sender.call{value: amount}("");
    require(success, "Transfer failed");
}`}
                </pre>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-orange-600 dark:text-orange-400 mb-3">
              Flash Loan Attack
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-sm text-gray-700 dark:text-gray-300">
{`// Flash Loan 공격 시나리오
contract FlashLoanAttack {
    function executeAttack() external {
        // 1. Flash Loan으로 대량 자금 대출
        uint256 loanAmount = 1000000 * 10**18;
        flashLender.flashLoan(loanAmount);
    }
    
    function onFlashLoan(uint256 amount) external {
        // 2. 가격 조작
        manipulatePrice(amount);
        
        // 3. 차익 거래
        arbitrage();
        
        // 4. 대출 상환
        repayLoan(amount);
    }
}`}
              </pre>
            </div>
            <div className="mt-3 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <p className="text-sm text-gray-700 dark:text-gray-300">
                <strong>방어:</strong> 오라클 사용, 가격 평균화, 시간 지연
              </p>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🛡️ 보안 베스트 프랙티스
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">개발 단계</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>✓ 최신 Solidity 버전 사용</li>
              <li>✓ OpenZeppelin 라이브러리 활용</li>
              <li>✓ 단위 테스트 100% 커버리지</li>
              <li>✓ Fuzzing 테스트 수행</li>
              <li>✓ 형식 검증 (Formal Verification)</li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">배포 전</h4>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>✓ 내부 코드 리뷰</li>
              <li>✓ 외부 감사 (2개 이상)</li>
              <li>✓ Bug Bounty 프로그램</li>
              <li>✓ 테스트넷 장기 운영</li>
              <li>✓ 단계적 배포 전략</li>
            </ul>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔍 감사 도구와 방법론
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3">
            자동화 도구
          </h4>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-purple-600 dark:text-purple-400 mb-2">Slither</h5>
              <pre className="text-xs text-gray-600 dark:text-gray-400">
{`# 정적 분석
slither . --print human-summary
slither . --detect reentrancy-eth`}
              </pre>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">Mythril</h5>
              <pre className="text-xs text-gray-600 dark:text-gray-400">
{`# 심볼릭 실행
myth analyze Contract.sol
myth --execution-timeout 60`}
              </pre>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h5 className="font-semibold text-green-600 dark:text-green-400 mb-2">Echidna</h5>
              <pre className="text-xs text-gray-600 dark:text-gray-400">
{`# Fuzzing
echidna-test . --contract Test
echidna-test . --test-limit 50000`}
              </pre>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          💰 Bug Bounty 프로그램
        </h3>
        
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">보상 체계</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Critical</span>
                  <span className="font-bold text-red-600">$50,000 - $1,000,000</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">High</span>
                  <span className="font-bold text-orange-600">$10,000 - $50,000</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Medium</span>
                  <span className="font-bold text-yellow-600">$1,000 - $10,000</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Low</span>
                  <span className="font-bold text-green-600">$100 - $1,000</span>
                </div>
              </div>
            </div>
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">플랫폼</h4>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• Immunefi</li>
                <li>• HackerOne</li>
                <li>• Code4rena</li>
                <li>• Sherlock</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}