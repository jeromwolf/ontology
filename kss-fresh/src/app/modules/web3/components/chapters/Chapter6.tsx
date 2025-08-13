'use client'

import { Layers, Shield, Zap, Users } from 'lucide-react'

export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Layer 2와 확장성 솔루션
        </h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            블록체인의 확장성 트릴레마를 해결하기 위한 Layer 2 솔루션들은
            메인체인의 보안성을 유지하면서 처리량과 속도를 대폭 개선합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔄 Rollup 기술
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              Optimistic Rollups
            </h4>
            <div className="space-y-3">
              <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-3">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>원리:</strong> 트랜잭션을 일단 유효하다고 가정
                </p>
              </div>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>• Fraud Proof 기반 검증</li>
                <li>• 7일 챌린지 기간</li>
                <li>• EVM 호환성 우수</li>
                <li>• Arbitrum, Optimism</li>
              </ul>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              ZK Rollups
            </h4>
            <div className="space-y-3">
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  <strong>원리:</strong> 영지식 증명으로 유효성 검증
                </p>
              </div>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>• Validity Proof 생성</li>
                <li>• 즉시 최종성</li>
                <li>• 높은 보안성</li>
                <li>• zkSync, StarkNet</li>
              </ul>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🌉 브릿지와 크로스체인
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3">
            브릿지 메커니즘
          </h4>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 mb-4">
            <pre className="text-sm text-gray-700 dark:text-gray-300">
{`// Lock & Mint 방식
contract Bridge {
    // L1 → L2
    function deposit(uint256 amount) external {
        // L1에서 토큰 잠금
        token.transferFrom(msg.sender, address(this), amount);
        
        // L2에 메시지 전송
        messenger.sendMessage(
            l2Bridge,
            abi.encode(msg.sender, amount)
        );
    }
    
    // L2 → L1
    function withdraw(uint256 amount) external {
        // L2에서 토큰 소각
        token.burn(msg.sender, amount);
        
        // L1 인출 요청
        initiateWithdrawal(msg.sender, amount);
    }
}`}
            </pre>
          </div>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
              <h5 className="font-semibold text-sm mb-1">Native Bridge</h5>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                공식 브릿지, 높은 보안, 느린 속도
              </p>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
              <h5 className="font-semibold text-sm mb-1">Third-party</h5>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                빠른 속도, 유동성 풀, 추가 위험
              </p>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
              <h5 className="font-semibold text-sm mb-1">Atomic Swap</h5>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                P2P 교환, 신뢰 불필요, 제한적
              </p>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          ⚖️ 확장성 트릴레마
        </h3>
        
        <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="w-16 h-16 bg-red-500 text-white rounded-full flex items-center justify-center mx-auto mb-2">
                <Shield className="w-8 h-8" />
              </div>
              <h4 className="font-bold text-gray-900 dark:text-white">보안성</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                네트워크 공격 저항력
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-500 text-white rounded-full flex items-center justify-center mx-auto mb-2">
                <Zap className="w-8 h-8" />
              </div>
              <h4 className="font-bold text-gray-900 dark:text-white">확장성</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                처리량과 속도
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-green-500 text-white rounded-full flex items-center justify-center mx-auto mb-2">
                <Users className="w-8 h-8" />
              </div>
              <h4 className="font-bold text-gray-900 dark:text-white">탈중앙화</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                노드 분산과 접근성
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}