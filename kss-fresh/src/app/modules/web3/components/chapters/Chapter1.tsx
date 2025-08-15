'use client';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          블록체인의 탄생과 진화
        </h2>
        
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            2008년 사토시 나카모토의 비트코인 백서로 시작된 블록체인 혁명은
            단순한 디지털 화폐를 넘어 신뢰의 인터넷(Internet of Trust)을 구현하는
            핵심 기술로 발전했습니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔗 블록체인의 핵심 구조
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">블록 구조</h4>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-indigo-500 rounded-full mt-2"></div>
                <div>
                  <span className="font-semibold">블록 헤더</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    이전 블록 해시, 타임스탬프, 논스, 머클 루트
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-indigo-500 rounded-full mt-2"></div>
                <div>
                  <span className="font-semibold">트랜잭션 데이터</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    송신자, 수신자, 금액, 서명 정보
                  </p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6 border border-indigo-200 dark:border-indigo-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">체인 연결</h4>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                <div>
                  <span className="font-semibold">암호학적 연결</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    각 블록은 이전 블록의 해시를 포함
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                <div>
                  <span className="font-semibold">불변성 보장</span>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    과거 블록 수정 시 모든 후속 블록 재계산 필요
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          ⚡ 합의 메커니즘
        </h3>
        
        <div className="space-y-4 mb-8">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              Proof of Work (PoW)
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              컴퓨팅 파워를 사용해 복잡한 수학 문제를 해결하여 블록을 생성
            </p>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <code className="text-sm text-indigo-600 dark:text-indigo-400">
                {`// PoW 예시: 특정 난이도의 해시 찾기
while (hash.substring(0, difficulty) !== Array(difficulty + 1).join("0")) {
  nonce++;
  hash = sha256(blockData + nonce);
}`}
              </code>
            </div>
            <div className="mt-3 flex items-center gap-4 text-sm">
              <span className="text-green-600 dark:text-green-400">✓ 높은 보안성</span>
              <span className="text-red-600 dark:text-red-400">✗ 높은 에너지 소비</span>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              Proof of Stake (PoS)
            </h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              보유한 토큰 양과 기간에 비례하여 블록 생성 권한 획득
            </p>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <code className="text-sm text-indigo-600 dark:text-indigo-400">
                {`// PoS 검증자 선택
validator = selectValidator(stakingPool, {
  stake: validator.stakedAmount,
  age: validator.stakingDuration,
  randomSeed: currentBlock.hash
});`}
              </code>
            </div>
            <div className="mt-3 flex items-center gap-4 text-sm">
              <span className="text-green-600 dark:text-green-400">✓ 에너지 효율적</span>
              <span className="text-green-600 dark:text-green-400">✓ 빠른 처리</span>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          💰 토큰 이코노미
        </h3>
        
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6 mb-6">
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">유틸리티 토큰</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                플랫폼 내 서비스 이용, 거버넌스 참여, 스테이킹 보상
              </p>
            </div>
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">거버넌스 토큰</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                프로토콜 업그레이드 투표, 파라미터 조정, 자금 운용 결정
              </p>
            </div>
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">보상 메커니즘</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                블록 보상, 트랜잭션 수수료, 유동성 마이닝
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}