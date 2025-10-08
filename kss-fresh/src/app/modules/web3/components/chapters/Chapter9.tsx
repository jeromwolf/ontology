'use client';

import { FileText, Cpu, Globe } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter9() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          블록체인 백서와 철학
        </h2>

        <div className="bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            블록체인 혁명은 두 개의 역사적 백서에서 시작되었습니다.
            사토시 나카모토의 비트코인 백서(2008)와 비탈릭 부테린의 이더리움 백서(2013)는
            탈중앙화 시스템의 철학과 기술적 기반을 제시했습니다.
          </p>
        </div>

        {/* 비트코인 백서 */}
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center gap-2">
          <FileText className="w-6 h-6 text-orange-600" />
          비트코인 백서 (2008)
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-2 border-orange-200 dark:border-orange-800 mb-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3 text-lg">
            "Bitcoin: A Peer-to-Peer Electronic Cash System"
          </h4>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 italic">
            - Satoshi Nakamoto, October 31, 2008 (9 pages)
          </p>

          <div className="space-y-4">
            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-red-800 dark:text-red-300 mb-2">
                🎯 핵심 문제: 이중 지불 (Double-Spending Problem)
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                디지털 화폐의 가장 큰 문제는 같은 돈을 두 번 쓸 수 있다는 것입니다.
                기존 해결책은 신뢰할 수 있는 제3자(은행)를 거치는 것이었습니다.
              </p>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <pre className="text-xs text-gray-700 dark:text-gray-300">
{`기존 방식:
Alice → Bank → Bob (은행이 중개)

비트코인 방식:
Alice → Blockchain → Bob (네트워크가 검증)`}
                </pre>
              </div>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-orange-800 dark:text-orange-300 mb-2">
                🔗 해결책: Proof of Work + Longest Chain Rule
              </h5>
              <div className="space-y-3">
                <div>
                  <h6 className="font-semibold text-sm mb-1">1. 작업 증명 (Proof of Work)</h6>
                  <p className="text-xs text-gray-700 dark:text-gray-300 mb-2">
                    채굴자들이 복잡한 수학 문제를 해결하여 블록 생성 권한을 획득
                  </p>
                  <div className="bg-white dark:bg-gray-800 rounded p-2">
                    <code className="text-xs text-indigo-600 dark:text-indigo-400">
{`SHA256(SHA256(Block Header)) < Target
// 목표값보다 작은 해시를 찾을 때까지 nonce 변경
// 평균 10분마다 1개 블록 생성`}
                    </code>
                  </div>
                </div>

                <div>
                  <h6 className="font-semibold text-sm mb-1">2. 최장 체인 규칙</h6>
                  <p className="text-xs text-gray-700 dark:text-gray-300">
                    가장 많은 계산량이 투입된 체인이 정당한 체인입니다.
                    악의적인 공격자가 과거 거래를 수정하려면 전체 네트워크의 51% 이상 해시파워 필요
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-green-800 dark:text-green-300 mb-2">
                🌳 머클 트리 (Merkle Tree)
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                수천 개의 트랜잭션을 하나의 해시로 요약하여 효율적인 검증이 가능합니다.
              </p>
              <div className="bg-white dark:bg-gray-800 rounded p-3 font-mono text-xs">
                <pre className="text-gray-700 dark:text-gray-300">
{`                Root Hash
                    |
        ┌───────────┴───────────┐
       Hash A               Hash B
        |                    |
    ┌───┴───┐            ┌───┴───┐
  Tx1     Tx2          Tx3     Tx4

// SPV (Simple Payment Verification)
// 전체 블록체인 다운로드 없이 거래 검증 가능`}
                </pre>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">
                💰 인센티브 메커니즘
              </h5>
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                <div>
                  <h6 className="font-semibold mb-1">블록 보상 (Block Reward)</h6>
                  <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 2009-2012: 50 BTC</li>
                    <li>• 2012-2016: 25 BTC</li>
                    <li>• 2016-2020: 12.5 BTC</li>
                    <li>• 2020-2024: 6.25 BTC</li>
                    <li>• 2024-2028: 3.125 BTC</li>
                    <li>• 총 공급량: 21,000,000 BTC</li>
                  </ul>
                </div>
                <div>
                  <h6 className="font-semibold mb-1">트랜잭션 수수료</h6>
                  <p className="text-xs text-gray-700 dark:text-gray-300">
                    블록 보상이 줄어들면 수수료가 주 수입원이 됩니다.
                    사용자가 원하는 수수료를 설정하여 우선순위 결정
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 이더리움 백서 */}
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center gap-2">
          <Cpu className="w-6 h-6 text-blue-600" />
          이더리움 백서 (2013)
        </h3>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border-2 border-blue-200 dark:border-blue-800 mb-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3 text-lg">
            "A Next-Generation Smart Contract and Decentralized Application Platform"
          </h4>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 italic">
            - Vitalik Buterin, November 2013
          </p>

          <div className="space-y-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
                🌍 비전: World Computer (세계 컴퓨터)
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                블록체인을 단순한 화폐 시스템에서 범용 계산 플랫폼으로 확장합니다.
                누구나 탈중앙화된 애플리케이션(DApp)을 개발하고 배포할 수 있습니다.
              </p>
              <div className="grid md:grid-cols-2 gap-3 text-xs">
                <div className="bg-white dark:bg-gray-800 rounded p-2">
                  <div className="font-semibold mb-1">비트코인</div>
                  <div className="text-gray-600 dark:text-gray-400">
                    특정 목적 (디지털 화폐)<br/>
                    제한된 스크립트 언어
                  </div>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded p-2">
                  <div className="font-semibold mb-1">이더리움</div>
                  <div className="text-gray-600 dark:text-gray-400">
                    범용 플랫폼 (무엇이든 가능)<br/>
                    튜링 완전 프로그래밍
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-indigo-800 dark:text-indigo-300 mb-2">
                📜 스마트 컨트랙트 (Smart Contracts)
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                "코드가 곧 법이다 (Code is Law)" - 블록체인에서 실행되는 자동화된 계약
              </p>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <pre className="text-xs text-gray-700 dark:text-gray-300">
{`// Solidity 예시: 간단한 토큰 계약
contract SimpleToken {
    mapping(address => uint256) public balances;

    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}

// 배포되면 누구도 수정 불가능
// 코드대로 자동 실행`}
                </pre>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-purple-800 dark:text-purple-300 mb-2">
                ⛽ Gas: 계산 비용 측정 단위
              </h5>
              <div className="space-y-3 text-sm">
                <p className="text-gray-700 dark:text-gray-300">
                  무한 루프나 악의적 코드로부터 네트워크를 보호하기 위해 모든 연산에 비용을 부과합니다.
                </p>
                <div className="bg-white dark:bg-gray-800 rounded p-3">
                  <pre className="text-xs text-gray-700 dark:text-gray-300">
{`Gas Limit × Gas Price = Transaction Fee

예시:
- Gas Limit: 21,000 (간단한 전송)
- Gas Price: 50 Gwei
- Total Fee: 0.00105 ETH (~$2)

복잡한 스마트 컨트랙트:
- Gas Limit: 300,000
- Total Fee: 0.015 ETH (~$30)`}
                  </pre>
                </div>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
              <h5 className="font-semibold text-green-800 dark:text-green-300 mb-2">
                🔄 Ethereum Virtual Machine (EVM)
              </h5>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                스마트 컨트랙트를 실행하는 분산 가상 머신입니다.
              </p>
              <div className="grid md:grid-cols-3 gap-2 text-xs">
                <div className="bg-white dark:bg-gray-800 rounded p-2">
                  <div className="font-semibold mb-1">튜링 완전</div>
                  <div className="text-gray-600 dark:text-gray-400">
                    모든 계산 가능한 함수를 실행할 수 있음
                  </div>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded p-2">
                  <div className="font-semibold mb-1">결정적 실행</div>
                  <div className="text-gray-600 dark:text-gray-400">
                    같은 입력 → 항상 같은 출력
                  </div>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded p-2">
                  <div className="font-semibold mb-1">샌드박스</div>
                  <div className="text-gray-600 dark:text-gray-400">
                    격리된 환경에서 안전하게 실행
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Blockchain Trilemma */}
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center gap-2">
          <Globe className="w-6 h-6 text-purple-600" />
          블록체인 트릴레마 (Scalability Trilemma)
        </h3>

        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Vitalik Buterin이 제시한 블록체인의 근본적 한계:
            <strong> 탈중앙화, 보안, 확장성 </strong> 중 2가지만 최적화 가능
          </p>

          <div className="grid md:grid-cols-3 gap-4 mb-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border-2 border-purple-200 dark:border-purple-700">
              <h5 className="font-bold text-purple-600 dark:text-purple-400 mb-2">
                🌐 탈중앙화 (Decentralization)
              </h5>
              <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 노드 운영 진입장벽 낮음</li>
                <li>• 누구나 검증자 참여 가능</li>
                <li>• 검열 저항성</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border-2 border-red-200 dark:border-red-700">
              <h5 className="font-bold text-red-600 dark:text-red-400 mb-2">
                🔒 보안 (Security)
              </h5>
              <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 51% 공격 방어</li>
                <li>• 이중 지불 방지</li>
                <li>• 불변성 보장</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border-2 border-blue-200 dark:border-blue-700">
              <h5 className="font-bold text-blue-600 dark:text-blue-400 mb-2">
                ⚡ 확장성 (Scalability)
              </h5>
              <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 높은 TPS (초당 거래)</li>
                <li>• 낮은 수수료</li>
                <li>• 빠른 최종성</li>
              </ul>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h5 className="font-semibold text-gray-900 dark:text-white mb-3">
              실제 사례 분석
            </h5>
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded">
                <span className="font-semibold">비트코인</span>
                <span className="text-xs">탈중앙화 ✅ | 보안 ✅ | 확장성 ❌ (7 TPS)</span>
              </div>
              <div className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded">
                <span className="font-semibold">이더리움 (PoW)</span>
                <span className="text-xs">탈중앙화 ✅ | 보안 ✅ | 확장성 ❌ (15 TPS)</span>
              </div>
              <div className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded">
                <span className="font-semibold">BSC (BNB Chain)</span>
                <span className="text-xs">탈중앙화 ❌ | 보안 ⚠️ | 확장성 ✅ (160 TPS)</span>
              </div>
              <div className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded">
                <span className="font-semibold">Solana</span>
                <span className="text-xs">탈중앙화 ⚠️ | 보안 ⚠️ | 확장성 ✅ (65,000 TPS)</span>
              </div>
              <div className="flex items-center justify-between p-2 bg-green-50 dark:bg-green-900/20 rounded">
                <span className="font-semibold">Ethereum L2 (Rollups)</span>
                <span className="text-xs">탈중앙화 ✅ | 보안 ✅ | 확장성 ✅ (해결 시도)</span>
              </div>
            </div>
          </div>
        </div>

        {/* 철학적 의미 */}
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🧠 블록체인의 철학
        </h3>

        <div className="space-y-4">
          <div className="bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-900/20 dark:to-yellow-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              탈중앙화 (Decentralization)
            </h4>
            <blockquote className="border-l-4 border-amber-500 pl-4 italic text-gray-700 dark:text-gray-300 mb-3">
              "권력을 중앙에서 가장자리로 이동시킨다"
            </blockquote>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              단일 실패 지점(Single Point of Failure)을 제거하고,
              모든 참여자가 동등한 권한을 가집니다.
              은행, 정부, 기업 없이도 신뢰할 수 있는 시스템 구축
            </p>
          </div>

          <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              무신뢰성 (Trustlessness)
            </h4>
            <blockquote className="border-l-4 border-cyan-500 pl-4 italic text-gray-700 dark:text-gray-300 mb-3">
              "Don't trust, Verify" - 믿지 말고 검증하라
            </blockquote>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              상대방을 신뢰할 필요 없이 수학과 암호학으로 보장됩니다.
              투명한 규칙과 검증 가능한 코드가 신뢰를 대체합니다.
            </p>
          </div>

          <div className="bg-gradient-to-r from-rose-50 to-red-50 dark:from-rose-900/20 dark:to-red-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              검열 저항성 (Censorship Resistance)
            </h4>
            <blockquote className="border-l-4 border-rose-500 pl-4 italic text-gray-700 dark:text-gray-300 mb-3">
              "누구도 당신의 거래를 막을 수 없다"
            </blockquote>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              정부나 기업이 특정 거래를 차단하거나 계정을 동결할 수 없습니다.
              개인의 금융 주권(Financial Sovereignty)을 보장합니다.
            </p>
          </div>
        </div>

        {/* 역사적 영향 */}
        <div className="bg-gradient-to-r from-gray-50 to-slate-50 dark:from-gray-900/50 dark:to-slate-900/50 rounded-xl p-6 mt-6">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            📚 블록체인의 역사적 영향
          </h3>
          <div className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
              <div>
                <strong>2008:</strong> 금융 위기 속에서 비트코인 백서 발표 - 중앙화된 금융 시스템에 대한 대안 제시
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
              <div>
                <strong>2013:</strong> 이더리움 백서로 블록체인의 활용 범위를 금융을 넘어 모든 산업으로 확장
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
              <div>
                <strong>2020-현재:</strong> DeFi, NFT, DAO, Web3 생태계로 발전하며 인터넷의 패러다임 전환 주도
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* References Section */}
      <References
        sections={[
          {
            title: '원본 백서 (Original Papers)',
            icon: 'paper',
            color: 'border-orange-500',
            items: [
              {
                authors: 'Satoshi Nakamoto',
                year: '2008',
                title: 'Bitcoin: A Peer-to-Peer Electronic Cash System',
                description: '비트코인의 기술적 기반을 제시한 역사적 문서',
                link: 'https://bitcoin.org/bitcoin.pdf'
              },
              {
                authors: 'Vitalik Buterin',
                year: '2013',
                title: 'Ethereum White Paper: A Next-Generation Smart Contract and Decentralized Application Platform',
                description: '스마트 컨트랙트 플랫폼의 청사진',
                link: 'https://ethereum.org/en/whitepaper/'
              },
              {
                authors: 'Gavin Wood',
                year: '2014',
                title: 'Ethereum Yellow Paper: A Secure Decentralised Generalised Transaction Ledger',
                description: 'EVM의 공식 기술 명세',
                link: 'https://ethereum.github.io/yellowpaper/paper.pdf'
              }
            ]
          },
          {
            title: '학술 논문 (Academic Papers)',
            icon: 'book',
            color: 'border-blue-500',
            items: [
              {
                authors: 'Leslie Lamport, Robert Shostak, Marshall Pease',
                year: '1982',
                title: 'The Byzantine Generals Problem',
                description: '분산 시스템의 합의 문제를 정의한 선구적 연구',
                link: 'https://lamport.azurewebsites.net/pubs/byz.pdf'
              },
              {
                authors: 'Adam Back',
                year: '2002',
                title: 'Hashcash - A Denial of Service Counter-Measure',
                description: 'Proof of Work의 기원이 된 연구',
                link: 'http://www.hashcash.org/papers/hashcash.pdf'
              },
              {
                authors: 'Joseph Bonneau et al.',
                year: '2015',
                title: 'SoK: Research Perspectives and Challenges for Bitcoin and Cryptocurrencies',
                description: '비트코인 보안 및 프라이버시 종합 분석',
                link: 'https://www.ieee-security.org/TC/SP2015/papers-archived/6949a104.pdf'
              }
            ]
          },
          {
            title: '산업 리포트 & 분석',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                authors: 'Messari',
                year: '2024',
                title: 'State of Crypto Report',
                description: '암호화폐 산업 동향 및 데이터 분석',
                link: 'https://messari.io/report-pdf'
              },
              {
                authors: 'Coinbase',
                year: '2024',
                title: 'Around the Block: Institutional Crypto Adoption',
                description: '기관 투자자의 암호화폐 도입 현황',
                link: 'https://www.coinbase.com/institutional'
              },
              {
                title: 'Ethereum Foundation - Research Papers',
                description: '이더리움 재단의 최신 연구 자료',
                link: 'https://ethereum.org/en/learn/'
              }
            ]
          },
          {
            title: '추가 학습 자료',
            icon: 'book',
            color: 'border-purple-500',
            items: [
              {
                title: 'Bitcoin Developer Documentation',
                description: '비트코인 개발자를 위한 공식 문서',
                link: 'https://developer.bitcoin.org/'
              },
              {
                title: 'Ethereum.org - Learn Hub',
                description: '이더리움 학습을 위한 종합 리소스',
                link: 'https://ethereum.org/en/developers/docs/'
              },
              {
                title: 'MIT OpenCourseWare - Blockchain and Money',
                description: 'Gary Gensler 교수의 블록체인 강의',
                link: 'https://ocw.mit.edu/courses/15-s12-blockchain-and-money-fall-2018/'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
