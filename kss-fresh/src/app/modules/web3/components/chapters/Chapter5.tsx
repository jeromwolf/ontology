'use client'

import { Code } from 'lucide-react'

export default function Chapter5() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Web3 개발 스택
        </h2>
        
        <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            Web3 애플리케이션 개발에는 스마트 컨트랙트, 프론트엔드, 
            블록체인 상호작용을 위한 다양한 도구와 라이브러리가 필요합니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🛠️ 개발 프레임워크
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">Hardhat</h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 mb-3">
              <pre className="text-sm text-gray-700 dark:text-gray-300">
{`// hardhat.config.js
module.exports = {
  solidity: "0.8.19",
  networks: {
    mainnet: {
      url: process.env.MAINNET_RPC,
      accounts: [process.env.PRIVATE_KEY]
    }
  },
  etherscan: {
    apiKey: process.env.ETHERSCAN_KEY
  }
};`}
              </pre>
            </div>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>✓ 로컬 노드 내장</li>
              <li>✓ 콘솔 디버깅</li>
              <li>✓ 플러그인 시스템</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">Foundry</h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 mb-3">
              <pre className="text-sm text-gray-700 dark:text-gray-300">
{`// Solidity로 테스트 작성
contract TokenTest is Test {
    function testTransfer() public {
        token.transfer(alice, 100);
        assertEq(token.balanceOf(alice), 100);
    }
}`}
              </pre>
            </div>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>✓ 빠른 컴파일 속도</li>
              <li>✓ Solidity 테스트</li>
              <li>✓ Fuzzing 지원</li>
            </ul>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🔗 Web3 라이브러리
        </h3>
        
        <div className="space-y-4 mb-8">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              Ethers.js vs Web3.js
            </h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h5 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-2">Ethers.js</h5>
                <pre className="text-sm text-gray-700 dark:text-gray-300">
{`import { ethers } from 'ethers';

// 프로바이더 연결
const provider = new ethers.providers.Web3Provider(
  window.ethereum
);

// 컨트랙트 인스턴스
const contract = new ethers.Contract(
  address, 
  abi, 
  provider.getSigner()
);

// 트랜잭션 실행
const tx = await contract.transfer(to, amount);
await tx.wait();`}
                </pre>
              </div>
              
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                <h5 className="font-semibold text-green-600 dark:text-green-400 mb-2">Web3.js</h5>
                <pre className="text-sm text-gray-700 dark:text-gray-300">
{`import Web3 from 'web3';

// 프로바이더 연결
const web3 = new Web3(window.ethereum);

// 컨트랙트 인스턴스
const contract = new web3.eth.Contract(
  abi, 
  address
);

// 트랜잭션 실행
await contract.methods.transfer(to, amount)
  .send({ from: account });`}
                </pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🦊 지갑 통합
        </h3>
        
        <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6 mb-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3">
            MetaMask 연결 구현
          </h4>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <pre className="text-sm text-gray-700 dark:text-gray-300">
{`async function connectWallet() {
  if (typeof window.ethereum !== 'undefined') {
    try {
      // 계정 요청
      const accounts = await window.ethereum.request({ 
        method: 'eth_requestAccounts' 
      });
      
      // 체인 ID 확인
      const chainId = await window.ethereum.request({ 
        method: 'eth_chainId' 
      });
      
      // 체인 전환 (필요시)
      if (chainId !== '0x1') {
        await window.ethereum.request({
          method: 'wallet_switchEthereumChain',
          params: [{ chainId: '0x1' }],
        });
      }
      
      // 이벤트 리스너
      window.ethereum.on('accountsChanged', handleAccountsChanged);
      window.ethereum.on('chainChanged', handleChainChanged);
      
      return accounts[0];
    } catch (error) {
      console.error('Connection failed:', error);
    }
  } else {
    alert('Please install MetaMask!');
  }
}`}
            </pre>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          📊 The Graph 프로토콜
        </h3>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3">
            서브그래프 쿼리
          </h4>
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <pre className="text-sm text-gray-700 dark:text-gray-300">
{`# GraphQL 쿼리
query GetUserTokens($user: String!) {
  tokens(where: { owner: $user }) {
    id
    tokenURI
    transfers(orderBy: timestamp, orderDirection: desc) {
      from
      to
      timestamp
      txHash
    }
  }
}`}
            </pre>
          </div>
        </div>
      </section>
    </div>
  )
}