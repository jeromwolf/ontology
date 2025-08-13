'use client'

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Ethereum과 스마트 컨트랙트
        </h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            Ethereum은 "월드 컴퓨터"를 목표로 하는 프로그래밍 가능한 블록체인입니다.
            스마트 컨트랙트를 통해 탈중앙화된 애플리케이션(DApp)을 구축할 수 있습니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🖥️ Ethereum Virtual Machine (EVM)
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">EVM 특징</h4>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• 튜링 완전한 가상 머신</li>
                <li>• 결정론적 실행 환경</li>
                <li>• Gas 기반 실행 비용 계산</li>
                <li>• 256비트 워드 크기</li>
                <li>• 스택 기반 아키텍처</li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">EVM 호환 체인</h4>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• Polygon (MATIC)</li>
                <li>• Binance Smart Chain</li>
                <li>• Avalanche C-Chain</li>
                <li>• Fantom</li>
                <li>• Arbitrum</li>
              </ul>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          📝 Solidity 프로그래밍
        </h3>
        
        <div className="space-y-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              기본 스마트 컨트랙트 구조
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-sm text-gray-700 dark:text-gray-300">
{`// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleStorage {
    // State variables
    uint256 private storedData;
    address public owner;
    
    // Events
    event DataStored(uint256 indexed data, address indexed user);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }
    
    // Constructor
    constructor() {
        owner = msg.sender;
    }
    
    // Functions
    function set(uint256 _data) public onlyOwner {
        storedData = _data;
        emit DataStored(_data, msg.sender);
    }
    
    function get() public view returns (uint256) {
        return storedData;
    }
}`}
              </pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              Gas 최적화 패턴
            </h4>
            <div className="space-y-4">
              <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2">
                  1. Storage 최적화
                </h5>
                <pre className="text-sm text-gray-700 dark:text-gray-300">
{`// ❌ 비효율적
contract Inefficient {
    uint8 a;   // Slot 0
    uint256 b; // Slot 1
    uint8 c;   // Slot 2
}

// ✅ 효율적 (Packing)
contract Efficient {
    uint8 a;   // Slot 0
    uint8 c;   // Slot 0
    uint256 b; // Slot 1
}`}
                </pre>
              </div>

              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2">
                  2. 루프 최적화
                </h5>
                <pre className="text-sm text-gray-700 dark:text-gray-300">
{`// ❌ 비효율적
for (uint i = 0; i < array.length; i++) {
    // array.length를 매번 읽음
}

// ✅ 효율적
uint256 length = array.length;
for (uint i = 0; i < length; ++i) {
    // 한 번만 읽고, ++i 사용
}`}
                </pre>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🚀 배포와 상호작용
        </h3>
        
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3">
            컨트랙트 배포 프로세스
          </h4>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
              <span>Solidity 코드 작성 및 컴파일</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
              <span>테스트넷에서 테스트 (Goerli, Sepolia)</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
              <span>보안 감사 수행</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
              <span>메인넷 배포</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">5</div>
              <span>Etherscan 검증</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}