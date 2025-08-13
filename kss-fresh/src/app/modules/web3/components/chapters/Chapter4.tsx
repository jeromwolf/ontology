'use client'

import { Image } from 'lucide-react'

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          NFT와 디지털 자산
        </h2>
        
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 mb-6">
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            NFT(Non-Fungible Token)는 블록체인 상의 고유한 디지털 자산입니다.
            예술, 게임, 메타버스, 신원 증명 등 다양한 분야에서 활용되고 있습니다.
          </p>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🎨 NFT 표준과 구현
        </h3>
        
        <div className="space-y-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              ERC-721 표준
            </h4>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <pre className="text-sm text-gray-700 dark:text-gray-300">
{`interface IERC721 {
    // 소유권 관련
    function ownerOf(uint256 tokenId) external view returns (address);
    function balanceOf(address owner) external view returns (uint256);
    
    // 전송 관련
    function transferFrom(address from, address to, uint256 tokenId) external;
    function safeTransferFrom(address from, address to, uint256 tokenId) external;
    
    // 승인 관련
    function approve(address to, uint256 tokenId) external;
    function getApproved(uint256 tokenId) external view returns (address);
    function setApprovalForAll(address operator, bool approved) external;
    
    // 이벤트
    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);
}`}
              </pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">
              ERC-1155 Multi-Token
            </h4>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2">장점</h5>
                <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                  <li>• Fungible + Non-Fungible</li>
                  <li>• 배치 전송 지원</li>
                  <li>• Gas 효율적</li>
                  <li>• 게임 아이템에 최적</li>
                </ul>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                <h5 className="font-semibold text-gray-900 dark:text-white mb-2">사용 사례</h5>
                <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                  <li>• 게임 아이템 (검, 포션)</li>
                  <li>• 티켓 시스템</li>
                  <li>• 멤버십 토큰</li>
                  <li>• 에디션 아트</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          💾 메타데이터와 IPFS
        </h3>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 mb-6">
          <h4 className="font-bold text-gray-900 dark:text-white mb-3">
            NFT 메타데이터 구조
          </h4>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <pre className="text-sm text-gray-700 dark:text-gray-300">
{`{
  "name": "Cosmic Explorer #1337",
  "description": "A rare cosmic explorer from the metaverse",
  "image": "ipfs://QmXxx.../image.png",
  "attributes": [
    {
      "trait_type": "Background",
      "value": "Nebula"
    },
    {
      "trait_type": "Rarity",
      "value": "Legendary"
    },
    {
      "trait_type": "Power",
      "value": 9500,
      "max_value": 10000
    }
  ],
  "animation_url": "ipfs://QmYyy.../animation.mp4",
  "external_url": "https://example.com/nft/1337"
}`}
            </pre>
          </div>
        </div>

        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3">
          🏪 NFT 마켓플레이스
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">주요 마켓플레이스</h4>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="font-medium">OpenSea</span>
                <span className="text-sm text-gray-600 dark:text-gray-400">종합 마켓</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="font-medium">Blur</span>
                <span className="text-sm text-gray-600 dark:text-gray-400">프로 트레이더</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="font-medium">Rarible</span>
                <span className="text-sm text-gray-600 dark:text-gray-400">커뮤니티 중심</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="font-medium">Foundation</span>
                <span className="text-sm text-gray-600 dark:text-gray-400">아트 특화</span>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg p-6">
            <h4 className="font-bold text-gray-900 dark:text-white mb-3">로열티 메커니즘</h4>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <pre className="text-sm text-gray-700 dark:text-gray-300">
{`// EIP-2981 로열티 표준
function royaltyInfo(uint256 tokenId, uint256 salePrice)
    external view returns (
        address receiver,
        uint256 royaltyAmount
    ) {
    uint256 royalty = (salePrice * royaltyBps) / 10000;
    return (creator, royalty);
}`}
              </pre>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}