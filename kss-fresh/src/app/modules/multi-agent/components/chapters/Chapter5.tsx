'use client';

import React from 'react';
import References from '@/components/common/References';

export default function Chapter5() {
  return (
    <div className="space-y-8">
      {/* 합의 알고리즘 */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          분산 합의 알고리즘
        </h2>
        <div className="prose prose-lg dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            멀티 에이전트 시스템에서 <strong>합의(Consensus)</strong>는 분산된 에이전트들이 
            공통의 결정에 도달하는 과정입니다. 중앙 조정자 없이도 일관된 의사결정을 가능하게 합니다.
          </p>
        </div>
      </section>

      <section className="bg-green-50 dark:bg-green-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          주요 합의 알고리즘
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-3">Voting Mechanisms</h4>
            <ul className="space-y-2 text-sm">
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>Majority Vote:</strong> 과반수 득표
              </li>
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>Weighted Vote:</strong> 가중치 투표
              </li>
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>Ranked Choice:</strong> 선호도 순위
              </li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold text-green-600 dark:text-green-400 mb-3">Byzantine Consensus</h4>
            <ul className="space-y-2 text-sm">
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>PBFT:</strong> Practical Byzantine Fault Tolerance
              </li>
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>Raft:</strong> 리더 기반 합의
              </li>
              <li className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>Paxos:</strong> 분산 합의 프로토콜
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          경매 기반 조정 메커니즘
        </h3>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
              <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">English Auction</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                가격이 점진적으로 상승하는 공개 경매
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
              <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">Dutch Auction</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                높은 가격에서 시작해 하락하는 경매
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
              <h4 className="font-semibold text-yellow-700 dark:text-yellow-300 mb-2">Vickrey Auction</h4>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                비공개 입찰, 차순위 가격 지불
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-green-100 to-blue-100 dark:from-green-900/20 dark:to-blue-900/20 rounded-xl p-6">
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          🎯 실전: 분산 자원 할당
        </h3>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
          <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
            클라우드 컴퓨팅 자원 할당 시나리오
          </h4>
          <div className="space-y-2 text-sm">
            <p className="text-gray-600 dark:text-gray-400">
              여러 에이전트가 제한된 컴퓨팅 자원(CPU, 메모리, 스토리지)을 경쟁
            </p>
            <div className="grid md:grid-cols-2 gap-2 mt-3">
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>문제:</strong> 자원 경쟁과 공정성
              </div>
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>해결:</strong> 경매 메커니즘 적용
              </div>
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>최적화:</strong> 전체 시스템 효율
              </div>
              <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded">
                <strong>공정성:</strong> 비례 할당 보장
              </div>
            </div>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Consensus Algorithms Foundations',
            icon: 'book',
            color: 'border-orange-500',
            items: [
              {
                title: 'Consensus in Distributed Systems',
                authors: 'Leslie Lamport',
                year: '1998',
                description: '분산 시스템 합의 알고리즘의 이론적 기초',
                link: 'https://lamport.azurewebsites.net/pubs/pubs.html#consensus'
              },
              {
                title: 'The Byzantine Generals Problem',
                authors: 'Leslie Lamport, Robert Shostak, Marshall Pease',
                year: '1982',
                description: 'Byzantine Fault Tolerance의 고전적 문제 정의',
                link: 'https://lamport.azurewebsites.net/pubs/byz.pdf'
              },
              {
                title: 'Paxos Made Simple',
                authors: 'Leslie Lamport',
                year: '2001',
                description: 'Paxos 알고리즘의 단순화된 설명',
                link: 'https://lamport.azurewebsites.net/pubs/paxos-simple.pdf'
              }
            ]
          },
          {
            title: 'Modern Consensus Protocols',
            icon: 'paper',
            color: 'border-purple-500',
            items: [
              {
                title: 'Practical Byzantine Fault Tolerance (PBFT)',
                authors: 'Miguel Castro, Barbara Liskov',
                year: '1999',
                description: '실용적 Byzantine Fault Tolerance 알고리즘',
                link: 'http://pmg.csail.mit.edu/papers/osdi99.pdf'
              },
              {
                title: 'In Search of an Understandable Consensus Algorithm (Raft)',
                authors: 'Diego Ongaro, John Ousterhout',
                year: '2014',
                description: '이해하기 쉬운 합의 알고리즘 Raft',
                link: 'https://raft.github.io/raft.pdf'
              },
              {
                title: 'HotStuff: BFT Consensus in the Lens of Blockchain',
                authors: 'Maofan Yin, Dahlia Malkhi, et al.',
                year: '2019',
                description: '블록체인을 위한 3-chain BFT 합의',
                link: 'https://arxiv.org/abs/1803.05069'
              },
              {
                title: 'Tendermint: Consensus without Mining',
                authors: 'Jae Kwon',
                year: '2014',
                description: 'PoS 기반 Byzantine Fault Tolerant 합의',
                link: 'https://tendermint.com/static/docs/tendermint.pdf'
              }
            ]
          },
          {
            title: 'Auction Theory & Mechanism Design',
            icon: 'web',
            color: 'border-blue-500',
            items: [
              {
                title: 'Auction Theory',
                authors: 'Vijay Krishna',
                year: '2009',
                description: '경매 이론의 포괄적 교과서',
                link: 'https://www.wiley.com/en-us/Auction+Theory%2C+2nd+Edition-p-9780123745071'
              },
              {
                title: 'Mechanism Design and Approximation',
                authors: 'Jason Hartline',
                year: '2013',
                description: '메커니즘 디자인과 근사 알고리즘',
                link: 'http://jasonhartline.com/MDnA/'
              },
              {
                title: 'Vickrey Auction & Second-Price Sealed Bid',
                description: 'Vickrey 경매의 이론과 실제',
                link: 'https://en.wikipedia.org/wiki/Vickrey_auction'
              }
            ]
          },
          {
            title: 'Implementation & Applications',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Raft Consensus Algorithm: Official Site',
                description: 'Raft 합의 알고리즘 공식 사이트 및 구현',
                link: 'https://raft.github.io/'
              },
              {
                title: 'etcd: Distributed Key-Value Store with Raft',
                description: 'Raft를 사용하는 분산 KV 스토어',
                link: 'https://etcd.io/'
              },
              {
                title: 'Consensus in Cloud Resource Allocation',
                description: '클라우드 자원 할당을 위한 합의 메커니즘',
                link: 'https://ieeexplore.ieee.org/document/8967348'
              },
              {
                title: 'Blockchain Consensus Mechanisms',
                description: '블록체인 합의 알고리즘 비교 분석',
                link: 'https://ethereum.org/en/developers/docs/consensus-mechanisms/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}