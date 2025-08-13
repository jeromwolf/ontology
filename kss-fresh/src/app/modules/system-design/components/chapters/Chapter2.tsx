'use client'

import React from 'react'
import { 
  Network, Database, GitBranch
} from 'lucide-react'

export default function Chapter2() {
  return (
    <div className="space-y-8">
      {/* Load Balancing */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Network className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          로드 밸런싱
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            로드 밸런서는 들어오는 트래픽을 여러 서버에 분산시켜 시스템의 가용성과 응답성을 향상시킵니다.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                L4 로드 밸런서
              </h3>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• Transport Layer (TCP/UDP)</li>
                <li>• IP 주소와 포트 기반 라우팅</li>
                <li>• 빠른 처리 속도</li>
                <li>• HAProxy, NGINX Plus</li>
              </ul>
            </div>
            
            <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                L7 로드 밸런서
              </h3>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• Application Layer (HTTP/HTTPS)</li>
                <li>• URL, 헤더, 쿠키 기반 라우팅</li>
                <li>• 콘텐츠 기반 라우팅 가능</li>
                <li>• NGINX, Apache, ALB</li>
              </ul>
            </div>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              로드 밸런싱 알고리즘
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  Round Robin
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  순차적으로 서버에 요청 분배
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  Least Connections
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  연결 수가 가장 적은 서버 선택
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  Weighted Round Robin
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  서버 성능에 따른 가중치 부여
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  IP Hash
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  클라이언트 IP 기반 서버 고정
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Database Sharding */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Database className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          데이터베이스 샤딩
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            샤딩은 대용량 데이터베이스를 여러 개의 작은 파티션(샤드)으로 분할하는 기술입니다.
          </p>
          
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
              샤딩 전략
            </h3>
            
            <div className="space-y-4">
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  Range-based Sharding
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  데이터 범위로 샤드 결정 (예: user_id 1-1000 → Shard 1)
                </p>
                <div className="bg-gray-100 dark:bg-gray-600 rounded p-2 font-mono text-xs">
                  if user_id &lt;= 1000: shard_1<br/>
                  elif user_id &lt;= 2000: shard_2<br/>
                  else: shard_3
                </div>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  Hash-based Sharding
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  해시 함수로 샤드 결정 (균등 분산)
                </p>
                <div className="bg-gray-100 dark:bg-gray-600 rounded p-2 font-mono text-xs">
                  shard_id = hash(user_id) % num_shards
                </div>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  Geographic Sharding
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  지역별로 데이터 분할 (데이터 지역성)
                </p>
                <div className="bg-gray-100 dark:bg-gray-600 rounded p-2 font-mono text-xs">
                  if region == 'US': shard_us<br/>
                  elif region == 'EU': shard_eu<br/>
                  else: shard_asia
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Consistent Hashing */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <GitBranch className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          Consistent Hashing
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            Consistent Hashing은 노드 추가/제거 시 최소한의 데이터 재분배만 필요한 해싱 기법입니다.
          </p>
          
          <div className="bg-yellow-50 dark:bg-yellow-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              작동 원리
            </h3>
            <ol className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>1. 해시 함수로 서버와 키를 링 위에 매핑</li>
              <li>2. 키는 시계방향으로 가장 가까운 서버에 할당</li>
              <li>3. 서버 추가/제거 시 인접 키만 재할당</li>
              <li>4. Virtual Nodes로 부하 균등 분산</li>
            </ol>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              장점
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 노드 추가/제거 시 K/N 개의 키만 재할당</li>
              <li>• 수평 확장에 유리</li>
              <li>• 핫스팟 문제 완화</li>
              <li>• Cassandra, DynamoDB에서 사용</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}