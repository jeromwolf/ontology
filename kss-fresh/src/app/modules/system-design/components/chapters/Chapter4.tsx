'use client';

import React from 'react';
import { 
  Database, Shield, CheckCircle, AlertCircle, Layers
} from 'lucide-react';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      {/* SQL vs NoSQL */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Database className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          SQL vs NoSQL
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              관계형 데이터베이스 (SQL)
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                <span>ACID 트랜잭션 보장</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                <span>복잡한 쿼리와 조인 지원</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                <span>스키마로 데이터 일관성</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 mt-0.5" />
                <span>수직 확장 위주</span>
              </li>
            </ul>
            <div className="mt-4 p-3 bg-white dark:bg-gray-700 rounded">
              <p className="text-sm font-medium text-gray-800 dark:text-gray-200">
                예시: PostgreSQL, MySQL, Oracle
              </p>
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              NoSQL 데이터베이스
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                <span>수평 확장 용이</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                <span>유연한 스키마</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                <span>대용량 데이터 처리</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-yellow-500 mt-0.5" />
                <span>일관성 트레이드오프</span>
              </li>
            </ul>
            <div className="mt-4 p-3 bg-white dark:bg-gray-700 rounded">
              <p className="text-sm font-medium text-gray-800 dark:text-gray-200">
                예시: MongoDB, Cassandra, Redis
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CAP Theorem */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Shield className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          CAP 이론
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            분산 시스템은 Consistency, Availability, Partition Tolerance 중 최대 2개만 보장할 수 있습니다.
          </p>
          
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-lg p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="w-20 h-20 mx-auto bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-xl mb-3">
                  C
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                  Consistency
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  모든 노드가 동일한 데이터를 보여줌
                </p>
              </div>
              
              <div className="text-center">
                <div className="w-20 h-20 mx-auto bg-green-500 rounded-full flex items-center justify-center text-white font-bold text-xl mb-3">
                  A
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                  Availability
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  시스템이 항상 응답 가능
                </p>
              </div>
              
              <div className="text-center">
                <div className="w-20 h-20 mx-auto bg-purple-500 rounded-full flex items-center justify-center text-white font-bold text-xl mb-3">
                  P
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                  Partition Tolerance
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  네트워크 분할 시에도 동작
                </p>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                CP 시스템
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                일관성 + 분할 내성
              </p>
              <p className="text-xs text-gray-500">
                예: MongoDB, HBase, Redis
              </p>
            </div>
            
            <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                AP 시스템
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                가용성 + 분할 내성
              </p>
              <p className="text-xs text-gray-500">
                예: Cassandra, DynamoDB, CouchDB
              </p>
            </div>
            
            <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                CA 시스템
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                일관성 + 가용성
              </p>
              <p className="text-xs text-gray-500">
                예: 단일 노드 RDBMS
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Replication */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Layers className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          데이터베이스 복제
        </h2>
        
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Master-Slave 복제
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li>• Master: 읽기/쓰기 모두 처리</li>
              <li>• Slave: 읽기 전용 (Master 데이터 복제)</li>
              <li>• 읽기 부하 분산 가능</li>
              <li>• Master 장애 시 Slave 승격 필요</li>
            </ul>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs">
              Write → [Master] → Replicate → [Slave1, Slave2, Slave3]<br/>
              Read ← [Master or Slaves]
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Master-Master 복제
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li>• 모든 노드가 읽기/쓰기 가능</li>
              <li>• 높은 가용성</li>
              <li>• 충돌 해결 메커니즘 필요</li>
              <li>• 복잡한 일관성 관리</li>
            </ul>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs">
              [Master1] ↔ [Master2] ↔ [Master3]<br/>
              ↑ Read/Write from any node ↑
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}