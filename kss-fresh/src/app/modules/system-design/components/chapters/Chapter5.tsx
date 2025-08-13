'use client'

import React from 'react'
import { 
  GitBranch, Network, Layers
} from 'lucide-react'

export default function Chapter5() {
  return (
    <div className="space-y-8">
      {/* Message Queue Overview */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <GitBranch className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          메시지 큐 개요
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            메시지 큐는 프로듀서와 컴슈머 간의 비동기 통신을 가능하게 하는 미들웨어입니다.
          </p>
          
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
              메시징 패턴
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-3">
                  Point-to-Point (Queue)
                </h4>
                <div className="bg-gray-100 dark:bg-gray-600 rounded p-3 font-mono text-xs mb-3">
                  Producer → [Queue] → Consumer<br/>
                  (메시지는 하나의 컴슈머만 처리)
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  작업 큐, 태스크 분배에 적합
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded-lg p-4">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-3">
                  Publish-Subscribe
                </h4>
                <div className="bg-gray-100 dark:bg-gray-600 rounded p-3 font-mono text-xs mb-3">
                  Publisher → [Topic] → Subscribers<br/>
                  (메시지를 모든 구독자가 수신)
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  이벤트 브로드캐스트, 알림에 적합
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Message Brokers Comparison */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Network className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          메시지 브로커 비교
        </h2>
        
        <div className="space-y-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b dark:border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-700 dark:text-gray-300">특징</th>
                  <th className="text-left py-3 px-4 text-gray-700 dark:text-gray-300">RabbitMQ</th>
                  <th className="text-left py-3 px-4 text-gray-700 dark:text-gray-300">Kafka</th>
                  <th className="text-left py-3 px-4 text-gray-700 dark:text-gray-300">Redis Pub/Sub</th>
                  <th className="text-left py-3 px-4 text-gray-700 dark:text-gray-300">AWS SQS</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">처리량</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">중간</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">매우 높음</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">높음</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">중간</td>
                </tr>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">메시지 보존</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">일시적</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">영구적</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">없음</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">14일</td>
                </tr>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">순서 보장</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">큐 단위</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">파티션 단위</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">없음</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">FIFO 큐</td>
                </tr>
                <tr>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">사용 사례</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">작업 큐</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">로그 수집</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">실시간 알림</td>
                  <td className="py-3 px-4 text-gray-600 dark:text-gray-400">분산 시스템</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Event Sourcing & CQRS */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Layers className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          이벤트 소싱과 CQRS
        </h2>
        
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Event Sourcing
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              애플리케이션 상태를 이벤트의 시퀀스로 저장하는 패턴
            </p>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs">
              <span className="text-green-600 dark:text-green-400">// 전통적 방식</span><br/>
              User {`{id: 1, balance: 100}`}<br/>
              <br/>
              <span className="text-green-600 dark:text-green-400">// 이벤트 소싱</span><br/>
              AccountCreated {`{id: 1, balance: 0}`}<br/>
              MoneyDeposited {`{id: 1, amount: 150}`}<br/>
              MoneyWithdrawn {`{id: 1, amount: 50}`}<br/>
              → Current State: balance = 100
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              CQRS (Command Query Responsibility Segregation)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              읽기와 쓰기 모델을 분리하는 아키텍처 패턴
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-700 rounded p-3">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  Command Side
                </h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 상태 변경 처리</li>
                  <li>• 비즈니스 로직 실행</li>
                  <li>• 이벤트 발생</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-700 rounded p-3">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  Query Side
                </h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 읽기 전용 모델</li>
                  <li>• 최적화된 뷰</li>
                  <li>• 캐싱 가능</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}