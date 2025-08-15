'use client';

import React from 'react';
import { 
  HardDrive, GitBranch, Cloud
} from 'lucide-react';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      {/* Cache Overview */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <HardDrive className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          캐싱 개요
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            캐싱은 자주 액세스되는 데이터를 빠른 저장소에 임시 저장하여 시스템 성능을 향상시키는 기술입니다.
          </p>
          
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
              캐시 계층 구조
            </h3>
            <div className="space-y-3">
              <div className="flex items-center gap-4">
                <div className="w-24 text-sm font-medium text-gray-700 dark:text-gray-300">브라우저</div>
                <div className="flex-1 bg-blue-200 dark:bg-blue-800 h-8 rounded flex items-center px-3 text-sm">
                  Browser Cache (가장 빠름)
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="w-24 text-sm font-medium text-gray-700 dark:text-gray-300">CDN</div>
                <div className="flex-1 bg-green-200 dark:bg-green-800 h-8 rounded flex items-center px-3 text-sm">
                  Edge Cache (지리적 분산)
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="w-24 text-sm font-medium text-gray-700 dark:text-gray-300">앱 서버</div>
                <div className="flex-1 bg-yellow-200 dark:bg-yellow-800 h-8 rounded flex items-center px-3 text-sm">
                  Application Cache (Redis/Memcached)
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="w-24 text-sm font-medium text-gray-700 dark:text-gray-300">데이터베이스</div>
                <div className="flex-1 bg-red-200 dark:bg-red-800 h-8 rounded flex items-center px-3 text-sm">
                  Database Cache (Query Cache)
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Caching Patterns */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <GitBranch className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          캐싱 패턴
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Cache-Aside (Lazy Loading)
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs mb-3">
              <span className="text-purple-600 dark:text-purple-400">// Read</span><br/>
              data = cache.get(key)<br/>
              if data == null:<br/>
              &nbsp;&nbsp;data = db.query(key)<br/>
              &nbsp;&nbsp;cache.set(key, data)<br/>
              return data
            </div>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>✅ 필요한 데이터만 캐싱</li>
              <li>✅ 노드 장애에 강함</li>
              <li>❌ 캐시 미스 시 지연</li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Write-Through
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs mb-3">
              <span className="text-purple-600 dark:text-purple-400">// Write</span><br/>
              cache.set(key, data)<br/>
              db.save(key, data)<br/>
              <br/>
              <span className="text-purple-600 dark:text-purple-400">// Read</span><br/>
              return cache.get(key)
            </div>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>✅ 캐시 항상 최신</li>
              <li>✅ 읽기 성능 우수</li>
              <li>❌ 쓰기 지연 증가</li>
            </ul>
          </div>
          
          <div className="bg-yellow-50 dark:bg-yellow-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Write-Behind (Write-Back)
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs mb-3">
              <span className="text-purple-600 dark:text-purple-400">// Write</span><br/>
              cache.set(key, data)<br/>
              <span className="text-green-600 dark:text-green-400">// 비동기로 DB 업데이트</span><br/>
              async_queue.add(key, data)
            </div>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>✅ 쓰기 성능 최고</li>
              <li>✅ 배치 처리 가능</li>
              <li>❌ 데이터 손실 위험</li>
            </ul>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Refresh-Ahead
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs mb-3">
              <span className="text-purple-600 dark:text-purple-400">// 만료 전 자동 갱신</span><br/>
              if ttl &lt; threshold:<br/>
              &nbsp;&nbsp;async_refresh(key)
            </div>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>✅ 캐시 미스 최소화</li>
              <li>✅ 일관된 성능</li>
              <li>❌ 예측 정확도 중요</li>
            </ul>
          </div>
        </div>
      </section>

      {/* CDN */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Cloud className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          CDN (Content Delivery Network)
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            CDN은 지리적으로 분산된 서버 네트워크로, 사용자와 가까운 위치에서 콘텐츠를 제공합니다.
          </p>
          
          <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-950/20 dark:to-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
              CDN 작동 방식
            </h3>
            <ol className="space-y-3 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-3">
                <span className="font-bold text-purple-600 dark:text-purple-400">1.</span>
                사용자가 콘텐츠 요청 (예: image.cdn.com/photo.jpg)
              </li>
              <li className="flex items-start gap-3">
                <span className="font-bold text-purple-600 dark:text-purple-400">2.</span>
                DNS가 가장 가까운 Edge 서버로 라우팅
              </li>
              <li className="flex items-start gap-3">
                <span className="font-bold text-purple-600 dark:text-purple-400">3.</span>
                Edge 서버에 캐시가 있으면 즉시 응답
              </li>
              <li className="flex items-start gap-3">
                <span className="font-bold text-purple-600 dark:text-purple-400">4.</span>
                캐시 미스 시 Origin 서버에서 가져와 캐싱
              </li>
            </ol>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                Push CDN
              </h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• Origin에서 CDN으로 콘텐츠 푸시</li>
                <li>• 업데이트 시점 제어 가능</li>
                <li>• 저장 공간 많이 사용</li>
                <li>• 정적 콘텐츠에 적합</li>
              </ul>
            </div>
            
            <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                Pull CDN
              </h3>
              <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li>• 요청 시 Origin에서 가져옴</li>
                <li>• 저장 공간 효율적</li>
                <li>• 첫 요청 시 지연</li>
                <li>• 트래픽 변동이 큰 경우 적합</li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}