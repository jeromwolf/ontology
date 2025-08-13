'use client'

import React from 'react'
import { 
  Lightbulb
} from 'lucide-react'

export default function Chapter8() {
  return (
    <div className="space-y-8">
      {/* URL Shortener */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Lightbulb className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          Case Study: URL 단축 서비스
        </h2>
        
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              요구사항 분석
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 일일 100M URL 단축 요청</li>
              <li>• 읽기:쓰기 = 100:1</li>
              <li>• 7자리 단축 URL (62^7 = 3.5조 조합)</li>
              <li>• 99.9% 가용성</li>
              <li>• &lt; 100ms 응답 시간</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
              시스템 설계
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-4 font-mono text-xs">
              <span className="text-green-600 dark:text-green-400"># 단축 URL 생성</span><br/>
              1. Counter Service → 고유 ID 생성<br/>
              2. Base62 Encoding → 7자리 문자열<br/>
              3. Cache + DB 저장<br/>
              <br/>
              <span className="text-green-600 dark:text-green-400"># URL 리다이렉트</span><br/>
              1. Cache 조회 (Redis)<br/>
              2. Cache Miss → DB 조회<br/>
              3. 301/302 Redirect<br/>
              <br/>
              <span className="text-green-600 dark:text-green-400"># 확장 전략</span><br/>
              • 다중 캐시 서버 (Consistent Hashing)<br/>
              • 읽기 복제본 DB<br/>
              • CDN for popular URLs
            </div>
          </div>
        </div>
      </section>

      {/* Real-time Chat System */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Lightbulb className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          Case Study: 실시간 채팅 시스템
        </h2>
        
        <div className="space-y-6">
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              핵심 기능
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 1:1 및 그룹 채팅</li>
              <li>• 온라인 상태 표시</li>
              <li>• 메시지 전달 확인</li>
              <li>• 미디어 파일 전송</li>
              <li>• 메시지 암호화</li>
            </ul>
          </div>
          
          <div className="bg-yellow-50 dark:bg-yellow-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              아키텍처 컴포넌트
            </h3>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-700 rounded p-3">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-1">
                  WebSocket Servers
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  실시간 양방향 통신, Sticky Session
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-3">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-1">
                  Message Queue (Kafka)
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  메시지 순서 보장, 오프라인 사용자 처리
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-3">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-1">
                  NoSQL DB (Cassandra)
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  메시지 이력 저장, 시계열 데이터
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-3">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-1">
                  Redis
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  온라인 상태, 세션 관리
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Social Media Feed */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Lightbulb className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          Case Study: 소셜 미디어 피드
        </h2>
        
        <div className="space-y-6">
          <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              피드 생성 전략
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-700 rounded p-3">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  Push Model (Write Heavy)
                </h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 포스트 작성 시 팔로워 피드에 푸시</li>
                  <li>• 읽기 빠름</li>
                  <li>• 유명인 문제 (팔로워 많으면 느림)</li>
                </ul>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-3">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  Pull Model (Read Heavy)
                </h4>
                <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 피드 요청 시 실시간 생성</li>
                  <li>• 쓰기 빠름</li>
                  <li>• 읽기 시 계산 비용</li>
                </ul>
              </div>
            </div>
            
            <div className="mt-4 bg-blue-50 dark:bg-blue-950/20 rounded p-3">
              <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                Hybrid Approach
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                • 일반 사용자: Push Model<br/>
                • 유명인 (팔로워 &gt; 10K): Pull Model<br/>
                • 최근 포스트는 캐시에 유지
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Video Streaming Platform */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Lightbulb className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          Case Study: 동영상 스트리밍 플랫폼
        </h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-950/20 dark:to-orange-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              비디오 처리 파이프라인
            </h3>
            <ol className="space-y-3 text-gray-700 dark:text-gray-300">
              <li className="flex items-start gap-3">
                <span className="font-bold text-purple-600 dark:text-purple-400">1.</span>
                <div>
                  <strong>업로드:</strong> 청크 단위 업로드, 재개 가능
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="font-bold text-purple-600 dark:text-purple-400">2.</span>
                <div>
                  <strong>인코딩:</strong> 다양한 해상도 (144p ~ 4K)
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="font-bold text-purple-600 dark:text-purple-400">3.</span>
                <div>
                  <strong>저장:</strong> Object Storage (S3)
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="font-bold text-purple-600 dark:text-purple-400">4.</span>
                <div>
                  <strong>CDN 배포:</strong> 글로벌 엣지 서버
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="font-bold text-purple-600 dark:text-purple-400">5.</span>
                <div>
                  <strong>스트리밍:</strong> Adaptive Bitrate Streaming
                </div>
              </li>
            </ol>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              핵심 기술
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>HLS/DASH:</strong> HTTP 기반 스트리밍 프로토콜</li>
              <li>• <strong>Transcoding:</strong> FFmpeg 기반 비디오 변환</li>
              <li>• <strong>DRM:</strong> 콘텐츠 보호</li>
              <li>• <strong>Analytics:</strong> 시청 패턴 분석</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}