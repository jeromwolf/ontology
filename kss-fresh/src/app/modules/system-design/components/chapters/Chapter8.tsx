'use client';

import React from 'react';
import {
  Lightbulb
} from 'lucide-react';
import References from '@/components/common/References';

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

      {/* References Section */}
      <References
        sections={[
          {
            title: '📚 핵심 서적 & 강의',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Designing Data-Intensive Applications',
                authors: 'Martin Kleppmann',
                year: '2017',
                description: '데이터 중심 애플리케이션 설계의 바이블. 확장성, 신뢰성, 유지보수성을 고려한 시스템 설계의 모든 것',
                link: 'https://dataintensive.net/'
              },
              {
                title: 'System Design Interview (Vol 1 & 2)',
                authors: 'Alex Xu',
                year: '2020, 2022',
                description: 'FAANG 시스템 디자인 인터뷰 대비 필독서. 실전 케이스 스터디와 단계별 설계 프로세스',
                link: 'https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF'
              },
              {
                title: 'Web Scalability for Startup Engineers',
                authors: 'Artur Ejsmont',
                year: '2015',
                description: '스타트업 엔지니어를 위한 웹 확장성 가이드. 실무 중심의 확장 전략과 패턴',
                link: 'https://www.amazon.com/Scalability-Startup-Engineers-Artur-Ejsmont/dp/0071843655'
              },
              {
                title: 'Building Microservices',
                authors: 'Sam Newman',
                year: '2021',
                description: '마이크로서비스 아키텍처의 설계, 구축, 배포. 분산 시스템의 모범 사례',
                link: 'https://www.oreilly.com/library/view/building-microservices-2nd/9781492034018/'
              },
              {
                title: 'The System Design Primer',
                authors: 'Donne Martin (GitHub)',
                year: '2024',
                description: '170K+ stars를 받은 오픈소스 시스템 디자인 학습 자료. 다이어그램과 예제가 풍부',
                link: 'https://github.com/donnemartin/system-design-primer'
              }
            ]
          },
          {
            title: '🏢 기술 블로그 & 아키텍처',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'High Scalability',
                authors: 'Todd Hoff',
                year: '지속 업데이트',
                description: '대규모 시스템 아키텍처 사례 연구. Netflix, Uber, Twitter 등 실전 아키텍처 분석',
                link: 'http://highscalability.com/'
              },
              {
                title: 'Netflix Tech Blog',
                authors: 'Netflix Engineering',
                year: '지속 업데이트',
                description: 'Netflix의 마이크로서비스 아키�ecture, 카오스 엔지니어링, 글로벌 CDN 전략',
                link: 'https://netflixtechblog.com/'
              },
              {
                title: 'Uber Engineering Blog',
                authors: 'Uber Engineering',
                year: '지속 업데이트',
                description: 'Uber의 실시간 데이터 처리, 위치 기반 서비스, 분산 시스템 설계',
                link: 'https://eng.uber.com/'
              },
              {
                title: 'AWS Architecture Blog',
                authors: 'AWS Solutions Architects',
                year: '지속 업데이트',
                description: 'AWS 기반 Well-Architected Framework, 참조 아키텍처, 베스트 프랙티스',
                link: 'https://aws.amazon.com/blogs/architecture/'
              },
              {
                title: 'Meta Engineering Blog',
                authors: 'Meta Engineering',
                year: '지속 업데이트',
                description: 'Facebook, Instagram의 대규모 소셜 미디어 시스템 아키텍처와 최적화',
                link: 'https://engineering.fb.com/'
              },
              {
                title: 'Google Cloud Architecture Center',
                authors: 'Google Cloud',
                year: '지속 업데이트',
                description: 'Google의 클라우드 아키텍처 패턴, 참조 아키텍처, 설계 원칙',
                link: 'https://cloud.google.com/architecture'
              }
            ]
          },
          {
            title: '🛠️ 도구 & 플랫폼',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'draw.io (diagrams.net)',
                authors: 'JGraph',
                year: '무료',
                description: '시스템 아키텍처 다이어그램 작성 도구. AWS, GCP, Azure 아이콘 라이브러리 제공',
                link: 'https://www.diagrams.net/'
              },
              {
                title: 'Excalidraw',
                authors: 'Excalidraw',
                year: '무료',
                description: '손그림 스타일의 아키텍처 다이어그램 도구. 협업 기능 내장',
                link: 'https://excalidraw.com/'
              },
              {
                title: 'Mermaid Live Editor',
                authors: 'Mermaid JS',
                year: '무료',
                description: '코드 기반 다이어그램 생성. Markdown과 통합 가능, Git friendly',
                link: 'https://mermaid.live/'
              },
              {
                title: 'ByteByteGo',
                authors: 'Alex Xu',
                year: '유료 ($29/월)',
                description: '시스템 디자인 인터뷰 준비 플랫폼. 비디오 강의, 다이어그램, 실전 문제',
                link: 'https://bytebytego.com/'
              },
              {
                title: 'Educative.io - Grokking System Design',
                authors: 'Educative',
                year: '유료',
                description: '인터랙티브 시스템 디자인 강의. 실전 케이스 스터디와 연습 문제',
                link: 'https://www.educative.io/courses/grokking-the-system-design-interview'
              },
              {
                title: 'System Design Interview Roadmap',
                authors: 'roadmap.sh',
                year: '무료',
                description: '시스템 디자인 학습 로드맵. 단계별 학습 경로와 리소스',
                link: 'https://roadmap.sh/system-design'
              }
            ]
          }
        ]}
      />
    </div>
  )
}