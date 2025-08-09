'use client'

import React from 'react'
import { 
  Server, Database, Cloud, Shield, Activity, 
  Layers, HardDrive, GitBranch, Box, Cpu,
  Network, Zap, AlertCircle, CheckCircle,
  ArrowRight, Code, BookOpen, Lightbulb
} from 'lucide-react'

interface ChapterContentProps {
  chapterId: string
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const content = getChapterContent(chapterId)
  
  if (!content) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <p className="text-gray-600 dark:text-gray-400">
          챕터 콘텐츠를 불러올 수 없습니다.
        </p>
      </div>
    )
  }
  
  return <>{content}</>
}

function getChapterContent(chapterId: string) {
  const contents: { [key: string]: JSX.Element } = {
    'fundamentals': <FundamentalsContent />,
    'scaling': <ScalingContent />,
    'caching': <CachingContent />,
    'database': <DatabaseContent />,
    'messaging': <MessagingContent />,
    'microservices': <MicroservicesContent />,
    'monitoring': <MonitoringContent />,
    'case-studies': <CaseStudiesContent />
  }
  
  return contents[chapterId]
}

function FundamentalsContent() {
  return (
    <div className="space-y-8">
      {/* Introduction */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Server className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          시스템 설계란?
        </h2>
        
        <div className="prose dark:prose-invert max-w-none">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            시스템 설계는 복잡한 소프트웨어 시스템의 아키텍처를 정의하는 과정입니다. 
            확장 가능하고, 신뢰할 수 있으며, 유지보수가 쉬운 시스템을 구축하기 위한 
            청사진을 만드는 것이 목표입니다.
          </p>
          
          <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-6 mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
              핵심 고려사항
            </h3>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                <span className="text-gray-700 dark:text-gray-300">
                  <strong>기능적 요구사항:</strong> 시스템이 수행해야 할 기능
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                <span className="text-gray-700 dark:text-gray-300">
                  <strong>비기능적 요구사항:</strong> 성능, 확장성, 가용성, 보안
                </span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="w-5 h-5 text-green-500 mt-0.5" />
                <span className="text-gray-700 dark:text-gray-300">
                  <strong>제약사항:</strong> 예산, 시간, 기술 스택, 팀 역량
                </span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Scalability */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Layers className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          확장성 (Scalability)
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              수직 확장 (Vertical Scaling)
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 단일 서버의 성능 향상</li>
              <li>• CPU, RAM, Storage 업그레이드</li>
              <li>• 구현이 간단함</li>
              <li>• 하드웨어 한계 존재</li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              수평 확장 (Horizontal Scaling)
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 서버 대수 증가</li>
              <li>• 로드 밸런싱 필요</li>
              <li>• 무한 확장 가능</li>
              <li>• 복잡도 증가</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Reliability & Availability */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Shield className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          신뢰성과 가용성
        </h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              가용성 목표 (SLA)
            </h3>
            <table className="w-full">
              <thead>
                <tr className="border-b dark:border-gray-700">
                  <th className="text-left py-2 text-gray-700 dark:text-gray-300">가용성</th>
                  <th className="text-left py-2 text-gray-700 dark:text-gray-300">연간 다운타임</th>
                  <th className="text-left py-2 text-gray-700 dark:text-gray-300">일반적 용도</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-2 text-gray-600 dark:text-gray-400">99%</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">3.65일</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">개인 프로젝트</td>
                </tr>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-2 text-gray-600 dark:text-gray-400">99.9%</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">8.77시간</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">일반 서비스</td>
                </tr>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-2 text-gray-600 dark:text-gray-400">99.99%</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">52.6분</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">핵심 서비스</td>
                </tr>
                <tr>
                  <td className="py-2 text-gray-600 dark:text-gray-400">99.999%</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">5.26분</td>
                  <td className="py-2 text-gray-600 dark:text-gray-400">금융/의료</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Back-of-the-envelope Calculation */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Cpu className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          백오브더엔벨로프 계산
        </h2>
        
        <div className="bg-yellow-50 dark:bg-yellow-950/20 rounded-lg p-6">
          <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
            시스템 용량 추정 예시
          </h3>
          
          <div className="bg-white dark:bg-gray-700 rounded-lg p-4 font-mono text-sm">
            <p className="text-gray-700 dark:text-gray-300">
              <span className="text-purple-600 dark:text-purple-400"># Twitter 타임라인 설계</span><br/>
              <br/>
              DAU (일일 활성 사용자): 150M<br/>
              사용자당 평균 트윗: 2개/일<br/>
              사용자당 평균 팔로우: 200명<br/>
              <br/>
              <span className="text-green-600 dark:text-green-400"># 일일 트윗 수</span><br/>
              150M × 2 = 300M 트윗/일<br/>
              <br/>
              <span className="text-green-600 dark:text-green-400"># 초당 트윗 (TPS)</span><br/>
              300M / 86,400초 ≈ 3,500 TPS (평균)<br/>
              피크 시간: 3,500 × 2 = 7,000 TPS<br/>
              <br/>
              <span className="text-green-600 dark:text-green-400"># 타임라인 읽기 요청</span><br/>
              150M × 50 (일일 조회) = 7.5B 읽기/일<br/>
              7.5B / 86,400 ≈ 87,000 RPS<br/>
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

function ScalingContent() {
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

function CachingContent() {
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

function DatabaseContent() {
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
              [Master1] ←→ [Master2] ←→ [Master3]<br/>
              ↑ Read/Write from any node ↑
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

function MessagingContent() {
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
            메시지 큐는 프로듀서와 컨슈머 간의 비동기 통신을 가능하게 하는 미들웨어입니다.
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
                  (메시지는 하나의 컨슈머만 처리)
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

function MicroservicesContent() {
  return (
    <div className="space-y-8">
      {/* Microservices Overview */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Box className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          마이크로서비스 아키텍처
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            마이크로서비스는 작고 독립적으로 배포 가능한 서비스들로 구성된 아키텍처 스타일입니다.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-red-50 dark:bg-red-950/20 rounded-lg p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                모놀리스 아키텍처
              </h3>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>✅ 개발 초기 단순함</li>
                <li>✅ 디버깅 용이</li>
                <li>✅ 트랜잭션 관리 간단</li>
                <li>❌ 확장성 제한</li>
                <li>❌ 기술 스택 고정</li>
                <li>❌ 배포 리스크 높음</li>
              </ul>
            </div>
            
            <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                마이크로서비스 아키텍처
              </h3>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>✅ 독립적 확장</li>
                <li>✅ 기술 다양성</li>
                <li>✅ 장애 격리</li>
                <li>❌ 운영 복잡도</li>
                <li>❌ 네트워크 지연</li>
                <li>❌ 데이터 일관성</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* API Gateway Pattern */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Network className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          API Gateway 패턴
        </h2>
        
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
              API Gateway 역할
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• 단일 진입점 제공</li>
                <li>• 인증/인가 처리</li>
                <li>• 요청 라우팅</li>
                <li>• 프로토콜 변환</li>
              </ul>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• Rate Limiting</li>
                <li>• 캐싱</li>
                <li>• 모니터링/분석</li>
                <li>• 응답 집계</li>
              </ul>
            </div>
          </div>
          
          <div className="bg-yellow-50 dark:bg-yellow-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              BFF (Backend for Frontend) 패턴
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              각 클라이언트 타입별로 최적화된 API Gateway 제공
            </p>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs">
              Mobile App → Mobile BFF → Microservices<br/>
              Web App → Web BFF → Microservices<br/>
              Desktop App → Desktop BFF → Microservices
            </div>
          </div>
        </div>
      </section>

      {/* Service Discovery */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Activity className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          서비스 디스커버리
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              클라이언트 사이드 디스커버리
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs mb-3">
              Client → Service Registry<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
              Client → Service Instance
            </div>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• 클라이언트가 직접 서비스 위치 조회</li>
              <li>• Netflix Eureka, Consul</li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              서버 사이드 디스커버리
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs mb-3">
              Client → Load Balancer<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Service Instance
            </div>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• 로드 밸런서가 서비스 라우팅</li>
              <li>• AWS ELB, Kubernetes Service</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Circuit Breaker Pattern */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Zap className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          Circuit Breaker 패턴
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            장애가 발생한 서비스로의 요청을 차단하여 연쇄 장애를 방지하는 패턴입니다.
          </p>
          
          <div className="bg-gradient-to-r from-green-50 to-yellow-50 to-red-50 dark:from-green-950/20 dark:via-yellow-950/20 dark:to-red-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
              Circuit Breaker 상태
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <div className="text-green-600 dark:text-green-400 font-semibold mb-2">
                  Closed (정상)
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  모든 요청 통과<br/>
                  실패 카운트 모니터링
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <div className="text-yellow-600 dark:text-yellow-400 font-semibold mb-2">
                  Half-Open (확인)
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  일부 요청만 통과<br/>
                  복구 여부 확인
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <div className="text-red-600 dark:text-red-400 font-semibold mb-2">
                  Open (차단)
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  모든 요청 차단<br/>
                  즉시 에러 반환
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Retry 전략
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• <strong>Exponential Backoff:</strong> 2초 → 4초 → 8초 → 16초</li>
              <li>• <strong>Jitter:</strong> 랜덤 지연 추가로 동시 재시도 방지</li>
              <li>• <strong>Retry Budget:</strong> 전체 요청의 10% 이하로 재시도 제한</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}

function MonitoringContent() {
  return (
    <div className="space-y-8">
      {/* Observability */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Activity className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          관측가능성 (Observability)
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            관측가능성은 시스템의 외부 출력을 통해 내부 상태를 이해할 수 있는 능력입니다.
          </p>
          
          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/20 dark:to-indigo-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">
              Three Pillars of Observability
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  📊 Metrics
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  시계열 수치 데이터<br/>
                  CPU, Memory, Latency, RPS
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  📝 Logs
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  이벤트 기록<br/>
                  Error, Warning, Info, Debug
                </p>
              </div>
              
              <div className="bg-white dark:bg-gray-700 rounded p-4">
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  🔍 Traces
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  요청 흐름 추적<br/>
                  분산 시스템 전체 경로
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Metrics & Monitoring */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Cpu className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          메트릭과 모니터링
        </h2>
        
        <div className="space-y-6">
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              핵심 메트릭 (Golden Signals)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  🚦 Latency
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  요청 처리 시간 (P50, P95, P99)
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  📈 Traffic
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  초당 요청 수 (RPS/QPS)
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  ❌ Errors
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  실패율 (4xx, 5xx)
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">
                  💾 Saturation
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  리소스 사용률 (CPU, Memory, Disk)
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Prometheus + Grafana
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs mb-3">
              <span className="text-green-600 dark:text-green-400"># PromQL 예시</span><br/>
              rate(http_requests_total[5m])<br/>
              histogram_quantile(0.95, http_request_duration_seconds)<br/>
              sum(rate(http_requests_total{`{status=~"5.."}`}[5m]))
            </div>
            <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
              <li>• Pull 기반 메트릭 수집</li>
              <li>• 시계열 데이터베이스</li>
              <li>• 강력한 쿼리 언어</li>
              <li>• 알림 규칙 설정</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Logging */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <BookOpen className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          로깅 시스템
        </h2>
        
        <div className="space-y-6">
          <div className="bg-yellow-50 dark:bg-yellow-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              ELK Stack
            </h3>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-16 font-bold text-purple-600 dark:text-purple-400">E</div>
                <div>
                  <strong>Elasticsearch:</strong> 로그 저장 및 검색
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-16 font-bold text-purple-600 dark:text-purple-400">L</div>
                <div>
                  <strong>Logstash:</strong> 로그 수집 및 처리
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-16 font-bold text-purple-600 dark:text-purple-400">K</div>
                <div>
                  <strong>Kibana:</strong> 시각화 및 대시보드
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              구조화된 로깅
            </h3>
            <div className="bg-white dark:bg-gray-700 rounded p-3 font-mono text-xs">
              {`{`}<br/>
              &nbsp;&nbsp;"timestamp": "2024-01-15T10:30:45Z",<br/>
              &nbsp;&nbsp;"level": "ERROR",<br/>
              &nbsp;&nbsp;"service": "payment-service",<br/>
              &nbsp;&nbsp;"trace_id": "abc123",<br/>
              &nbsp;&nbsp;"user_id": "user_456",<br/>
              &nbsp;&nbsp;"message": "Payment failed",<br/>
              &nbsp;&nbsp;"error": "Insufficient funds"<br/>
              {`}`}
            </div>
          </div>
        </div>
      </section>

      {/* Distributed Tracing */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Network className="w-7 h-7 text-purple-600 dark:text-purple-400" />
          분산 트레이싱
        </h2>
        
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            마이크로서비스 환경에서 요청이 여러 서비스를 거치는 전체 경로를 추적합니다.
          </p>
          
          <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-950/20 dark:to-green-950/20 rounded-lg p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              OpenTelemetry
            </h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• Trace ID: 전체 요청 추적</li>
              <li>• Span ID: 개별 작업 추적</li>
              <li>• Context Propagation: 서비스 간 컨텍스트 전달</li>
              <li>• Auto-instrumentation: 자동 계측</li>
            </ul>
            
            <div className="mt-4 bg-white dark:bg-gray-700 rounded p-3">
              <div className="text-sm font-mono">
                API Gateway → [2ms]<br/>
                └─ Auth Service → [5ms]<br/>
                └─ User Service → [8ms]<br/>
                &nbsp;&nbsp;&nbsp;└─ Database → [15ms]<br/>
                └─ Payment Service → [12ms]<br/>
                Total: 42ms
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

function CaseStudiesContent() {
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