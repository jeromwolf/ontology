'use client';

import References from '@/components/common/References';

// Chapter 2: AWS 핵심 서비스
export default function Chapter2() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">AWS (Amazon Web Services) 소개</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          AWS는 2006년 출시 이후 전 세계적으로 가장 널리 사용되는 클라우드 플랫폼입니다.
          200개 이상의 완전한 기능을 갖춘 서비스를 32개 리전과 102개 가용 영역에서 제공합니다.
        </p>
        <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
          <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-3">AWS 핵심 강점</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-2">
              <span className="text-orange-600 dark:text-orange-400">•</span>
              <div><strong>최대 시장 점유율</strong>: 글로벌 클라우드 시장의 32% 차지 (2024)</div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-orange-600 dark:text-orange-400">•</span>
              <div><strong>광범위한 서비스</strong>: 컴퓨팅, 스토리지, AI/ML, IoT 등 모든 영역 커버</div>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-orange-600 dark:text-orange-400">•</span>
              <div><strong>성숙한 생태계</strong>: 파트너, 마켓플레이스, 커뮤니티 지원</div>
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">핵심 컴퓨팅 서비스</h2>
        <div className="grid gap-6">
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 border-l-4 border-orange-500">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-3 text-xl">
              Amazon EC2 (Elastic Compute Cloud)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              AWS의 가장 기본적인 서비스로, 확장 가능한 가상 서버를 제공합니다.
              500개 이상의 인스턴스 타입으로 모든 워크로드를 지원합니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-4">
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">인스턴스 패밀리:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• <strong>범용 (T, M)</strong>: 웹 서버, 개발 환경</li>
                    <li>• <strong>컴퓨팅 최적화 (C)</strong>: 고성능 컴퓨팅, 배치 처리</li>
                    <li>• <strong>메모리 최적화 (R, X)</strong>: 데이터베이스, 캐싱</li>
                    <li>• <strong>스토리지 최적화 (I, D)</strong>: 빅데이터, DW</li>
                    <li>• <strong>GPU (P, G)</strong>: 머신러닝, 비디오 인코딩</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">구매 옵션:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• <strong>온디맨드</strong>: 시간당 과금, 유연성</li>
                    <li>• <strong>예약 인스턴스</strong>: 1-3년 약정, 최대 72% 절감</li>
                    <li>• <strong>스팟 인스턴스</strong>: 최대 90% 절감, 중단 가능</li>
                    <li>• <strong>Savings Plans</strong>: 유연한 약정 모델</li>
                  </ul>
                </div>
              </div>
            </div>
            <div className="bg-orange-100 dark:bg-orange-900/30 rounded p-4">
              <p className="text-sm text-orange-800 dark:text-orange-200">
                <strong>Auto Scaling</strong>: CPU 사용률, 네트워크 트래픽 등에 따라 자동으로 인스턴스 수 조절
              </p>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-3 text-xl">
              AWS Lambda (서버리스 컴퓨팅)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              서버 관리 없이 코드를 실행할 수 있는 이벤트 기반 서버리스 컴퓨팅 서비스입니다.
              밀리초 단위로 과금되며, 자동으로 확장됩니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-4">
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">지원 언어:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• Node.js, Python, Java, Go</li>
                    <li>• .NET Core, Ruby</li>
                    <li>• 커스텀 런타임 (Rust, PHP 등)</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-gray-900 dark:text-white block mb-2">주요 사용 사례:</strong>
                  <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                    <li>• 이미지/비디오 처리 (S3 트리거)</li>
                    <li>• API 백엔드 (API Gateway 연동)</li>
                    <li>• 실시간 스트림 처리 (Kinesis)</li>
                    <li>• 예약 작업 (EventBridge)</li>
                  </ul>
                </div>
              </div>
            </div>
            <div className="bg-blue-100 dark:bg-blue-900/30 rounded p-4">
              <p className="text-sm text-blue-800 dark:text-blue-200">
                <strong>요금</strong>: 요청 100만 건당 $0.20 + 실행 시간(GB-초)당 $0.0000166667
              </p>
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3 text-xl">
              Amazon ECS & EKS (컨테이너 서비스)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              Docker 컨테이너를 실행하고 관리하는 완전 관리형 서비스입니다.
            </p>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div className="bg-white dark:bg-gray-800 rounded p-4">
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">ECS (Elastic Container Service)</h4>
                <p className="text-gray-700 dark:text-gray-300 mb-2">AWS 자체 오케스트레이션</p>
                <ul className="space-y-1 text-gray-600 dark:text-gray-400">
                  <li>✓ AWS 네이티브 통합</li>
                  <li>✓ Fargate로 서버리스 실행</li>
                  <li>✓ 간단한 설정</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-4">
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">EKS (Elastic Kubernetes Service)</h4>
                <p className="text-gray-700 dark:text-gray-300 mb-2">관리형 Kubernetes</p>
                <ul className="space-y-1 text-gray-600 dark:text-gray-400">
                  <li>✓ 표준 Kubernetes API</li>
                  <li>✓ 멀티 클라우드 이식성</li>
                  <li>✓ 풍부한 생태계</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">스토리지 서비스</h2>
        <div className="grid gap-6">
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border-l-4 border-green-500">
            <h3 className="font-semibold text-green-800 dark:text-green-200 mb-3 text-xl">
              Amazon S3 (Simple Storage Service)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              무제한 용량의 객체 스토리지로, 99.999999999% (11 9's)의 내구성을 보장합니다.
              정적 웹사이트 호스팅, 데이터 레이크, 백업 등 다양한 용도로 사용됩니다.
            </p>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-4">
              <strong className="text-gray-900 dark:text-white block mb-3">스토리지 클래스 선택 가이드:</strong>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <span className="text-gray-700 dark:text-gray-300"><strong>S3 Standard</strong>: 자주 액세스</span>
                  <span className="text-gray-600 dark:text-gray-400">$0.023/GB</span>
                </div>
                <div className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <span className="text-gray-700 dark:text-gray-300"><strong>S3 Intelligent-Tiering</strong>: 자동 최적화</span>
                  <span className="text-gray-600 dark:text-gray-400">$0.023/GB + 모니터링</span>
                </div>
                <div className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <span className="text-gray-700 dark:text-gray-300"><strong>S3 Standard-IA</strong>: 드문 액세스</span>
                  <span className="text-gray-600 dark:text-gray-400">$0.0125/GB</span>
                </div>
                <div className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <span className="text-gray-700 dark:text-gray-300"><strong>S3 Glacier Flexible</strong>: 아카이빙</span>
                  <span className="text-gray-600 dark:text-gray-400">$0.004/GB</span>
                </div>
                <div className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-700 rounded">
                  <span className="text-gray-700 dark:text-gray-300"><strong>S3 Glacier Deep Archive</strong>: 장기 보관</span>
                  <span className="text-gray-600 dark:text-gray-400">$0.00099/GB</span>
                </div>
              </div>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-green-100 dark:bg-green-900/30 rounded p-3">
                <p className="text-sm text-green-800 dark:text-green-200">
                  <strong>S3 Versioning</strong>: 객체의 모든 버전 보관, 실수로 삭제 방지
                </p>
              </div>
              <div className="bg-green-100 dark:bg-green-900/30 rounded p-3">
                <p className="text-sm text-green-800 dark:text-green-200">
                  <strong>S3 Lifecycle</strong>: 자동으로 스토리지 클래스 전환 또는 삭제
                </p>
              </div>
            </div>
          </div>

          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6 border-l-4 border-indigo-500">
            <h3 className="font-semibold text-indigo-800 dark:text-indigo-200 mb-3 text-xl">
              Amazon EBS (Elastic Block Store)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              EC2 인스턴스에 연결되는 블록 레벨 스토리지입니다. 데이터베이스, 파일 시스템 등에 사용됩니다.
            </p>
            <div className="grid md:grid-cols-3 gap-3 text-sm">
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <h4 className="font-semibold text-gray-900 dark:text-white mb-1">gp3 (범용 SSD)</h4>
                <p className="text-gray-600 dark:text-gray-400">최대 16,000 IOPS</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <h4 className="font-semibold text-gray-900 dark:text-white mb-1">io2 (프로비저닝 IOPS)</h4>
                <p className="text-gray-600 dark:text-gray-400">최대 64,000 IOPS</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <h4 className="font-semibold text-gray-900 dark:text-white mb-1">st1 (처리량 최적화 HDD)</h4>
                <p className="text-gray-600 dark:text-gray-400">빅데이터, 로그</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">데이터베이스 서비스</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-blue-200 dark:border-blue-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">🗄️</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Amazon RDS</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
              관리형 관계형 데이터베이스 서비스. 자동 백업, 패치, 모니터링 제공.
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• MySQL, PostgreSQL, MariaDB</li>
              <li>• Oracle, SQL Server</li>
              <li>• Multi-AZ 고가용성</li>
              <li>• Read Replica 읽기 확장</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-purple-200 dark:border-purple-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">⚡</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Amazon Aurora</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
              MySQL/PostgreSQL 호환 고성능 데이터베이스. 표준 MySQL 대비 5배 빠름.
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• 자동 스토리지 확장 (최대 128TB)</li>
              <li>• 6개 복제본 (3개 AZ)</li>
              <li>• Aurora Serverless: 자동 확장</li>
              <li>• Global Database: 1초 미만 복제</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-orange-200 dark:border-orange-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">📊</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Amazon DynamoDB</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
              완전 관리형 NoSQL 데이터베이스. 밀리초 단위 지연시간, 무제한 확장.
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• Key-Value & Document 스토어</li>
              <li>• 자동 확장 (On-Demand 모드)</li>
              <li>• DynamoDB Streams: 변경 캡처</li>
              <li>• Global Tables: 다중 리전 복제</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-red-200 dark:border-red-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center">
                <span className="text-xl">💨</span>
              </div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-lg">Amazon ElastiCache</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3 text-sm">
              인메모리 캐싱 서비스. 애플리케이션 성능을 대폭 향상시킵니다.
            </p>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• Redis (데이터 구조, 지속성)</li>
              <li>• Memcached (단순 캐싱)</li>
              <li>• 마이크로초 단위 지연시간</li>
              <li>• 클러스터 모드로 확장</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">네트워킹 & CDN</h2>
        <div className="space-y-4">
          <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-lg p-6">
            <h3 className="font-bold text-cyan-900 dark:text-cyan-100 mb-3 text-lg">Amazon VPC (Virtual Private Cloud)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              AWS 클라우드 내에서 격리된 가상 네트워크를 정의합니다. 서브넷, 라우팅 테이블, 보안 그룹 등을 완전히 제어할 수 있습니다.
            </p>
            <div className="grid md:grid-cols-3 gap-3 text-sm">
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <strong className="text-gray-900 dark:text-white">Subnet</strong>
                <p className="text-gray-600 dark:text-gray-400 mt-1">Public / Private 분리</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <strong className="text-gray-900 dark:text-white">Security Group</strong>
                <p className="text-gray-600 dark:text-gray-400 mt-1">상태 저장 방화벽</p>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded p-3">
                <strong className="text-gray-900 dark:text-white">NAT Gateway</strong>
                <p className="text-gray-600 dark:text-gray-400 mt-1">Private 아웃바운드</p>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Amazon CloudFront (CDN)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                전 세계 400+ 엣지 로케이션을 통한 콘텐츠 전송 네트워크
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• 낮은 지연시간, 높은 전송 속도</li>
                <li>• DDoS 보호 (AWS Shield 통합)</li>
                <li>• Lambda@Edge로 엣지 컴퓨팅</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-5 border border-gray-200 dark:border-gray-700">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Elastic Load Balancing</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                트래픽을 여러 대상에 자동으로 분산
              </p>
              <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                <li>• ALB: HTTP/HTTPS (L7)</li>
                <li>• NLB: TCP/UDP (L4, 초고성능)</li>
                <li>• GLB: 네트워크 가상 어플라이언스</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-900/20 dark:to-amber-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-orange-800 dark:text-orange-200">📚 핵심 정리</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-orange-600 dark:text-orange-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>컴퓨팅:</strong> EC2 (가상 서버), Lambda (서버리스), ECS/EKS (컨테이너)
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-orange-600 dark:text-orange-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>스토리지:</strong> S3 (객체), EBS (블록), 5가지 S3 스토리지 클래스
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-orange-600 dark:text-orange-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>데이터베이스:</strong> RDS, Aurora, DynamoDB, ElastiCache
            </span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-orange-600 dark:text-orange-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">
              <strong>네트워킹:</strong> VPC, CloudFront, Elastic Load Balancing
            </span>
          </li>
        </ul>
      </section>

      <References
        sections={[
          {
            title: '📚 AWS 공식 문서',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'AWS Documentation',
                description: '모든 AWS 서비스에 대한 공식 문서 (한국어 지원)',
                link: 'https://docs.aws.amazon.com/'
              },
              {
                title: 'AWS EC2 User Guide',
                description: 'EC2 인스턴스 타입, Auto Scaling, 네트워킹 완벽 가이드',
                link: 'https://docs.aws.amazon.com/ec2/'
              },
              {
                title: 'AWS Lambda Developer Guide',
                description: '서버리스 함수 개발, 배포, 모니터링 전체 가이드',
                link: 'https://docs.aws.amazon.com/lambda/'
              },
              {
                title: 'Amazon S3 User Guide',
                description: 'S3 버킷 관리, 스토리지 클래스, 보안 설정, Lifecycle 정책',
                link: 'https://docs.aws.amazon.com/s3/'
              }
            ]
          },
          {
            title: '🛠️ 실습 가이드',
            icon: 'tools',
            color: 'border-blue-500',
            items: [
              {
                title: 'AWS Hands-On Tutorials',
                description: '10분 안에 완료하는 AWS 실습 튜토리얼 (무료)',
                link: 'https://aws.amazon.com/getting-started/hands-on/'
              },
              {
                title: 'AWS Well-Architected Labs',
                description: 'AWS 모범 사례 실습 랩 - 보안, 성능, 비용 최적화',
                link: 'https://www.wellarchitectedlabs.com/'
              },
              {
                title: 'AWS Serverless Patterns Collection',
                description: 'Lambda 기반 서버리스 아키텍처 패턴 400+ 개',
                link: 'https://serverlessland.com/patterns'
              },
              {
                title: 'AWS Architecture Center',
                description: '실제 사용 사례별 참조 아키텍처 및 다이어그램',
                link: 'https://aws.amazon.com/architecture/'
              }
            ]
          },
          {
            title: '📖 학습 리소스',
            icon: 'book',
            color: 'border-purple-500',
            items: [
              {
                title: 'AWS Certified Solutions Architect - Associate',
                description: '가장 인기 있는 AWS 자격증 - 핵심 서비스 마스터',
                link: 'https://aws.amazon.com/certification/certified-solutions-architect-associate/'
              },
              {
                title: 'AWS Skill Builder (무료)',
                description: '500+ 무료 AWS 클라우드 학습 코스 및 실습 랩',
                link: 'https://skillbuilder.aws/'
              },
              {
                title: 'AWS re:Invent Videos',
                description: 'AWS 최신 기술 세미나 및 발표 영상 아카이브',
                link: 'https://www.youtube.com/user/AmazonWebServices'
              },
              {
                title: 'AWS This Week',
                description: 'AWS 주간 뉴스 및 신규 서비스 업데이트',
                link: 'https://aws.amazon.com/blogs/aws/'
              }
            ]
          },
          {
            title: '⚡ 비용 최적화',
            icon: 'tools',
            color: 'border-emerald-500',
            items: [
              {
                title: 'AWS Pricing Calculator',
                description: 'AWS 서비스 비용 예측 및 견적 생성 도구',
                link: 'https://calculator.aws/'
              },
              {
                title: 'AWS Cost Explorer',
                description: '비용 분석, 예측, 리소스 사용량 시각화',
                link: 'https://aws.amazon.com/aws-cost-management/aws-cost-explorer/'
              },
              {
                title: 'AWS Trusted Advisor',
                description: '비용 최적화, 보안, 성능 개선 권장 사항',
                link: 'https://aws.amazon.com/premiumsupport/technology/trusted-advisor/'
              }
            ]
          }
        ]}
      />
    </div>
  )
}
