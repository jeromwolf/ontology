export const metadata = {
  id: 'system-design',
  name: 'System Design',
  description: '대규모 분산 시스템 설계의 핵심 원칙과 실전 패턴을 학습합니다',
  icon: 'Server',
  color: 'purple',
  level: 'advanced',
  duration: '20시간',
  prerequisites: ['컴퓨터 과학 기초', '데이터베이스', '네트워크'],
  chapters: [
    {
      id: 'fundamentals',
      number: 1,
      title: '시스템 설계 기초',
      description: '확장성, 신뢰성, 가용성의 핵심 개념',
      duration: '2시간',
      objectives: [
        '시스템 설계의 기본 원칙 이해',
        '비기능적 요구사항 분석',
        'Trade-off 의사결정',
        '백오브더엔벨로프 계산법'
      ]
    },
    {
      id: 'scaling',
      number: 2,
      title: '확장성 설계 패턴',
      description: '수평/수직 확장, 로드 밸런싱, 샤딩 전략',
      duration: '3시간',
      objectives: [
        '수평 vs 수직 확장 전략',
        '로드 밸런서 종류와 알고리즘',
        '데이터베이스 샤딩 기법',
        'Consistent Hashing 이해'
      ]
    },
    {
      id: 'caching',
      number: 3,
      title: '캐싱 전략과 CDN',
      description: '캐시 계층, 캐싱 패턴, CDN 활용',
      duration: '2.5시간',
      objectives: [
        '캐시 계층 설계',
        'Cache-aside, Write-through, Write-behind 패턴',
        'Redis vs Memcached 비교',
        'CDN 아키텍처와 활용'
      ]
    },
    {
      id: 'database',
      number: 4,
      title: '데이터베이스 설계',
      description: 'SQL vs NoSQL, 복제, 파티셔닝, CAP 이론',
      duration: '3시간',
      objectives: [
        'ACID vs BASE 트레이드오프',
        'Master-Slave, Master-Master 복제',
        'NoSQL 데이터베이스 선택 기준',
        'CAP 이론과 실제 적용'
      ]
    },
    {
      id: 'messaging',
      number: 5,
      title: '메시징 시스템과 큐',
      description: '비동기 처리, 메시지 큐, 이벤트 스트리밍',
      duration: '2.5시간',
      objectives: [
        'Message Queue vs Pub/Sub 패턴',
        'Kafka, RabbitMQ, SQS 비교',
        '이벤트 소싱과 CQRS',
        '백프레셔와 흐름 제어'
      ]
    },
    {
      id: 'microservices',
      number: 6,
      title: '마이크로서비스 아키텍처',
      description: '서비스 분해, API Gateway, 서비스 메시',
      duration: '3시간',
      objectives: [
        '모놀리스 vs 마이크로서비스',
        'API Gateway 패턴',
        '서비스 디스커버리와 레지스트리',
        'Circuit Breaker와 Retry 패턴'
      ]
    },
    {
      id: 'monitoring',
      number: 7,
      title: '모니터링과 로깅',
      description: '분산 트레이싱, 메트릭 수집, 로그 집계',
      duration: '2시간',
      objectives: [
        '관측가능성(Observability) 구축',
        'Prometheus와 Grafana 활용',
        'ELK 스택 구성',
        '분산 트레이싱과 OpenTelemetry'
      ]
    },
    {
      id: 'case-studies',
      number: 8,
      title: '실전 시스템 설계',
      description: 'URL 단축기, 채팅 시스템, 뉴스피드 설계',
      duration: '4시간',
      objectives: [
        'URL 단축 서비스 설계',
        '실시간 채팅 시스템 구현',
        '소셜 미디어 피드 설계',
        '동영상 스트리밍 플랫폼 아키텍처'
      ]
    }
  ],
  simulators: [
    {
      id: 'load-balancer',
      title: '로드 밸런서 시뮬레이터',
      description: '다양한 로드 밸런싱 알고리즘 시각화'
    },
    {
      id: 'cache-simulator',
      title: '캐시 전략 시뮬레이터',
      description: 'LRU, LFU, FIFO 캐시 정책 비교'
    },
    {
      id: 'cap-theorem',
      title: 'CAP 이론 시각화',
      description: 'Consistency, Availability, Partition Tolerance 트레이드오프'
    },
    {
      id: 'sharding-visualizer',
      title: '샤딩 전략 시각화',
      description: 'Range, Hash, Geographic 샤딩 시뮬레이션'
    },
    {
      id: 'rate-limiter',
      title: 'Rate Limiter 구현',
      description: 'Token Bucket, Sliding Window 알고리즘'
    },
    {
      id: 'architecture-builder',
      title: '아키텍처 빌더',
      description: '드래그 앤 드롭으로 시스템 설계하기'
    }
  ]
}