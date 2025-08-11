export const moduleMetadata = {
  id: 'data-engineering',
  title: 'Data Engineering',
  description: '현대적인 데이터 엔지니어링 - EDA부터 실시간 처리까지 완벽 마스터',
  icon: '🗃️',
  gradient: 'from-indigo-600 to-blue-700',
  category: 'Data',
  difficulty: 'Advanced',
  estimatedHours: 48,
  students: 1850,
  rating: 4.8,
  lastUpdated: '2025-08-10',
  prerequisites: ['Python 중급', 'SQL 기본', '리눅스 기초'],
  skills: [
    'ETL/ELT 파이프라인 설계',
    '탐색적 데이터 분석 (EDA)',
    '실시간 스트림 처리',
    '데이터 레이크하우스 구축',
    'Apache Spark 최적화',
    '데이터 품질 관리',
    'MLOps 파이프라인 통합',
    '클라우드 데이터 플랫폼'
  ],
  chapters: [
    {
      id: 'data-engineering-foundations',
      title: '데이터 엔지니어링 기초와 생태계',
      description: '데이터 엔지니어의 역할, 최신 트렌드, 커리어 패스',
      estimatedMinutes: 120,
    },
    {
      id: 'exploratory-data-analysis',
      title: '탐색적 데이터 분석 (EDA) 완벽 가이드',
      description: 'Pandas, Polars로 하는 현대적 EDA, 시각화, 통계적 분석',
      estimatedMinutes: 240,
    },
    {
      id: 'data-architecture-patterns',
      title: '현대적 데이터 아키텍처 패턴',
      description: '람다/카파 아키텍처, 데이터 메시, 레이크하우스 설계',
      estimatedMinutes: 180,
    },
    {
      id: 'batch-processing',
      title: '배치 데이터 처리와 ETL/ELT',
      description: 'Apache Spark, dbt, Airflow를 활용한 대규모 데이터 처리',
      estimatedMinutes: 300,
    },
    {
      id: 'stream-processing',
      title: '실시간 스트림 처리 마스터',
      description: 'Kafka, Flink, Spark Streaming으로 실시간 파이프라인 구축',
      estimatedMinutes: 300,
    },
    {
      id: 'data-modeling-warehousing',
      title: '데이터 모델링과 웨어하우징',
      description: 'Kimball vs Inmon, Star Schema, Data Vault 2.0',
      estimatedMinutes: 240,
    },
    {
      id: 'data-quality-governance',
      title: '데이터 품질과 거버넌스',
      description: 'Great Expectations, dbt tests, 데이터 계보 추적',
      estimatedMinutes: 180,
    },
    {
      id: 'cloud-data-platforms',
      title: '클라우드 데이터 플랫폼 실전',
      description: 'Snowflake, BigQuery, Databricks, AWS/Azure/GCP 비교',
      estimatedMinutes: 240,
    },
    {
      id: 'data-orchestration',
      title: '데이터 오케스트레이션',
      description: 'Airflow, Dagster, Prefect - 워크플로우 자동화',
      estimatedMinutes: 240,
    },
    {
      id: 'performance-optimization',
      title: '성능 최적화와 비용 관리',
      description: '쿼리 최적화, 파티셔닝, 인덱싱, 클라우드 비용 절감',
      estimatedMinutes: 180,
    },
    {
      id: 'mlops-data-engineering',
      title: 'MLOps를 위한 데이터 엔지니어링',
      description: 'Feature Store, ML 파이프라인, 모델 서빙 데이터 준비',
      estimatedMinutes: 240,
    },
    {
      id: 'real-world-projects',
      title: '실전 프로젝트와 케이스 스터디',
      description: 'Netflix, Uber, Airbnb의 데이터 플랫폼 분석',
      estimatedMinutes: 180,
    },
  ],
  simulators: [
    {
      id: 'eda-playground',
      title: '탐색적 데이터 분석 플레이그라운드',
      description: '인터랙티브 EDA - 데이터셋 업로드, 시각화, 통계 분석, 이상치 탐지',
      component: 'EDAPlayground'
    },
    {
      id: 'etl-pipeline-designer',
      title: 'ETL/ELT 파이프라인 디자이너',
      description: '드래그 앤 드롭으로 데이터 파이프라인 설계 및 실행',
      component: 'ETLPipelineDesigner'
    },
    {
      id: 'stream-processing-lab',
      title: '실시간 스트림 처리 실습실',
      description: 'Kafka + Spark Streaming 실시간 데이터 처리 시뮬레이션',
      component: 'StreamProcessingLab'
    },
    {
      id: 'data-lakehouse-architect',
      title: '데이터 레이크하우스 아키텍트',
      description: 'Delta Lake, Iceberg를 활용한 레이크하우스 설계',
      component: 'DataLakehouseArchitect'
    },
    {
      id: 'airflow-dag-builder',
      title: 'Airflow DAG 빌더',
      description: '비주얼 DAG 작성 및 워크플로우 오케스트레이션',
      component: 'AirflowDAGBuilder'
    },
    {
      id: 'spark-optimizer',
      title: 'Spark 성능 최적화 도구',
      description: 'Spark Job 분석, 최적화 제안, 실행 계획 시각화',
      component: 'SparkOptimizer'
    },
    {
      id: 'data-quality-suite',
      title: '데이터 품질 관리 스위트',
      description: 'Great Expectations 기반 데이터 품질 검증 및 모니터링',
      component: 'DataQualitySuite'
    },
    {
      id: 'cloud-cost-calculator',
      title: '클라우드 데이터 비용 계산기',
      description: 'AWS/GCP/Azure 데이터 서비스 비용 최적화 시뮬레이터',
      component: 'CloudCostCalculator'
    },
    {
      id: 'data-lineage-explorer',
      title: '데이터 계보 탐색기',
      description: '데이터 흐름 추적, 영향도 분석, 의존성 시각화',
      component: 'DataLineageExplorer'
    },
    {
      id: 'sql-performance-tuner',
      title: 'SQL 쿼리 성능 튜너',
      description: '쿼리 실행 계획 분석, 인덱스 추천, 최적화 가이드',
      component: 'SQLPerformanceTuner'
    },
  ],
  tools: [
    {
      id: 'data-profiler',
      title: '데이터 프로파일러',
      description: '데이터셋 자동 분석 및 품질 리포트 생성',
      icon: '📊',
    },
    {
      id: 'schema-generator',
      title: '스키마 생성기',
      description: 'JSON/CSV에서 SQL DDL 자동 생성',
      icon: '🏗️',
    },
    {
      id: 'pipeline-monitor',
      title: '파이프라인 모니터',
      description: '실시간 파이프라인 상태 대시보드',
      icon: '📡',
    },
    {
      id: 'data-dictionary',
      title: '데이터 사전 관리자',
      description: '메타데이터 문서화 및 카탈로그',
      icon: '📚',
    },
  ],
  learningPath: [
    {
      stage: 'Foundation',
      description: '데이터 엔지니어링 기초와 EDA',
      chapters: ['data-engineering-foundations', 'exploratory-data-analysis', 'data-architecture-patterns']
    },
    {
      stage: 'Core Skills',
      description: '핵심 데이터 처리 기술',
      chapters: ['batch-processing', 'stream-processing', 'data-modeling-warehousing']
    },
    {
      stage: 'Advanced',
      description: '고급 주제와 최적화',
      chapters: ['data-quality-governance', 'cloud-data-platforms', 'data-orchestration', 'performance-optimization']
    },
    {
      stage: 'Professional',
      description: '실무 적용과 통합',
      chapters: ['mlops-data-engineering', 'real-world-projects']
    }
  ]
};