import References from '@/components/common/References'

export default function Section6() {
  return (
    <>
      {/* Section 6: 실전 구현 가이드 */}
      <section className="bg-gradient-to-r from-indigo-500 to-cyan-600 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-6">실전 구현 가이드</h2>

        <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
          <h3 className="font-bold mb-4">🚀 프로덕션 체크리스트</h3>

          <div className="space-y-4">
            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">📋 인프라 준비</h4>
              <ol className="space-y-2 text-sm">
                <li>✅ Kubernetes 클러스터 (최소 10 노드)</li>
                <li>✅ 고성능 SSD 스토리지 (NVMe 권장)</li>
                <li>✅ 100Gbps 네트워크 대역폭</li>
                <li>✅ GPU 노드 풀 (임베딩 계산용)</li>
                <li>✅ Multi-AZ 배포</li>
              </ol>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">🔧 필수 구성 요소</h4>
              <ul className="space-y-2 text-sm">
                <li>• <strong>Vector DB:</strong> Milvus 2.3+ 또는 Qdrant 1.7+</li>
                <li>• <strong>Message Queue:</strong> Pulsar 또는 Kafka</li>
                <li>• <strong>Cache:</strong> Redis Cluster 7.0+</li>
                <li>• <strong>Monitoring:</strong> Prometheus + Grafana</li>
                <li>• <strong>Service Mesh:</strong> Istio 또는 Linkerd</li>
              </ul>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">💡 최적화 팁</h4>
              <ul className="space-y-1 text-sm">
                <li>• 임베딩 차원을 384로 줄여 메모리 30% 절약</li>
                <li>• Product Quantization으로 인덱스 크기 75% 감소</li>
                <li>• 배치 처리로 처리량 10배 향상</li>
                <li>• Edge 캐싱으로 글로벌 레이턴시 50% 감소</li>
              </ul>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">🚨 주의사항</h4>
              <p className="text-sm">
                대규모 분산 시스템은 복잡도가 매우 높습니다.
                처음부터 완벽한 시스템을 구축하려 하지 말고,
                점진적으로 확장하며 각 단계에서 충분한 테스트를 수행하세요.
                특히 Chaos Engineering은 충분한 준비 후에 도입하세요.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 분산 벡터 데이터베이스',
            icon: 'web' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'Weaviate Distributed Architecture',
                authors: 'Weaviate',
                year: '2025',
                description: '분산 벡터 DB - 수평 스케일링, Multi-tenancy, Replication',
                link: 'https://weaviate.io/developers/weaviate/concepts/cluster'
              },
              {
                title: 'Milvus Clustering & Sharding',
                authors: 'Zilliz',
                year: '2025',
                description: 'Kubernetes 기반 클러스터 - 10억+ 벡터, 샤딩 전략',
                link: 'https://milvus.io/docs/scaleout.md'
              },
              {
                title: 'Qdrant Distributed Mode',
                authors: 'Qdrant',
                year: '2024',
                description: 'Rust 기반 분산 벡터 검색 - Raft 컨센서스, 자동 샤드 밸런싱',
                link: 'https://qdrant.tech/documentation/guides/distributed_deployment/'
              },
              {
                title: 'Pinecone Serverless',
                authors: 'Pinecone',
                year: '2024',
                description: '완전 관리형 분산 벡터 DB - 자동 스케일링, 99.99% SLA',
                link: 'https://docs.pinecone.io/docs/architecture'
              },
              {
                title: 'Elasticsearch Vector Search at Scale',
                authors: 'Elastic',
                year: '2025',
                description: '분산 검색 엔진 - HNSW 인덱싱, Cross-cluster 검색',
                link: 'https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html'
              }
            ]
          },
          {
            title: '📖 대규모 시스템 설계 & 사례',
            icon: 'research' as const,
            color: 'border-cyan-500',
            items: [
              {
                title: 'Uber Michelangelo: ML Platform',
                authors: 'Uber Engineering',
                year: '2023',
                description: '분산 ML 추론 시스템 - Feature Store, Model Serving, 40억 QPS',
                link: 'https://www.uber.com/blog/michelangelo-machine-learning-platform/'
              },
              {
                title: 'Netflix Recommendation Infrastructure',
                authors: 'Netflix Tech Blog',
                year: '2024',
                description: '실시간 개인화 추천 - Cassandra, Kafka, 수억 사용자',
                link: 'https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8'
              },
              {
                title: 'Airbnb Search Ranking & Personalization',
                authors: 'Airbnb Engineering',
                year: '2023',
                description: '분산 검색 시스템 - Embedding 기반 검색, Real-time 업데이트',
                link: 'https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789'
              },
              {
                title: 'Designing Data-Intensive Applications',
                authors: 'Martin Kleppmann',
                year: '2017',
                description: '분산 시스템 바이블 - Replication, Partitioning, Consistency',
                link: 'https://dataintensive.net/'
              }
            ]
          },
          {
            title: '🛠️ 분산 시스템 인프라 도구',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'Kubernetes for ML',
                authors: 'CNCF',
                year: '2025',
                description: 'Container 오케스트레이션 - Auto-scaling, Load Balancing, Service Mesh',
                link: 'https://kubernetes.io/docs/concepts/workloads/'
              },
              {
                title: 'Apache Kafka: Event Streaming',
                authors: 'Confluent',
                year: '2025',
                description: '분산 메시지 큐 - Partitioning, Replication, Exactly-once Delivery',
                link: 'https://kafka.apache.org/documentation/'
              },
              {
                title: 'Consul: Service Discovery',
                authors: 'HashiCorp',
                year: '2024',
                description: 'Service Mesh - Health Checking, Load Balancing, Multi-datacenter',
                link: 'https://developer.hashicorp.com/consul/docs'
              },
              {
                title: 'Prometheus + Grafana',
                authors: 'CNCF / Grafana Labs',
                year: '2025',
                description: '분산 모니터링 - 시계열 메트릭, 알람, 대시보드',
                link: 'https://prometheus.io/docs/introduction/overview/'
              },
              {
                title: 'Chaos Mesh: Chaos Engineering',
                authors: 'PingCAP',
                year: '2024',
                description: 'Kubernetes 장애 주입 - Pod Failure, Network 지연, IO 스트레스',
                link: 'https://chaos-mesh.org/docs/'
              }
            ]
          }
        ]}
      />
    </>
  )
}
