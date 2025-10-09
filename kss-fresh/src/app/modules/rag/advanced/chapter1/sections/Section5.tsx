import References from '@/components/common/References'

export default function Section5() {
  return (
    <>
      {/* 실습 과제 */}
      <section className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-6">실습 과제</h2>

        <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
          <h3 className="font-bold mb-4">GraphRAG 시스템 구축 및 평가</h3>

          <div className="space-y-4">
            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">🏗️ 구현 단계</h4>
              <ol className="space-y-2 text-sm">
                <li>1. Microsoft GraphRAG 라이브러리 설치 및 환경 구성</li>
                <li>2. 위키피디아 문서 100개로 지식 그래프 구축</li>
                <li>3. Neo4j와의 연동을 통한 그래프 시각화</li>
                <li>4. 커뮤니티 탐지 알고리즘 비교 (Louvain vs Leiden)</li>
                <li>5. 복잡한 질문에 대한 GraphRAG vs 기존 RAG 성능 비교</li>
              </ol>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">🎯 평가 기준</h4>
              <ul className="space-y-1 text-sm">
                <li>• 엔티티 추출 정확도 (F1-Score &gt; 0.8)</li>
                <li>• 커뮤니티 응집도 (Modularity &gt; 0.3)</li>
                <li>• 질의응답 품질 (BLEU Score, 사람 평가)</li>
                <li>• 시스템 확장성 (처리 시간, 메모리 사용량)</li>
              </ul>
            </div>

            <div className="bg-white/10 p-4 rounded-lg">
              <h4 className="font-medium mb-2">🚀 심화 과제</h4>
              <p className="text-sm">
                다국어 지식 그래프 구축: 영어-한국어 문서를 통합한 cross-lingual GraphRAG 시스템을 구축하고,
                언어 간 엔티티 매칭 및 관계 추론 성능을 분석해보세요.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 GraphRAG 공식 문서 & 도구',
            icon: 'web' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Microsoft GraphRAG Official Repository',
                authors: 'Microsoft Research',
                year: '2024',
                description: 'GraphRAG Python 라이브러리 - Community Detection, Hierarchical Summarization 공식 구현',
                link: 'https://github.com/microsoft/graphrag'
              },
              {
                title: 'Neo4j Graph Database',
                authors: 'Neo4j',
                year: '2025',
                description: '지식 그래프 저장 및 쿼리 - Cypher 언어, APOC 라이브러리',
                link: 'https://neo4j.com/docs/'
              },
              {
                title: 'NetworkX Documentation',
                authors: 'NetworkX Developers',
                year: '2024',
                description: '파이썬 그래프 분석 - Louvain, Leiden 커뮤니티 탐지 알고리즘',
                link: 'https://networkx.org/documentation/stable/'
              },
              {
                title: 'LangChain GraphRAG Integration',
                authors: 'LangChain',
                year: '2024',
                description: 'GraphRAG + LangChain 통합 - RAG 파이프라인 간편 구축',
                link: 'https://python.langchain.com/docs/use_cases/graph/graph_rag'
              },
              {
                title: 'OpenAI Entity Extraction Guide',
                authors: 'OpenAI',
                year: '2024',
                description: 'GPT-4를 활용한 엔티티 추출 - Few-shot 프롬프트 최적화',
                link: 'https://platform.openai.com/docs/guides/entity-extraction'
              }
            ]
          },
          {
            title: '📖 GraphRAG & Knowledge Graph 연구',
            icon: 'research' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'From Local to Global: GraphRAG for Complex Question Answering',
                authors: 'Edge et al., Microsoft Research',
                year: '2024',
                description: 'GraphRAG 원본 논문 - 41% 답변 포괄성 향상, Community-based Reasoning',
                link: 'https://arxiv.org/abs/2404.16130'
              },
              {
                title: 'Fast Unfolding of Communities in Large Networks (Louvain)',
                authors: 'Blondel et al., Université catholique de Louvain',
                year: '2008',
                description: 'Louvain 커뮤니티 탐지 - O(n log n) 시간복잡도, Modularity 최적화',
                link: 'https://arxiv.org/abs/0803.0476'
              },
              {
                title: 'From Louvain to Leiden: Better Community Detection',
                authors: 'Traag et al., Leiden University',
                year: '2019',
                description: 'Leiden 알고리즘 - Louvain 개선, 더 나은 커뮤니티 품질',
                link: 'https://www.nature.com/articles/s41598-019-41695-z'
              },
              {
                title: 'Knowledge Graph Embedding: A Survey',
                authors: 'Wang et al., Tsinghua University',
                year: '2023',
                description: 'TransE, DistMult, ComplEx 등 KG 임베딩 기법 종합',
                link: 'https://arxiv.org/abs/2002.00388'
              }
            ]
          },
          {
            title: '🛠️ 지식 그래프 구축 도구',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'spaCy for Entity Recognition',
                authors: 'Explosion AI',
                year: '2024',
                description: '고성능 NER - 다국어 지원, 커스텀 엔티티 학습',
                link: 'https://spacy.io/usage/linguistic-features#named-entities'
              },
              {
                title: 'Python-Louvain Library',
                authors: 'Thomas Aynaud',
                year: '2024',
                description: 'Louvain 커뮤니티 탐지 파이썬 구현 - NetworkX 호환',
                link: 'https://python-louvain.readthedocs.io/'
              },
              {
                title: 'Gephi Graph Visualization',
                authors: 'Gephi Consortium',
                year: '2024',
                description: '지식 그래프 시각화 - Force-directed 레이아웃, 커뮤니티 색상 표시',
                link: 'https://gephi.org/'
              },
              {
                title: 'Py2neo Neo4j Toolkit',
                authors: 'Nigel Small',
                year: '2024',
                description: 'Neo4j Python 드라이버 - OGM(Object-Graph Mapping) 지원',
                link: 'https://py2neo.org/2021.1/'
              },
              {
                title: 'LlamaIndex Knowledge Graph Index',
                authors: 'LlamaIndex',
                year: '2024',
                description: 'GraphRAG 구현 - Neo4j, Kuzu 통합, 자동 엔티티 추출',
                link: 'https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/'
              }
            ]
          }
        ]}
      />
    </>
  )
}
