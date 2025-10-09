import { Code } from 'lucide-react'

export default function Section3() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
          <Code className="text-green-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.3 GraphRAG 파이썬 구현</h2>
          <p className="text-gray-600 dark:text-gray-400">Microsoft GraphRAG SDK와 Neo4j 통합 구현</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
          <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">완전한 GraphRAG 시스템 구현</h3>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`import asyncio
import networkx as nx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from community import community_louvain
import pandas as pd

@dataclass
class Entity:
    name: str
    type: str
    description: str
    mentions: int

@dataclass
class Relationship:
    source: str
    target: str
    relation_type: str
    strength: float
    description: str

@dataclass
class Community:
    id: int
    entities: List[Entity]
    summary: str
    level: int
    parent_community: Optional[int] = None

class GraphRAGSystem:
    def __init__(self, openai_api_key: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        # OpenAI 클라이언트 초기화
        self.openai_client = openai.OpenAI(api_key=openai_api_key)

        # Neo4j 드라이버 초기화
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )

        # NetworkX 그래프 초기화
        self.knowledge_graph = nx.Graph()

        # 커뮤니티 저장소
        self.communities: Dict[int, Community] = {}

        # 텍스트 벡터화
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )

    async def extract_entities_and_relationships(self, document: str) -> tuple[List[Entity], List[Relationship]]:
        """LLM을 사용한 엔티티 및 관계 추출"""

        extraction_prompt = f"""
        다음 텍스트에서 엔티티와 관계를 추출하세요.

        텍스트: {document}

        추출 형식:
        ENTITIES:
        - [엔티티명] | [타입] | [설명]

        RELATIONSHIPS:
        - [소스 엔티티] -> [관계 타입] -> [대상 엔티티] | [강도 0-1] | [설명]

        중요한 엔티티와 관계만 추출하고, 명확하고 구체적으로 작성하세요.
        """

        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 텍스트에서 지식 그래프를 구축하는 전문가입니다."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.1
        )

        return self._parse_extraction_result(response.choices[0].message.content)

    def _parse_extraction_result(self, result: str) -> tuple[List[Entity], List[Relationship]]:
        """추출 결과 파싱"""
        entities = []
        relationships = []

        lines = result.split('\\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('ENTITIES:'):
                current_section = 'entities'
                continue
            elif line.startswith('RELATIONSHIPS:'):
                current_section = 'relationships'
                continue

            if current_section == 'entities' and line.startswith('-'):
                parts = line[1:].split('|')
                if len(parts) >= 3:
                    entities.append(Entity(
                        name=parts[0].strip(),
                        type=parts[1].strip(),
                        description=parts[2].strip(),
                        mentions=1
                    ))

            elif current_section == 'relationships' and '->' in line:
                # [소스] -> [관계] -> [대상] | [강도] | [설명] 파싱
                parts = line.split('|')
                if len(parts) >= 2:
                    rel_parts = parts[0].split('->')
                    if len(rel_parts) >= 3:
                        relationships.append(Relationship(
                            source=rel_parts[0].strip().replace('-', '').strip(),
                            relation_type=rel_parts[1].strip(),
                            target=rel_parts[2].strip(),
                            strength=float(parts[1].strip()) if len(parts) > 1 else 0.5,
                            description=parts[2].strip() if len(parts) > 2 else ""
                        ))

        return entities, relationships

    def build_knowledge_graph(self, entities: List[Entity], relationships: List[Relationship]):
        """지식 그래프 구축"""

        # 엔티티 노드 추가
        for entity in entities:
            self.knowledge_graph.add_node(
                entity.name,
                type=entity.type,
                description=entity.description,
                mentions=entity.mentions
            )

        # 관계 엣지 추가
        for rel in relationships:
            if rel.source in self.knowledge_graph and rel.target in self.knowledge_graph:
                self.knowledge_graph.add_edge(
                    rel.source,
                    rel.target,
                    relation_type=rel.relation_type,
                    strength=rel.strength,
                    description=rel.description
                )

    def detect_communities(self) -> Dict[int, List[str]]:
        """Louvain 알고리즘을 사용한 커뮤니티 탐지"""

        # 가중치가 있는 그래프로 변환
        weighted_graph = nx.Graph()
        for u, v, data in self.knowledge_graph.edges(data=True):
            weight = data.get('strength', 0.5)
            weighted_graph.add_edge(u, v, weight=weight)

        # 커뮤니티 탐지
        partition = community_louvain.best_partition(weighted_graph)

        # 커뮤니티별 노드 그룹화
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)

        return communities

    async def generate_community_summary(self, community_entities: List[str]) -> str:
        """커뮤니티 요약 생성"""

        # 커뮤니티 내 엔티티들의 정보 수집
        entity_info = []
        for entity in community_entities:
            if entity in self.knowledge_graph:
                node_data = self.knowledge_graph.nodes[entity]
                entity_info.append(f"- {entity} ({node_data.get('type', 'Unknown')}): {node_data.get('description', '')}")

        # 관계 정보 수집
        relationships = []
        for i, entity1 in enumerate(community_entities):
            for entity2 in community_entities[i+1:]:
                if self.knowledge_graph.has_edge(entity1, entity2):
                    edge_data = self.knowledge_graph.edges[entity1, entity2]
                    relationships.append(f"- {entity1} --[{edge_data.get('relation_type', 'related')}]--> {entity2}")

        summary_prompt = f"""
        다음 엔티티들과 관계들로 구성된 지식 커뮤니티를 요약하세요:

        엔티티들:
        {chr(10).join(entity_info)}

        관계들:
        {chr(10).join(relationships)}

        이 커뮤니티의 핵심 주제, 주요 인사이트, 그리고 중요한 연결점들을 포함한
        간결하면서도 포괄적인 요약을 작성하세요.
        """

        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 지식 그래프 분석 전문가입니다."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )

        return response.choices[0].message.content

    async def process_documents(self, documents: List[str]):
        """문서 집합을 처리하여 GraphRAG 인덱스 구축"""

        all_entities = []
        all_relationships = []

        # 각 문서에서 엔티티와 관계 추출
        for i, document in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}")
            entities, relationships = await self.extract_entities_and_relationships(document)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # 지식 그래프 구축
        self.build_knowledge_graph(all_entities, all_relationships)

        # 커뮤니티 탐지
        community_nodes = self.detect_communities()

        # 각 커뮤니티에 대한 요약 생성
        for community_id, nodes in community_nodes.items():
            if len(nodes) >= 2:  # 최소 2개 이상의 엔티티
                summary = await self.generate_community_summary(nodes)

                # Community 객체 생성
                community_entities = [
                    Entity(name=node,
                          type=self.knowledge_graph.nodes[node].get('type', 'Unknown'),
                          description=self.knowledge_graph.nodes[node].get('description', ''),
                          mentions=self.knowledge_graph.nodes[node].get('mentions', 1))
                    for node in nodes
                ]

                self.communities[community_id] = Community(
                    id=community_id,
                    entities=community_entities,
                    summary=summary,
                    level=0
                )

        print(f"GraphRAG 인덱스 구축 완료:")
        print(f"- 엔티티: {len(self.knowledge_graph.nodes)}")
        print(f"- 관계: {len(self.knowledge_graph.edges)}")
        print(f"- 커뮤니티: {len(self.communities)}")

    async def query(self, question: str, max_communities: int = 5) -> str:
        """GraphRAG 쿼리 처리"""

        # 질문과 관련된 커뮤니티 찾기
        relevant_communities = await self._find_relevant_communities(question, max_communities)

        # 컨텍스트 구성
        context_parts = []
        for community_id, relevance_score in relevant_communities:
            community = self.communities[community_id]
            context_parts.append(f"커뮤니티 {community_id} (관련도: {relevance_score:.2f}):\\n{community.summary}")

        context = "\\n\\n".join(context_parts)

        # 최종 답변 생성
        answer_prompt = f"""
        다음 지식 커뮤니티 정보를 바탕으로 질문에 답하세요:

        질문: {question}

        지식 컨텍스트:
        {context}

        답변 요구사항:
        1. 제공된 지식을 종합하여 포괄적인 답변을 작성하세요
        2. 여러 관점이 있다면 모두 포함하세요
        3. 구체적인 예시나 증거가 있다면 인용하세요
        4. 확실하지 않은 부분은 명시하세요
        """

        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 지식 그래프 기반 질의응답 전문가입니다."},
                {"role": "user", "content": answer_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        return response.choices[0].message.content

    async def _find_relevant_communities(self, question: str, max_communities: int) -> List[tuple[int, float]]:
        """질문과 관련된 커뮤니티 찾기"""

        # 간단한 TF-IDF 기반 유사도 계산
        question_vector = self.vectorizer.fit_transform([question])

        community_scores = []
        for community_id, community in self.communities.items():
            # 커뮤니티 텍스트 (요약 + 엔티티 이름들)
            community_text = community.summary + " " + " ".join([e.name for e in community.entities])
            community_vector = self.vectorizer.transform([community_text])

            # 코사인 유사도 계산
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(question_vector, community_vector)[0][0]

            community_scores.append((community_id, similarity))

        # 상위 관련 커뮤니티 반환
        community_scores.sort(key=lambda x: x[1], reverse=True)
        return community_scores[:max_communities]

    def save_to_neo4j(self):
        """Neo4j에 지식 그래프 저장"""

        with self.neo4j_driver.session() as session:
            # 기존 데이터 삭제
            session.run("MATCH (n) DETACH DELETE n")

            # 엔티티 노드 생성
            for node_id, node_data in self.knowledge_graph.nodes(data=True):
                session.run(
                    """
                    CREATE (e:Entity {
                        name: $name,
                        type: $type,
                        description: $description,
                        mentions: $mentions
                    })
                    """,
                    name=node_id,
                    type=node_data.get('type', 'Unknown'),
                    description=node_data.get('description', ''),
                    mentions=node_data.get('mentions', 1)
                )

            # 관계 엣지 생성
            for source, target, edge_data in self.knowledge_graph.edges(data=True):
                session.run(
                    """
                    MATCH (a:Entity {name: $source})
                    MATCH (b:Entity {name: $target})
                    CREATE (a)-[:RELATED {
                        type: $relation_type,
                        strength: $strength,
                        description: $description
                    }]->(b)
                    """,
                    source=source,
                    target=target,
                    relation_type=edge_data.get('relation_type', 'related'),
                    strength=edge_data.get('strength', 0.5),
                    description=edge_data.get('description', '')
                )

            # 커뮤니티 정보 저장
            for community_id, community in self.communities.items():
                session.run(
                    """
                    CREATE (c:Community {
                        id: $id,
                        summary: $summary,
                        level: $level
                    })
                    """,
                    id=community_id,
                    summary=community.summary,
                    level=community.level
                )

                # 커뮤니티-엔티티 관계 생성
                for entity in community.entities:
                    session.run(
                        """
                        MATCH (c:Community {id: $community_id})
                        MATCH (e:Entity {name: $entity_name})
                        CREATE (e)-[:BELONGS_TO]->(c)
                        """,
                        community_id=community_id,
                        entity_name=entity.name
                    )

    def close(self):
        """리소스 정리"""
        self.neo4j_driver.close()

# 사용 예시
async def main():
    # GraphRAG 시스템 초기화
    graph_rag = GraphRAGSystem(
        openai_api_key="your-openai-key",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )

    # 문서 처리 (예시)
    documents = [
        "인공지능 기술이 급속도로 발전하면서 많은 산업에 혁신을 가져오고 있다...",
        "기계학습은 데이터에서 패턴을 찾아 예측하는 기술이다...",
        # 더 많은 문서들...
    ]

    # GraphRAG 인덱스 구축
    await graph_rag.process_documents(documents)

    # Neo4j에 저장
    graph_rag.save_to_neo4j()

    # 질의응답 테스트
    response = await graph_rag.query("인공지능이 산업에 미치는 영향은 무엇인가?")
    print("답변:", response)

    # 리소스 정리
    graph_rag.close()

# 실행
if __name__ == "__main__":
    asyncio.run(main())`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
