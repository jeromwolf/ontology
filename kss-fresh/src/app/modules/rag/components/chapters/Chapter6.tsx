'use client'

import Link from 'next/link'
import { Network, Sparkles } from 'lucide-react'

// Chapter 6: Advanced RAG
export default function Chapter6() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">고급 RAG 기법</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          기본 RAG를 넘어서 더 높은 성능과 정확도를 달성하기 위한 고급 기법들을 살펴봅니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Multi-hop Reasoning</h2>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700 mb-6">
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            복잡한 질문에 답하기 위해 여러 단계의 검색과 추론을 거치는 기법입니다.
          </p>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="bg-emerald-100 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-200 px-2 py-1 rounded text-sm font-medium">
                Step 1
              </span>
              <div>
                <strong>초기 검색</strong>: 질문과 직접 관련된 문서 검색
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="bg-emerald-100 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-200 px-2 py-1 rounded text-sm font-medium">
                Step 2
              </span>
              <div>
                <strong>추가 질문 생성</strong>: 초기 결과를 바탕으로 추가 정보가 필요한 부분 파악
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="bg-emerald-100 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-200 px-2 py-1 rounded text-sm font-medium">
                Step 3
              </span>
              <div>
                <strong>반복 검색</strong>: 추가 질문으로 더 깊은 정보 검색
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="bg-emerald-100 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-200 px-2 py-1 rounded text-sm font-medium">
                Step 4
              </span>
              <div>
                <strong>종합 답변</strong>: 모든 정보를 종합하여 최종 답변 생성
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Reranking 전략</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Cross-Encoder Reranking</h3>
            <p className="text-gray-700 dark:text-gray-300 text-sm mb-3">
              쿼리와 문서를 함께 인코딩하여 더 정확한 관련성 점수 계산
            </p>
            <div className="bg-gray-100 dark:bg-gray-800 rounded p-3 text-sm">
              <strong>장점</strong>: 높은 정확도<br/>
              <strong>단점</strong>: 계산 비용 높음
            </div>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">LLM-based Reranking</h3>
            <p className="text-gray-700 dark:text-gray-300 text-sm mb-3">
              LLM을 사용하여 검색 결과의 관련성을 재평가
            </p>
            <div className="bg-gray-100 dark:bg-gray-800 rounded p-3 text-sm">
              <strong>장점</strong>: 문맥 이해 우수<br/>
              <strong>단점</strong>: 지연 시간 증가
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RAG 시스템 평가</h2>
        <div className="bg-gradient-to-r from-emerald-50 to-green-50 dark:from-gray-800/50 dark:to-gray-900/50 rounded-lg p-6">
          <h3 className="font-semibold text-gray-900 dark:text-white mb-4">RAGAS 평가 프레임워크</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">Context Relevancy</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                검색된 컨텍스트가 질문과 얼마나 관련있는지
              </p>
            </div>
            <div>
              <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">Answer Relevancy</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                생성된 답변이 질문에 얼마나 적절한지
              </p>
            </div>
            <div>
              <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">Faithfulness</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                답변이 제공된 컨텍스트에 얼마나 충실한지
              </p>
            </div>
            <div>
              <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-2">Answer Correctness</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                답변의 사실적 정확성
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">GraphRAG: 지식 그래프 기반 RAG</h2>
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-4">Microsoft GraphRAG의 혁신</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            GraphRAG는 문서를 단순한 텍스트 청크가 아닌 <strong>지식 그래프</strong>로 변환하여 
            더 깊은 이해와 추론을 가능하게 합니다.
          </p>
          
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-medium text-purple-700 dark:text-purple-300 mb-2">일반 RAG의 한계</h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 단순 키워드 매칭에 의존</li>
                <li>• 문서 간 관계 파악 어려움</li>
                <li>• 전체적인 맥락 이해 부족</li>
                <li>• 복잡한 추론 불가능</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-medium text-purple-700 dark:text-purple-300 mb-2">GraphRAG의 강점</h4>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 엔티티와 관계 기반 검색</li>
                <li>• 다중 홉 추론 가능</li>
                <li>• 전체 문서의 구조적 이해</li>
                <li>• 커뮤니티 기반 요약</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-purple-100 dark:bg-purple-900/30 rounded-lg p-6 mb-6">
          <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-3 flex items-center gap-2">
            <Sparkles className="w-5 h-5" />
            GraphRAG 체험하기
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            직접 문서를 지식 그래프로 변환하고 Neo4j 스타일의 시각화를 체험해보세요!
          </p>
          <Link 
            href="/modules/rag/simulators/graphrag-explorer"
            className="inline-flex items-center px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
          >
            <Network className="w-4 h-4 mr-2" />
            GraphRAG 탐색기 시작하기
          </Link>
        </div>

        <div className="space-y-6">
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">GraphRAG 파이프라인</h3>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 px-3 py-1 rounded-full text-sm font-medium">
                  1
                </span>
                <div>
                  <strong>엔티티 추출</strong>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    LLM을 사용하여 문서에서 인물, 조직, 장소, 개념 등 주요 엔티티 추출
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 px-3 py-1 rounded-full text-sm font-medium">
                  2
                </span>
                <div>
                  <strong>관계 추출</strong>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    엔티티 간의 관계를 파악하고 타입 지정 (예: "근무하다", "소유하다", "위치하다")
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 px-3 py-1 rounded-full text-sm font-medium">
                  3
                </span>
                <div>
                  <strong>그래프 구축</strong>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Neo4j 등의 그래프 DB에 엔티티(노드)와 관계(엣지) 저장
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 px-3 py-1 rounded-full text-sm font-medium">
                  4
                </span>
                <div>
                  <strong>커뮤니티 감지</strong>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Leiden 알고리즘으로 관련 엔티티를 클러스터링하여 주제별 그룹 형성
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <span className="bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 px-3 py-1 rounded-full text-sm font-medium">
                  5
                </span>
                <div>
                  <strong>계층적 요약</strong>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    각 커뮤니티와 전체 그래프에 대한 요약 생성
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">GraphRAG 쿼리 유형</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-4">
                <h4 className="font-medium text-indigo-700 dark:text-indigo-300 mb-2">글로벌 쿼리</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  전체 문서 컬렉션에 대한 포괄적인 질문
                </p>
                <div className="bg-white dark:bg-gray-800 rounded p-2 text-xs font-mono">
                  "이 회사의 전체적인 사업 전략은?"
                </div>
              </div>
              
              <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-4">
                <h4 className="font-medium text-indigo-700 dark:text-indigo-300 mb-2">로컬 쿼리</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                  특정 엔티티나 관계에 대한 세부 질문
                </p>
                <div className="bg-white dark:bg-gray-800 rounded p-2 text-xs font-mono">
                  "홍길동과 협업한 모든 프로젝트는?"
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Neo4j와의 통합</h3>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
              <pre className="text-sm overflow-x-auto">
{`# GraphRAG with Neo4j 예시
from neo4j import GraphDatabase
import openai

class GraphRAG:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def extract_entities_relations(self, text):
        # LLM을 사용한 엔티티/관계 추출
        prompt = f"""
        텍스트에서 엔티티와 관계를 추출하세요.
        
        텍스트: {text}
        
        형식:
        엔티티: [(이름, 타입), ...]
        관계: [(주체, 관계, 객체), ...]
        """
        
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt
        )
        return parse_response(response)
    
    def build_graph(self, entities, relations):
        with self.driver.session() as session:
            # 엔티티(노드) 생성
            for name, entity_type in entities:
                session.run(
                    "MERGE (e:Entity {name: $name, type: $type})",
                    name=name, type=entity_type
                )
            
            # 관계(엣지) 생성
            for subj, rel, obj in relations:
                session.run(f"""
                    MATCH (a:Entity {{name: $subj}})
                    MATCH (b:Entity {{name: $obj}})
                    MERGE (a)-[r:{rel}]->(b)
                """, subj=subj, obj=obj)
    
    def query_graph(self, question):
        # 질문을 Cypher 쿼리로 변환
        cypher = self.question_to_cypher(question)
        
        with self.driver.session() as session:
            result = session.run(cypher)
            return process_result(result)`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실시간 업데이트 아키텍처</h2>
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">증분 인덱싱</h3>
            <p className="text-gray-700 dark:text-gray-300 text-sm">
              새로운 문서만 임베딩하고 인덱싱하여 효율성 향상. 
              변경된 문서는 이전 버전을 삭제하고 새로 추가.
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">버전 관리</h3>
            <p className="text-gray-700 dark:text-gray-300 text-sm">
              문서의 버전을 추적하여 시점별 정보 제공 가능.
              타임스탬프와 버전 번호를 메타데이터로 저장.
            </p>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">캐시 전략</h3>
            <p className="text-gray-700 dark:text-gray-300 text-sm">
              자주 검색되는 쿼리와 결과를 캐싱하여 응답 속도 향상.
              문서 업데이트 시 관련 캐시 무효화.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}