'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Rocket } from 'lucide-react'
import References from '@/components/common/References'

export default function Section4() {
  return (
    <>
      <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
            <Rocket className="text-purple-600" size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">6.4 RAG의 미래: 2025년과 그 이후</h2>
            <p className="text-gray-600 dark:text-gray-400">차세대 RAG 기술의 발전 방향</p>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
            <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">🚀 2025년 RAG 기술 전망</h3>

            <div className="space-y-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <h4 className="font-medium text-purple-600 dark:text-purple-400 mb-2">1. Agentic RAG</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  RAG 시스템이 단순 검색을 넘어 능동적으로 정보를 수집, 검증, 업데이트하는
                  자율 에이전트로 진화. 필요시 외부 API 호출, 실시간 데이터 수집,
                  정보 신뢰도 자동 평가 등을 수행.
                </p>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <h4 className="font-medium text-blue-600 dark:text-blue-400 mb-2">2. Continual Learning RAG</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  사용자 피드백과 새로운 정보를 실시간으로 학습하여 지속적으로 개선되는 RAG.
                  Catastrophic forgetting 없이 새로운 지식을 통합하고,
                  오래된 정보를 자동으로 업데이트.
                </p>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <h4 className="font-medium text-green-600 dark:text-green-400 mb-2">3. Personalized RAG</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  개인의 선호도, 전문성 수준, 문맥을 이해하여 맞춤형 정보를 제공하는 RAG.
                  프라이버시를 보장하면서도 개인화된 지식 그래프를 구축하고 활용.
                </p>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <h4 className="font-medium text-orange-600 dark:text-orange-400 mb-2">4. Quantum RAG</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  양자 컴퓨팅을 활용한 초고속 벡터 검색과 양자 중첩을 이용한
                  다차원 의미 공간 탐색. 기존 RAG 대비 1000배 이상의 검색 속도 향상 예상.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
            <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">🎯 연구자를 위한 오픈 문제들</h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">이론적 도전과제</h4>
                <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                  <li>• RAG의 이론적 한계 증명</li>
                  <li>• 최적 검색-생성 균형점</li>
                  <li>• 정보 이론적 관점의 RAG</li>
                  <li>• 환각 현상의 수학적 모델링</li>
                </ul>
              </div>

              <div className="bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg">
                <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">실용적 도전과제</h4>
                <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                  <li>• 실시간 지식 업데이트</li>
                  <li>• 다국어 크로스링구얼 RAG</li>
                  <li>• 에너지 효율적 RAG</li>
                  <li>• 엣지 디바이스용 경량 RAG</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Research Papers and Resources */}
      <section className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-6">추천 논문 및 리소스</h2>

        <div className="space-y-4">
          <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
            <h3 className="font-bold mb-4">📚 필독 논문 (2023-2024)</h3>

            <div className="space-y-3">
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-1">Self-RAG: Learning to Retrieve, Generate, and Critique</h4>
                <p className="text-sm opacity-90">Asai et al., 2023 - Washington University</p>
                <a href="#" className="text-xs underline">arXiv:2310.11511</a>
              </div>

              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-1">RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval</h4>
                <p className="text-sm opacity-90">Sarthi et al., 2024 - Stanford University</p>
                <a href="#" className="text-xs underline">arXiv:2401.18059</a>
              </div>

              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-1">Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models</h4>
                <p className="text-sm opacity-90">Jeong et al., 2024 - KAIST</p>
                <a href="#" className="text-xs underline">arXiv:2403.14403</a>
              </div>
            </div>
          </div>

          <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
            <h3 className="font-bold mb-4">🛠️ 오픈소스 프로젝트</h3>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white/10 p-3 rounded">
                <h4 className="font-medium text-sm">LlamaIndex</h4>
                <p className="text-xs opacity-90">최신 RAG 기법 구현체</p>
              </div>
              <div className="bg-white/10 p-3 rounded">
                <h4 className="font-medium text-sm">LangChain</h4>
                <p className="text-xs opacity-90">프로덕션 RAG 파이프라인</p>
              </div>
              <div className="bg-white/10 p-3 rounded">
                <h4 className="font-medium text-sm">RAGAS</h4>
                <p className="text-xs opacity-90">RAG 평가 프레임워크</p>
              </div>
              <div className="bg-white/10 p-3 rounded">
                <h4 className="font-medium text-sm">Haystack</h4>
                <p className="text-xs opacity-90">엔터프라이즈 RAG 솔루션</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 2024년 최신 RAG 논문',
            icon: 'research' as const,
            color: 'border-violet-500',
            items: [
              {
                title: 'Self-RAG: Learning to Retrieve, Generate and Critique',
                authors: 'Asai et al., University of Washington',
                year: '2024',
                description: '자기 성찰 RAG - Reflection Tokens로 검색 필요성 판단, 35% 성능 향상',
                link: 'https://arxiv.org/abs/2310.11511'
              },
              {
                title: 'CRAG: Corrective RAG',
                authors: 'Yan et al., Salesforce',
                year: '2024',
                description: '검색 결과 교정 - Relevance Evaluator, Knowledge Refinement',
                link: 'https://arxiv.org/abs/2401.15884'
              },
              {
                title: 'RAG-Fusion: Next Generation Retrieval',
                authors: 'Rackauckas, MIT',
                year: '2024',
                description: '다중 쿼리 생성 + 상호 순위 융합 - Reciprocal Rank Fusion',
                link: 'https://arxiv.org/abs/2402.03367'
              },
              {
                title: 'HyDE: Hypothetical Document Embeddings',
                authors: 'Gao et al., Microsoft',
                year: '2024',
                description: '가상 문서 생성 후 검색 - Zero-shot Dense Retrieval',
                link: 'https://arxiv.org/abs/2212.10496'
              },
              {
                title: 'RAPTOR: Recursive Tree-based RAG',
                authors: 'Sarthi et al., Stanford',
                year: '2024',
                description: '계층적 문서 요약 - Bottom-up Clustering, Tree Traversal',
                link: 'https://arxiv.org/abs/2401.18059'
              }
            ]
          },
          {
            title: '🔬 Advanced RAG 기법 연구',
            icon: 'research' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Adaptive-RAG: Dynamic Retrieval Strategies',
                authors: 'Jeong et al., KAIST',
                year: '2024',
                description: '질의 복잡도에 따른 동적 검색 - Simple vs Complex Query Classification',
                link: 'https://arxiv.org/abs/2403.14403'
              },
              {
                title: 'Query Rewriting for Retrieval-Augmented LLMs',
                authors: 'Ma et al., Tsinghua',
                year: '2024',
                description: 'LLM 기반 쿼리 재작성 - Rewrite-Retrieve-Read 패턴',
                link: 'https://arxiv.org/abs/2305.14283'
              },
              {
                title: 'Long-Context RAG: Challenges and Solutions',
                authors: 'Liu et al., Meta',
                year: '2024',
                description: '100K+ 토큰 컨텍스트 활용 - Retrieval vs Long-Context 트레이드오프',
                link: 'https://arxiv.org/abs/2404.00909'
              },
              {
                title: 'RAG with Self-Refinement',
                authors: 'Shinn et al., Google DeepMind',
                year: '2024',
                description: '반복적 답변 개선 - Self-Critique, Multi-Round Retrieval',
                link: 'https://arxiv.org/abs/2303.17651'
              }
            ]
          },
          {
            title: '🚀 RAG 오픈소스 & 도구',
            icon: 'tools' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'LlamaIndex Advanced Techniques',
                authors: 'LlamaIndex',
                year: '2024',
                description: 'Router, Sub-Question, Knowledge Graph 통합 RAG',
                link: 'https://docs.llamaindex.ai/en/stable/examples/'
              },
              {
                title: 'LangChain RAG Cookbook',
                authors: 'LangChain',
                year: '2024',
                description: 'Production RAG Patterns - Self-Query, Multi-Query, Parent Document',
                link: 'https://python.langchain.com/docs/use_cases/question_answering/'
              },
              {
                title: 'AutoRAG: Automated RAG Optimization',
                authors: 'Marker-Inc',
                year: '2024',
                description: 'RAG 파이프라인 자동 최적화 - Hyperparameter Tuning',
                link: 'https://github.com/Marker-Inc-Korea/AutoRAG'
              },
              {
                title: 'RAGFlow: Visual RAG Builder',
                authors: 'InfiniFlow',
                year: '2024',
                description: 'No-code RAG 구축 - Workflow Designer, Template Library',
                link: 'https://github.com/infiniflow/ragflow'
              },
              {
                title: 'Flowise: RAG Flow Builder',
                authors: 'FlowiseAI',
                year: '2024',
                description: 'Low-code RAG - Drag & Drop, Multi-modal Support',
                link: 'https://flowiseai.com/'
              }
            ]
          }
        ]}
      />

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced/chapter5"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: RAG 평가와 모니터링
          </Link>

          <Link
            href="/modules/rag/advanced"
            className="inline-flex items-center gap-2 bg-violet-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-violet-600 transition-colors"
          >
            고급 과정 완료
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </>
  )
}
