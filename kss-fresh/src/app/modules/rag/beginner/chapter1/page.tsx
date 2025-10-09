'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, BookOpen, AlertTriangle, Code, Database } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter1Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/beginner"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          초급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-green-500 to-emerald-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <BookOpen size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 1: LLM의 한계점 이해하기</h1>
              <p className="text-emerald-100 text-lg">왜 RAG가 필요한지 체험해보세요</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Hallucination */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
              <AlertTriangle className="text-red-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.1 환각(Hallucination) 현상</h2>
              <p className="text-gray-600 dark:text-gray-400">LLM이 그럴듯한 거짓말을 하는 이유</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl border border-red-200 dark:border-red-700">
              <h3 className="font-bold text-red-800 dark:text-red-200 mb-3">실제 사례 체험</h3>
              <p className="text-red-700 dark:text-red-300 mb-4">
                다음 질문들을 ChatGPT나 Claude에게 물어보세요. 매우 그럴듯하게 답하지만 모두 가짜입니다.
              </p>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <p className="font-medium text-gray-900 dark:text-white mb-2">❌ 존재하지 않는 정보 질문:</p>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• "2024년 노벨물리학상 수상자의 연구 내용을 설명해주세요"</li>
                    <li>• "김철수 교수의 'Quantum RAG Theory' 논문을 요약해주세요"</li>
                    <li>• "서울대학교 AI연구소의 2025년 연구 계획은?"</li>
                  </ul>
                </div>
                
                <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg border border-emerald-200 dark:border-emerald-700">
                  <p className="font-medium text-emerald-800 dark:text-emerald-200 mb-2">✅ RAG가 있다면:</p>
                  <p className="text-sm text-emerald-700 dark:text-emerald-300">
                    "죄송합니다. 제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 정직하게 답변할 것입니다.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-3">왜 이런 일이 발생할까요?</h3>
              <div className="space-y-3 text-blue-700 dark:text-blue-300">
                <div className="flex items-start gap-3">
                  <span className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold mt-0.5">1</span>
                  <p className="text-sm"><strong>확률적 생성:</strong> LLM은 다음 단어를 확률로 예측하므로, 가장 그럴듯한 조합을 생성</p>
                </div>
                <div className="flex items-start gap-3">
                  <span className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold mt-0.5">2</span>
                  <p className="text-sm"><strong>학습 데이터 한계:</strong> 학습 시점 이후의 정보나 내부 문서는 알 수 없음</p>
                </div>
                <div className="flex items-start gap-3">
                  <span className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs font-bold mt-0.5">3</span>
                  <p className="text-sm"><strong>패턴 매칭:</strong> 비슷한 패턴을 조합해서 새로운 정보를 만들어냄</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: Knowledge Cutoff */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-orange-100 dark:bg-orange-900/20 flex items-center justify-center">
              <Database className="text-orange-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.2 실시간 정보의 부재</h2>
              <p className="text-gray-600 dark:text-gray-400">LLM은 과거에 멈춰있습니다</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl border border-orange-200 dark:border-orange-700">
              <h3 className="font-bold text-orange-800 dark:text-orange-200 mb-4">❌ LLM이 모르는 것들</h3>
              <ul className="space-y-2 text-sm text-orange-700 dark:text-orange-300">
                <li>• 오늘의 코스피 지수</li>
                <li>• 현재 비트코인 가격</li>
                <li>• 어제 발표된 애플 신제품</li>
                <li>• 이번주 날씨 예보</li>
                <li>• 방금 업데이트된 회사 정책</li>
              </ul>
            </div>
            
            <div className="bg-emerald-50 dark:bg-emerald-900/20 p-6 rounded-xl border border-emerald-200 dark:border-emerald-700">
              <h3 className="font-bold text-emerald-800 dark:text-emerald-200 mb-4">✅ RAG로 해결 가능</h3>
              <ul className="space-y-2 text-sm text-emerald-700 dark:text-emerald-300">
                <li>• 실시간 API에서 주식 시세 조회</li>
                <li>• 최신 뉴스 사이트 크롤링</li>
                <li>• 회사 내부 데이터베이스 검색</li>
                <li>• 방금 업로드된 문서 검색</li>
                <li>• 개인 파일에서 정보 추출</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Section 3: Enterprise Knowledge */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <Code className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">1.3 기업 내부 지식의 활용 불가</h2>
              <p className="text-gray-600 dark:text-gray-400">가장 중요한 정보는 공개되지 않습니다</p>
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
            <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">실제 비즈니스 시나리오</h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <p className="font-medium text-gray-900 dark:text-white mb-2">🏢 법무팀 질문:</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    "우리 회사 계약서 템플릿에서 Force Majeure 조항은?"
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <p className="font-medium text-gray-900 dark:text-white mb-2">👥 인사팀 질문:</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    "연차 사용 규정 중 이월 가능 일수는?"
                  </p>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <p className="font-medium text-gray-900 dark:text-white mb-2">💻 개발팀 질문:</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    "우리 API의 rate limiting 정책은?"
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <p className="font-medium text-gray-900 dark:text-white mb-2">📈 영업팀 질문:</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    "작년 3분기 매출 데이터와 전년 대비 성장률은?"
                  </p>
                </div>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-emerald-100 dark:bg-emerald-900/40 rounded-lg border border-emerald-200 dark:border-emerald-700">
              <p className="text-emerald-800 dark:text-emerald-200 font-medium">
                💡 <strong>RAG의 핵심 가치:</strong> 이런 내부 지식을 LLM과 연결하여 직원들이 쉽게 정보를 찾을 수 있게 하는 것입니다.
              </p>
            </div>
          </div>
        </section>

        {/* Section 4: Summary */}
        <section className="bg-gradient-to-r from-emerald-500 to-green-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">핵심 요약</h2>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-white/10 rounded-xl p-4">
              <h3 className="font-bold mb-3">🚫 LLM의 한계</h3>
              <ul className="text-sm space-y-1 text-emerald-100">
                <li>• 환각 현상</li>
                <li>• 지식 컷오프</li>
                <li>• 내부 정보 접근 불가</li>
              </ul>
            </div>

            <div className="bg-white/10 rounded-xl p-4">
              <h3 className="font-bold mb-3">✅ RAG의 해결책</h3>
              <ul className="text-sm space-y-1 text-emerald-100">
                <li>• 외부 지식 검색</li>
                <li>• 실시간 정보 활용</li>
                <li>• 신뢰할 수 있는 답변</li>
              </ul>
            </div>

            <div className="bg-white/10 rounded-xl p-4">
              <h3 className="font-bold mb-3">🎯 다음 단계</h3>
              <ul className="text-sm space-y-1 text-emerald-100">
                <li>• RAG 기본 원리 학습</li>
                <li>• 파이프라인 구조 이해</li>
                <li>• 실습 프로젝트 시작</li>
              </ul>
            </div>
          </div>
        </section>

        {/* References */}
        <References
          sections={[
            {
              title: '📚 공식 문서 & 튜토리얼',
              icon: 'web' as const,
              color: 'border-emerald-500',
              items: [
                {
                  title: 'LangChain RAG Tutorial',
                  authors: 'LangChain',
                  year: '2025',
                  description: 'RAG 구현 완전 가이드 - 문서 로딩부터 답변 생성까지',
                  link: 'https://python.langchain.com/docs/tutorials/rag'
                },
                {
                  title: 'OpenAI RAG Guide',
                  authors: 'OpenAI',
                  year: '2024',
                  description: 'GPT-4를 활용한 프로덕션 RAG 시스템 구축',
                  link: 'https://platform.openai.com/docs/guides/retrieval-augmented-generation'
                },
                {
                  title: 'Anthropic Prompt Engineering',
                  authors: 'Anthropic',
                  year: '2025',
                  description: 'Claude를 활용한 환각 방지 프롬프트 기법',
                  link: 'https://docs.anthropic.com/claude/docs/prompt-engineering'
                },
                {
                  title: 'Vector Database Comparison',
                  authors: 'Awesome AI',
                  year: '2024',
                  description: 'Pinecone, Weaviate, Qdrant, Chroma 비교 분석',
                  link: 'https://github.com/facebookresearch/faiss/wiki'
                }
              ]
            },
            {
              title: '📖 핵심 논문',
              icon: 'research' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
                  authors: 'Lewis, P., Perez, E., et al.',
                  year: '2020',
                  description: 'RAG의 원조 논문 - Meta AI Research',
                  link: 'https://arxiv.org/abs/2005.11401'
                },
                {
                  title: 'Lost in the Middle: How Language Models Use Long Contexts',
                  authors: 'Liu, N., Lin, K., et al.',
                  year: '2023',
                  description: 'LLM 컨텍스트 길이의 한계 연구 - Stanford',
                  link: 'https://arxiv.org/abs/2307.03172'
                },
                {
                  title: 'Self-RAG: Learning to Retrieve, Generate and Critique',
                  authors: 'Asai, A., Wu, Z., et al.',
                  year: '2023',
                  description: '자체 검증 메커니즘을 갖춘 차세대 RAG',
                  link: 'https://arxiv.org/abs/2310.11511'
                },
                {
                  title: 'Active Retrieval Augmented Generation',
                  authors: 'Jiang, Z., Xu, F., et al.',
                  year: '2023',
                  description: '동적 검색 전략을 통한 RAG 효율성 향상',
                  link: 'https://arxiv.org/abs/2305.06983'
                }
              ]
            },
            {
              title: '🛠️ 실전 리소스',
              icon: 'tools' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'LlamaIndex RAG Tutorial',
                  authors: 'LlamaIndex',
                  year: '2025',
                  description: '5분만에 RAG 시스템 구축하기',
                  link: 'https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html'
                },
                {
                  title: 'Pinecone RAG Handbook',
                  authors: 'Pinecone',
                  year: '2024',
                  description: '벡터 검색 기반 RAG 완전 정복',
                  link: 'https://www.pinecone.io/learn/retrieval-augmented-generation/'
                },
                {
                  title: 'Weaviate RAG Examples',
                  authors: 'Weaviate',
                  year: '2024',
                  description: '실전 코드 예제 모음 (Python, JavaScript)',
                  link: 'https://weaviate.io/developers/weaviate/tutorials'
                },
                {
                  title: 'Hugging Face RAG Tutorial',
                  authors: 'Hugging Face',
                  year: '2024',
                  description: 'Transformers 라이브러리로 RAG 구현',
                  link: 'https://huggingface.co/docs/transformers/model_doc/rag'
                },
                {
                  title: 'LangChain RAG Cookbook',
                  authors: 'LangChain Community',
                  year: '2025',
                  description: '80+ RAG 패턴과 레시피 모음',
                  link: 'https://github.com/langchain-ai/langchain/tree/master/cookbook'
                }
              ]
            }
          ]}
        />
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/beginner"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            초급 과정으로
          </Link>
          
          <Link
            href="/modules/rag/beginner/chapter2"
            className="inline-flex items-center gap-2 bg-emerald-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-emerald-600 transition-colors"
          >
            다음: 문서 처리와 청킹
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}