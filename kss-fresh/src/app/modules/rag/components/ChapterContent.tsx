'use client'

import { ReactNode } from 'react'
import Link from 'next/link'
import { Network, Sparkles } from 'lucide-react'
import ChunkingDemo from './ChunkingDemo'
import EmbeddingVisualizer from './EmbeddingVisualizer'
import VectorSearchDemo from './VectorSearchDemo'

interface ChapterContentProps {
  chapterId: string
}

// 챕터별 콘텐츠를 렌더링하는 컴포넌트
export default function ChapterContent({ chapterId }: ChapterContentProps) {
  // 챕터별 콘텐츠 매핑
  const renderContent = (): ReactNode => {
    switch (chapterId) {
      case '01-what-is-rag':
        return <WhatIsRAGContent />
      case '02-document-processing':
        return <DocumentProcessingContent />
      case '03-embeddings':
        return <EmbeddingsContent />
      case '04-vector-search':
        return <VectorSearchContent />
      case '05-answer-generation':
        return <AnswerGenerationContent />
      case '06-advanced-rag':
        return <AdvancedRAGContent />
      default:
        return <ComingSoonContent />
    }
  }

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {renderContent()}
    </div>
  )
}

// Chapter 1: What is RAG?
function WhatIsRAGContent() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">LLM의 한계</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          대규모 언어 모델(LLM)은 놀라운 능력을 보여주지만, 몇 가지 근본적인 한계가 있습니다:
        </p>
        
        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2">할루시네이션</h3>
            <p className="text-gray-700 dark:text-gray-300">
              학습하지 않은 정보에 대해 그럴듯하지만 틀린 답변을 생성하는 현상
            </p>
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">최신 정보 부족</h3>
            <p className="text-gray-700 dark:text-gray-300">
              학습 데이터 기준일 이후의 정보는 알 수 없음
            </p>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">도메인 특화 지식</h3>
            <p className="text-gray-700 dark:text-gray-300">
              기업 내부 문서나 특정 도메인 지식은 학습되지 않음
            </p>
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">소스 추적 불가</h3>
            <p className="text-gray-700 dark:text-gray-300">
              생성된 답변의 출처를 확인할 수 없어 신뢰성 검증 어려움
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RAG의 등장</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          RAG(Retrieval-Augmented Generation)는 이러한 LLM의 한계를 극복하기 위해 등장했습니다.
          외부 지식 베이스에서 관련 정보를 검색하여 LLM에 제공함으로써 더 정확하고 신뢰할 수 있는 답변을 생성합니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-xl p-6">
          <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-4">RAG의 핵심 아이디어</h3>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">1.</span>
              <div>
                <strong>검색(Retrieval)</strong>: 사용자 질문과 관련된 문서를 지식 베이스에서 찾기
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">2.</span>
              <div>
                <strong>증강(Augmentation)</strong>: 검색된 문서를 LLM의 컨텍스트로 제공
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-emerald-600 dark:text-emerald-400 font-bold">3.</span>
              <div>
                <strong>생성(Generation)</strong>: 컨텍스트를 바탕으로 정확한 답변 생성
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RAG vs Fine-tuning</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-800">
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">비교 항목</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">RAG</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">Fine-tuning</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">지식 업데이트</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-green-600 dark:text-green-400">실시간 가능</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-red-600 dark:text-red-400">재학습 필요</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800/50">
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">비용</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-green-600 dark:text-green-400">상대적으로 저렴</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-red-600 dark:text-red-400">GPU 비용 높음</td>
              </tr>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">소스 추적</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-green-600 dark:text-green-400">가능</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-red-600 dark:text-red-400">불가능</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800/50">
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 font-medium">정확도 제어</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-green-600 dark:text-green-400">문서 기반 100%</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-yellow-600 dark:text-yellow-400">확률적</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실제 RAG 시스템 사례</h2>
        <div className="grid gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Microsoft Copilot</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Office 문서, 이메일, 캘린더 등 기업 데이터를 활용한 업무 보조 AI
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Perplexity AI</h3>
            <p className="text-gray-600 dark:text-gray-400">
              실시간 웹 검색을 통해 최신 정보를 제공하는 AI 검색 엔진
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">ChatGPT with Browsing</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Bing 검색을 통해 실시간 정보를 보강한 답변 생성
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

// Chapter 2: Document Processing
function DocumentProcessingContent() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">문서 처리의 중요성</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          RAG 시스템의 성능은 문서를 얼마나 효과적으로 처리하고 저장하는지에 크게 좌우됩니다.
          적절한 청킹 전략은 검색 정확도와 답변 품질에 직접적인 영향을 미칩니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">청킹 전략 체험하기</h2>
        <ChunkingDemo />
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">다양한 문서 형식 처리</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">구조화된 문서</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• PDF: PyPDF2, pdfplumber로 텍스트 추출</li>
              <li>• Word: python-docx로 단락별 처리</li>
              <li>• HTML: BeautifulSoup으로 태그 파싱</li>
              <li>• Markdown: 헤더 기준 자동 분할</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">비구조화된 문서</h3>
            <ul className="space-y-2 text-gray-700 dark:text-gray-300">
              <li>• 일반 텍스트: 문장/단락 기준 분할</li>
              <li>• 코드: AST 파싱으로 의미 단위 추출</li>
              <li>• 테이블: 행/열 구조 보존</li>
              <li>• 이미지: OCR + 캡션 생성</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}

// Chapter 3: Embeddings
function EmbeddingsContent() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">임베딩이란?</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          임베딩은 텍스트를 고차원 벡터 공간의 점으로 변환하는 과정입니다.
          의미가 유사한 텍스트는 벡터 공간에서 가까이 위치하게 됩니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">임베딩 시각화</h2>
        <EmbeddingVisualizer />
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">임베딩 모델 비교</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-800">
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">모델</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">차원</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">특징</th>
                <th className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left">비용</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">OpenAI text-embedding-3</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">1536-3072</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">높은 정확도</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">$0.00002/1K tokens</td>
              </tr>
              <tr className="bg-gray-50 dark:bg-gray-800/50">
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">Cohere embed-v3</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">1024</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">다국어 지원</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">$0.00010/1K tokens</td>
              </tr>
              <tr>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">BGE-M3</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">1024</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">오픈소스</td>
                <td className="border border-gray-300 dark:border-gray-600 px-4 py-2">무료 (자체 호스팅)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}

// Chapter 4: Vector Search
function VectorSearchContent() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">벡터 검색의 원리</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          벡터 검색은 고차원 공간에서 가장 가까운 이웃을 찾는 과정입니다.
          쿼리 벡터와 문서 벡터 간의 거리를 계산하여 가장 유사한 문서를 찾습니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">벡터 검색 실습</h2>
        <VectorSearchDemo />
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">주요 벡터 데이터베이스 비교</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Pinecone</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 완전 관리형 서비스</li>
              <li>✓ 실시간 업데이트</li>
              <li>✓ 하이브리드 검색 지원</li>
              <li>✗ 클라우드 전용</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Weaviate</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 오픈소스</li>
              <li>✓ GraphQL API</li>
              <li>✓ 모듈식 아키텍처</li>
              <li>✓ 온프레미스 가능</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Chroma</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ 경량 임베디드 DB</li>
              <li>✓ Python 네이티브</li>
              <li>✓ 개발자 친화적</li>
              <li>✗ 대규모 확장 제한</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Qdrant</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>✓ Rust 기반 고성능</li>
              <li>✓ 필터링 기능 강력</li>
              <li>✓ 클라우드 & 온프레미스</li>
              <li>✓ gRPC API</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  )
}

// Chapter 5: Answer Generation
function AnswerGenerationContent() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">효과적인 프롬프트 설계</h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          RAG 시스템에서 LLM에게 검색된 컨텍스트를 효과적으로 전달하는 것이 중요합니다.
          프롬프트 설계는 답변의 품질을 크게 좌우합니다.
        </p>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RAG 프롬프트 템플릿</h2>
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 font-mono text-sm">
          <pre className="whitespace-pre-wrap">
{`시스템: 당신은 주어진 컨텍스트를 바탕으로 질문에 답변하는 AI 어시스턴트입니다.

컨텍스트:
{context}

질문: {query}

지침:
1. 컨텍스트에 있는 정보만을 사용하여 답변하세요.
2. 확실하지 않은 경우 "주어진 정보로는 답변할 수 없습니다"라고 말하세요.
3. 답변에 사용한 정보의 출처를 명시하세요.

답변:`}
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">컨텍스트 관리 전략</h2>
        <div className="grid gap-4">
          <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">
              1. 컨텍스트 순서 최적화
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              가장 관련성 높은 문서를 앞쪽에 배치합니다. LLM은 프롬프트의 시작과 끝 부분에 더 주의를 기울입니다.
            </p>
          </div>
          
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">
              2. 메타데이터 활용
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              문서의 출처, 날짜, 저자 등 메타데이터를 포함하여 신뢰성을 높입니다.
            </p>
          </div>
          
          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6">
            <h3 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">
              3. 컨텍스트 압축
            </h3>
            <p className="text-gray-700 dark:text-gray-300">
              토큰 제한을 고려하여 핵심 정보만 추출하거나 요약하여 전달합니다.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">답변 품질 향상 기법</h2>
        <ul className="space-y-3 text-gray-700 dark:text-gray-300">
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400">•</span>
            <div>
              <strong>Chain-of-Thought</strong>: 단계별 추론 과정을 포함하도록 유도
            </div>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400">•</span>
            <div>
              <strong>Self-Consistency</strong>: 여러 번 생성하여 일관된 답변 선택
            </div>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400">•</span>
            <div>
              <strong>Citation</strong>: 답변에 사용된 소스를 명확히 인용
            </div>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400">•</span>
            <div>
              <strong>Confidence Score</strong>: 답변의 확신도를 함께 제공
            </div>
          </li>
        </ul>
      </section>
    </div>
  )
}

// Chapter 6: Advanced RAG
function AdvancedRAGContent() {
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

// Coming Soon
function ComingSoonContent() {
  return (
    <div className="text-center py-16">
      <div className="text-6xl mb-4">🚧</div>
      <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
        콘텐츠 준비 중
      </h2>
      <p className="text-gray-600 dark:text-gray-400">
        이 챕터의 콘텐츠는 곧 업데이트될 예정입니다.
      </p>
    </div>
  )
}