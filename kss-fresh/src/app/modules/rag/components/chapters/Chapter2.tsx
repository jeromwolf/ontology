'use client';

import ChunkingDemo from '../ChunkingDemo';
import References from '@/components/common/References';

// Chapter 2: Document Processing
export default function Chapter2() {
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

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-emerald-800 dark:text-emerald-200">📚 이 챕터에서 배운 것</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">다양한 문서 형식 처리 기법 (PDF, Word, HTML, Markdown)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">청킹 전략의 종류 (고정 크기, 의미 단위, 중첩 방식)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">메타데이터 보존과 활용 방법</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">전처리 최적화 및 성능 향상 기법</span>
          </li>
        </ul>
      </section>

      <References
        sections={[
          {
            title: '📚 핵심 논문 (Core Papers)',
            icon: 'paper',
            color: 'border-emerald-500',
            items: [
              {
                title: 'Lost in the Middle: How Language Models Use Long Contexts',
                authors: 'Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang',
                year: '2023',
                description: '긴 컨텍스트에서 정보 손실 문제와 청킹의 중요성을 분석한 연구',
                link: 'https://arxiv.org/abs/2307.03172'
              },
              {
                title: 'Precise Zero-Shot Dense Retrieval without Relevance Labels',
                authors: 'Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan',
                year: '2023',
                description: '효과적인 문서 분할이 검색 성능에 미치는 영향 분석',
                link: 'https://arxiv.org/abs/2212.10496'
              },
              {
                title: 'Text Splitting Methods for RAG Systems',
                authors: 'Various Contributors',
                year: '2024',
                description: '다양한 텍스트 분할 방법론과 성능 비교 연구',
                link: 'https://python.langchain.com/docs/how_to/recursive_text_splitter'
              }
            ]
          },
          {
            title: '🛠️ 실무 도구 & 라이브러리 (Tools & Libraries)',
            icon: 'tools',
            color: 'border-blue-500',
            items: [
              {
                title: 'LangChain Text Splitters',
                description: '11가지 전문 텍스트 분할 도구 (RecursiveCharacter, Semantic, Token-based 등)',
                link: 'https://python.langchain.com/docs/modules/data_connection/document_transformers/'
              },
              {
                title: 'LlamaIndex Node Parser',
                description: '의미론적 청킹과 계층적 문서 구조 파싱',
                link: 'https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/'
              },
              {
                title: 'Unstructured.io',
                description: 'PDF, Word, PPT, 이미지 등 20+ 형식 자동 파싱',
                link: 'https://unstructured.io/'
              },
              {
                title: 'PyPDF2 & pdfplumber',
                description: 'PDF 텍스트 추출 및 레이아웃 보존 도구',
                link: 'https://github.com/py-pdf/pypdf'
              },
              {
                title: 'python-docx',
                description: 'Word 문서 구조 보존 파싱',
                link: 'https://python-docx.readthedocs.io/'
              }
            ]
          },
          {
            title: '📖 청킹 전략 가이드 (Chunking Strategy Guides)',
            icon: 'book',
            color: 'border-purple-500',
            items: [
              {
                title: 'Pinecone: Chunking Strategies for LLM Applications',
                description: '고정 크기 vs 의미 단위 vs 중첩 방식 비교 및 실습 가이드',
                link: 'https://www.pinecone.io/learn/chunking-strategies/'
              },
              {
                title: 'Greg Kamradt: 5 Levels of Text Splitting',
                description: '캐릭터 → 토큰 → 문장 → 의미 → Agent 기반 분할까지',
                link: 'https://github.com/FullStackRetrieval-com/RetrievalTutorials'
              },
              {
                title: 'OpenAI Cookbook: Text Embedding',
                description: '토큰 제한을 고려한 최적 청크 크기 결정 방법',
                link: 'https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb'
              },
              {
                title: 'Weaviate: Chunking Best Practices',
                description: '메타데이터 보존과 컨텍스트 윈도우 최적화',
                link: 'https://weaviate.io/blog/chunking-strategies'
              }
            ]
          },
          {
            title: '⚡ 성능 최적화 (Performance Optimization)',
            icon: 'web',
            color: 'border-orange-500',
            items: [
              {
                title: 'Benchmarking Chunk Sizes',
                description: '128, 256, 512, 1024 토큰 청크 크기별 성능 벤치마크',
                link: 'https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5'
              },
              {
                title: 'Semantic Chunking with Embeddings',
                description: '임베딩 유사도 기반 자동 경계 감지',
                link: 'https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker'
              },
              {
                title: 'Contextual Chunk Headers',
                description: '각 청크에 문서 컨텍스트 자동 추가하여 검색 성능 30% 향상',
                link: 'https://d-id.com/blog/improving-rag-with-contextual-chunk-headers/'
              }
            ]
          }
        ]}
      />
    </div>
  )
}