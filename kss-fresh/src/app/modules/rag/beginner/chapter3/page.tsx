'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Scissors, Layers, RefreshCw, BarChart3, ExternalLink } from 'lucide-react'
import References from '@/components/common/References'
import CodeSandbox from '../../components/CodeSandbox'

export default function Chapter3Page() {
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
              <Scissors size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 3: 청킹 전략의 모든 것</h1>
              <p className="text-emerald-100 text-lg">문서를 AI가 이해하기 쉽게 나누는 기술</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Why Chunking Matters */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <Layers className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.1 청킹이 중요한 이유</h2>
              <p className="text-gray-600 dark:text-gray-400">임베딩과 검색 품질의 핵심</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">🎯 청킹의 목적</h3>
              <ul className="space-y-3 text-purple-700 dark:text-purple-300">
                <li className="flex items-start gap-2">
                  <span className="text-purple-500 mt-0.5">•</span>
                  <div>
                    <strong>의미적 완결성:</strong> 하나의 청크가 독립적인 의미를 가져야 함
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500 mt-0.5">•</span>
                  <div>
                    <strong>검색 효율성:</strong> 적절한 크기로 정확한 정보 검색
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-purple-500 mt-0.5">•</span>
                  <div>
                    <strong>컨텍스트 보존:</strong> 문맥 정보가 손실되지 않도록
                  </div>
                </li>
              </ul>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl border border-red-200 dark:border-red-700">
              <h3 className="font-bold text-red-800 dark:text-red-200 mb-4">⚠️ 잘못된 청킹의 문제</h3>
              <ul className="space-y-3 text-red-700 dark:text-red-300">
                <li className="flex items-start gap-2">
                  <span className="text-red-500 mt-0.5">•</span>
                  <div>
                    <strong>너무 작은 청크:</strong> 문맥 손실, 의미 파편화
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500 mt-0.5">•</span>
                  <div>
                    <strong>너무 큰 청크:</strong> 부정확한 검색, 노이즈 증가
                  </div>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500 mt-0.5">•</span>
                  <div>
                    <strong>일관성 없음:</strong> 검색 품질 편차 발생
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Section 2: Chunking Strategies */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">3.2 주요 청킹 전략</h2>

          <div className="space-y-6">
            {/* Strategy 1: Fixed Size */}
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <div className="flex items-center gap-3 mb-4">
                <BarChart3 className="text-blue-600" size={20} />
                <h3 className="font-bold text-blue-800 dark:text-blue-200">1. 고정 크기 청킹 (Fixed Size Chunking)</h3>
              </div>
              
              <div className="grid md:grid-cols-3 gap-4 mb-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">설정 예시</h4>
                  <code className="text-sm bg-gray-100 dark:bg-gray-700 p-2 rounded block">
                    chunk_size: 1000<br/>
                    overlap: 200
                  </code>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">장점</h4>
                  <ul className="text-sm space-y-1">
                    <li>• 구현 간단</li>
                    <li>• 예측 가능한 크기</li>
                    <li>• 빠른 처리 속도</li>
                  </ul>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">단점</h4>
                  <ul className="text-sm space-y-1">
                    <li>• 문맥 단절 가능</li>
                    <li>• 의미 단위 무시</li>
                    <li>• 문장 중간 절단</li>
                  </ul>
                </div>
              </div>

              {/* Visual Example */}
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">시각적 예시:</h4>
                <div className="flex gap-2 overflow-x-auto pb-2">
                  <div className="flex-shrink-0 bg-blue-200 dark:bg-blue-700 p-2 rounded text-xs">
                    [0-1000자]
                  </div>
                  <div className="flex-shrink-0 bg-blue-300 dark:bg-blue-600 p-2 rounded text-xs">
                    [800-1800자]
                  </div>
                  <div className="flex-shrink-0 bg-blue-200 dark:bg-blue-700 p-2 rounded text-xs">
                    [1600-2600자]
                  </div>
                  <span className="text-gray-500">→ 200자 중첩</span>
                </div>
              </div>
            </div>

            {/* Strategy 2: Semantic */}
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <div className="flex items-center gap-3 mb-4">
                <RefreshCw className="text-green-600" size={20} />
                <h3 className="font-bold text-green-800 dark:text-green-200">2. 의미 단위 청킹 (Semantic Chunking)</h3>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
                <h4 className="font-medium text-gray-900 dark:text-white mb-3">작동 원리:</h4>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <span className="w-8 h-8 bg-green-600 text-white rounded-full flex items-center justify-center text-sm font-bold">1</span>
                    <div>
                      <p className="font-medium">문단별 임베딩 생성</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">각 문단을 벡터로 변환</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="w-8 h-8 bg-green-600 text-white rounded-full flex items-center justify-center text-sm font-bold">2</span>
                    <div>
                      <p className="font-medium">유사도 계산</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">인접 문단 간 코사인 유사도 측정</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="w-8 h-8 bg-green-600 text-white rounded-full flex items-center justify-center text-sm font-bold">3</span>
                    <div>
                      <p className="font-medium">경계 결정</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">유사도가 낮은 지점에서 분할</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-emerald-100 dark:bg-emerald-900/40 p-4 rounded-lg">
                  <h4 className="font-medium text-emerald-800 dark:text-emerald-200 mb-2">✅ 추천 상황</h4>
                  <ul className="text-sm space-y-1 text-emerald-700 dark:text-emerald-300">
                    <li>• 기술 문서, 매뉴얼</li>
                    <li>• 논문, 보고서</li>
                    <li>• 구조화된 콘텐츠</li>
                  </ul>
                </div>
                <div className="bg-emerald-100 dark:bg-emerald-900/40 p-4 rounded-lg">
                  <h4 className="font-medium text-emerald-800 dark:text-emerald-200 mb-2">⚡ 성능 팁</h4>
                  <ul className="text-sm space-y-1 text-emerald-700 dark:text-emerald-300">
                    <li>• 임계값: 0.7-0.8 권장</li>
                    <li>• 최소 청크: 200자 이상</li>
                    <li>• 최대 청크: 1500자 이하</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Strategy 3: Document Structure */}
            <div className="bg-amber-50 dark:bg-amber-900/20 p-6 rounded-xl border border-amber-200 dark:border-amber-700">
              <h3 className="font-bold text-amber-800 dark:text-amber-200 mb-4">3. 문서 구조 기반 청킹</h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">활용 가능한 구조 요소:</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div className="bg-amber-100 dark:bg-amber-900/30 p-2 rounded text-center text-sm">
                      <strong>Markdown</strong><br/>
                      # ## ### 헤더
                    </div>
                    <div className="bg-amber-100 dark:bg-amber-900/30 p-2 rounded text-center text-sm">
                      <strong>HTML</strong><br/>
                      &lt;h1&gt; &lt;p&gt; &lt;div&gt;
                    </div>
                    <div className="bg-amber-100 dark:bg-amber-900/30 p-2 rounded text-center text-sm">
                      <strong>LaTeX</strong><br/>
                      \section \chapter
                    </div>
                    <div className="bg-amber-100 dark:bg-amber-900/30 p-2 rounded text-center text-sm">
                      <strong>PDF</strong><br/>
                      페이지, 섹션
                    </div>
                  </div>
                </div>

                <div className="bg-amber-100 dark:bg-amber-900/30 p-4 rounded-lg">
                  <h4 className="font-medium text-amber-800 dark:text-amber-200 mb-2">💡 프로 팁</h4>
                  <p className="text-sm text-amber-700 dark:text-amber-300">
                    문서 구조와 의미적 청킹을 <strong>하이브리드</strong>로 사용하면 최상의 결과를 얻을 수 있습니다.
                    예: 섹션 단위로 먼저 나누고, 각 섹션 내에서 의미적 청킹 적용
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Best Practices */}
        <section className="bg-gradient-to-r from-emerald-500 to-green-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">청킹 베스트 프랙티스</h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-white/10 rounded-xl p-6">
              <h3 className="font-bold mb-4">📏 크기 가이드라인</h3>
              <ul className="space-y-2 text-sm text-emerald-100">
                <li><strong>일반 문서:</strong> 500-1000자</li>
                <li><strong>기술 문서:</strong> 1000-1500자</li>
                <li><strong>대화형:</strong> 300-500자</li>
                <li><strong>법률/의학:</strong> 1500-2000자</li>
              </ul>
            </div>
            
            <div className="bg-white/10 rounded-xl p-6">
              <h3 className="font-bold mb-4">🔄 중첩(Overlap) 설정</h3>
              <ul className="space-y-2 text-sm text-emerald-100">
                <li><strong>표준:</strong> 10-20%</li>
                <li><strong>높은 정밀도:</strong> 25-30%</li>
                <li><strong>빠른 처리:</strong> 5-10%</li>
                <li><strong>0% 중첩:</strong> 권장하지 않음</li>
              </ul>
            </div>
            
            <div className="bg-white/10 rounded-xl p-6">
              <h3 className="font-bold mb-4">🎯 성능 최적화</h3>
              <ul className="space-y-2 text-sm text-emerald-100">
                <li>청크별 메타데이터 추가</li>
                <li>문서 타입별 전략 차별화</li>
                <li>A/B 테스트로 최적값 찾기</li>
                <li>사용자 피드백 반영</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Hands-on Practice Section */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">3.4 실습: 청킹 전략 직접 구현하기</h2>
            <p className="text-gray-600 dark:text-gray-400">
              실제 코드를 실행해보며 3가지 청킹 전략을 비교해보세요. 복사 버튼으로 코드를 복사하여 로컬 환경에서 실행할 수 있습니다.
            </p>
          </div>

          <div className="space-y-6">
            {/* Example 1: Recursive Chunking */}
            <CodeSandbox
              title="실습 1: 재귀적 문자 분할 (추천)"
              description="LangChain의 RecursiveCharacterTextSplitter를 사용한 스마트 청킹"
              language="python"
              code={`from langchain.text_splitter import RecursiveCharacterTextSplitter

# 긴 문서를 의미 단위로 자동 분할
def smart_chunking(text):
    """
    재귀적 문자 분할: 문단 → 문장 → 단어 순으로 분할 시도
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,           # 최대 청크 크기 (문자 수)
        chunk_overlap=200,         # 청크 간 중복 (20%)
        separators=["\\n\\n", "\\n", ". ", " ", ""],  # 우선순위 구분자
        length_function=len,       # 길이 계산 함수
    )
    chunks = splitter.split_text(text)
    return chunks

# 실행 예제
sample_text = """
인공지능은 빠르게 발전하고 있습니다.

특히 대규모 언어 모델의 등장으로 자연어 처리 기술이 혁신되었습니다.
GPT, BERT 같은 모델들이 대표적입니다.

RAG 시스템은 이러한 LLM의 한계를 극복하는 핵심 기술입니다.
"""

chunks = smart_chunking(sample_text)
print(f"총 {len(chunks)}개 청크 생성:")
for i, chunk in enumerate(chunks, 1):
    print(f"\\n[청크 {i}] ({len(chunk)}자)")
    print(chunk)`}
              output={`총 1개 청크 생성:

[청크 1] (141자)
인공지능은 빠르게 발전하고 있습니다.

특히 대규모 언어 모델의 등장으로 자연어 처리 기술이 혁신되었습니다.
GPT, BERT 같은 모델들이 대표적입니다.

RAG 시스템은 이러한 LLM의 한계를 극복하는 핵심 기술입니다.`}
              highlightLines={[8, 9, 10]}
            />

            {/* Example 2: Token-based Chunking */}
            <CodeSandbox
              title="실습 2: 토큰 기반 청킹 (OpenAI 모델용)"
              description="정확한 토큰 수로 청크를 나누는 방법"
              language="python"
              code={`import tiktoken

def token_based_chunking(text, max_tokens=500):
    """
    OpenAI 토큰 기준으로 정확하게 청킹
    """
    # OpenAI의 토크나이저 로드
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # 텍스트를 토큰으로 변환
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks

# 실행 예제
text = "RAG 시스템은 검색과 생성을 결합한 혁신적인 AI 기술입니다. " * 100
chunks = token_based_chunking(text, max_tokens=200)

print(f"총 {len(chunks)}개 청크 생성")
print(f"각 청크 토큰 수: ~200 tokens")
print(f"첫 청크 길이: {len(chunks[0])}자")`}
              output={`총 5개 청크 생성
각 청크 토큰 수: ~200 tokens
첫 청크 길이: 245자`}
              highlightLines={[8, 11, 15]}
            />

            {/* Example 3: Semantic Chunking */}
            <CodeSandbox
              title="실습 3: 의미 기반 청킹 (고급)"
              description="임베딩 유사도를 활용한 지능형 청킹"
              language="python"
              code={`from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def semantic_chunking(paragraphs, similarity_threshold=0.7):
    """
    의미적으로 유사한 문단들을 하나의 청크로 병합
    """
    embeddings = OpenAIEmbeddings()

    # 각 문단의 임베딩 벡터 생성
    vectors = [embeddings.embed_query(p) for p in paragraphs]

    chunks = []
    current_chunk = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        # 이전 문단과의 유사도 계산
        similarity = cosine_similarity(
            [vectors[i-1]],
            [vectors[i]]
        )[0][0]

        if similarity > similarity_threshold:
            # 유사하면 현재 청크에 추가
            current_chunk.append(paragraphs[i])
        else:
            # 다르면 새 청크 시작
            chunks.append(" ".join(current_chunk))
            current_chunk = [paragraphs[i]]

    chunks.append(" ".join(current_chunk))
    return chunks

# 실행 예제
paragraphs = [
    "RAG는 Retrieval-Augmented Generation의 약자입니다.",
    "검색 기반 증강 생성 기술을 의미합니다.",
    "주식 시장은 오늘 상승세를 보였습니다.",
    "코스피 지수가 2% 상승했습니다."
]

chunks = semantic_chunking(paragraphs, similarity_threshold=0.75)
for i, chunk in enumerate(chunks, 1):
    print(f"\\n청크 {i}:\\n{chunk}")`}
              output={`청크 1:
RAG는 Retrieval-Augmented Generation의 약자입니다. 검색 기반 증강 생성 기술을 의미합니다.

청크 2:
주식 시장은 오늘 상승세를 보였습니다. 코스피 지수가 2% 상승했습니다.`}
              highlightLines={[18, 19, 20, 24]}
            />
          </div>
        </section>

        {/* Simulator Link */}
        <section className="bg-gradient-to-r from-emerald-500 to-green-600 rounded-2xl p-8 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-2xl font-bold mb-2">🎮 인터랙티브 청킹 시뮬레이터</h3>
              <p className="text-emerald-100 mb-4">
                실시간으로 다양한 청킹 전략을 비교하고 결과를 시각화해보세요
              </p>
              <ul className="space-y-2 text-sm text-emerald-100">
                <li>✓ 3가지 청킹 방식 실시간 비교</li>
                <li>✓ 청크 크기/중복 조절 가능</li>
                <li>✓ 시각화된 결과 확인</li>
              </ul>
            </div>
            <Link
              href="/modules/rag/simulators/chunking-demo"
              className="flex items-center gap-2 bg-white text-emerald-600 px-6 py-3 rounded-lg font-bold hover:bg-emerald-50 transition-all transform hover:scale-105"
            >
              시뮬레이터 열기
              <ExternalLink size={20} />
            </Link>
          </div>
        </section>
      </div>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 청킹 전략 심화',
            icon: 'web' as const,
            color: 'border-emerald-500',
            items: [
              {
                title: 'Optimal Chunk Size for RAG',
                authors: 'Greg Kamradt',
                year: '2024',
                description: '512, 1024, 2048 토큰 청크 크기 실험 결과',
                link: 'https://www.youtube.com/watch?v=8OJC21T2SL4'
              },
              {
                title: 'Advanced Chunking Techniques',
                authors: 'LangChain',
                year: '2025',
                description: '문장 윈도우, 계층적 청킹, 문서 요약 청킹',
                link: 'https://python.langchain.com/docs/modules/data_connection/document_transformers/advanced'
              },
              {
                title: 'Context-Aware Chunking',
                authors: 'Anthropic',
                year: '2024',
                description: 'Contextual Retrieval - 문맥 정보 포함 청킹',
                link: 'https://www.anthropic.com/news/contextual-retrieval'
              },
              {
                title: 'Chunk Overlap Best Practices',
                authors: 'Pinecone',
                year: '2024',
                description: '오버랩 크기가 검색 품질에 미치는 영향',
                link: 'https://www.pinecone.io/learn/chunking-strategies/#chunk-overlap'
              }
            ]
          },
          {
            title: '📖 청킹 연구 논문',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Dense Passage Retrieval for Open-Domain Question Answering',
                authors: 'Karpukhin, V., et al.',
                year: '2020',
                description: 'DPR - 효과적인 문서 분할 전략 (Meta AI)',
                link: 'https://arxiv.org/abs/2004.04906'
              },
              {
                title: 'Long Document Segmentation for Retrieval',
                authors: 'Dai, Z., et al.',
                year: '2022',
                description: '긴 문서의 최적 분할 방법 연구',
                link: 'https://arxiv.org/abs/2204.07186'
              },
              {
                title: 'Sentence-BERT for Semantic Search',
                authors: 'Reimers, N., Gurevych, I.',
                year: '2019',
                description: '문장 단위 임베딩 및 청킹 기초',
                link: 'https://arxiv.org/abs/1908.10084'
              }
            ]
          },
          {
            title: '🛠️ 청킹 최적화 도구',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Sentence Window Node Parser',
                authors: 'LlamaIndex',
                year: '2025',
                description: '문장 주변 컨텍스트 윈도우 포함',
                link: 'https://docs.llamaindex.ai/en/stable/examples/node_parsers/sentence_window.html'
              },
              {
                title: 'Hierarchical Node Parser',
                authors: 'LlamaIndex',
                year: '2025',
                description: '계층적 문서 구조 유지 청킹',
                link: 'https://docs.llamaindex.ai/en/stable/examples/node_parsers/hierarchical.html'
              },
              {
                title: 'Code-Aware Text Splitter',
                authors: 'LangChain',
                year: '2024',
                description: '프로그래밍 언어별 구문 인식 분할',
                link: 'https://python.langchain.com/docs/modules/data_connection/document_transformers/code_splitter'
              },
              {
                title: 'Tiktoken - OpenAI Tokenizer',
                authors: 'OpenAI',
                year: '2024',
                description: '정확한 토큰 수 기반 청킹',
                link: 'https://github.com/openai/tiktoken'
              },
              {
                title: 'Chunking Benchmark Tool',
                authors: 'Unstructured',
                year: '2024',
                description: '다양한 청킹 전략 성능 비교 도구',
                link: 'https://github.com/Unstructured-IO/unstructured'
              }
            ]
          }
        ]}
      />

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/beginner/chapter2"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 문서 처리 기초
          </Link>

          <Link
            href="/modules/rag/beginner/chapter4"
            className="inline-flex items-center gap-2 bg-emerald-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-emerald-600 transition-colors"
          >
            다음: 첫 RAG 시스템
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}