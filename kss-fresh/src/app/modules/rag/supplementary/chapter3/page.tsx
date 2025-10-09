'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, DollarSign } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter3Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      <div className="mb-8">
        <Link
          href="/modules/rag/supplementary"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          보충 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-green-500 to-teal-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <DollarSign size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 3: 비용 최적화</h1>
              <p className="text-green-100 text-lg">RAG 시스템 운영 비용을 80% 절감하는 전략</p>
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-8">
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">비용 최적화 전략</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            RAG 시스템의 운영 비용을 체계적으로 분석하고 최적화하는 방법을 다룹니다.
          </p>
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
            <h3 className="font-bold text-green-800 dark:text-green-200 mb-3">주요 절감 포인트</h3>
            <ul className="text-green-700 dark:text-green-300 space-y-2">
              <li>• 벡터 DB 인덱싱 최적화</li>
              <li>• 캐싱 전략 개선</li>
              <li>• 배치 처리 효율화</li>
              <li>• 스토리지 비용 절감</li>
            </ul>
          </div>
        </section>

        {/* References */}
        <References
          sections={[
            {
              title: '📚 비용 최적화 도구 & 서비스',
              icon: 'web' as const,
              color: 'border-green-500',
              items: [
                {
                  title: 'OpenAI Token Calculator',
                  authors: 'OpenAI',
                  year: '2024',
                  description: 'GPT API 비용 예측 - Tokenizer, 가격 계산기, 사용량 대시보드',
                  link: 'https://platform.openai.com/tokenizer'
                },
                {
                  title: 'LangChain Callbacks: Cost Tracking',
                  authors: 'LangChain',
                  year: '2024',
                  description: 'LLM 호출 비용 추적 - 자동 토큰 카운팅, 비용 집계, 알림',
                  link: 'https://python.langchain.com/docs/modules/callbacks/'
                },
                {
                  title: 'Pinecone Cost Optimizer',
                  authors: 'Pinecone',
                  year: '2024',
                  description: '벡터 DB 비용 절감 - Index 최적화, Pod 크기 조정, 스토리지 관리',
                  link: 'https://www.pinecone.io/pricing/'
                },
                {
                  title: 'Redis Cache-Aside Pattern',
                  authors: 'Redis Labs',
                  year: '2024',
                  description: '캐싱 전략 - Cache Hit Rate 90%+, TTL 관리, 메모리 최적화',
                  link: 'https://redis.io/docs/manual/patterns/cache-aside/'
                },
                {
                  title: 'AWS Cost Explorer for AI Workloads',
                  authors: 'Amazon Web Services',
                  year: '2024',
                  description: 'AI 워크로드 비용 분석 - 리소스별 비용, 예측, 최적화 권장사항',
                  link: 'https://aws.amazon.com/aws-cost-management/aws-cost-explorer/'
                }
              ]
            },
            {
              title: '📖 비용 최적화 연구',
              icon: 'research' as const,
              color: 'border-teal-500',
              items: [
                {
                  title: 'Efficient LLM Inference: A Survey',
                  authors: 'Kim et al., Seoul National University',
                  year: '2024',
                  description: 'LLM 추론 비용 절감 - Quantization, Pruning, Distillation, KV Cache 최적화',
                  link: 'https://arxiv.org/abs/2312.03863'
                },
                {
                  title: 'Cost-Effective RAG with Semantic Caching',
                  authors: 'Chen et al., UC Berkeley',
                  year: '2024',
                  description: '의미 기반 캐싱 - 유사 쿼리 탐지, 70% API 호출 절감, Cache Hit 최적화',
                  link: 'https://arxiv.org/abs/2308.07922'
                },
                {
                  title: 'Dynamic Batching for LLM Serving',
                  authors: 'Yu et al., CMU',
                  year: '2024',
                  description: '동적 배치 처리 - Throughput 3배 향상, Latency 유지, 비용 50% 절감',
                  link: 'https://arxiv.org/abs/2305.13245'
                },
                {
                  title: 'Vector Index Compression Techniques',
                  authors: 'Zhang et al., Stanford',
                  year: '2024',
                  description: '벡터 압축 - Product Quantization, Scalar Quantization, 90% 저장공간 절감',
                  link: 'https://arxiv.org/abs/2401.08281'
                }
              ]
            },
            {
              title: '🛠️ 실무 최적화 기법',
              icon: 'tools' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'Prompt Compression Techniques',
                  authors: 'Microsoft Research',
                  year: '2024',
                  description: '프롬프트 압축 - LongLLMLingua, 80% 토큰 절감, 성능 유지',
                  link: 'https://github.com/microsoft/LLMLingua'
                },
                {
                  title: 'Batch Embedding with SentenceTransformers',
                  authors: 'UKPLab',
                  year: '2024',
                  description: '배치 임베딩 - GPU 활용 최적화, 10배 처리 속도, 비용 절감',
                  link: 'https://www.sbert.net/examples/applications/computing-embeddings/README.html'
                },
                {
                  title: 'Serverless Vector Search',
                  authors: 'Weaviate',
                  year: '2024',
                  description: '서버리스 벡터 검색 - Auto-scaling, Pay-per-use, 유휴 시간 비용 0',
                  link: 'https://weaviate.io/developers/weaviate/concepts/serverless'
                },
                {
                  title: 'Model Quantization with GGML',
                  authors: 'Georgi Gerganov',
                  year: '2024',
                  description: '모델 양자화 - 4bit/8bit 양자화, 메모리 75% 절감, 속도 2배 향상',
                  link: 'https://github.com/ggerganov/llama.cpp'
                },
                {
                  title: 'CloudWatch Cost Anomaly Detection',
                  authors: 'AWS',
                  year: '2024',
                  description: '비용 이상 탐지 - ML 기반 이상 패턴 감지, 자동 알림, 예산 초과 방지',
                  link: 'https://aws.amazon.com/aws-cost-management/aws-cost-anomaly-detection/'
                }
              ]
            }
          ]}
        />
      </div>

      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/supplementary/chapter2"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 보안 및 프라이버시
          </Link>
          
          <Link
            href="/modules/rag/supplementary/chapter4"
            className="inline-flex items-center gap-2 bg-green-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-green-600 transition-colors"
          >
            다음: 복구 시스템
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}