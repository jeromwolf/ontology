'use client';

import React from 'react';
import LLMLimitations from './sections/LLMLimitations';
import RAGIntroduction from './sections/RAGIntroduction';
import RAGVsFineTuning from './sections/RAGVsFineTuning';
import RAGExamples from './sections/RAGExamples';
import References from '@/components/common/References';

// Chapter 1: RAG의 필요성과 개념
export default function Chapter1() {
  return (
    <div className="space-y-8">
      {/* 페이지 헤더 */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-4">Chapter 1: RAG가 필요한 이유</h1>
        <p className="text-lg text-gray-600 dark:text-gray-400">
          LLM의 한계를 극복하고 실시간 정보와 전문 지식을 활용하는 차세대 AI 기술
        </p>
      </div>

      {/* 섹션 컴포넌트들 */}
      <LLMLimitations />
      <RAGIntroduction />
      <RAGVsFineTuning />
      <RAGExamples />

      {/* 학습 요약 */}
      <section className="bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 rounded-xl p-6 mt-8">
        <h2 className="text-xl font-bold mb-4 text-emerald-800 dark:text-emerald-200">📚 이 챕터에서 배운 것</h2>
        <ul className="space-y-2">
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">LLM의 4가지 주요 한계점 (할루시네이션, 최신성, 도메인 지식, 소스 추적)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">RAG의 핵심 작동 원리 (Retrieval → Augmentation → Generation)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">RAG와 Fine-tuning의 차이점과 각각의 장단점</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-600 dark:text-emerald-400 mt-0.5">✓</span>
            <span className="text-gray-700 dark:text-gray-300">실제 서비스에서 활용되는 RAG 시스템 사례</span>
          </li>
        </ul>
      </section>

      {/* 다음 챕터 안내 */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 text-center">
        <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">🚀 다음 챕터</h3>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          Chapter 2에서는 RAG 시스템의 첫 단계인 문서 처리와 청킹 전략을 실습을 통해 배워보겠습니다.
        </p>
        <button className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          다음 챕터로
        </button>
      </div>

      <References
        sections={[
          {
            title: '원본 논문 (Original Papers)',
            icon: 'paper',
            color: 'border-emerald-500',
            items: [
              {
                title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
                authors: 'Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela',
                year: '2020',
                description: 'RAG를 최초로 제안한 역사적 논문 - Meta AI Research',
                link: 'https://arxiv.org/abs/2005.11401'
              },
              {
                title: 'In-Context Retrieval-Augmented Language Models',
                authors: 'Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, Yoav Shoham',
                year: '2023',
                description: 'In-Context Learning과 RAG를 결합한 최신 연구',
                link: 'https://arxiv.org/abs/2302.00083'
              },
              {
                title: 'Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection',
                authors: 'Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi',
                year: '2023',
                description: 'Self-reflection을 통한 RAG 품질 향상 기법',
                link: 'https://arxiv.org/abs/2310.11511'
              },
              {
                title: 'Dense Passage Retrieval for Open-Domain Question Answering',
                authors: 'Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih',
                year: '2020',
                description: '밀집 벡터 검색의 기초가 된 DPR 논문',
                link: 'https://arxiv.org/abs/2004.04906'
              }
            ]
          },
          {
            title: 'LLM 한계 연구 (LLM Limitations Research)',
            icon: 'paper',
            color: 'border-red-500',
            items: [
              {
                title: 'On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?',
                authors: 'Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell',
                year: '2021',
                description: 'LLM의 근본적 한계와 위험성을 다룬 선구적 논문',
                link: 'https://dl.acm.org/doi/10.1145/3442188.3445922'
              },
              {
                title: 'Survey of Hallucination in Natural Language Generation',
                authors: 'Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, Pascale Fung',
                year: '2023',
                description: 'LLM 할루시네이션 현상에 대한 포괄적 서베이',
                link: 'https://arxiv.org/abs/2202.03629'
              },
              {
                title: 'Retrieval Augmentation Reduces Hallucination in Conversation',
                authors: 'Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, Jason Weston',
                year: '2021',
                description: 'RAG를 통한 할루시네이션 감소 효과 검증',
                link: 'https://arxiv.org/abs/2104.07567'
              }
            ]
          },
          {
            title: '실무 가이드 (Practical Guides)',
            icon: 'book',
            color: 'border-blue-500',
            items: [
              {
                title: 'LangChain RAG Documentation',
                description: '가장 널리 사용되는 RAG 프레임워크의 공식 문서',
                link: 'https://python.langchain.com/docs/use_cases/question_answering/'
              },
              {
                title: 'LlamaIndex: Data Framework for LLM Applications',
                description: 'RAG 시스템 구축을 위한 종합 프레임워크',
                link: 'https://docs.llamaindex.ai/'
              },
              {
                title: 'Pinecone: Vector Database Guide',
                description: '벡터 데이터베이스와 RAG 구현 실무 가이드',
                link: 'https://www.pinecone.io/learn/retrieval-augmented-generation/'
              },
              {
                title: 'OpenAI: RAG Best Practices',
                description: 'OpenAI 공식 RAG 구현 모범 사례',
                link: 'https://platform.openai.com/docs/guides/retrieval-augmented-generation'
              }
            ]
          },
          {
            title: '산업 응용 사례 (Industry Applications)',
            icon: 'web',
            color: 'border-purple-500',
            items: [
              {
                title: 'Microsoft Bing: Combining Search and LLM',
                description: 'Microsoft의 검색 엔진과 LLM 통합 사례',
                link: 'https://blogs.microsoft.com/blog/2023/02/07/reinventing-search-with-a-new-ai-powered-microsoft-bing-and-edge-your-copilot-for-the-web/'
              },
              {
                title: 'Perplexity AI: Conversational Search Engine',
                description: 'RAG 기반 대화형 검색 엔진의 선두주자',
                link: 'https://www.perplexity.ai/'
              },
              {
                title: 'Notion AI: Document-based Q&A',
                description: '문서 기반 AI 어시스턴트 구현 사례',
                link: 'https://www.notion.so/product/ai'
              }
            ]
          }
        ]}
      />
    </div>
  );
}