'use client';

import React from 'react';
import LLMLimitations from './sections/LLMLimitations';
import RAGIntroduction from './sections/RAGIntroduction';
import RAGVsFineTuning from './sections/RAGVsFineTuning';
import RAGExamples from './sections/RAGExamples';

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
    </div>
  );
}