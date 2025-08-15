'use client';

import Link from 'next/link';
import { FlaskConical } from 'lucide-react';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold text-indigo-800 dark:text-indigo-200 mb-4">
          프롬프트 엔지니어링 마스터
        </h2>
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-6">
          <p className="text-lg text-gray-700 dark:text-gray-300">
            효과적인 프롬프트 설계는 LLM의 성능을 극대화하는 핵심 기술입니다.
          </p>
        </div>
      </section>

      <section>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">프롬프트 기법들</h3>
        
        {/* Prompt Playground 시뮬레이터 링크 */}
        <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-900/20 dark:to-teal-900/20 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-semibold text-green-900 dark:text-green-200 mb-1">🎮 Prompt Engineering Playground</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                다양한 프롬프트 기법을 실험하고 결과를 비교해보세요
              </p>
            </div>
            <Link 
              href="/modules/llm/simulators/prompt-playground"
              className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              <FlaskConical className="w-4 h-4" />
              시뮬레이터 실행
            </Link>
          </div>
        </div>
        
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">Zero-shot Prompting</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">예시 없이 작업 설명만으로 수행</p>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border">
              <pre className="text-sm">다음 텍스트를 한국어로 번역해주세요: "Hello, world!"</pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">Few-shot Prompting</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">몇 개의 예시를 제공하여 패턴 학습</p>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border">
              <pre className="text-sm whitespace-pre-wrap">{`영어 -> 한국어 번역:
Hello -> 안녕하세요
Thank you -> 감사합니다
Goodbye -> 안녕히 가세요

Good morning -> ?`}</pre>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border">
            <h4 className="font-semibold text-indigo-600 dark:text-indigo-400 mb-3">Chain-of-Thought (CoT)</h4>
            <p className="text-gray-700 dark:text-gray-300 mb-3">단계별 추론 과정을 명시적으로 안내</p>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border">
              <pre className="text-sm whitespace-pre-wrap">{`문제를 단계별로 해결해보겠습니다:

1. 주어진 정보 파악
2. 필요한 공식 확인  
3. 계산 수행
4. 결과 검증`}</pre>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}