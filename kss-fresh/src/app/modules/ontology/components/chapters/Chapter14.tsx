'use client';

export default function Chapter14() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 14: 온톨로지와 AI의 만남</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            온톨로지는 AI 시스템에 구조화된 지식을 제공하여 더 똑똑하고 설명 가능한 AI를 만듭니다. 
            이번 챕터에서는 온톨로지와 최신 AI 기술의 통합을 살펴봅니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Knowledge-Enhanced AI</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">기존 AI의 한계</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 블랙박스 문제: 결정 과정 설명 불가</li>
              <li>• 상식 부족: 기본적인 지식 결여</li>
              <li>• 일반화 어려움: 새로운 상황 대처 미흡</li>
              <li>• 데이터 의존성: 대량의 학습 데이터 필요</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-3">온톨로지의 해결책</h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 설명 가능성: 추론 과정 추적 가능</li>
              <li>• 지식 주입: 도메인 지식 활용</li>
              <li>• 제약사항: 논리적 일관성 보장</li>
              <li>• 효율성: 적은 데이터로 학습 가능</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">온톨로지 + LLM</h2>
        
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-4">Knowledge-Grounded Language Models</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-medium mb-2">1. Retrieval-Augmented Generation (RAG)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                온톨로지에서 관련 지식을 검색하여 LLM의 프롬프트에 추가
              </p>
              <div className="mt-2 font-mono text-xs bg-gray-50 dark:bg-gray-900 p-2 rounded">
                Query → Ontology Search → Context + Query → LLM → Answer
              </div>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-medium mb-2">2. Constrained Decoding</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                온톨로지의 제약사항을 활용하여 LLM의 출력을 제한
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-medium mb-2">3. Knowledge Distillation</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                온톨로지의 구조화된 지식을 LLM에 학습시키기
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Knowledge Graph Embedding</h2>
        
        <div className="space-y-4">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">임베딩 기법</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div>
                <h4 className="font-medium text-indigo-600 dark:text-indigo-400">TransE</h4>
                <p className="text-sm">관계를 벡터 변환으로 모델링</p>
                <p className="text-xs font-mono mt-1">h + r ≈ t</p>
              </div>
              <div>
                <h4 className="font-medium text-indigo-600 dark:text-indigo-400">ComplEx</h4>
                <p className="text-sm">복소수 공간에서 표현</p>
                <p className="text-xs font-mono mt-1">Re(⟨h, r, t̄⟩)</p>
              </div>
              <div>
                <h4 className="font-medium text-indigo-600 dark:text-indigo-400">ConvE</h4>
                <p className="text-sm">CNN 기반 임베딩</p>
                <p className="text-xs font-mono mt-1">f(vec(h, r) * Ω)</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실제 응용 사례</h2>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-orange-600 dark:text-orange-400 mb-3">1. 의료 진단 AI</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              의료 온톨로지 + 딥러닝으로 정확한 진단과 설명 제공
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 font-mono text-xs">
              환자 증상 → 온톨로지 매칭 → 가능한 질병 후보 → AI 진단 → 근거 설명
            </div>
          </div>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-3">2. 대화형 AI 어시스턴트</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              도메인 온톨로지로 전문적이고 정확한 답변 생성
            </p>
            <div className="bg-gray-50 dark:bg-gray-900 rounded p-4">
              <p className="text-sm">• 법률 상담 챗봇</p>
              <p className="text-sm">• 금융 자문 시스템</p>
              <p className="text-sm">• 교육 튜터링 봇</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Neuro-Symbolic AI</h2>
        
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-3">신경망과 기호 추론의 통합</h3>
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="text-center flex-1">
              <div className="w-20 h-20 mx-auto bg-blue-500 text-white rounded-full flex items-center justify-center mb-2">
                🧠
              </div>
              <p className="font-medium">Neural Networks</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">패턴 인식, 학습</p>
            </div>
            <div className="text-3xl">+</div>
            <div className="text-center flex-1">
              <div className="w-20 h-20 mx-auto bg-green-500 text-white rounded-full flex items-center justify-center mb-2">
                🔤
              </div>
              <p className="font-medium">Symbolic AI</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">논리, 추론</p>
            </div>
            <div className="text-3xl">=</div>
            <div className="text-center flex-1">
              <div className="w-20 h-20 mx-auto bg-purple-500 text-white rounded-full flex items-center justify-center mb-2">
                🚀
              </div>
              <p className="font-medium">Neuro-Symbolic</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">최고의 성능</p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">🔮</span>
          미래 전망
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li>• 대규모 언어 모델과 지식 그래프의 완전한 통합</li>
          <li>• 자동 온톨로지 학습 및 업데이트</li>
          <li>• 멀티모달 지식 표현 (텍스트, 이미지, 비디오)</li>
          <li>• 분산 온톨로지와 연합 학습</li>
          <li>• 양자 컴퓨팅을 활용한 초고속 추론</li>
        </ul>
      </section>
    </div>
  )
}