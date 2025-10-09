'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, BookOpen, MessageSquare, Brain, AlertCircle, Repeat, Lightbulb } from 'lucide-react'
import References from '@/components/common/References'
import CodeSandbox from '../../components/CodeSandbox'

export default function Chapter3Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/intermediate"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          중급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <MessageSquare size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 3: RAG를 위한 프롬프트 엔지니어링</h1>
              <p className="text-pink-100 text-lg">검색 증강 생성을 최적화하는 프롬프트 전략</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Chain of Thought for Retrieval */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <Brain className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.1 검색을 위한 Chain of Thought</h2>
              <p className="text-gray-600 dark:text-gray-400">단계별 사고를 통한 검색 품질 향상</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">기본 CoT 검색 프롬프트</h3>
              
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border mb-4">
                <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-2">❌ 단순한 접근</p>
                <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
                  <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono whitespace-pre-wrap">
{`사용자 질문: {query}
검색된 문서: {documents}
위 정보를 바탕으로 답변하세요.`}
                  </pre>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-2">✅ Chain of Thought 접근</p>
                <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
                  <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono whitespace-pre-wrap">
{`당신은 검색 증강 AI 어시스턴트입니다. 다음 단계를 따라 사용자 질문에 답변하세요:

사용자 질문: {query}

검색된 문서:
{documents}

답변 프로세스:
1. 먼저 사용자의 핵심 의도를 파악하세요
2. 검색된 각 문서의 관련성을 평가하세요
3. 가장 관련성 높은 정보를 추출하세요
4. 정보 간의 모순이나 차이점이 있는지 확인하세요
5. 종합적인 답변을 구성하세요

단계별 분석:
<thinking>
1. 사용자 의도: [여기에 분석]
2. 문서별 관련성: 
   - 문서1: [관련도 및 핵심 정보]
   - 문서2: [관련도 및 핵심 정보]
3. 정보 종합: [추출한 핵심 정보들]
4. 모순 확인: [있다면 설명, 없다면 "없음"]
</thinking>

최종 답변:
[종합적이고 명확한 답변]`}
                  </pre>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
              <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">고급 CoT 기법: Self-Ask</h3>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
                <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`class SelfAskRAGPrompt:
    def __init__(self):
        self.template = """
사용자 질문을 분석하고 필요한 하위 질문들을 생성한 후, 
검색된 정보를 활용해 답변하세요.

원본 질문: {original_query}

<self_ask>
이 질문에 답하려면 어떤 정보가 필요한가?
1. [하위 질문 1]
2. [하위 질문 2]
3. [하위 질문 3]
</self_ask>

각 하위 질문에 대한 검색 결과:
{sub_query_results}

<integration>
하위 답변들을 어떻게 통합할 것인가?
- 정보의 신뢰도 평가
- 시간적 순서나 인과관계 고려
- 모순되는 정보 처리 방법
</integration>

최종 답변:
[통합된 종합 답변]
"""
    
    def generate_sub_queries(self, original_query):
        # LLM을 사용해 하위 질문 생성
        prompt = f"""
다음 질문을 답하기 위해 필요한 3-5개의 구체적인 하위 질문을 생성하세요:
"{original_query}"

하위 질문들:
"""
        return self.llm.generate(prompt)
    
    def search_and_answer(self, query):
        # 1. 하위 질문 생성
        sub_queries = self.generate_sub_queries(query)
        
        # 2. 각 하위 질문에 대해 검색
        sub_results = []
        for sub_q in sub_queries:
            docs = self.retriever.search(sub_q, k=3)
            sub_results.append({
                "question": sub_q,
                "documents": docs
            })
        
        # 3. 통합 답변 생성
        final_prompt = self.template.format(
            original_query=query,
            sub_query_results=sub_results
        )
        
        return self.llm.generate(final_prompt)`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: Few-shot Prompting with Context */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-pink-100 dark:bg-pink-900/20 flex items-center justify-center">
              <Repeat className="text-pink-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.2 컨텍스트를 활용한 Few-shot 프롬프팅</h2>
              <p className="text-gray-600 dark:text-gray-400">예시를 통한 RAG 응답 품질 향상</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-pink-50 dark:bg-pink-900/20 p-6 rounded-xl border border-pink-200 dark:border-pink-700">
              <h3 className="font-bold text-pink-800 dark:text-pink-200 mb-4">효과적인 Few-shot 예시 구성</h3>
              
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                <div className="bg-slate-50 dark:bg-slate-800 p-3 rounded border border-slate-200 dark:border-slate-700">
                  <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono whitespace-pre-wrap">
{`시스템: 당신은 검색된 문서를 바탕으로 정확하고 도움이 되는 답변을 제공하는 AI입니다.
다음 예시들을 참고하여 답변 스타일을 학습하세요.

예시 1:
질문: "Python에서 리스트와 튜플의 차이점은?"
검색 문서:
- 문서1: "리스트는 가변(mutable) 객체로 생성 후 수정 가능..."
- 문서2: "튜플은 불변(immutable) 객체로 한번 생성되면..."
- 문서3: "메모리 효율성 면에서 튜플이 리스트보다..."

답변: 검색된 문서를 바탕으로 Python의 리스트와 튜플의 주요 차이점을 설명하겠습니다:

1. **가변성 (Mutability)**
   - 리스트: 가변 객체로 요소 추가, 삭제, 수정 가능
   - 튜플: 불변 객체로 생성 후 변경 불가

2. **메모리 효율성**
   - 튜플이 리스트보다 메모리를 적게 사용
   - 고정된 데이터에는 튜플이 효율적

[출처: 문서1, 문서2, 문서3]

예시 2:
질문: "RESTful API의 주요 원칙은?"
검색 문서:
- 문서1: "REST는 6가지 제약 조건을 따르는..."
- 문서2: "Stateless 특성으로 서버는 클라이언트 상태를..."
- 문서3: "Uniform Interface는 리소스 식별과..."

답변: 검색 결과를 종합하여 RESTful API의 핵심 원칙을 정리하면:

1. **무상태성 (Stateless)**
   - 각 요청은 독립적이며 서버는 클라이언트 상태를 저장하지 않음

2. **균일한 인터페이스 (Uniform Interface)**
   - 리소스는 URI로 식별
   - 표준 HTTP 메서드 사용 (GET, POST, PUT, DELETE)

3. **클라이언트-서버 분리**
   - 관심사의 분리로 확장성 향상

[출처: 문서1, 문서2, 문서3]

---

이제 실제 질문에 답변하세요:
질문: {user_query}
검색 문서:
{retrieved_documents}

답변:`}
                  </pre>
                </div>
              </div>
            </div>

            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl border border-indigo-200 dark:border-indigo-700">
              <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">도메인별 Few-shot 템플릿</h3>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">🏥 의료 도메인</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                    특징: 정확성 강조, 주의사항 포함, 전문 용어 설명
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
                    <pre>
{`답변 형식:
1. 의학적 정의
2. 증상/원인
3. 치료 방법
4. ⚠️ 주의사항
[참고 문헌 명시]`}
                    </pre>
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">⚖️ 법률 도메인</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                    특징: 조항 인용, 판례 참조, 면책 조항
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
                    <pre>
{`답변 형식:
1. 관련 법령
2. 핵심 내용
3. 판례/해석
4. 💡 실무 팁
[법령/판례 출처]`}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: System Prompts Optimization */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
              <Lightbulb className="text-green-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.3 시스템 프롬프트 최적화</h2>
              <p className="text-gray-600 dark:text-gray-400">RAG 시스템의 기본 동작 정의</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">RAG 최적화된 시스템 프롬프트</h3>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
                <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`SYSTEM_PROMPT = """
당신은 검색 증강 생성(RAG) 시스템을 갖춘 AI 어시스턴트입니다.

핵심 원칙:
1. **정확성**: 검색된 문서에 있는 정보만을 사용하여 답변
2. **투명성**: 정보의 출처를 명확히 표시
3. **완전성**: 관련된 모든 정보를 종합적으로 제공
4. **신뢰성**: 불확실한 정보는 그 불확실성을 명시

답변 가이드라인:

## 정보 처리
- 검색된 문서를 신중히 분석하여 관련 정보 추출
- 여러 문서 간 정보가 상충할 경우 모든 관점 제시
- 시간에 민감한 정보는 날짜/버전 명시

## 답변 구조
1. 핵심 답변 (1-2문장 요약)
2. 상세 설명 (구조화된 정보)
3. 추가 고려사항 (있을 경우)
4. 출처 표시

## 특별 지침
- 검색 결과가 불충분한 경우: "제공된 문서에는 충분한 정보가 없습니다"
- 전문 용어 사용 시: 간단한 설명 추가
- 코드/명령어 포함 시: 명확한 포맷팅과 주석

## 금지 사항
- 검색되지 않은 정보를 추측하거나 생성하지 않기
- 개인적 의견이나 추천을 하지 않기 (문서 기반이 아닌 경우)
- 확실하지 않은 정보를 확실한 것처럼 표현하지 않기
"""

class RAGSystemPromptOptimizer:
    def __init__(self, domain=None):
        self.base_prompt = SYSTEM_PROMPT
        self.domain = domain
        
    def get_optimized_prompt(self, query_type=None):
        prompt = self.base_prompt
        
        # 도메인별 추가 지침
        if self.domain:
            prompt += f"\\n\\n도메인: {self.domain}\\n"
            prompt += self.get_domain_specific_rules()
        
        # 쿼리 타입별 조정
        if query_type == "factual":
            prompt += "\\n정확한 사실 확인에 중점을 두세요."
        elif query_type == "analytical":
            prompt += "\\n여러 관점을 분석하고 비교하세요."
        elif query_type == "procedural":
            prompt += "\\n단계별로 명확하게 설명하세요."
        
        return prompt
    
    def get_domain_specific_rules(self):
        rules = {
            "medical": "의학적 조언은 제공하지 않으며, 전문의 상담을 권고하세요.",
            "legal": "법률 자문이 아님을 명시하고, 전문가 상담을 권고하세요.",
            "financial": "투자 조언이 아님을 명시하고, 리스크를 설명하세요.",
            "technical": "코드 예제는 테스트를 거쳐야 함을 명시하세요."
        }
        return rules.get(self.domain, "")`}
                </pre>
              </div>
            </div>

            <div className="bg-amber-50 dark:bg-amber-900/20 p-6 rounded-xl border border-amber-200 dark:border-amber-700">
              <h3 className="font-bold text-amber-800 dark:text-amber-200 mb-4">성능 측정 지표</h3>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
                  <p className="text-2xl font-bold text-amber-600">85%</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">정확도</p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
                  <p className="text-2xl font-bold text-amber-600">92%</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">출처 명시율</p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
                  <p className="text-2xl font-bold text-amber-600">3.2초</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">평균 응답시간</p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
                  <p className="text-2xl font-bold text-amber-600">4.6/5</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">사용자 만족도</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: Error Handling Prompts */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
              <AlertCircle className="text-red-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.4 에러 처리 프롬프트</h2>
              <p className="text-gray-600 dark:text-gray-400">우아한 실패와 사용자 가이드</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl border border-red-200 dark:border-red-700">
              <h3 className="font-bold text-red-800 dark:text-red-200 mb-4">상황별 에러 처리 템플릿</h3>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">📭 검색 결과 없음</h4>
                  <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-sm">
                    <pre className="whitespace-pre-wrap">
{`죄송합니다. "{query}"에 대한 관련 정보를 찾을 수 없습니다.

다음과 같이 시도해보세요:
• 다른 키워드나 동의어를 사용해보세요
• 더 구체적이거나 일반적인 용어로 검색해보세요
• 철자와 띄어쓰기를 확인해보세요

예시: "{suggested_query}"`}
                    </pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">🔀 모순된 정보</h4>
                  <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-sm">
                    <pre className="whitespace-pre-wrap">
{`검색된 문서들에서 상충하는 정보가 발견되었습니다:

관점 1: [출처: 문서A]
"{contradicting_info_1}"

관점 2: [출처: 문서B]
"{contradicting_info_2}"

💡 이러한 차이는 다음과 같은 이유일 수 있습니다:
• 정보의 업데이트 시점 차이
• 서로 다른 맥락이나 조건
• 출처의 관점 차이

최신 정보나 공식 출처를 확인하시기 바랍니다.`}
                    </pre>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">⚠️ 부분적 정보</h4>
                  <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded text-sm">
                    <pre className="whitespace-pre-wrap">
{`요청하신 정보의 일부만 찾을 수 있었습니다:

✅ 찾은 정보:
{found_information}

❌ 찾지 못한 정보:
{missing_information}

추가 정보가 필요하시면:
1. 더 구체적인 질문을 해주세요
2. 다른 측면에서 접근해보세요
3. 관련 전문가나 공식 문서를 참조하세요`}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 5: Multi-turn Conversation */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-teal-100 dark:bg-teal-900/20 flex items-center justify-center">
              <MessageSquare className="text-teal-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.5 다중 턴 대화 관리</h2>
              <p className="text-gray-600 dark:text-gray-400">컨텍스트를 유지하는 연속 대화</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-teal-50 dark:bg-teal-900/20 p-6 rounded-xl border border-teal-200 dark:border-teal-700">
              <h3 className="font-bold text-teal-800 dark:text-teal-200 mb-4">대화 컨텍스트 관리 시스템</h3>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
                <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`class MultiTurnRAGManager:
    def __init__(self, max_history=5):
        self.conversation_history = []
        self.retrieved_context = {}
        self.max_history = max_history
        
    def process_turn(self, user_query, turn_number):
        # 대화 기록 기반 쿼리 개선
        enhanced_query = self.enhance_query_with_context(user_query)
        
        # 이전 검색 결과 재사용 여부 결정
        should_reuse = self.check_context_relevance(user_query)
        
        if should_reuse:
            # 기존 컨텍스트 활용
            documents = self.retrieved_context.get('documents', [])
            prompt = self.build_contextual_prompt(
                user_query, 
                documents, 
                self.conversation_history
            )
        else:
            # 새로운 검색 수행
            documents = self.retriever.search(enhanced_query)
            self.retrieved_context = {
                'query': enhanced_query,
                'documents': documents,
                'turn': turn_number
            }
            prompt = self.build_fresh_prompt(user_query, documents)
        
        return prompt
    
    def build_contextual_prompt(self, query, documents, history):
        return f"""
이전 대화 내용:
{self.format_history(history[-3:])}  # 최근 3개 턴만

현재 질문: {query}

관련 컨텍스트 (이전에 검색된 정보):
{self.format_documents(documents)}

지침:
1. 이전 대화의 맥락을 고려하여 답변하세요
2. 이미 언급된 내용은 간략히 참조만 하세요
3. 새로운 관점이나 추가 정보에 집중하세요
4. 대화의 흐름을 자연스럽게 이어가세요

답변:
"""
    
    def enhance_query_with_context(self, current_query):
        if not self.conversation_history:
            return current_query
            
        # 대명사 해결 및 컨텍스트 추가
        context_keywords = self.extract_keywords(self.conversation_history[-2:])
        
        # 쿼리에 컨텍스트가 부족한 경우 보강
        if self.is_query_ambiguous(current_query):
            enhanced = f"{context_keywords} {current_query}"
            return enhanced
        
        return current_query
    
    def check_context_relevance(self, current_query):
        """이전 검색 결과의 재사용 가능성 판단"""
        if not self.retrieved_context:
            return False
            
        # 의미적 유사도 계산
        prev_query = self.retrieved_context.get('query', '')
        similarity = self.calculate_similarity(prev_query, current_query)
        
        # 임계값 이상이면 재사용
        return similarity > 0.7

# 사용 예시
manager = MultiTurnRAGManager()

# Turn 1
response1 = manager.process_turn("파이썬의 장점은 무엇인가요?", 1)

# Turn 2 - 컨텍스트 활용
response2 = manager.process_turn("그럼 단점은요?", 2)  # "그럼"이 파이썬을 지칭

# Turn 3 - 새로운 검색 필요
response3 = manager.process_turn("자바와 비교하면 어떤가요?", 3)`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Section 5: Hands-on Code Examples */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-emerald-100 dark:bg-emerald-900/20 flex items-center justify-center">
              <BookOpen className="text-emerald-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">3.5 실전 코드 예제</h2>
              <p className="text-gray-600 dark:text-gray-400">RAG 프롬프트 엔지니어링 실무 구현</p>
            </div>
          </div>

          <div className="space-y-6">
            <CodeSandbox
              title="실습 1: Chain of Thought RAG 프롬프트"
              description="단계별 사고 과정을 포함한 고급 RAG 시스템"
              language="python"
              code={`from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser

# Chain of Thought RAG 프롬프트 템플릿
cot_rag_prompt = ChatPromptTemplate.from_template("""
당신은 검색 증강 AI 어시스턴트입니다. 다음 단계를 따라 답변하세요:

사용자 질문: {query}

검색된 문서:
{documents}

답변 프로세스 (반드시 모든 단계를 거쳐야 합니다):

<thinking>
1. 사용자 의도 분석:
   - 핵심 질문: [질문의 본질을 한 문장으로]
   - 요구 정보: [필요한 정보 유형]
   - 답변 형식: [예상되는 답변 형태]

2. 문서 관련성 평가:
   문서1: [0-10점] - [평가 이유]
   문서2: [0-10점] - [평가 이유]
   문서3: [0-10점] - [평가 이유]

3. 핵심 정보 추출:
   - [문서에서 발췌한 핵심 사실들]

4. 정보 일관성 확인:
   - 모순 여부: [있음/없음]
   - 확신도: [높음/중간/낮음]
</thinking>

최종 답변:
[종합적이고 명확한 답변. 불확실한 경우 솔직히 표현]

출처: [사용한 문서 번호]
""")

# LLM 체인 구성
llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = cot_rag_prompt | llm | StrOutputParser()

# 실행 예제
query = "벡터 데이터베이스의 주요 장점은 무엇인가요?"
documents = """
문서1: 벡터 데이터베이스는 의미 기반 검색을 지원하여 키워드 매칭의 한계를 극복합니다.
문서2: 고차원 벡터를 효율적으로 저장하고 유사도 검색을 빠르게 수행할 수 있습니다.
문서3: 전통적인 DB와 달리 근사 최근접 이웃 검색(ANN)을 사용하여 대규모 데이터에서도 빠릅니다.
"""

response = chain.invoke({
    "query": query,
    "documents": documents
})

print(response)`}
              output={`<thinking>
1. 사용자 의도 분석:
   - 핵심 질문: 벡터 DB의 핵심 이점 파악
   - 요구 정보: 기술적 장점과 실용적 가치
   - 답변 형식: 구조화된 장점 목록

2. 문서 관련성 평가:
   문서1: 9점 - 의미 검색의 핵심 가치 설명
   문서2: 8점 - 기술적 효율성 강조
   문서3: 9점 - 성능 이점과 알고리즘 설명

3. 핵심 정보 추출:
   - 의미 기반 검색 지원 (vs 키워드 매칭)
   - 고차원 벡터 효율적 저장
   - ANN 알고리즘으로 빠른 검색

4. 정보 일관성 확인:
   - 모순 여부: 없음
   - 확신도: 높음
</thinking>

최종 답변:
벡터 데이터베이스의 주요 장점은 다음과 같습니다:

1. 의미 기반 검색: 키워드가 정확히 일치하지 않아도 문맥상 유사한 내용을 찾을 수 있습니다.
2. 고효율 저장: 고차원 벡터를 최적화된 방식으로 저장하여 메모리를 효율적으로 사용합니다.
3. 빠른 검색 속도: ANN(근사 최근접 이웃) 알고리즘을 활용하여 대규모 데이터에서도 밀리초 단위로 검색합니다.

출처: 문서1, 문서2, 문서3`}
              highlightLines={[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]}
            />

            <CodeSandbox
              title="실습 2: 도메인별 프롬프트 템플릿"
              description="의료, 법률, 기술 도메인 전문 RAG 프롬프트"
              language="python"
              code={`from enum import Enum
from typing import Dict

class Domain(Enum):
    MEDICAL = "medical"
    LEGAL = "legal"
    TECHNICAL = "technical"

# 도메인별 시스템 프롬프트
DOMAIN_PROMPTS: Dict[Domain, str] = {
    Domain.MEDICAL: """
당신은 의료 정보 전문 AI 어시스턴트입니다.

중요 원칙:
- 항상 의학적 근거를 제시하세요
- 진단이나 처방은 절대 금지입니다
- 불확실한 정보는 명확히 표시하세요
- 응급 상황 시 즉시 의료진 상담을 권장하세요

검색된 문서: {documents}
사용자 질문: {query}

답변 형식:
1. 의학적 배경
2. 검색된 정보 요약
3. 주의사항
4. 참고 출처

⚠️ 면책: 이 정보는 교육 목적이며, 전문 의료 상담을 대체할 수 없습니다.
""",

    Domain.LEGAL: """
당신은 법률 정보 전문 AI 어시스턴트입니다.

중요 원칙:
- 법적 조언은 제공할 수 없으며, 일반적인 정보만 제공합니다
- 관련 법령과 판례를 명시하세요
- 지역별 법률 차이를 고려하세요
- 최신 법령 개정 여부를 확인하세요

검색된 문서: {documents}
사용자 질문: {query}

답변 형식:
1. 관련 법령 (법률명, 조항)
2. 일반적인 법적 해석
3. 주의사항 및 예외 케이스
4. 전문가 상담 필요성

⚠️ 면책: 실제 법적 문제는 변호사와 상담하시기 바랍니다.
""",

    Domain.TECHNICAL: """
당신은 기술 문서 전문 AI 어시스턴트입니다.

중요 원칙:
- 정확한 기술 용어를 사용하세요
- 코드 예제는 실행 가능한 상태로 제공하세요
- 버전 호환성 정보를 명시하세요
- 공식 문서 링크를 포함하세요

검색된 문서: {documents}
사용자 질문: {query}

답변 형식:
1. 개념 설명
2. 코드 예제 (주석 포함)
3. 주요 파라미터 설명
4. 일반적인 에러와 해결법
5. 공식 문서 참조
"""
}

# 사용 예제
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

def get_domain_chain(domain: Domain):
    """도메인별 RAG 체인 생성"""
    prompt = ChatPromptTemplate.from_template(DOMAIN_PROMPTS[domain])
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    return prompt | llm

# 의료 도메인 예제
medical_chain = get_domain_chain(Domain.MEDICAL)
response = medical_chain.invoke({
    "query": "비타민 D 결핍 증상은 무엇인가요?",
    "documents": "비타민 D가 부족하면 뼈 건강에 영향을 미칠 수 있습니다..."
})

print(f"의료 도메인 답변:\\n{response.content}")`}
              output={`의료 도메인 답변:
1. 의학적 배경
비타민 D는 칼슘 흡수와 뼈 건강에 필수적인 지용성 비타민입니다. 햇빛 노출을 통해 피부에서 합성되며, 음식이나 보충제로도 섭취 가능합니다.

2. 검색된 정보 요약
비타민 D 결핍 시 주요 증상:
- 뼈 통증 및 근육 약화
- 피로감 및 무기력증
- 면역력 저하
- 우울감 증가 (일부 연구)

3. 주의사항
- 증상이 비특이적이어서 다른 질환과 혼동 가능
- 혈액 검사(25-OH 비타민 D)를 통한 정확한 진단 필요
- 자가 진단 및 과다 복용 금지

4. 참고 출처
검색된 의학 문서

⚠️ 면책: 이 정보는 교육 목적이며, 전문 의료 상담을 대체할 수 없습니다.`}
              highlightLines={[10, 11, 12, 13, 14, 15, 16, 17]}
            />
          </div>
        </section>

        {/* Section 6: Simulator Link */}
        <section className="bg-gradient-to-r from-emerald-500 to-green-600 rounded-2xl p-8 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-2xl font-bold mb-2">🎮 프롬프트 엔지니어링 시뮬레이터</h3>
              <p className="text-emerald-100">다양한 프롬프트 기법을 실시간으로 테스트하세요</p>
            </div>
            <Link
              href="/modules/rag/simulators/prompt-engineering-lab"
              className="inline-flex items-center gap-2 bg-white text-emerald-600 px-6 py-3 rounded-lg font-semibold hover:bg-emerald-50 transition-colors shadow-lg"
            >
              시뮬레이터 열기
              <ArrowRight size={20} />
            </Link>
          </div>
        </section>

        {/* Section 7: Practical Exercise */}
        <section className="bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">실습 과제</h2>
          
          <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
            <h3 className="font-bold mb-4">RAG 프롬프트 엔지니어링 실습</h3>
            
            <div className="space-y-4">
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">📋 과제 1: 도메인별 프롬프트 템플릿 구축</h4>
                <ol className="space-y-2 text-sm">
                  <li>1. 3개 이상의 도메인 선택 (예: 의료, 법률, 기술)</li>
                  <li>2. 각 도메인별 시스템 프롬프트 작성</li>
                  <li>3. Few-shot 예시 3개씩 준비</li>
                  <li>4. 에러 처리 시나리오 정의</li>
                  <li>5. 실제 질문으로 테스트 및 평가</li>
                </ol>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">🎯 과제 2: Chain of Thought 최적화</h4>
                <ul className="space-y-1 text-sm">
                  <li>• Self-Ask 방식으로 복잡한 질문 분해</li>
                  <li>• 각 단계별 추론 과정 명시화</li>
                  <li>• 검색 효율성과 답변 품질 측정</li>
                  <li>• A/B 테스트로 개선 효과 검증</li>
                </ul>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">💡 과제 3: 다중 턴 대화 시뮬레이션</h4>
                <p className="text-sm mb-2">
                  고객 지원 챗봇 시나리오로 다음을 구현:
                </p>
                <ul className="space-y-1 text-sm">
                  <li>• 5턴 이상의 연속 대화 처리</li>
                  <li>• 컨텍스트 유지 및 참조 해결</li>
                  <li>• 대화 기록 기반 개인화</li>
                  <li>• 만족도 평가 시스템 구축</li>
                </ul>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">📊 평가 지표</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="font-medium">정량적 지표:</p>
                    <ul className="mt-1">
                      <li>• 응답 정확도</li>
                      <li>• 처리 시간</li>
                      <li>• 토큰 사용량</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-medium">정성적 지표:</p>
                    <ul className="mt-1">
                      <li>• 답변의 자연스러움</li>
                      <li>• 컨텍스트 이해도</li>
                      <li>• 에러 처리 품질</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* References */}
        <References
          sections={[
            {
              title: '📚 프롬프트 엔지니어링 가이드',
              icon: 'web' as const,
              color: 'border-teal-500',
              items: [
                {
                  title: 'OpenAI Prompt Engineering Guide',
                  authors: 'OpenAI',
                  year: '2025',
                  description: 'GPT 모델을 위한 공식 프롬프트 작성 가이드 - 6가지 전략',
                  link: 'https://platform.openai.com/docs/guides/prompt-engineering'
                },
                {
                  title: 'Anthropic Prompt Engineering',
                  authors: 'Anthropic',
                  year: '2025',
                  description: 'Claude를 위한 프롬프트 최적화 - Chain of Thought 강조',
                  link: 'https://docs.anthropic.com/claude/docs/prompt-engineering'
                },
                {
                  title: 'LangChain Prompt Templates',
                  authors: 'LangChain',
                  year: '2025',
                  description: 'RAG용 프롬프트 템플릿 라이브러리 - 재사용 가능',
                  link: 'https://python.langchain.com/docs/modules/model_io/prompts/'
                },
                {
                  title: 'Prompt Engineering by DAIR.AI',
                  authors: 'DAIR.AI',
                  year: '2024',
                  description: '실전 프롬프트 엔지니어링 가이드 - 40+ 예제',
                  link: 'https://www.promptingguide.ai/'
                },
                {
                  title: 'AWS RAG Prompt Best Practices',
                  authors: 'AWS',
                  year: '2025',
                  description: 'Bedrock을 위한 RAG 프롬프트 최적화 패턴',
                  link: 'https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-engineering-guidelines.html'
                }
              ]
            },
            {
              title: '📖 Chain of Thought & Few-shot 연구',
              icon: 'research' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'Chain-of-Thought Prompting',
                  authors: 'Wei et al., Google Research',
                  year: '2022',
                  description: 'CoT 프롬프팅으로 추론 성능 대폭 향상 - 540B 모델',
                  link: 'https://arxiv.org/abs/2201.11903'
                },
                {
                  title: 'Self-Ask: Decomposing Complex Questions',
                  authors: 'Press et al., UW & AI2',
                  year: '2023',
                  description: '복잡한 질문을 하위 질문으로 분해 - 정확도 30% 향상',
                  link: 'https://arxiv.org/abs/2210.03350'
                },
                {
                  title: 'Few-Shot Parameter-Efficient Fine-Tuning',
                  authors: 'Liu et al., Stanford',
                  year: '2022',
                  description: 'In-Context Learning의 이론적 기반 - GPT-3',
                  link: 'https://arxiv.org/abs/2012.15723'
                },
                {
                  title: 'Tree of Thoughts (ToT)',
                  authors: 'Yao et al., Princeton',
                  year: '2023',
                  description: '트리 구조로 사고 확장 - CoT의 고급 버전',
                  link: 'https://arxiv.org/abs/2305.10601'
                }
              ]
            },
            {
              title: '🛠️ RAG 프롬프트 실전 도구',
              icon: 'tools' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'LlamaIndex Prompt Optimizer',
                  authors: 'LlamaIndex',
                  year: '2025',
                  description: 'RAG 프롬프트 자동 최적화 - A/B 테스트 내장',
                  link: 'https://docs.llamaindex.ai/en/stable/module_guides/querying/prompts/'
                },
                {
                  title: 'Haystack PromptNode',
                  authors: 'deepset',
                  year: '2025',
                  description: '다양한 LLM용 프롬프트 통합 관리 - 템플릿 시스템',
                  link: 'https://docs.haystack.deepset.ai/docs/prompt_node'
                },
                {
                  title: 'Guidance (Microsoft)',
                  authors: 'Microsoft Research',
                  year: '2024',
                  description: '구조화된 프롬프트 생성 - 제약 조건 적용',
                  link: 'https://github.com/microsoft/guidance'
                },
                {
                  title: 'Prompttools',
                  authors: 'Hegel AI',
                  year: '2024',
                  description: '프롬프트 실험 및 평가 프레임워크 - 벤치마킹',
                  link: 'https://github.com/hegelai/prompttools'
                },
                {
                  title: 'LangSmith Prompt Hub',
                  authors: 'LangChain',
                  year: '2025',
                  description: '커뮤니티 검증된 프롬프트 템플릿 저장소',
                  link: 'https://smith.langchain.com/hub'
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
            href="/modules/rag/intermediate/chapter2"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: 하이브리드 검색 전략
          </Link>
          
          <Link
            href="/modules/rag/intermediate/chapter4"
            className="inline-flex items-center gap-2 bg-purple-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-purple-600 transition-colors"
          >
            다음: RAG 성능 최적화
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}