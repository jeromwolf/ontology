import { MessageSquare } from 'lucide-react'

export default function Section5() {
  return (
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
  )
}
