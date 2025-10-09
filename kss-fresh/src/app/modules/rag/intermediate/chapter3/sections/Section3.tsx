import { Lightbulb } from 'lucide-react'

export default function Section3() {
  return (
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
  )
}
