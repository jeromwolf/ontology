import { BookOpen } from 'lucide-react'
import CodeSandbox from '../../../components/CodeSandbox'

export default function Section6() {
  return (
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
  )
}
