import { Repeat } from 'lucide-react'

export default function Section2() {
  return (
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
  )
}
