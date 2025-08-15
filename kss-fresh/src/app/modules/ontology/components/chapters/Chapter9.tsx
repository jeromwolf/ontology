'use client';

export default function Chapter9() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 9: 온톨로지 설계 방법론</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            성공적인 온톨로지 개발을 위해서는 체계적인 방법론이 필요합니다. 
            이번 챕터에서는 검증된 온톨로지 개발 방법론과 7단계 프로세스를 학습합니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">주요 온톨로지 개발 방법론</h2>
        
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-blue-600 dark:text-blue-400 mb-3">METHONTOLOGY</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              가장 널리 사용되는 온톨로지 개발 방법론으로, 소프트웨어 공학 원칙을 적용
            </p>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 개발 프로세스와 생명주기를 명확히 정의</li>
              <li>• 각 단계별 상세한 기법과 활동 제시</li>
              <li>• 품질 보증 및 문서화 강조</li>
              <li>• IEEE 표준 준수</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-green-600 dark:text-green-400 mb-3">On-To-Knowledge</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              기업 환경에서의 지식 관리를 위한 실용적 접근법
            </p>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 비즈니스 요구사항 중심 설계</li>
              <li>• 기존 정보 시스템과의 통합 고려</li>
              <li>• 반복적이고 점진적인 개발</li>
              <li>• ROI 측정 및 평가 포함</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-purple-600 dark:text-purple-400 mb-3">NeOn Methodology</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              네트워크 환경에서의 협업적 온톨로지 개발
            </p>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li>• 분산 팀 협업 지원</li>
              <li>• 온톨로지 재사용 강조</li>
              <li>• 시나리오 기반 개발</li>
              <li>• 9가지 유연한 개발 시나리오 제공</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">7단계 온톨로지 개발 프로세스</h2>
        
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-8">
          <div className="space-y-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">1</div>
              <div className="flex-1">
                <h4 className="font-semibold mb-2">요구사항 명세 (Requirements Specification)</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  온톨로지의 목적, 범위, 사용자를 명확히 정의
                </p>
                <div className="bg-white dark:bg-gray-800 rounded p-3 text-sm">
                  <strong>산출물:</strong> 요구사항 문서, 역량 질문(Competency Questions)
                </div>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-green-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">2</div>
              <div className="flex-1">
                <h4 className="font-semibold mb-2">지식 획득 (Knowledge Acquisition)</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  도메인 전문가와의 인터뷰, 문서 분석, 브레인스토밍
                </p>
                <div className="bg-white dark:bg-gray-800 rounded p-3 text-sm">
                  <strong>산출물:</strong> 용어집, 개념 맵, 도메인 지식 문서
                </div>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-purple-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">3</div>
              <div className="flex-1">
                <h4 className="font-semibold mb-2">개념화 (Conceptualization)</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  핵심 개념과 관계를 식별하고 구조화
                </p>
                <div className="bg-white dark:bg-gray-800 rounded p-3 text-sm">
                  <strong>산출물:</strong> 개념 계층구조, 관계 다이어그램, 속성 목록
                </div>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-orange-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">4</div>
              <div className="flex-1">
                <h4 className="font-semibold mb-2">통합 (Integration)</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  기존 온톨로지와의 연계 및 재사용
                </p>
                <div className="bg-white dark:bg-gray-800 rounded p-3 text-sm">
                  <strong>산출물:</strong> 통합 계획, 매핑 테이블, 재사용 분석 보고서
                </div>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-red-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">5</div>
              <div className="flex-1">
                <h4 className="font-semibold mb-2">구현 (Implementation)</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  온톨로지 언어(OWL, RDF)로 형식화
                </p>
                <div className="bg-white dark:bg-gray-800 rounded p-3 text-sm">
                  <strong>산출물:</strong> OWL 파일, SPARQL 쿼리, 추론 규칙
                </div>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-indigo-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">6</div>
              <div className="flex-1">
                <h4 className="font-semibold mb-2">평가 (Evaluation)</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  일관성, 완전성, 정확성 검증
                </p>
                <div className="bg-white dark:bg-gray-800 rounded p-3 text-sm">
                  <strong>산출물:</strong> 평가 보고서, 테스트 결과, 개선 사항 목록
                </div>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">7</div>
              <div className="flex-1">
                <h4 className="font-semibold mb-2">문서화 및 유지보수 (Documentation & Maintenance)</h4>
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                  사용자 가이드 작성, 버전 관리, 지속적 개선
                </p>
                <div className="bg-white dark:bg-gray-800 rounded p-3 text-sm">
                  <strong>산출물:</strong> 사용자 매뉴얼, API 문서, 변경 이력
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">역량 질문 (Competency Questions)</h2>
        
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
            역량 질문은 온톨로지가 답할 수 있어야 하는 질문들로, 요구사항을 명확히 하는 핵심 도구입니다.
          </p>
          
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 space-y-3">
            <h4 className="font-semibold">예시: 의료 온톨로지의 역량 질문</h4>
            <ul className="space-y-2 text-sm">
              <li>• 특정 증상을 보이는 모든 질병은 무엇인가?</li>
              <li>• 특정 약물의 부작용은 무엇인가?</li>
              <li>• 두 약물 간의 상호작용이 있는가?</li>
              <li>• 특정 질병의 위험 요인은 무엇인가?</li>
              <li>• 환자의 병력에 따른 금기 약물은?</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">개발 생명주기 모델</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">폭포수 모델</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              요구사항이 명확하고 변경이 적을 때 적합
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span> 단계별 명확한 산출물
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span> 체계적인 문서화
              </div>
              <div className="flex items-center gap-2">
                <span className="text-red-600">✗</span> 변경에 유연하지 않음
              </div>
            </div>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">반복적 모델</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              요구사항이 진화하고 피드백이 중요할 때 적합
            </p>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span> 빠른 프로토타입
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span> 지속적인 개선
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span> 사용자 피드백 반영
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">품질 평가 기준</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3">구조적 품질</h4>
              <ul className="space-y-2 text-sm">
                <li>• <strong>일관성:</strong> 논리적 모순이 없는가?</li>
                <li>• <strong>완전성:</strong> 필요한 개념이 모두 포함되었는가?</li>
                <li>• <strong>간결성:</strong> 불필요한 중복이 없는가?</li>
                <li>• <strong>확장성:</strong> 새로운 개념 추가가 용이한가?</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-3">기능적 품질</h4>
              <ul className="space-y-2 text-sm">
                <li>• <strong>정확성:</strong> 도메인을 올바르게 표현하는가?</li>
                <li>• <strong>명확성:</strong> 개념이 명확히 정의되었는가?</li>
                <li>• <strong>계산 효율성:</strong> 추론이 효율적인가?</li>
                <li>• <strong>사용성:</strong> 사용자가 이해하기 쉬운가?</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">💡</span>
          실습: 미니 프로젝트
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-4">
          7단계 프로세스를 따라 간단한 "스마트 홈" 온톨로지를 설계해보세요:
        </p>
        <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
          <li>요구사항: 가전제품 제어, 에너지 관리, 보안 시스템</li>
          <li>역량 질문 5개 이상 작성</li>
          <li>핵심 개념 10개 이상 도출</li>
          <li>각 단계별 산출물 작성</li>
        </ul>
      </section>
    </div>
  )
}