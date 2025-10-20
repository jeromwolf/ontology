import React from 'react';
import { Shield, AlertTriangle, Eye, Lock, Users, CheckCircle } from 'lucide-react';
import References from '../References';

export default function Chapter1() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-6 text-gray-900 dark:text-white">AI 윤리의 중요성</h1>

      <div className="bg-gradient-to-r from-rose-100 to-pink-100 dark:from-rose-900/30 dark:to-pink-900/30 p-6 rounded-lg mb-8">
        <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
          AI 기술이 사회 전반에 깊숙이 침투하면서, 윤리적 고려사항은 더 이상 선택이 아닌 필수가 되었습니다.
          2024-2025년 현재, ChatGPT, Claude, Gemini 등 거대 AI 모델들이 일상에 통합되면서
          편향, 투명성, 책임성, 프라이버시 문제가 전례 없이 중요해졌습니다.
        </p>
      </div>

      {/* AI 윤리 5대 원칙 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Shield className="w-8 h-8 text-rose-600" />
          AI 윤리 5대 원칙
        </h2>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-rose-500">
            <div className="flex items-center gap-3 mb-3">
              <Users className="w-6 h-6 text-rose-600" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">1. 공정성 (Fairness)</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              AI 시스템은 인종, 성별, 나이, 사회경제적 지위 등에 따른 차별 없이 모든 사용자에게 공평하게 작동해야 합니다.
            </p>
            <div className="bg-rose-50 dark:bg-rose-900/20 p-3 rounded">
              <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-1">핵심 질문:</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                "이 AI가 특정 그룹에 불리한 결과를 생성하지는 않는가?"
              </p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-blue-500">
            <div className="flex items-center gap-3 mb-3">
              <Eye className="w-6 h-6 text-blue-600" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">2. 투명성 (Transparency)</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              AI의 의사결정 과정을 사용자가 이해할 수 있어야 하며, "블랙박스" 문제를 해결해야 합니다.
            </p>
            <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
              <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-1">핵심 질문:</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                "사용자가 AI의 결정 근거를 이해할 수 있는가?"
              </p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-purple-500">
            <div className="flex items-center gap-3 mb-3">
              <AlertTriangle className="w-6 h-6 text-purple-600" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">3. 책임성 (Accountability)</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              AI 시스템의 결과에 대해 명확한 책임 주체가 존재해야 하며, 문제 발생 시 책임 소재를 추적할 수 있어야 합니다.
            </p>
            <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded">
              <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-1">핵심 질문:</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                "AI가 잘못된 결정을 내렸을 때 누가 책임지는가?"
              </p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-green-500">
            <div className="flex items-center gap-3 mb-3">
              <Lock className="w-6 h-6 text-green-600" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">4. 프라이버시 (Privacy)</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              개인 데이터는 안전하게 보호되어야 하며, GDPR, CCPA 등 규제를 준수해야 합니다.
            </p>
            <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded">
              <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-1">핵심 질문:</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                "사용자 데이터가 동의 없이 수집·활용되지 않는가?"
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-orange-500">
          <div className="flex items-center gap-3 mb-3">
            <CheckCircle className="w-6 h-6 text-orange-600" />
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">5. 안전성 (Safety)</h3>
          </div>
          <p className="text-gray-700 dark:text-gray-300 mb-3">
            AI 시스템은 물리적·심리적 해를 끼치지 않아야 하며, 악용 가능성을 최소화해야 합니다.
          </p>
          <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded">
            <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-1">핵심 질문:</p>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              "이 AI가 악의적으로 사용될 가능성은 없는가?"
            </p>
          </div>
        </div>
      </section>

      {/* 2024-2025 주요 AI 윤리 사건 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">2024-2025 주요 AI 윤리 사건</h2>

        <div className="space-y-6">
          <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg border-l-4 border-red-500">
            <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-white">OpenAI ChatGPT 데이터 유출 논란 (2024.03)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              ChatGPT가 학습 데이터에 포함된 저작권 보호 콘텐츠를 재생성하면서 뉴욕타임스, Getty Images 등과
              법적 분쟁 발생. AI 학습에서의 저작권 경계에 대한 논쟁 심화.
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <p className="text-sm font-semibold text-gray-800 dark:text-gray-200">핵심 쟁점:</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 mt-2 space-y-1">
                <li>AI 학습 데이터의 저작권 범위</li>
                <li>"공정 사용(Fair Use)" vs 저작권 침해</li>
                <li>생성 콘텐츠의 원저작자 보상 문제</li>
              </ul>
            </div>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg border-l-4 border-purple-500">
            <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-white">Google Gemini 이미지 생성 편향 논란 (2024.02)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              역사적 인물 이미지 생성 시 과도한 "다양성 보정"으로 역사적 사실 왜곡 발생.
              나치 군인을 유색인종으로 묘사하는 등 문제 발생 후 이미지 생성 기능 일시 중단.
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <p className="text-sm font-semibold text-gray-800 dark:text-gray-200">교훈:</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 mt-2 space-y-1">
                <li>편향 완화 노력도 과하면 역효과 발생</li>
                <li>역사적 맥락 이해 필요</li>
                <li>투명한 의사결정 공개의 중요성</li>
              </ul>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border-l-4 border-blue-500">
            <h3 className="text-xl font-bold mb-2 text-gray-900 dark:text-white">Anthropic Claude 헌법적 AI 접근법 (2024-2025)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              "Constitutional AI" 방법론으로 인간 피드백(RLHF) 의존도를 줄이고,
              명시적 윤리 원칙을 AI에 내재화. 투명성과 안전성에서 업계 표준 제시.
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <p className="text-sm font-semibold text-gray-800 dark:text-gray-200">혁신 포인트:</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 mt-2 space-y-1">
                <li>AI 스스로 윤리 기준 학습</li>
                <li>RLHF 편향 감소</li>
                <li>확장 가능한 안전성(Scalable Safety)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* UNESCO AI 윤리 권고안 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">UNESCO AI 윤리 권고안 (2021, 193개국 채택)</h2>

        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-lg mb-6">
          <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">10대 핵심 원칙</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">1. 비례성과 무해성</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">AI는 해를 끼치지 않아야 함</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">2. 안전과 보안</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">물리적·디지털 안전 보장</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">3. 프라이버시 권리</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">개인정보 보호 철저</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">4. 다차원적 책임</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">명확한 책임 체계 구축</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">5. 투명성과 설명가능성</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">의사결정 과정 공개</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">6. 인간 감독과 결정</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">최종 결정권은 인간에게</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">7. 지속가능성</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">환경 영향 최소화</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">8. 인식과 이해</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">AI 리터러시 증진</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">9. 다층 거버넌스</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">다양한 이해관계자 참여</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">10. 적응적 거버넌스</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">기술 변화에 대응</p>
            </div>
          </div>
        </div>
      </section>

      {/* 코드 예제: 윤리 체크리스트 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">실전 코드: AI 윤리 체크리스트 구현</h2>

        <div className="bg-gray-900 dark:bg-gray-950 p-6 rounded-lg overflow-x-auto">
          <pre className="text-sm text-gray-100">
            <code>{`# AI Ethics Checklist Implementation
class AIEthicsChecker:
    """
    AI 시스템 배포 전 윤리 점검 도구
    UNESCO 권고안 및 업계 모범 사례 기반
    """

    def __init__(self):
        self.criteria = {
            'fairness': {
                'name': '공정성',
                'checks': [
                    '다양한 인구통계 그룹에서 테스트했는가?',
                    '보호된 속성(인종, 성별 등)이 예측에 영향을 미치는가?',
                    '소수 그룹에서 에러율이 높지 않은가?'
                ]
            },
            'transparency': {
                'name': '투명성',
                'checks': [
                    'AI 사용 사실을 사용자에게 고지하는가?',
                    '의사결정 근거를 설명할 수 있는가?',
                    '모델 아키텍처와 데이터 출처를 문서화했는가?'
                ]
            },
            'accountability': {
                'name': '책임성',
                'checks': [
                    '오류 발생 시 책임 주체가 명확한가?',
                    '사용자 피드백 수집 메커니즘이 있는가?',
                    '정기적 성능 감사 계획이 있는가?'
                ]
            },
            'privacy': {
                'name': '프라이버시',
                'checks': [
                    'GDPR/CCPA 등 규제를 준수하는가?',
                    '최소 데이터 수집 원칙을 따르는가?',
                    '데이터 암호화 및 익명화를 적용했는가?'
                ]
            },
            'safety': {
                'name': '안전성',
                'checks': [
                    '악의적 사용 시나리오를 검토했는가?',
                    '예상치 못한 동작에 대한 안전장치가 있는가?',
                    '고위험 결정(의료, 법률)에서 인간 검토를 포함하는가?'
                ]
            }
        }

    def evaluate(self, responses: dict) -> dict:
        """
        윤리 체크리스트 평가

        Args:
            responses: 각 체크 항목에 대한 응답 (True/False)

        Returns:
            카테고리별 점수 및 전체 준수율
        """
        results = {}
        total_checks = 0
        passed_checks = 0

        for category, data in self.criteria.items():
            category_score = 0
            category_total = len(data['checks'])

            for i, check in enumerate(data['checks']):
                key = f"{category}_{i}"
                if responses.get(key, False):
                    category_score += 1
                    passed_checks += 1
                total_checks += 1

            results[category] = {
                'name': data['name'],
                'score': category_score,
                'total': category_total,
                'percentage': (category_score / category_total) * 100
            }

        overall_compliance = (passed_checks / total_checks) * 100

        return {
            'category_results': results,
            'overall_compliance': overall_compliance,
            'recommendation': self._get_recommendation(overall_compliance)
        }

    def _get_recommendation(self, compliance: float) -> str:
        """배포 권고사항 결정"""
        if compliance >= 90:
            return "✅ 배포 승인 - 우수한 윤리 준수율"
        elif compliance >= 70:
            return "⚠️ 조건부 승인 - 미흡 항목 개선 후 배포"
        elif compliance >= 50:
            return "❌ 배포 불가 - 주요 윤리 기준 미달"
        else:
            return "🚫 긴급 중단 - 근본적 재설계 필요"

# 사용 예제
checker = AIEthicsChecker()

# 실제 AI 시스템 평가 (예시)
responses = {
    'fairness_0': True,   # 다양한 그룹 테스트 완료
    'fairness_1': True,   # 보호 속성 영향 검토
    'fairness_2': False,  # 소수 그룹 에러율 높음
    'transparency_0': True,
    'transparency_1': True,
    'transparency_2': False,
    'accountability_0': True,
    'accountability_1': True,
    'accountability_2': True,
    'privacy_0': True,
    'privacy_1': False,
    'privacy_2': True,
    'safety_0': True,
    'safety_1': True,
    'safety_2': True
}

result = checker.evaluate(responses)

print(f"전체 준수율: {result['overall_compliance']:.1f}%")
print(f"권고사항: {result['recommendation']}")
print("\\n카테고리별 점수:")
for category, data in result['category_results'].items():
    print(f"  {data['name']}: {data['score']}/{data['total']} ({data['percentage']:.1f}%)")

# 출력 예시:
# 전체 준수율: 80.0%
# 권고사항: ⚠️ 조건부 승인 - 미흡 항목 개선 후 배포
#
# 카테고리별 점수:
#   공정성: 2/3 (66.7%)
#   투명성: 2/3 (66.7%)
#   책임성: 3/3 (100.0%)
#   프라이버시: 2/3 (66.7%)
#   안전성: 3/3 (100.0%)`}</code>
          </pre>
        </div>

        <div className="mt-4 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <p className="text-sm text-gray-700 dark:text-gray-300">
            <strong>활용 팁:</strong> 이 체크리스트를 CI/CD 파이프라인에 통합하여 모델 배포 전 자동 윤리 검증을 수행할 수 있습니다.
            80% 이상 준수율이 배포 최소 기준으로 권장됩니다.
          </p>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 공식 문서 & 가이드라인',
            icon: 'docs' as const,
            color: 'border-rose-500',
            items: [
              {
                title: 'UNESCO AI Ethics Recommendation',
                url: 'https://www.unesco.org/en/artificial-intelligence/recommendation-ethics',
                description: '193개국이 채택한 AI 윤리 권고안 전문 (2021)'
              },
              {
                title: 'EU Ethics Guidelines for Trustworthy AI',
                url: 'https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai',
                description: 'EU 집행위원회의 신뢰할 수 있는 AI 윤리 가이드라인'
              },
              {
                title: 'Google AI Principles',
                url: 'https://ai.google/responsibility/principles/',
                description: 'Google의 7대 AI 개발 원칙 및 적용 사례'
              },
              {
                title: 'Microsoft Responsible AI',
                url: 'https://www.microsoft.com/en-us/ai/responsible-ai',
                description: 'Microsoft의 책임 있는 AI 프레임워크 및 도구'
              },
              {
                title: 'OpenAI Usage Policies',
                url: 'https://openai.com/policies/usage-policies',
                description: 'OpenAI의 사용 정책 및 윤리 가이드 (2024 업데이트)'
              }
            ]
          },
          {
            title: '📖 핵심 연구 논문',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Constitutional AI: Harmlessness from AI Feedback',
                url: 'https://arxiv.org/abs/2212.08073',
                description: 'Anthropic의 헌법적 AI 방법론 - RLHF 대안 제시'
              },
              {
                title: 'Model Cards for Model Reporting (Google, 2019)',
                url: 'https://arxiv.org/abs/1810.03993',
                description: 'ML 모델 투명성을 위한 문서화 프레임워크'
              },
              {
                title: 'Datasheets for Datasets (Microsoft, 2021)',
                url: 'https://arxiv.org/abs/1803.09010',
                description: '데이터셋 투명성 및 책임성 문서화'
              },
              {
                title: 'AI Ethics in 2024: A Practitioner\'s Guide',
                url: 'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4658912',
                description: '2024년 AI 윤리 실무 가이드 (법률·기술 통합)'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 리소스',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'IBM AI Fairness 360',
                url: 'https://aif360.mybluemix.net/',
                description: '편향 탐지 및 완화를 위한 오픈소스 툴킷'
              },
              {
                title: 'Google What-If Tool',
                url: 'https://pair-code.github.io/what-if-tool/',
                description: 'ML 모델 동작 분석 및 공정성 검증 도구'
              },
              {
                title: 'Fairlearn (Microsoft)',
                url: 'https://fairlearn.org/',
                description: 'Python 기반 공정성 평가 및 완화 라이브러리'
              },
              {
                title: 'AI Incident Database',
                url: 'https://incidentdatabase.ai/',
                description: '전 세계 AI 사고 사례 데이터베이스 (1000+ 사건)'
              },
              {
                title: 'Hugging Face Ethics and Society',
                url: 'https://huggingface.co/spaces/huggingface/ethics-soc',
                description: 'AI 모델 윤리 평가 커뮤니티 도구'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
