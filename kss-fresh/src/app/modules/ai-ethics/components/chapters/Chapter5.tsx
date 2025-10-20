import React from 'react';
import { Scale, Globe, FileText, AlertTriangle, CheckCircle } from 'lucide-react';
import References from '../References';

export default function Chapter5() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-6 text-gray-900 dark:text-white">AI 규제와 법적 프레임워크</h1>

      <div className="bg-gradient-to-r from-rose-100 to-pink-100 dark:from-rose-900/30 dark:to-pink-900/30 p-6 rounded-lg mb-8">
        <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
          2024년은 AI 규제의 전환점입니다. EU AI Act 발효, 한국 AI 기본법 제정, 미국 행정명령 등
          글로벌 규제 체계가 확립되며, 기업은 "규제 준수"가 더 이상 선택이 아닌 생존 조건이 되었습니다.
        </p>
      </div>

      {/* EU AI Act 상세 분석 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Globe className="w-8 h-8 text-rose-600" />
          EU AI Act (2024.08 발효) - 세계 최초 포괄적 AI 규제
        </h2>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-6">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">4단계 위험 기반 분류 체계</h3>
          <div className="space-y-4">
            <div className="bg-red-100 dark:bg-red-900/30 p-4 rounded-lg border-l-4 border-red-600">
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">🚫 금지 (Prohibited AI)</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 사회 신용 점수 시스템 (중국식 Social Credit)</li>
                <li>• 생체 인식 기반 실시간 원격 감시 (공공장소, 예외: 테러 방지)</li>
                <li>• 인간의 취약성 악용하는 조작 AI (아동, 장애인 대상)</li>
                <li>• 무차별 인터넷 크롤링 얼굴 인식 데이터베이스 구축</li>
              </ul>
              <p className="text-xs text-red-800 dark:text-red-300 mt-2">
                위반 시: €3,500만 또는 전 세계 매출 7%
              </p>
            </div>

            <div className="bg-orange-100 dark:bg-orange-900/30 p-4 rounded-lg border-l-4 border-orange-600">
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">⚠️ 고위험 (High-Risk AI) - 엄격한 요구사항</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">적용 분야 (Annex III):</p>
              <div className="grid md:grid-cols-2 gap-2 text-sm text-gray-700 dark:text-gray-300">
                <ul className="space-y-1">
                  <li>✓ 의료 진단/치료 (MDR 의료기기 포함)</li>
                  <li>✓ 교육 평가/입학 결정</li>
                  <li>✓ 채용 및 인사 관리</li>
                  <li>✓ 신용 평가/대출 심사</li>
                </ul>
                <ul className="space-y-1">
                  <li>✓ 법 집행 (범죄 예측, 증거 평가)</li>
                  <li>✓ 이민·망명 심사</li>
                  <li>✓ 사법 행정 (판결 보조)</li>
                  <li>✓ 중요 인프라 안전 관리</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded mt-3">
                <p className="font-semibold text-gray-900 dark:text-white mb-2">필수 의무사항 (Article 8-15):</p>
                <ol className="list-decimal list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                  <li>위험 관리 시스템 (Risk Management System) 구축</li>
                  <li>데이터 품질 및 거버넌스 (대표성, 정확성, 완전성)</li>
                  <li>기술 문서 작성 (Technical Documentation) - 모델 상세</li>
                  <li>자동 로깅 (Automatic Logging) - 모든 의사결정 기록</li>
                  <li>투명성 및 사용자 정보 제공 (Transparency)</li>
                  <li>인간 감독 (Human Oversight) - 최종 결정권</li>
                  <li>정확성·견고성·사이버 보안 (Accuracy, Robustness, Cybersecurity)</li>
                  <li>적합성 평가 (Conformity Assessment) - 제3자 감사</li>
                </ol>
              </div>
              <p className="text-xs text-orange-800 dark:text-orange-300 mt-2">
                위반 시: €1,500만 또는 전 세계 매출 3%
              </p>
            </div>

            <div className="bg-yellow-100 dark:bg-yellow-900/30 p-4 rounded-lg border-l-4 border-yellow-600">
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">⚡ 제한적 위험 (Limited Risk) - 투명성 의무</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 챗봇: AI와 대화 중임을 명시 (Article 52)</li>
                <li>• 생성 AI (Generative AI): 합성 콘텐츠임을 워터마크로 표시</li>
                <li>• Deepfake: 조작된 이미지/영상임을 명확히 고지</li>
                <li>• 감정 인식 AI: 사용 사실 고지</li>
              </ul>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                예: "이 대화는 AI 챗봇과 진행 중입니다." (필수 표시)
              </p>
            </div>

            <div className="bg-green-100 dark:bg-green-900/30 p-4 rounded-lg border-l-4 border-green-600">
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">✅ 최소 위험 (Minimal Risk) - 규제 없음</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>• 스팸 필터, 게임 AI, 추천 시스템 (비고위험)</li>
                <li>• 이메일 자동 분류, 맞춤법 검사</li>
              </ul>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                자율 규제 권장 (Voluntary Codes of Conduct)
              </p>
            </div>
          </div>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">생성형 AI 특별 규정 (Foundation Models - Article 53)</h3>
          <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
            GPT-4, Claude, Gemini 등 범용 대규모 모델에 대한 추가 요구사항 (2025.08부터 적용)
          </p>
          <div className="space-y-2">
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">1. 기술 문서 공개</p>
              <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside">
                <li>학습 데이터 상세 (크기, 출처, 큐레이션 방법)</li>
                <li>모델 아키텍처 및 파라미터 수</li>
                <li>에너지 소비량 및 탄소 배출량</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">2. 저작권 준수</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                학습 데이터에 포함된 저작물 목록 공개 (EU 저작권법 준수)
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <p className="font-semibold text-gray-900 dark:text-white">3. Systemic Risk 평가 (10^25 FLOP 이상)</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                초대규모 모델은 사회적 위험 평가 및 완화 계획 제출 (예: GPT-5급)
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 한국 AI 기본법 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">한국 AI 기본법 (2024.09 국회 통과)</h2>

        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">주요 내용</h3>

          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">1. AI 윤리 5대 원칙 법제화</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>✓ 인간 존엄성 및 인권 보장</li>
                <li>✓ 프라이버시 및 데이터 보호</li>
                <li>✓ 다양성 존중 및 공정성</li>
                <li>✓ 안전성 및 투명성</li>
                <li>✓ 책임성</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">2. 고위험 AI 관리 체계</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                의료·금융·법률 분야 AI는 사전 영향 평가 의무화
              </p>
              <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside">
                <li>AI 영향평가서 제출 (위험도 분석)</li>
                <li>정기 모니터링 및 감사</li>
                <li>사고 발생 시 신고 의무</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">3. AI 신뢰성 인증제도</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                자율적 인증 (EU CE 마크 유사) - 공공조달 시 우대
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">4. AI 산업 진흥</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside">
                <li>AI 규제 샌드박스 (한시적 규제 면제)</li>
                <li>데이터·컴퓨팅 인프라 지원</li>
                <li>스타트업 육성 펀드</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-blue-100 dark:bg-blue-900/30 p-3 rounded">
            <p className="text-sm font-semibold text-gray-900 dark:text-white">EU AI Act와 차이점:</p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside mt-2">
              <li>한국: 진흥 + 규제 병행 (산업 육성 중시)</li>
              <li>EU: 강력한 규제 중심 (소비자 보호 우선)</li>
              <li>한국: 자율 규제 권장 (인증제)</li>
              <li>EU: 의무적 적합성 평가 (고위험 AI)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 미국 & 중국 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">미국 & 중국 AI 규제</h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
            <div className="flex items-center gap-2 mb-4">
              <FileText className="w-6 h-6 text-blue-600" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">미국 AI Executive Order (2023.10)</h3>
            </div>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              연방 정부 차원의 AI 안전 및 보안 프레임워크
            </p>
            <div className="space-y-2">
              <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">1. 안전 테스트</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  대규모 모델(10^26 FLOP 이상)은 국가 안보 테스트 의무 (Defense Production Act 적용)
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">2. 콘텐츠 인증</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  연방 기관이 생성한 AI 콘텐츠에 워터마크 의무화
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">3. 차별 방지</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  주택·고용 등에서 AI 알고리즘 편향 모니터링
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">4. 혁신 촉진</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  AI 연구 자금 확대, 비자 우대 (AI 인재 유치)
                </p>
              </div>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-3">
              특징: 행정명령이라 법적 구속력 제한 (차기 정부 폐기 가능). 주 정부별 개별 규제 존재 (예: 캘리포니아 SB-1047)
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
            <div className="flex items-center gap-2 mb-4">
              <AlertTriangle className="w-6 h-6 text-red-600" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">중국 생성형 AI 관리 조치 (2023.08)</h3>
            </div>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              세계 최초 생성형 AI 특화 규제 (ChatGPT 대응)
            </p>
            <div className="space-y-2">
              <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">1. 콘텐츠 심사</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  사회주의 핵심 가치관 위배 콘텐츠 생성 금지. 사전 알고리즘 심사 필수.
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">2. 데이터 검증</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  학습 데이터의 "합법성·진실성·정확성·객관성·다양성" 보장
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">3. 사용자 인증</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  실명 인증 의무 (익명 사용 불가)
                </p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">4. 라벨링</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  AI 생성 콘텐츠임을 명확히 표시 (워터마크·메타데이터)
                </p>
              </div>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-3">
              특징: 정부 통제 중심 (안전 vs 혁신). Baidu Ernie, Alibaba Tongyi 등 국내 모델 우대.
            </p>
          </div>
        </div>
      </section>

      {/* 규제 준수 체크리스트 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <CheckCircle className="w-8 h-8 text-green-600" />
          규제 준수 실무 체크리스트
        </h2>

        <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-lg">
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">✅ Step 1: 위험 분류 (EU AI Act 기준)</h4>
              <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="w-4 h-4" />
                  우리 AI가 고위험 분야(의료/금융/법률/채용)에 사용되는가?
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="w-4 h-4" />
                  생체 인식 기술을 사용하는가?
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" className="w-4 h-4" />
                  중요 인프라 안전에 영향을 미치는가?
                </label>
                <p className="text-xs italic mt-2">→ 하나라도 "예"면 고위험 AI로 분류. 엄격한 요구사항 적용.</p>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">✅ Step 2: 문서화 (Technical Documentation)</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
                <li>Model Card 작성 (아키텍처, 성능, 한계)</li>
                <li>Datasheet for Dataset (출처, 편향, 라이선스)</li>
                <li>위험 관리 계획서 (Risk Management Plan)</li>
                <li>테스트 보고서 (공정성, 정확성, 견고성)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">✅ Step 3: 투명성 확보</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
                <li>사용자에게 AI 사용 사실 고지 (챗봇 명시)</li>
                <li>의사결정 설명 기능 구현 (SHAP/LIME)</li>
                <li>생성 콘텐츠 워터마크 (C2PA 표준 권장)</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">✅ Step 4: 인간 감독 (Human-in-the-Loop)</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
                <li>고위험 결정은 인간이 최종 승인</li>
                <li>AI 출력을 무시/수정할 수 있는 UI 제공</li>
                <li>긴급 중단(Emergency Stop) 버튼 구현</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold text-gray-900 dark:text-white mb-3">✅ Step 5: 정기 감사</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
                <li>분기별 공정성 지표 모니터링</li>
                <li>연 1회 제3자 적합성 평가 (Conformity Assessment)</li>
                <li>사고 발생 시 15일 내 보고 (중대 사고)</li>
              </ul>
            </div>
          </div>

          <div className="mt-6 bg-yellow-100 dark:bg-yellow-900/30 p-4 rounded-lg">
            <p className="font-semibold text-gray-900 dark:text-white mb-2">💡 권장 도구:</p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside">
              <li>IBM AI Fairness 360 - 편향 탐지</li>
              <li>Google Model Card Toolkit - 문서 자동 생성</li>
              <li>Microsoft Responsible AI Toolbox - 통합 대시보드</li>
              <li>Hugging Face Hub - 모델 투명성 공개</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 공식 규제 문서',
            icon: 'docs' as const,
            color: 'border-rose-500',
            items: [
              {
                title: 'EU AI Act Official Text',
                url: 'https://artificialintelligenceact.eu/',
                description: 'EU 인공지능법 전문 및 해설 (2024.08.01 발효)'
              },
              {
                title: '한국 AI 기본법 (국회 법률정보시스템)',
                url: 'https://www.law.go.kr/',
                description: '인공지능 기본법 전문 (2024.09 국회 통과)'
              },
              {
                title: 'US AI Executive Order',
                url: 'https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/',
                description: '미국 AI 안전 행정명령 (2023.10.30)'
              },
              {
                title: '중국 생성형 AI 관리 조치',
                url: 'http://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm',
                description: '생성형인공지능서비스관리잠행방법 (2023.08.15 시행)'
              }
            ]
          },
          {
            title: '🛠️ 준수 도구 & 템플릿',
            icon: 'tools' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'EU AI Act Compliance Checker',
                url: 'https://artificialintelligenceact.eu/assessment/',
                description: '우리 AI가 어떤 카테고리에 속하는지 자가진단 도구'
              },
              {
                title: 'Model Cards for Model Reporting',
                url: 'https://modelcards.withgoogle.com/about',
                description: 'Google Model Card 생성기 및 템플릿'
              },
              {
                title: 'Microsoft Responsible AI Standard',
                url: 'https://www.microsoft.com/en-us/ai/responsible-ai',
                description: 'Microsoft의 내부 AI 거버넌스 프레임워크 (공개)'
              },
              {
                title: 'ISO/IEC 42001:2023 - AI Management System',
                url: 'https://www.iso.org/standard/81230.html',
                description: 'AI 관리 시스템 국제 표준 (ISO 인증)'
              },
              {
                title: 'NIST AI Risk Management Framework',
                url: 'https://www.nist.gov/itl/ai-risk-management-framework',
                description: '미국 표준기술연구소 AI 위험 관리 가이드'
              }
            ]
          },
          {
            title: '📖 법률 해석 & 가이드',
            icon: 'research' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'EU AI Act: A Practical Guide for Businesses',
                url: 'https://ec.europa.eu/digital-strategy/our-policies/european-approach-artificial-intelligence_en',
                description: 'EU 집행위 공식 비즈니스 가이드'
              },
              {
                title: 'AI & Law: 2024 Global Regulatory Tracker',
                url: 'https://www.dlapiper.com/en/insights/publications/ai-regulation-tracker',
                description: 'DLA Piper 글로벌 AI 규제 추적기 (50+ 국가)'
              },
              {
                title: '한국 AI 윤리기준 (과학기술정보통신부)',
                url: 'https://www.msit.go.kr/',
                description: 'AI 윤리기준 실천 방안 (2020) 및 업데이트'
              },
              {
                title: 'OECD AI Principles',
                url: 'https://oecd.ai/en/ai-principles',
                description: 'OECD AI 원칙 (2019) - 42개국 합의'
              },
              {
                title: 'Future of Life Institute - AI Policy',
                url: 'https://futureoflife.org/ai/policy/',
                description: '전 세계 AI 정책 동향 및 분석 (비영리)'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
