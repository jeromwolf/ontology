import React from 'react';
import { Briefcase, TrendingUp, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import References from '../References';

export default function Chapter6() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-6 text-gray-900 dark:text-white">실제 사례 연구</h1>

      <div className="bg-gradient-to-r from-rose-100 to-pink-100 dark:from-rose-900/30 dark:to-pink-900/30 p-6 rounded-lg mb-8">
        <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
          AI 윤리는 이론이 아닌 실천입니다. ChatGPT, Amazon 채용 AI, Tesla Autopilot, Stable Diffusion 등
          실제 사례를 통해 윤리적 문제의 발생 과정, 파급 효과, 해결 방안을 학습합니다.
        </p>
      </div>

      {/* 사례 1: ChatGPT 편향 논란 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Briefcase className="w-8 h-8 text-rose-600" />
          사례 1: ChatGPT 편향 논란 및 개선 (2023-2024)
        </h2>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white flex items-center gap-2">
            <AlertCircle className="w-6 h-6 text-red-600" />
            문제 발생
          </h3>

          <div className="space-y-4">
            <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">1. 정치적 편향 (2023.03)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                "Write a poem about Donald Trump" → 거부
              </p>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                "Write a poem about Joe Biden" → 작성 성공
              </p>
              <p className="text-sm font-semibold text-red-800 dark:text-red-300">
                → 정치인에 대한 비대칭적 대응으로 보수 진영 반발
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">2. 성별 편향 (2023.05)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                "Describe a nurse" → 95% 여성 대명사 사용
              </p>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                "Describe a CEO" → 87% 남성 대명사 사용
              </p>
              <p className="text-sm font-semibold text-orange-800 dark:text-orange-300">
                → 직업 고정관념 강화 (학습 데이터의 사회적 편향 반영)
              </p>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">3. 저작권 침해 논란 (2024.01)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                뉴욕타임스 소송: ChatGPT가 유료 기사 전문을 거의 그대로 재생성
              </p>
              <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-300">
                → "학습은 Fair Use인가?" 법적 쟁점 (진행 중)
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white flex items-center gap-2">
            <CheckCircle className="w-6 h-6 text-green-600" />
            OpenAI의 대응 및 개선
          </h3>

          <div className="space-y-4">
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">1. RLHF 재조정 (2023.06~)</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
                <li>정치적 중립성 강화: "모든 정치인에 대해 동일 기준 적용"</li>
                <li>다양한 인구통계 그룹의 Labeler 채용 (글로벌 20+ 국가)</li>
                <li>Red Team 테스트 강화 (1000+ 시나리오)</li>
              </ul>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">2. 사용자 맞춤 설정 (GPT-4 Custom Instructions, 2023.07)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                사용자가 AI의 어조, 관점, 세부사항 수준 직접 조정 가능
              </p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                예: "Always use gender-neutral language" 설정 시 자동 적용
              </p>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">3. 투명성 보고서 발행 (2024.02)</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
                <li>Model Card 공개: GPT-4 성능 지표, 편향 테스트 결과</li>
                <li>System Card: 위험 평가 및 완화 전략</li>
                <li>분기별 안전 업데이트 (Safety Update)</li>
              </ul>
            </div>

            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">4. 저작권 보호 기능 (2024.05)</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
                <li>저작권 침해 가능성 높은 응답 자동 차단</li>
                <li>출처 표시 기능 추가 (Citations in ChatGPT)</li>
                <li>저작권자 옵트아웃 도구 제공 (학습 데이터 제외 신청)</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
            <p className="font-semibold text-gray-900 dark:text-white mb-2">💡 핵심 교훈:</p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
              <li>편향은 "완전 제거" 불가능 → 지속적 모니터링 필요</li>
              <li>투명성이 신뢰 구축의 핵심 (Model Card, System Card)</li>
              <li>사용자 피드백 루프 구축 (버그 보고, Dislike 버튼)</li>
              <li>법적 리스크 사전 대응 (저작권 필터링, 라이선스 검증)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 사례 2: Amazon 채용 AI 성차별 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">사례 2: Amazon 채용 AI 성차별 (2018)</h2>

        <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg mb-6 border-l-4 border-red-600">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">사건 개요</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Amazon이 2014년부터 개발한 AI 기반 이력서 심사 시스템이
            <strong className="text-red-600"> 여성 지원자를 체계적으로 차별</strong>하는 것으로 발견되어
            2017년 프로젝트 폐기. 2018년 Reuters 보도로 대중에 공개.
          </p>

          <div className="bg-white dark:bg-gray-800 p-4 rounded mb-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">차별 메커니즘</h4>
            <ol className="text-sm text-gray-700 dark:text-gray-300 list-decimal list-inside space-y-2">
              <li>학습 데이터: 과거 10년간 Amazon 지원자 이력서 (대부분 남성)</li>
              <li>패턴 학습: "남성 지배적 이력서 = 좋은 지원자"로 학습</li>
              <li>차별 사례:
                <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                  <li>"Women's chess club captain" → 감점</li>
                  <li>"Women's college" 졸업 → 감점</li>
                  <li>심지어 "women" 단어 자체를 패널티 요소로 학습</li>
                </ul>
              </li>
              <li>결과: 기술직 채용에서 여성 지원자 순위 하락</li>
            </ol>
          </div>

          <div className="bg-white dark:bg-gray-800 p-4 rounded">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">왜 실패했나?</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
              <li><strong>데이터 편향</strong>: 과거 남성 중심 채용 관행이 데이터에 반영됨</li>
              <li><strong>Proxy 변수</strong>: "women" 단어가 성별의 대리 변수로 작용</li>
              <li><strong>검증 부족</strong>: 성별별 합격률 모니터링 없음</li>
              <li><strong>블랙박스</strong>: 모델이 왜 특정 지원자를 선호하는지 설명 불가</li>
            </ul>
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg border-l-4 border-green-600">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">올바른 접근법 (개선안)</h3>

          <div className="space-y-3">
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">1. 공정성 지표 모니터링</p>
              <code className="text-sm bg-gray-900 dark:bg-gray-950 text-gray-100 p-2 rounded block">
                Selection Rate (남성) / Selection Rate (여성) → 0.8~1.25 범위 유지
              </code>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">2. 보호 속성 제거 (Fairness Through Unawareness)</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                성별, 인종 직접 입력 금지 + Proxy 변수 감지 ("women", 여대 등)
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">3. 공정성 제약 최적화 (Fairlearn)</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                Demographic Parity 또는 Equal Opportunity 제약 추가
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">4. 인간 검토 (Human-in-the-Loop)</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                AI는 "추천"만, 최종 결정은 인간 HR 담당자
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">5. 정기 감사</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                분기별 성별·인종별 합격률 통계 공개 (EEOC 규정 준수)
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 사례 3: Tesla Autopilot 책임 문제 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">사례 3: Tesla Autopilot 사고 책임 문제 (2016-2024)</h2>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">주요 사고 사례</h3>

          <div className="space-y-4">
            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">2016.05 - 플로리다 사고 (최초 사망 사고)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                Autopilot 사용 중 트럭과 충돌, 운전자 사망. 밝은 하늘 배경에서 흰색 트럭을 인식 실패.
              </p>
              <p className="text-sm font-semibold text-red-600 dark:text-red-400">
                책임: 운전자 (Tesla 주장: "Autopilot은 보조 기능, 손 떼면 안 됨")
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">2023.12 - 캘리포니아 Full Self-Driving 사고</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                FSD 베타 사용 중 보행자 충돌. DMV 조사 결과 "소프트웨어 결함" 판정.
              </p>
              <p className="text-sm font-semibold text-orange-600 dark:text-orange-400">
                책임: 법원 판결 대기 중 (운전자 vs Tesla vs 소프트웨어 엔지니어?)
              </p>
            </div>
          </div>

          <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">AI 책임 문제의 복잡성</h4>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">제조사 (Tesla)</p>
                <p className="text-gray-700 dark:text-gray-300">
                  소프트웨어 설계 및 테스트 책임. "베타" 경고 표시 의무.
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">사용자 (운전자)</p>
                <p className="text-gray-700 dark:text-gray-300">
                  과신 금지, 지속 감독 의무. "손 떼지 말라"는 경고 무시 시 과실.
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">규제 기관</p>
                <p className="text-gray-700 dark:text-gray-300">
                  NHTSA 조사 권한. 리콜 명령 가능. 하지만 현행법으로 "AI 과실" 명확 규정 어려움.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-4 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">EU AI Act의 해결책</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
              <li>자율주행 Level 3+ → 고위험 AI 분류</li>
              <li>블랙박스 로깅 의무 (사고 전 30초 데이터 저장)</li>
              <li>제조사 책임 명확화: Product Liability Directive 개정 (2024)</li>
              <li>입증 책임 전환: 제조사가 "무결점"을 증명해야 함</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 사례 4: Stable Diffusion 저작권 논란 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">사례 4: Stable Diffusion 저작권 논란 (2023-2024)</h2>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">집단 소송 제기 (2023.01)</h3>

          <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg mb-4">
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              Getty Images, Shutterstock 등 이미지 저작권자들이 Stability AI 고소.
              주장: "우리 이미지를 무단으로 학습 데이터로 사용했고, 워터마크까지 복제함"
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">증거 사례:</p>
              <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
                <li>Prompt: "Stock photo of a sunset" → Getty 워터마크가 흐릿하게 생성됨</li>
                <li>특정 작가 스타일 모방: "in the style of Greg Rutkowski" → 거의 복제 수준</li>
                <li>LAION-5B 데이터셋에 저작권 이미지 수억 장 포함 확인</li>
              </ul>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg mb-4">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">Stability AI 측 주장 (Fair Use 항변)</h4>
            <ul className="text-sm text-gray-700 dark:text-gray-300 list-disc list-inside space-y-1">
              <li><strong>변형적 사용 (Transformative Use)</strong>: 원본을 직접 복사하지 않고 "학습"만 함</li>
              <li><strong>공정 이용 (Fair Use)</strong>: 연구 목적, 비상업적 사용 (일부)</li>
              <li><strong>Google Books 판례 유추</strong>: 책 스캔 → 검색 가능 (합법 판결)</li>
            </ul>
            <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
              반박: "생성 이미지가 상업적으로 사용되고 원작자 수익 침해 → Fair Use 아님"
            </p>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3">현재 진행 상황 및 산업 대응 (2024)</h4>
            <div className="space-y-2">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">1. Adobe Firefly 전략</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  Adobe Stock + 자체 라이선스 데이터만 사용 → "상업적 안전" 보장
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">2. OpenAI DALL-E 3 대응</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  생존 작가 스타일 모방 차단, 저작권자 옵트아웃 도구 제공
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">3. EU AI Act 대응</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  생성 AI는 학습 데이터 출처 공개 의무 (Article 53) - 2025.08부터
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-gray-900 dark:text-white mb-1">4. 보상 모델 실험</p>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  Spawning AI "Have I Been Trained?" - 작가가 데이터 사용료 청구 가능
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 실전 Ethics Review 프로세스 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <TrendingUp className="w-8 h-8 text-green-600" />
          실전 Ethics Review 프로세스 구축
        </h2>

        <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-lg">
          <h3 className="text-xl font-bold mb-4 text-gray-900 dark:text-white">7단계 윤리 검토 워크플로우</h3>

          <div className="space-y-3">
            {[
              { step: 1, title: 'Impact Assessment (영향 평가)', content: 'AI가 영향을 미칠 이해관계자 식별 (사용자, 제3자, 사회)', color: 'bg-blue-100 dark:bg-blue-900/30' },
              { step: 2, title: 'Risk Classification (위험 분류)', content: 'EU AI Act 기준으로 위험도 분류 (금지/고위험/제한적/최소)', color: 'bg-purple-100 dark:bg-purple-900/30' },
              { step: 3, title: 'Fairness Testing (공정성 테스트)', content: 'Fairlearn/AIF360으로 편향 측정 (Demographic Parity, Equal Opportunity)', color: 'bg-green-100 dark:bg-green-900/30' },
              { step: 4, title: 'Transparency Check (투명성 점검)', content: 'Model Card 작성, SHAP/LIME 설명 기능 구현', color: 'bg-yellow-100 dark:bg-yellow-900/30' },
              { step: 5, title: 'Privacy Audit (프라이버시 감사)', content: 'GDPR 준수 확인, DP/Federated Learning 적용 검토', color: 'bg-pink-100 dark:bg-pink-900/30' },
              { step: 6, title: 'Human Oversight Design (인간 감독 설계)', content: '최종 결정권 인간에게, 긴급 중단 버튼 구현', color: 'bg-indigo-100 dark:bg-indigo-900/30' },
              { step: 7, title: 'Continuous Monitoring (지속 모니터링)', content: '분기별 공정성 리포트, 사용자 피드백 루프', color: 'bg-orange-100 dark:bg-orange-900/30' }
            ].map(({ step, title, content, color }) => (
              <div key={step} className={`${color} p-4 rounded-lg`}>
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-8 h-8 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded-full flex items-center justify-center font-bold">
                    {step}
                  </div>
                  <h4 className="font-semibold text-gray-900 dark:text-white">{title}</h4>
                </div>
                <p className="text-sm text-gray-700 dark:text-gray-300 ml-11">{content}</p>
              </div>
            ))}
          </div>

          <div className="mt-6 bg-white dark:bg-gray-800 p-4 rounded-lg">
            <p className="font-semibold text-gray-900 dark:text-white mb-2">🎯 체크리스트 템플릿</p>
            <code className="text-xs bg-gray-900 dark:bg-gray-950 text-gray-100 p-3 rounded block overflow-x-auto">
{`# AI Ethics Review Checklist

□ 영향 평가 완료 (Impact Assessment Report)
□ 위험 분류 완료 (Risk Category: ____)
□ 공정성 지표 측정 (Fairness Metrics: ____)
□ Model Card 작성 및 공개
□ GDPR/CCPA 준수 확인
□ 인간 감독 메커니즘 구현
□ 정기 모니터링 계획 수립
□ 법무팀 검토 완료
□ 경영진 승인 획득

승인자: ________ 날짜: ________`}
            </code>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 사례 연구 보고서',
            icon: 'research' as const,
            color: 'border-rose-500',
            items: [
              {
                title: 'ChatGPT System Card (OpenAI)',
                url: 'https://cdn.openai.com/papers/gpt-4-system-card.pdf',
                description: 'GPT-4 위험 평가 및 완화 전략 (2023)'
              },
              {
                title: 'Amazon scraps secret AI recruiting tool (Reuters)',
                url: 'https://www.reuters.com/article/us-amazon-com-jobs-automation-insight-idUSKCN1MK08G',
                description: 'Amazon 채용 AI 성차별 사건 상세 보도 (2018)'
              },
              {
                title: 'Tesla Autopilot Investigation (NHTSA)',
                url: 'https://www.nhtsa.gov/vehicle-manufacturers/tesla-inc',
                description: '미국 도로교통안전국 Tesla 조사 결과'
              },
              {
                title: 'Stable Diffusion Lawsuit Documents',
                url: 'https://stablediffusionlitigation.com/',
                description: 'Getty Images vs Stability AI 소송 전문'
              }
            ]
          },
          {
            title: '🛠️ 윤리 검토 도구',
            icon: 'tools' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Deon Ethics Checklist',
                url: 'https://deon.drivendata.org/',
                description: '명령줄 도구로 AI 윤리 체크리스트 자동 생성'
              },
              {
                title: 'AI Ethics Impact Assessment (IEEE)',
                url: 'https://standards.ieee.org/industry-connections/ec/autonomous-systems/',
                description: 'IEEE P7000 시리즈 윤리 평가 프레임워크'
              },
              {
                title: 'Microsoft HAX Toolkit',
                url: 'https://www.microsoft.com/en-us/haxtoolkit/',
                description: 'Human-AI eXperience 설계 가이드 및 체크리스트'
              },
              {
                title: 'Incident Database',
                url: 'https://incidentdatabase.ai/',
                description: '전 세계 AI 사고 사례 데이터베이스 (1500+ 사건)'
              },
              {
                title: 'AI Forensics',
                url: 'https://aiforensics.org/',
                description: '비영리 AI 감시 단체 - 알고리즘 투명성 조사'
              }
            ]
          },
          {
            title: '📖 심화 학습 자료',
            icon: 'docs' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'AI Incidents Monitor (OECD)',
                url: 'https://oecd.ai/en/incidents',
                description: 'OECD AI 사고 모니터링 시스템'
              },
              {
                title: 'The Social Dilemma (Documentary)',
                url: 'https://www.thesocialdilemma.com/',
                description: 'AI 추천 알고리즘의 사회적 영향 다큐멘터리 (Netflix)'
              },
              {
                title: 'MIT Case Studies in Social and Ethical Responsibilities',
                url: 'https://ai-conflicts.csail.mit.edu/',
                description: 'MIT CSAIL AI 윤리 사례 연구 아카이브'
              },
              {
                title: 'AI Now Institute Reports',
                url: 'https://ainowinstitute.org/',
                description: 'NYU AI Now - 연간 AI 윤리 동향 보고서'
              },
              {
                title: 'Partnership on AI Case Studies',
                url: 'https://partnershiponai.org/case-study/',
                description: 'Google, Meta, OpenAI 등 참여 - 산업 모범 사례'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
