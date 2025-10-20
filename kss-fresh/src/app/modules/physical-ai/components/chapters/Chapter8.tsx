'use client';

import React from 'react';
import { Bot, Zap, Rocket, Factory, Home, Brain, Users, TrendingUp } from 'lucide-react';

export default function Chapter8() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl p-8 mb-8 border border-purple-200 dark:border-purple-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-purple-500 rounded-xl flex items-center justify-center">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">휴머노이드 로봇의 시대</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          Tesla Bot, Figure AI, 1X - 인간형 로봇이 현실이 되다
        </p>
      </div>

      {/* Introduction */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Rocket className="text-pink-600" />
          왜 휴머노이드인가?
        </h2>

        <div className="bg-gradient-to-r from-pink-50 to-purple-50 dark:from-pink-900/20 dark:to-purple-900/20 p-6 rounded-lg border-l-4 border-pink-500 mb-6">
          <h3 className="text-xl font-bold mb-4">🤖 휴머노이드 로봇의 결정적 장점</h3>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg">
              <h4 className="font-bold mb-3 text-blue-600">1. 기존 인프라 활용</h4>
              <p className="text-sm mb-3">
                세상은 <strong>인간을 위해 설계</strong>되어 있습니다.
                계단, 문손잡이, 엘리베이터 버튼, 자동차 - 모두 인간의 신체 구조에 최적화되어 있죠.
              </p>
              <ul className="text-sm space-y-1">
                <li>✅ 공장 재설계 불필요</li>
                <li>✅ 기존 도구 그대로 사용</li>
                <li>✅ 인간 작업 공간에 바로 투입</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg">
              <h4 className="font-bold mb-3 text-green-600">2. 범용성 (Generality)</h4>
              <p className="text-sm mb-3">
                특정 작업만 하는 전문 로봇 대신, <strong>모든 작업을 학습 가능</strong>한 범용 로봇.
                하나의 휴머노이드가 수십 가지 업무를 수행할 수 있습니다.
              </p>
              <ul className="text-sm space-y-1">
                <li>✅ 청소 → 물건 옮기기 → 조립</li>
                <li>✅ 작업 교체 시 재프로그래밍 불필요</li>
                <li>✅ 새로운 작업을 관찰로 학습</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg">
              <h4 className="font-bold mb-3 text-purple-600">3. 인간과의 협업</h4>
              <p className="text-sm mb-3">
                인간과 같은 형태이므로 <strong>자연스러운 소통과 협업</strong>이 가능합니다.
                언어, 제스처, 눈맞춤으로 의사소통할 수 있습니다.
              </p>
              <ul className="text-sm space-y-1">
                <li>✅ "저기 공구 좀 가져다줘" 이해 가능</li>
                <li>✅ 인간의 행동을 관찰하며 학습</li>
                <li>✅ 심리적 거부감 최소화</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg">
              <h4 className="font-bold mb-3 text-orange-600">4. 대량 생산 경제성</h4>
              <p className="text-sm mb-3">
                휴머노이드는 <strong>표준화된 설계</strong>로 대량 생산 가능.
                수백 가지 전문 로봇 대신 하나의 플랫폼으로 통합됩니다.
              </p>
              <ul className="text-sm space-y-1">
                <li>✅ 부품 공통화로 비용 절감</li>
                <li>✅ 소프트웨어 업데이트로 기능 확장</li>
                <li>✅ 규모의 경제 실현</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Tesla Bot (Optimus) */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Zap className="text-red-600" />
          1. Tesla Bot (Optimus) - 일론 머스크의 야심작
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-5 rounded-lg mb-6">
            <h3 className="text-2xl font-bold mb-3">🚗 Tesla의 자율주행 기술을 로봇에 적용</h3>
            <p className="text-lg mb-4">
              Tesla는 수백만 대의 차량에서 <strong>실제 주행 데이터</strong>를 수집하며
              세계 최고 수준의 <strong>비전 AI</strong>를 개발했습니다.
              이 기술을 그대로 휴머노이드에 이식합니다.
            </p>

            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center">
                <div className="text-3xl font-bold text-blue-600 mb-2">8개</div>
                <p className="text-sm">카메라로 360도 인식<br/>(FSD와 동일)</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">28</div>
                <p className="text-sm">자유도 (DOF)<br/>인간에 가까운 관절</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg text-center">
                <div className="text-3xl font-bold text-purple-600 mb-2">73kg</div>
                <p className="text-sm">무게 (성인 남성 평균)<br/>173cm 신장</p>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-l-4 border-blue-500">
              <h4 className="font-bold mb-2">🎯 핵심 전략: Sim2Real + Fleet Learning</h4>
              <p className="text-sm mb-3">
                Tesla는 <strong>가상 환경에서 무한 학습</strong> 후 실제 로봇에 배포합니다.
                수만 대의 Optimus가 배포되면, 각각의 경험이 <strong>클라우드로 공유</strong>되어
                전체 플릿(Fleet)이 함께 진화합니다.
              </p>
              <div className="bg-white dark:bg-gray-800 p-3 rounded text-sm">
                <strong>예시</strong>: 한 Optimus가 창고에서 새로운 상자 쌓기 방법을 학습하면,
                전 세계 모든 Optimus가 다음 날 아침 그 방법을 알게 됩니다.
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
              <h4 className="font-bold mb-2">🏭 목표 시장: 제조업 + 가정용</h4>
              <div className="grid md:grid-cols-2 gap-4 mt-3">
                <div>
                  <h5 className="font-bold text-sm mb-2">Phase 1: Tesla 공장 (2024-2025)</h5>
                  <ul className="text-sm space-y-1">
                    <li>• 부품 조립, 품질 검사</li>
                    <li>• 물류 자동화</li>
                    <li>• 위험 작업 대체</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-bold text-sm mb-2">Phase 2: 대중 시장 (2026+)</h5>
                  <ul className="text-sm space-y-1">
                    <li>• 가정용 가사 도우미</li>
                    <li>• 노인 돌봄 서비스</li>
                    <li>• 목표 가격: $20,000-$25,000</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border-l-4 border-purple-500">
              <h4 className="font-bold mb-2">⚡ Tesla의 차별화: 수직 통합</h4>
              <p className="text-sm mb-3">
                Tesla는 AI 칩, 배터리, 액추에이터, 소프트웨어를 <strong>모두 자체 개발</strong>합니다.
                이는 대량 생산 시 엄청난 비용 우위를 제공합니다.
              </p>
              <div className="grid grid-cols-4 gap-2 text-xs text-center">
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <strong>Dojo 칩</strong><br/>AI 학습
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <strong>4680 배터리</strong><br/>5시간 작동
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <strong>맞춤 액추에이터</strong><br/>28개 관절
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded">
                  <strong>FSD Stack</strong><br/>비전 AI
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Figure AI */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Brain className="text-cyan-600" />
          2. Figure AI - OpenAI와의 파트너십
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 p-5 rounded-lg mb-6">
            <h3 className="text-2xl font-bold mb-3">🤝 OpenAI + Figure = Physical GPT</h3>
            <p className="text-lg mb-4">
              Figure AI는 <strong>OpenAI와 전략적 파트너십</strong>을 맺고,
              GPT-4의 언어 이해 능력을 로봇에 통합했습니다.
              대화형 AI가 물리적 행동을 수행하는 <strong>최초의 상용 휴머노이드</strong>입니다.
            </p>
          </div>

          <div className="space-y-4">
            <div className="bg-cyan-50 dark:bg-cyan-900/20 p-4 rounded-lg border-l-4 border-cyan-500">
              <h4 className="font-bold mb-2">💡 Figure 01 - 1세대 모델 (2023)</h4>
              <div className="grid md:grid-cols-2 gap-4 mt-3">
                <div>
                  <h5 className="font-bold text-sm mb-2">사양</h5>
                  <ul className="text-sm space-y-1">
                    <li>• 신장: 167cm, 무게: 60kg</li>
                    <li>• 최대 하중: 20kg</li>
                    <li>• 배터리: 5시간 작동</li>
                    <li>• 보행 속도: 시속 1.2m/s</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-bold text-sm mb-2">핵심 기술</h5>
                  <ul className="text-sm space-y-1">
                    <li>• <strong>GPT-4V 통합</strong>: 음성 대화</li>
                    <li>• <strong>Vision-Language-Action</strong>: "사과 좀 줘" 이해</li>
                    <li>• <strong>Real-time Planning</strong>: 즉시 행동 계획</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-l-4 border-blue-500">
              <h4 className="font-bold mb-2">🏢 BMW 공장 파일럿 (2024년 1월 발표)</h4>
              <p className="text-sm mb-3">
                Figure AI는 <strong>BMW 제조 공장</strong>에 휴머노이드를 배치하는 계약을 체결했습니다.
                자동차 부품 조립, 품질 검사 등의 작업을 수행합니다.
              </p>
              <div className="bg-white dark:bg-gray-800 p-3 rounded text-sm">
                <strong>목표</strong>: 2025년까지 BMW 미국 공장에 <strong>수백 대</strong> 배치.
                성공 시 글로벌 확대 예정.
              </div>
            </div>

            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg border-l-4 border-indigo-500">
              <h4 className="font-bold mb-2">🎤 데모 하이라이트 (2024년 3월)</h4>
              <p className="text-sm mb-3">
                Figure 01이 사람과 <strong>자연어로 대화하며 작업</strong>하는 영상이 공개되었습니다:
              </p>
              <div className="bg-white dark:bg-gray-800 p-3 rounded space-y-2 text-sm">
                <div><strong>사람:</strong> "배고픈데 뭐 먹을 거 있어?"</div>
                <div><strong>Figure 01:</strong> "사과가 있네요. 드릴게요." (사과를 집어서 건넴)</div>
                <div><strong>사람:</strong> "설거지 좀 해줄래?"</div>
                <div><strong>Figure 01:</strong> "네, 지금 할게요." (컵과 접시를 싱크대로 옮김)</div>
              </div>
              <p className="text-sm mt-3 italic text-gray-600 dark:text-gray-400">
                → 사전 프로그래밍 없이, <strong>맥락을 이해하고 적절히 행동</strong>하는 첫 사례
              </p>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border-l-4 border-purple-500">
              <h4 className="font-bold mb-2">💰 투자 현황</h4>
              <p className="text-sm mb-3">
                2024년 기준, Figure AI는 <strong>$675M (약 9,000억 원)</strong> 투자를 유치했습니다.
              </p>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-center">
                  <strong>OpenAI</strong><br/>AI 기술 파트너
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-center">
                  <strong>NVIDIA</strong><br/>GPU 공급
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded text-center">
                  <strong>Jeff Bezos</strong><br/>개인 투자
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 1X Technologies NEO */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Home className="text-teal-600" />
          3. 1X Technologies NEO - 가정용 휴머노이드
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <div className="bg-gradient-to-r from-teal-50 to-green-50 dark:from-teal-900/20 dark:to-green-900/20 p-5 rounded-lg mb-6">
            <h3 className="text-2xl font-bold mb-3">🏠 가정에 들어갈 첫 휴머노이드</h3>
            <p className="text-lg mb-4">
              노르웨이 스타트업 1X는 <strong>가정용 특화</strong> 휴머노이드를 개발합니다.
              공장보다는 <strong>집안일, 노인 돌봄</strong>에 최적화된 설계입니다.
            </p>
          </div>

          <div className="space-y-4">
            <div className="bg-teal-50 dark:bg-teal-900/20 p-4 rounded-lg border-l-4 border-teal-500">
              <h4 className="font-bold mb-2">🤖 NEO 사양 (2024년 프로토타입)</h4>
              <div className="grid md:grid-cols-2 gap-4 mt-3">
                <div>
                  <h5 className="font-bold text-sm mb-2">디자인 철학</h5>
                  <ul className="text-sm space-y-1">
                    <li>• <strong>안전 제일</strong>: 부드러운 표면, 저속 동작</li>
                    <li>• <strong>조용한 작동</strong>: 가정 환경에 맞춤</li>
                    <li>• <strong>친근한 외형</strong>: 위협적이지 않은 디자인</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-bold text-sm mb-2">주요 작업</h5>
                  <ul className="text-sm space-y-1">
                    <li>• 청소, 빨래 개기, 설거지</li>
                    <li>• 물건 정리 및 배치</li>
                    <li>• 노인 모니터링 (낙상 감지)</li>
                    <li>• 간단한 대화 상대</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
              <h4 className="font-bold mb-2">🧠 핵심 기술: Embodied AI</h4>
              <p className="text-sm mb-3">
                1X는 <strong>EVE</strong>라는 이전 모델로 창고 자동화에서 검증받았습니다.
                NEO는 EVE의 기술을 가정용으로 진화시킨 버전입니다.
              </p>
              <ul className="text-sm space-y-2">
                <li>
                  <strong className="text-green-700 dark:text-green-300">• 학습 데이터 수집</strong>
                  <p className="mt-1">인간이 VR 컨트롤러로 원격 조종하며 작업을 시연하면,
                  NEO가 그 행동을 관찰하고 학습합니다 (Imitation Learning).</p>
                </li>
                <li>
                  <strong className="text-green-700 dark:text-green-300">• 자율 실행</strong>
                  <p className="mt-1">충분한 데이터가 쌓이면 NEO가 독립적으로 작업을 수행합니다.</p>
                </li>
              </ul>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-l-4 border-blue-500">
              <h4 className="font-bold mb-2">💰 비즈니스 모델: RaaS (Robot as a Service)</h4>
              <p className="text-sm mb-3">
                1X는 로봇을 <strong>판매가 아닌 구독형</strong>으로 제공할 계획입니다.
              </p>
              <div className="bg-white dark:bg-gray-800 p-3 rounded text-sm space-y-2">
                <div><strong>예상 가격:</strong> 월 $300-$500 구독 (로봇 무상 제공)</div>
                <div><strong>포함 서비스:</strong> 유지보수, 소프트웨어 업데이트, 고장 시 교체</div>
                <div><strong>장점:</strong> 초기 비용 부담 없이, 가정에서 휴머노이드 경험 가능</div>
              </div>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border-l-4 border-orange-500">
              <h4 className="font-bold mb-2">🎯 목표 시장: 고령화 사회</h4>
              <p className="text-sm mb-3">
                일본, 한국, 유럽 등 <strong>고령화가 심각한 국가</strong>가 1차 목표입니다.
              </p>
              <div className="grid md:grid-cols-2 gap-3 mt-3 text-sm">
                <div className="bg-white dark:bg-gray-800 p-3 rounded">
                  <strong>독거노인 돌봄</strong><br/>
                  <span className="text-xs">낙상 감지, 약 복용 알림, 응급 호출</span>
                </div>
                <div className="bg-white dark:bg-gray-800 p-3 rounded">
                  <strong>가사 부담 경감</strong><br/>
                  <span className="text-xs">맞벌이 가정의 청소, 정리 자동화</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Boston Dynamics Atlas (Electric) */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Factory className="text-orange-600" />
          4. Boston Dynamics Atlas (Electric) - 전기 구동 혁명
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 p-5 rounded-lg mb-6">
            <h3 className="text-2xl font-bold mb-3">🔋 유압 → 전기: 게임 체인저</h3>
            <p className="text-lg mb-4">
              2024년 4월, Boston Dynamics는 <strong>30년 유압 기술을 버리고</strong>
              완전 전기 구동 Atlas를 공개했습니다. 이는 휴머노이드 역사의 전환점입니다.
            </p>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded">
                <h5 className="font-bold mb-2 text-red-600">유압 Atlas (구형)</h5>
                <ul className="text-sm space-y-1">
                  <li>❌ 무게: 89kg (매우 무거움)</li>
                  <li>❌ 작동 시간: 1시간 미만</li>
                  <li>❌ 유압 펌프 소음</li>
                  <li>❌ 유지보수 복잡</li>
                  <li>✅ 강력한 힘 (백플립 가능)</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded border-2 border-green-500">
                <h5 className="font-bold mb-2 text-green-600">전기 Atlas (신형, 2024)</h5>
                <ul className="text-sm space-y-1">
                  <li>✅ 무게: 예상 60kg대 (30% 경량화)</li>
                  <li>✅ 작동 시간: 4-5시간</li>
                  <li>✅ 조용한 작동</li>
                  <li>✅ 유지보수 간편</li>
                  <li>✅ 더 정밀한 제어</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border-l-4 border-orange-500">
              <h4 className="font-bold mb-2">🤸 전기 Atlas의 충격적인 기능</h4>
              <p className="text-sm mb-3">
                2024년 4월 공개 영상에서 Atlas는 <strong>인간의 신체 한계를 초월</strong>하는 동작을 보여줬습니다:
              </p>
              <ul className="text-sm space-y-2">
                <li>
                  <strong>• 360도 관절 회전</strong>
                  <p className="mt-1">인간은 불가능한 방향으로 다리와 팔을 회전시킵니다.
                  넘어진 상태에서 몸을 뒤집지 않고 그대로 일어섭니다.</p>
                </li>
                <li>
                  <strong>• 180도 고개 회전</strong>
                  <p className="mt-1">앞을 보면서 뒤로 걷거나, 뒤를 보면서 앞으로 작업 가능.</p>
                </li>
                <li>
                  <strong>• 유연한 신체 제어</strong>
                  <p className="mt-1">상체와 하체가 독립적으로 움직이며, 좁은 공간도 자유롭게 통과.</p>
                </li>
              </ul>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-l-4 border-red-500">
              <h4 className="font-bold mb-2">🏭 목표 시장: 극한 환경 작업</h4>
              <p className="text-sm mb-3">
                Boston Dynamics는 <strong>위험하고 반복적인 작업</strong>에 Atlas를 투입할 계획입니다.
              </p>
              <div className="grid md:grid-cols-3 gap-3 mt-3 text-sm">
                <div className="bg-white dark:bg-gray-800 p-3 rounded">
                  <strong>자동차 공장</strong><br/>
                  <span className="text-xs">무거운 부품 조립, 용접</span>
                </div>
                <div className="bg-white dark:bg-gray-800 p-3 rounded">
                  <strong>재난 현장</strong><br/>
                  <span className="text-xs">붕괴 건물 탐색, 구조 작업</span>
                </div>
                <div className="bg-white dark:bg-gray-800 p-3 rounded">
                  <strong>우주 탐사</strong><br/>
                  <span className="text-xs">화성 기지 건설 (NASA 협력)</span>
                </div>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-l-4 border-blue-500">
              <h4 className="font-bold mb-2">🤝 Hyundai Motor Group 소유</h4>
              <p className="text-sm mb-3">
                2021년 현대차그룹이 Boston Dynamics를 <strong>$1.1B (약 1조 5,000억 원)</strong>에 인수했습니다.
                현대차의 생산 라인에 Atlas를 직접 투입할 예정입니다.
              </p>
              <div className="bg-white dark:bg-gray-800 p-3 rounded text-sm">
                <strong>시너지</strong>: 현대차의 전기차 배터리 기술 + Boston Dynamics의 로봇 제어 기술
                = 세계 최고 성능의 휴머노이드
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Future of Humanoids */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <TrendingUp className="text-indigo-600" />
          5. 휴머노이드의 미래 - AGI로 가는 길
        </h2>

        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6 shadow-lg mb-6 border-l-4 border-indigo-500">
          <h3 className="text-2xl font-bold mb-4">🧠 Physical AI = AGI의 마지막 피스</h3>
          <p className="text-lg mb-4">
            많은 AI 연구자들은 <strong>진정한 AGI (범용 인공지능)</strong>는
            물리적 세계와 상호작용하며 학습해야 달성 가능하다고 믿습니다.
          </p>

          <div className="bg-white dark:bg-gray-800 p-5 rounded-lg mb-4">
            <h4 className="font-bold mb-3">💡 왜 Physical AI가 AGI에 필수인가?</h4>
            <div className="space-y-3 text-sm">
              <div className="flex items-start gap-2">
                <span className="text-indigo-600 font-bold">1.</span>
                <div>
                  <strong>Grounding Problem (접지 문제) 해결</strong>
                  <p className="mt-1">ChatGPT는 "무거움"을 텍스트로만 알지만,
                  로봇은 <strong>실제로 무거운 물체를 들며</strong> 진정한 의미를 학습합니다.</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-indigo-600 font-bold">2.</span>
                <div>
                  <strong>상식(Common Sense) 획득</strong>
                  <p className="mt-1">인간은 물리 세계와 상호작용하며 상식을 배웁니다.
                  AI도 로봇 몸을 통해 중력, 마찰, 관성을 직접 경험해야 합니다.</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-indigo-600 font-bold">3.</span>
                <div>
                  <strong>Multi-Modal 학습</strong>
                  <p className="mt-1">시각, 촉각, 청각을 동시에 활용해 학습하면
                  훨씬 빠르고 강건한 AI가 됩니다.</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h4 className="font-bold mb-4">🚀 향후 10년 로드맵 (2025-2035)</h4>

          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <div className="bg-blue-500 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold flex-shrink-0">
                2025
              </div>
              <div className="flex-1">
                <strong>공장 배치 시작</strong>
                <p className="text-sm mt-1">Tesla, Figure, Boston Dynamics가 제조 현장에 수백 대 배치.
                초기 작업: 단순 반복 작업 (부품 조립, 검사, 포장)</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="bg-green-500 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold flex-shrink-0">
                2027
              </div>
              <div className="flex-1">
                <strong>가정용 파일럿 프로그램</strong>
                <p className="text-sm mt-1">1X NEO가 고소득 가정, 노인 가구를 대상으로 베타 테스트.
                가격: $50,000-$80,000 (얼리어답터 대상)</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="bg-purple-500 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold flex-shrink-0">
                2030
              </div>
              <div className="flex-1">
                <strong>대량 생산 & 가격 하락</strong>
                <p className="text-sm mt-1">Tesla Optimus가 연간 <strong>100만 대 생산</strong> 달성.
                가격이 $20,000 이하로 내려가며 중산층도 구매 가능.</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="bg-orange-500 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold flex-shrink-0">
                2033
              </div>
              <div className="flex-1">
                <strong>사회 인프라 통합</strong>
                <p className="text-sm mt-1">편의점, 음식점, 호텔에서 휴머노이드가 일상적으로 근무.
                "로봇 직원"이 특별하지 않은 시대.</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="bg-red-500 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold flex-shrink-0">
                2035
              </div>
              <div className="flex-1">
                <strong>AGI 돌파 가능성</strong>
                <p className="text-sm mt-1">수억 대의 휴머노이드가 축적한 <strong>Physical World Data</strong>로
                AGI 수준의 범용 지능 출현 가능.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Challenges & Ethics */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Users className="text-red-600" />
          6. 도전과제 & 윤리적 고려사항
        </h2>

        <div className="space-y-4">
          <div className="bg-red-50 dark:bg-red-900/20 p-5 rounded-lg border-l-4 border-red-500">
            <h4 className="font-bold mb-3">⚠️ 일자리 대체 문제</h4>
            <p className="text-sm mb-3">
              McKinsey 추정: 휴머노이드가 <strong>전 세계 일자리의 30%</strong>를 대체 가능 (2030년대).
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded text-sm">
              <strong>대응 방안</strong>:
              <ul className="mt-2 space-y-1">
                <li>• 재교육 프로그램: 로봇 유지보수, AI 트레이닝 전문가 양성</li>
                <li>• 기본소득(UBI) 논의 가속화</li>
                <li>• 인간은 창의적 업무로 이동 (디자인, 기획, 예술)</li>
              </ul>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 p-5 rounded-lg border-l-4 border-orange-500">
            <h4 className="font-bold mb-3">🔒 안전성 & 제어 문제</h4>
            <p className="text-sm mb-3">
              휴머노이드는 인간 주변에서 작동하므로 <strong>절대적 안전</strong>이 필요합니다.
            </p>
            <ul className="text-sm space-y-2">
              <li>• <strong>Fail-Safe 메커니즘</strong>: 오작동 시 즉시 정지</li>
              <li>• <strong>힘 제한</strong>: 인간을 다치게 할 만한 힘 출력 방지</li>
              <li>• <strong>투명한 의사결정</strong>: AI가 왜 그 행동을 했는지 설명 가능</li>
            </ul>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border-l-4 border-purple-500">
            <h4 className="font-bold mb-3">🤝 인간-로봇 관계</h4>
            <p className="text-sm mb-3">
              휴머노이드가 가정에 들어오면 <strong>새로운 사회적 관계</strong>가 형성됩니다.
            </p>
            <ul className="text-sm space-y-2">
              <li>• 로봇에 대한 <strong>감정적 애착</strong> 발생 가능 (Companion Robot)</li>
              <li>• 노인, 장애인에게는 <strong>삶의 질 향상</strong> 도구</li>
              <li>• 하지만 <strong>과도한 의존</strong>으로 인간 간 소통 감소 우려</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Summary */}
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 border-l-4 border-purple-500 p-6 rounded-lg my-8">
        <h3 className="text-xl font-bold mb-3">📌 핵심 요약</h3>
        <ul className="space-y-2 text-sm">
          <li>✅ <strong>Tesla Bot (Optimus)</strong>: FSD 기술 활용, $20K 목표, Fleet Learning</li>
          <li>✅ <strong>Figure AI</strong>: OpenAI GPT-4 통합, BMW 공장 배치, 대화형 로봇</li>
          <li>✅ <strong>1X NEO</strong>: 가정용 특화, RaaS 모델, 고령화 시장 공략</li>
          <li>✅ <strong>Boston Dynamics Atlas</strong>: 전기 구동 전환, 360도 관절, 현대차 소유</li>
          <li>✅ <strong>2025-2035</strong>: 공장 → 가정 → 사회 인프라 → AGI 돌파 로드맵</li>
          <li>⚠️ <strong>과제</strong>: 일자리 대체, 안전성, 윤리적 관계 정립</li>
        </ul>
      </div>

      {/* Next Chapter */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-6 rounded-lg">
        <h3 className="text-xl font-bold mb-2">다음 단계: Physical AI와 미래 사회</h3>
        <p className="text-gray-700 dark:text-gray-300">
          다음 챕터에서는 Physical AI가 <strong>산업, 사회, 경제</strong>에 미칠 장기적 영향과
          인류가 준비해야 할 사항들을 종합적으로 다룹니다.
        </p>
      </div>
    </div>
  )
}