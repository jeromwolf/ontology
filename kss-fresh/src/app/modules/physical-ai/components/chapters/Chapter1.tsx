'use client';

import React from 'react';
import { Brain, Cpu, Factory, Rocket, TrendingUp, Globe, Zap, Target } from 'lucide-react';

export default function Chapter1() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-2xl p-8 mb-8 border border-purple-200 dark:border-purple-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-purple-500 rounded-xl flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">Physical AI 개요와 미래</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          50조 달러 시장, 디지털에서 현실로 - AI의 패러다임 전환
        </p>
      </div>

      {/* 1. Physical AI Definition */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Cpu className="text-purple-600" />
          1. Physical AI란 무엇인가?
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <p className="text-lg mb-4">
            <strong>Physical AI</strong>는 현실 세계와 직접 상호작용하는 인공지능 시스템입니다.
            디지털 환경에서만 작동하는 전통적인 AI와 달리, Physical AI는 <strong>센서, 로봇,
            액추에이터</strong>를 통해 물리적 세계를 인식하고 조작합니다.
          </p>

          <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 p-5 rounded-lg border-l-4 border-purple-500">
            <h4 className="font-bold mb-3">🧠 Brain in a Jar vs Brain in a Body</h4>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-gray-800 p-5 rounded-lg">
                <h5 className="font-bold text-blue-600 mb-2">Digital AI (Brain in a Jar)</h5>
                <p className="text-sm mb-3 italic">병 속의 뇌 - 디지털 영역에만 존재</p>
                <ul className="text-sm space-y-2">
                  <li>💬 <strong>ChatGPT, GPT-4</strong>: 텍스트 생성, 대화</li>
                  <li>🎨 <strong>이미지 생성 AI</strong>: DALL-E, Midjourney</li>
                  <li>🎵 <strong>음악 생성 AI</strong>: Suno, Udio</li>
                  <li>📊 <strong>데이터 분석 AI</strong>: 예측, 추천</li>
                  <li>⚠️ <strong>한계</strong>: 가상 공간에서만 작동, 물리적 행동 불가</li>
                </ul>
              </div>

              <div className="bg-white dark:bg-gray-800 p-5 rounded-lg border-2 border-purple-500">
                <h5 className="font-bold text-purple-600 mb-2">Physical AI (Brain in a Body)</h5>
                <p className="text-sm mb-3 italic">몸을 가진 뇌 - 현실 세계와 직접 상호작용</p>
                <ul className="text-sm space-y-2">
                  <li>🤖 <strong>휴머노이드 로봇</strong>: Tesla Bot, Figure AI</li>
                  <li>🦾 <strong>산업 로봇</strong>: 제조, 물류, 건설</li>
                  <li>🚗 <strong>자율주행 차량</strong>: Waymo, Tesla FSD</li>
                  <li>✈️ <strong>드론</strong>: 배송, 감시, 구조</li>
                  <li>✅ <strong>혁신</strong>: 물리적 세계에서 실제 행동, 환경과 상호작용</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-500 p-6 rounded-lg">
          <h4 className="font-bold mb-2 flex items-center gap-2">
            <Zap className="text-orange-600" size={20} />
            왜 지금 Physical AI인가?
          </h4>
          <ul className="space-y-2 text-sm">
            <li>✅ <strong>AI 모델의 발전</strong>: ChatGPT로 입증된 대규모 언어 모델 기술을 로봇에 적용</li>
            <li>✅ <strong>하드웨어 성능 향상</strong>: NVIDIA GPU, 센서 기술, 배터리 혁신</li>
            <li>✅ <strong>시뮬레이션 기술</strong>: 가상 환경에서 무한 학습 가능 (Sim2Real)</li>
            <li>✅ <strong>비용 감소</strong>: 로봇 부품 가격 하락, 클라우드 AI 접근성 향상</li>
            <li>✅ <strong>사회적 니즈</strong>: 고령화, 인력 부족, 위험 작업 자동화 필요성</li>
          </ul>
        </div>
      </section>

      {/* 2. 50 Trillion Dollar Market */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <TrendingUp className="text-green-600" />
          2. 50조 달러 시장의 탄생
        </h2>

        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6 shadow-lg mb-6 border-l-4 border-green-500">
          <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Rocket className="text-green-600" />
            젠슨 황 CEO, CES 2025 기조연설
          </h3>

          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Physical AI 시장 규모 예측</p>
              <div className="text-5xl font-bold text-green-600 mb-2">50조 달러</div>
              <p className="text-xl font-bold text-gray-900 dark:text-white mb-2">약 7경 2,000조 원</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">2025-2035 예상 시장 가치</p>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Citi 전망 (2050년)</p>
              <div className="text-5xl font-bold text-blue-600 mb-2">40억 대</div>
              <p className="text-xl font-bold text-gray-900 dark:text-white mb-2">AI 로봇</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">전 세계 배포 예상 대수</p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-5 rounded-lg">
            <h4 className="font-bold mb-3">📊 시장 규모 비교 (참고용)</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded">
                <span className="font-medium">🤖 Physical AI 시장 (예측)</span>
                <span className="text-green-600 font-bold">50조 달러</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded">
                <span className="font-medium">🌍 전 세계 GDP (2024년)</span>
                <span className="text-blue-600 font-bold">약 100조 달러</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded">
                <span className="font-medium">💰 미국 GDP (2024년)</span>
                <span className="text-purple-600 font-bold">약 28조 달러</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded">
                <span className="font-medium">🏭 전 세계 제조업 (2024년)</span>
                <span className="text-orange-600 font-bold">약 16조 달러</span>
              </div>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-4 italic">
              → Physical AI 시장은 <strong>현재 제조업 전체의 3배 이상</strong> 규모로 예상됨
            </p>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h4 className="font-bold mb-4 flex items-center gap-2">
            <Target className="text-indigo-600" />
            왜 50조 달러인가?
          </h4>

          <div className="space-y-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-l-4 border-blue-500">
              <h5 className="font-bold mb-2">1️⃣ 모든 산업의 혁명</h5>
              <p className="text-sm">
                Physical AI는 제조업, 물류, 건설, 농업, 의료, 국방, 우주 등
                <strong>거의 모든 산업에 적용 가능</strong>합니다.
                단순 자동화를 넘어 <strong>자율적 의사결정과 적응</strong>까지 구현합니다.
              </p>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
              <h5 className="font-bold mb-2">2️⃣ 인력 부족 해결</h5>
              <p className="text-sm">
                전 세계적인 <strong>고령화와 저출산</strong>으로 인한 노동력 감소 문제를
                Physical AI가 해결할 수 있습니다.
                위험하거나 반복적인 작업은 로봇이, 창의적 업무는 인간이 담당합니다.
              </p>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border-l-4 border-purple-500">
              <h5 className="font-bold mb-2">3️⃣ 생산성 폭발적 증가</h5>
              <p className="text-sm">
                AI 로봇은 <strong>24시간 작업 가능</strong>, 인간보다 빠르고 정확하며,
                학습을 통해 계속 개선됩니다.
                생산성이 10배, 100배 증가하면 새로운 경제 가치가 창출됩니다.
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border-l-4 border-orange-500">
              <h5 className="font-bold mb-2">4️⃣ 새로운 비즈니스 모델</h5>
              <p className="text-sm">
                <strong>RaaS (Robots as a Service)</strong>: 로봇을 소유하지 않고 구독형으로 이용<br/>
                <strong>디지털 트윈</strong>: 가상 시뮬레이션으로 제품 개발 비용 90% 절감<br/>
                <strong>자율 공급망</strong>: AI가 생산-물류-배송을 완전 자동화
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 3. NVIDIA COSMOS Vision */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Globe className="text-indigo-600" />
          3. NVIDIA COSMOS - Physical AI의 미래 설계도
        </h2>

        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-2xl font-bold mb-4">🌌 COSMOS란?</h3>
          <p className="text-lg mb-4">
            <strong>COSMOS (Compute Orchestration System for Multimodal Operating Systems)</strong>는
            NVIDIA가 제시하는 Physical AI를 위한 <strong>통합 플랫폼 비전</strong>입니다.
            디지털 세계와 물리 세계를 완벽히 연결하는 AI 생태계를 구축합니다.
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg">
              <h4 className="font-bold mb-3 text-indigo-600">🎯 COSMOS의 핵심 개념</h4>
              <ul className="space-y-3 text-sm">
                <li>
                  <strong className="text-purple-600">🌐 Omniverse</strong>
                  <p className="mt-1">물리 법칙을 완벽히 시뮬레이션하는 가상 세계 플랫폼.
                  실제 공장, 도시, 로봇을 디지털로 복제하여 테스트합니다.</p>
                </li>
                <li>
                  <strong className="text-blue-600">🔄 Digital Twin</strong>
                  <p className="mt-1">현실 세계의 디지털 쌍둥이. 실시간 센서 데이터를
                  받아 가상 모델이 현실과 동기화됩니다.</p>
                </li>
                <li>
                  <strong className="text-green-600">🤖 Sim2Real</strong>
                  <p className="mt-1">시뮬레이션에서 학습한 AI를 현실 로봇에 즉시 배포.
                  가상에서 100만 번 실패해도 안전합니다.</p>
                </li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg">
              <h4 className="font-bold mb-3 text-orange-600">⚡ COSMOS가 가능하게 하는 것</h4>
              <ul className="space-y-3 text-sm">
                <li>
                  <strong>1. 무한 학습 환경</strong>
                  <p className="mt-1">가상 공장에서 로봇이 하루에 수천 년치 경험을 쌓을 수 있습니다.</p>
                </li>
                <li>
                  <strong>2. 제로 리스크 테스트</strong>
                  <p className="mt-1">위험한 상황, 극한 환경을 가상에서 먼저 테스트해
                  사고를 사전에 방지합니다.</p>
                </li>
                <li>
                  <strong>3. 협업 시뮬레이션</strong>
                  <p className="mt-1">전 세계 엔지니어가 동일한 디지털 트윈에 접속해
                  함께 설계하고 개선합니다.</p>
                </li>
                <li>
                  <strong>4. AI 진화의 가속화</strong>
                  <p className="mt-1">실제 환경에서는 1년 걸릴 학습을 1주일로 단축합니다.</p>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h4 className="font-bold mb-4">🏭 COSMOS 활용 사례</h4>

          <div className="space-y-4">
            <div className="bg-cyan-50 dark:bg-cyan-900/20 p-4 rounded-lg">
              <h5 className="font-bold text-cyan-700 dark:text-cyan-300 mb-2">BMW 가상 공장</h5>
              <p className="text-sm">
                BMW는 NVIDIA Omniverse를 사용해 <strong>완전한 디지털 공장</strong>을 구축했습니다.
                실제 공장을 짓기 전에 가상에서 모든 생산 라인, 로봇 배치, 물류 흐름을
                시뮬레이션하여 <strong>개발 시간 30% 단축, 비용 20% 절감</strong>을 달성했습니다.
              </p>
            </div>

            <div className="bg-teal-50 dark:bg-teal-900/20 p-4 rounded-lg">
              <h5 className="font-bold text-teal-700 dark:text-teal-300 mb-2">Amazon 물류 로봇</h5>
              <p className="text-sm">
                Amazon은 Sim2Real 기술로 창고 로봇을 훈련합니다.
                <strong>가상 창고에서 수백만 개의 물건을 분류</strong>하며 학습한 후,
                실제 물류센터에 배포합니다. 학습 기간을 <strong>6개월에서 2주로 단축</strong>했습니다.
              </p>
            </div>

            <div className="bg-violet-50 dark:bg-violet-900/20 p-4 rounded-lg">
              <h5 className="font-bold text-violet-700 dark:text-violet-300 mb-2">Foxconn Virtual-First 전략</h5>
              <p className="text-sm">
                세계 최대 전자제품 제조사 Foxconn은 <strong>실제 공장을 짓기 전에
                디지털 트윈을 먼저 구축</strong>합니다. 가상 환경에서 수천 번의 시뮬레이션을
                실행해 완벽한 설계를 만든 후 현실에 건설합니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Physical AI Ecosystem */}
      <section className="my-8">
        <h2 className="flex items-center gap-2">
          <Factory className="text-blue-600" />
          4. Physical AI 생태계
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h4 className="font-bold mb-4">🔗 Physical AI를 구성하는 핵심 요소</h4>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">👁️</div>
              <h5 className="font-bold mb-2">Perception (인지)</h5>
              <p className="text-sm">카메라, 라이다, 레이더로 환경을 3D 이해</p>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">🧠</div>
              <h5 className="font-bold mb-2">Cognition (사고)</h5>
              <p className="text-sm">AI 모델이 상황을 분석하고 결정</p>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">🦾</div>
              <h5 className="font-bold mb-2">Action (행동)</h5>
              <p className="text-sm">로봇 팔, 바퀴, 드론이 물리적 작업 수행</p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">🔄</div>
              <h5 className="font-bold mb-2">Learning (학습)</h5>
              <p className="text-sm">경험을 통해 지속적으로 개선</p>
            </div>

            <div className="bg-cyan-50 dark:bg-cyan-900/20 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">☁️</div>
              <h5 className="font-bold mb-2">Cloud (클라우드)</h5>
              <p className="text-sm">대규모 AI 모델 학습 및 배포</p>
            </div>

            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg text-center">
              <div className="text-3xl mb-2">⚡</div>
              <h5 className="font-bold mb-2">Edge (엣지)</h5>
              <p className="text-sm">실시간 의사결정을 위한 현장 AI 칩</p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-lg border-l-4 border-purple-500">
          <h4 className="font-bold mb-3">🚀 Physical AI의 발전 단계</h4>

          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <div className="bg-purple-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">1</div>
              <div>
                <strong>Level 1: 고정형 자동화 (2010s)</strong>
                <p className="text-sm mt-1">정해진 프로그램만 반복 실행. 환경 변화 대응 불가.</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">2</div>
              <div>
                <strong>Level 2: 센서 기반 적응 (2020s 초반)</strong>
                <p className="text-sm mt-1">센서로 환경을 인식하고 기본적인 조정 가능.</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">3</div>
              <div>
                <strong>Level 3: AI 기반 자율성 (현재)</strong>
                <p className="text-sm mt-1">AI가 상황을 이해하고 자율적으로 판단. Sim2Real 학습.</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="bg-orange-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">4</div>
              <div>
                <strong>Level 4: 협업 지능 (2025-2030)</strong>
                <p className="text-sm mt-1">인간과 자연스럽게 협업. 언어로 소통하고 의도 파악.</p>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <div className="bg-red-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold flex-shrink-0">5</div>
              <div>
                <strong>Level 5: 완전 자율 시스템 (2030+)</strong>
                <p className="text-sm mt-1">공장, 도시, 물류망 전체를 AI가 자율 운영.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Next Steps */}
      <div className="bg-indigo-50 dark:bg-indigo-900/20 border-l-4 border-indigo-500 p-6 rounded-lg my-8">
        <h3 className="text-xl font-bold mb-2">다음 단계: Physical AI 핵심 기술 Deep Dive</h3>
        <p className="text-gray-700 dark:text-gray-300">
          다음 챕터에서는 Physical AI를 구현하는 <strong>센서 융합, 컴퓨터 비전,
          강화학습, IoT</strong> 등의 핵심 기술을 상세히 학습합니다.
        </p>
      </div>
    </div>
  )
}