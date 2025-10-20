'use client';

import React from 'react';
import { Factory, TrendingDown, TrendingUp, Globe, Users, Zap, Shield, CheckCircle, AlertTriangle, Building2, Cpu, LineChart } from 'lucide-react';

export default function Chapter7() {
  return (
    <div className="prose prose-lg max-w-none dark:prose-invert">
      <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-2xl p-8 mb-8 border border-orange-200 dark:border-orange-800">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-12 h-12 bg-orange-500 rounded-xl flex items-center justify-center">
            <Factory className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white m-0">산업 자동화와 스마트 팩토리</h1>
        </div>
        <p className="text-xl text-gray-700 dark:text-gray-300 m-0">
          Physical AI가 만드는 제조업의 미래 - 한국 제조업의 현실과 혁신 전략
        </p>
      </div>

      {/* 한국 제조업 현황 */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <LineChart className="text-red-600" />
          한국 제조업이 직면한 구조적 위기
        </h2>

        <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg mb-6 border-l-4 border-red-500">
          <h3 className="font-bold mb-3">위기의 실체: 수십 년 성장 엔진의 한계</h3>
          <p className="mb-4">
            한국 제조업은 수십 년간 국가 경제의 성장 엔진 역할을 해왔지만, 이제 근본적인 구조적 변화에 직면했습니다.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold mb-2 text-red-600">경쟁 과열 상태</h4>
              <div className="text-3xl font-bold mb-2">80%</div>
              <p className="text-sm">국내 제조기업 10곳 중 8곳이 자사의 주력 제품 시장이 이미 경쟁 과열 상태에 진입했다고 평가</p>
              <p className="text-xs mt-2 text-gray-600 dark:text-gray-400">출처: 대한상공회의소 조사</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-bold mb-2 text-orange-600">경쟁 우위 상실</h4>
              <div className="text-3xl font-bold mb-2">83.9%</div>
              <p className="text-sm">경쟁 우위가 거의 없거나 추월당한 것으로 인식하는 기업 비율</p>
              <p className="text-xs mt-2 text-gray-600 dark:text-gray-400">글로벌 경쟁에서 뒤처지는 현실</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">3대 복합 위기 요인</h3>
          <div className="space-y-4">
            <div className="flex items-start gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <Users className="text-blue-600 flex-shrink-0 mt-1" size={24} />
              <div>
                <h4 className="font-bold mb-1">1️⃣ 내부 구조 요인: 급격한 인구구조 변화</h4>
                <p className="text-sm">노동력 부족 심화, 고령화로 인한 생산성 저하, 젊은 인재의 제조업 기피 현상</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <Globe className="text-purple-600 flex-shrink-0 mt-1" size={24} />
              <div>
                <h4 className="font-bold mb-1">2️⃣ 외부 환경 충격: 글로벌 환경 변화</h4>
                <p className="text-sm">미·중 기술패권 경쟁, 공급망 재편, 보호무역주의 강화, 탄소중립 압력</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
              <Zap className="text-orange-600 flex-shrink-0 mt-1" size={24} />
              <div>
                <h4 className="font-bold mb-1">3️⃣ 기술 패러다임 전환: AI로 인한 제조 혁명</h4>
                <p className="text-sm">Physical AI의 등장, 디지털 트윈과 가상 시운전, 자율화·무인화 생산 시스템</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Physical AI 개념 */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Cpu className="text-indigo-600" />
          Physical AI: 병 속의 뇌에서 몸을 가진 뇌로
        </h2>

        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-6 rounded-lg mb-6">
          <h3 className="font-bold mb-4">Physical AI의 정의</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg">
              <h4 className="font-bold mb-2 text-gray-600 dark:text-gray-400">기존 SW 기반 AI</h4>
              <div className="text-center py-6">
                <div className="inline-block bg-blue-100 dark:bg-blue-900/40 rounded-full p-4 mb-3">
                  <span className="text-4xl">🧠</span>
                </div>
                <p className="font-bold text-lg">Brain in a Jar</p>
                <p className="text-sm mt-2">병 속의 뇌 (디지털 영역에만 존재)</p>
              </div>
              <ul className="text-sm space-y-1 mt-4">
                <li>• ChatGPT, GPT-4</li>
                <li>• 이미지 생성 AI</li>
                <li>• 데이터 분석 AI</li>
                <li>• 가상 공간에서만 작동</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg border-2 border-indigo-500">
              <h4 className="font-bold mb-2 text-indigo-600">Physical AI</h4>
              <div className="text-center py-6">
                <div className="inline-block bg-indigo-100 dark:bg-indigo-900/40 rounded-full p-4 mb-3">
                  <span className="text-4xl">🤖</span>
                </div>
                <p className="font-bold text-lg">Brain in a Body</p>
                <p className="text-sm mt-2">몸을 가진 뇌 (현실 세계와 직접 상호작용)</p>
              </div>
              <ul className="text-sm space-y-1 mt-4">
                <li>• 로봇 팔, 휴머노이드</li>
                <li>• 자율주행 차량</li>
                <li>• 드론, AGV</li>
                <li>• 물리적 세계에서 행동</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">거대한 시장의 탄생</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg text-center">
              <p className="text-sm mb-2">엔비디아 CEO 젠슨 황 (CES 2025)</p>
              <div className="text-4xl font-bold text-green-600 mb-2">50조 달러</div>
              <p className="text-lg font-bold mb-2">약 7경 2,000조 원</p>
              <p className="text-sm">Physical AI 시장 규모 예측</p>
            </div>
            <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg text-center">
              <p className="text-sm mb-2">Citi 전망 (2050년)</p>
              <div className="text-4xl font-bold text-blue-600 mb-2">40억 대</div>
              <p className="text-lg font-bold mb-2">AI 로봇</p>
              <p className="text-sm">전 세계 활동 예상 대수</p>
            </div>
          </div>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg">
          <h3 className="font-bold mb-3">인류 문명의 새로운 전환점</h3>
          <p className="mb-4">
            Physical AI는 단순한 기술적 진보가 아닌 인류 문명의 패러다임 전환을 의미합니다.
          </p>
          <div className="grid md:grid-cols-3 gap-3">
            <div className="bg-white dark:bg-gray-800 p-3 rounded text-center">
              <p className="font-bold text-sm">AI 모델</p>
              <p className="text-xs mt-1">대규모 언어/비전 모델</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-3 rounded text-center">
              <p className="font-bold text-sm">시뮬레이션</p>
              <p className="text-xs mt-1">디지털 트윈, 물리 엔진</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-3 rounded text-center">
              <p className="font-bold text-sm">로봇 하드웨어</p>
              <p className="text-xs mt-1">액추에이터, 센서</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-3 rounded text-center">
              <p className="font-bold text-sm">첨단 소재</p>
              <p className="text-xs mt-1">경량 복합소재, 스마트 소재</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-3 rounded text-center">
              <p className="font-bold text-sm">정밀 부품</p>
              <p className="text-xs mt-1">모터, 기어, 제어기</p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-3 rounded text-center">
              <p className="font-bold text-sm">통신/에너지</p>
              <p className="text-xs mt-1">5G/6G, 배터리, 충전</p>
            </div>
          </div>
          <p className="text-sm mt-4 text-center font-bold">
            → 거대한 융합 생태계 형성으로 시너지 효과 창출
          </p>
        </div>
      </section>

      {/* 제조 패러다임 전환 */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Zap className="text-yellow-600" />
          제조 패러다임의 근본적 전환
        </h2>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">로봇의 진화: 프로그램에서 파트너로</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-gray-50 dark:bg-gray-900 p-5 rounded-lg">
              <h4 className="font-bold mb-3 text-gray-600">기존 산업 로봇</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-red-500 flex-shrink-0">❌</span>
                  <span><strong>고정된 프로그래밍</strong>: 정해진 동작만 반복</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500 flex-shrink-0">❌</span>
                  <span><strong>환경 변화 대응 불가</strong>: 예외 상황 처리 불가능</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500 flex-shrink-0">❌</span>
                  <span><strong>사전 프로그래밍 필수</strong>: 엔지니어의 수동 설정</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-red-500 flex-shrink-0">❌</span>
                  <span><strong>유연성 부족</strong>: 다품종 소량 생산 어려움</span>
                </li>
              </ul>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border-2 border-green-500">
              <h4 className="font-bold mb-3 text-green-600">Physical AI 로봇</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-green-500 flex-shrink-0">✅</span>
                  <span><strong>자율적 판단</strong>: 상황에 맞게 스스로 결정</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500 flex-shrink-0">✅</span>
                  <span><strong>환경 적응</strong>: 변화하는 제조 환경에 실시간 대응</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500 flex-shrink-0">✅</span>
                  <span><strong>지속 학습</strong>: 경험을 통해 성능 개선</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-500 flex-shrink-0">✅</span>
                  <span><strong>유연한 생산</strong>: 다양한 제품에 즉시 적응</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg mb-6">
          <h3 className="text-xl font-bold mb-4">의사결정 구조의 재편</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-orange-50 dark:bg-orange-900/20 p-5 rounded-lg">
              <h4 className="font-bold mb-3">과거: 인간 주도 + 기계 보조</h4>
              <div className="space-y-3 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-16 font-bold">설계</div>
                  <div className="flex-1 bg-blue-200 dark:bg-blue-800 rounded p-2 text-center">인간</div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-16 font-bold">운영</div>
                  <div className="flex-1 bg-blue-200 dark:bg-blue-800 rounded p-2 text-center">인간</div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-16 font-bold">실행</div>
                  <div className="flex-1 bg-gray-300 dark:bg-gray-700 rounded p-2 text-center">기계</div>
                </div>
                <p className="text-xs mt-3">→ 사람이 모든 결정을 내리고 기계는 단순 실행만 담당</p>
              </div>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border-2 border-green-500">
              <h4 className="font-bold mb-3">현재: AI 주도 + 인간 감독</h4>
              <div className="space-y-3 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-16 font-bold">설계</div>
                  <div className="flex-1 bg-purple-200 dark:bg-purple-800 rounded p-2 text-center">인간 (창의적)</div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-16 font-bold">최적화</div>
                  <div className="flex-1 bg-indigo-300 dark:bg-indigo-700 rounded p-2 text-center">AI (실시간)</div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-16 font-bold">실행</div>
                  <div className="flex-1 bg-indigo-300 dark:bg-indigo-700 rounded p-2 text-center">AI 로봇</div>
                </div>
                <p className="text-xs mt-3">→ AI가 공정 최적화를 주도하고 사람은 전략적 감독에 집중</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg">
          <h3 className="font-bold mb-3">Physical AI의 실질적 경제 효과</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span><strong>24시간 무인 운영</strong>: 야간/주말 생산 가능, 인력 부족 해소</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span><strong>오류율 감소</strong>: 정밀 작업, 품질 일관성 향상</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span><strong>안전 모니터링</strong>: 위험 지역 작업, 산재 예방</span>
              </li>
            </ul>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span><strong>에너지 효율성</strong>: 최적 경로 계산, 전력 절감</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span><strong>예지 보전</strong>: 고장 전 사전 감지, 다운타임 최소화</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle className="text-green-500 flex-shrink-0 mt-0.5" size={16} />
                <span><strong>유연 생산</strong>: 빠른 제품 전환, 맞춤형 생산</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 글로벌 사례 */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Building2 className="text-purple-600" />
          "움직이는 모든 것은 로봇이 된다" - 글로벌 다크 팩토리 사례
        </h2>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg mb-6">
          <p className="font-bold mb-2">엔비디아 CEO 젠슨 황의 통찰</p>
          <p className="text-lg italic">
            "Physical AI 도입으로 공장 자체를 하나의 거대한 로봇으로 만들 수 있습니다."
          </p>
        </div>

        <div className="space-y-6">
          {/* 샤오미 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-start gap-4 mb-4">
              <div className="w-12 h-12 bg-orange-500 rounded-lg flex items-center justify-center text-white font-bold text-xl flex-shrink-0">
                小米
              </div>
              <div>
                <h3 className="text-xl font-bold">중국 샤오미 - 창핑 다크 팩토리</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">Complete Automation Factory</p>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-5 rounded-lg mb-4">
              <h4 className="font-bold mb-3">3無 시스템 (Three Nothings)</h4>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded text-center">
                  <div className="text-3xl mb-2">💡</div>
                  <p className="font-bold text-sm">無 조명 (No Lights)</p>
                  <p className="text-xs mt-1">사람이 없으므로 조명 불필요</p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded text-center">
                  <div className="text-3xl mb-2">❄️</div>
                  <p className="font-bold text-sm">無 냉난방 (No HVAC)</p>
                  <p className="text-xs mt-1">쾌적함이 필요 없음</p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded text-center">
                  <div className="text-3xl mb-2">👤</div>
                  <p className="font-bold text-sm">無 인력 (No Workers)</p>
                  <p className="text-xs mt-1">완전 무인 자동화</p>
                </div>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded">
                <h4 className="font-bold mb-2">핵심 기술</h4>
                <ul className="text-sm space-y-1">
                  <li>• Physical AI 기반 무인 자동화</li>
                  <li>• 자율 이동 로봇 (AMR) 물류</li>
                  <li>• 완전 자동 조립 라인</li>
                  <li>• 실시간 품질 검사 AI</li>
                </ul>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded">
                <h4 className="font-bold mb-2">성과</h4>
                <ul className="text-sm space-y-1">
                  <li>• 인건비 90% 이상 절감</li>
                  <li>• 에너지 비용 70% 감소</li>
                  <li>• 생산 효율 300% 향상</li>
                  <li>• 불량률 0.1% 이하 달성</li>
                </ul>
              </div>
            </div>
          </div>

          {/* 화낙 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-start gap-4 mb-4">
              <div className="w-12 h-12 bg-yellow-500 rounded-lg flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
                FANUC
              </div>
              <div>
                <h3 className="text-xl font-bold">일본 화낙 (FANUC) - 로봇이 로봇을 만드는 공장</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">Robots Building Robots</p>
              </div>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-5 rounded-lg mb-4">
              <h4 className="font-bold mb-3">완전 자동화 생산 시스템</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded">
                  <div className="text-2xl font-bold text-yellow-600 mb-2">50대/일</div>
                  <p className="text-sm">최첨단 제조 로봇 일일 생산량</p>
                </div>
                <div className="bg-white dark:bg-gray-800 p-4 rounded">
                  <div className="text-2xl font-bold text-green-600 mb-2">30일</div>
                  <p className="text-sm">인간 개입 없이 가동 가능 기간</p>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded">
              <h4 className="font-bold mb-2">시사점</h4>
              <p className="text-sm">
                제조업의 최종 진화 형태: 로봇이 로봇을 생산하는 완전 자율화 시스템.
                인간은 전략적 설계와 감독에만 집중하며, 실제 생산은 AI와 로봇이 주도합니다.
              </p>
            </div>
          </div>

          {/* 폭스콘 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-start gap-4 mb-4">
              <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
                鴻海
              </div>
              <div>
                <h3 className="text-xl font-bold">대만 폭스콘 - AI·디지털 트윈 공장</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">Virtual First, Physical Second</p>
              </div>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg mb-4">
              <h4 className="font-bold mb-3">정반대 접근법: 가상공간 우선</h4>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <p className="font-bold mb-2 text-red-600">기존 방식</p>
                  <ol className="text-sm space-y-2">
                    <li>1️⃣ 실제 공장 먼저 건설</li>
                    <li>2️⃣ 가동 후 문제점 발견</li>
                    <li>3️⃣ 시행착오 거치며 수정</li>
                    <li>4️⃣ 막대한 시간/비용 소모</li>
                  </ol>
                </div>
                <div className="border-l-2 border-blue-300 pl-6">
                  <p className="font-bold mb-2 text-blue-600">폭스콘 방식</p>
                  <ol className="text-sm space-y-2">
                    <li>1️⃣ 디지털 공장 먼저 구축</li>
                    <li>2️⃣ 수많은 시뮬레이션 실행</li>
                    <li>3️⃣ 가상에서 모든 문제 해결</li>
                    <li>4️⃣ 완벽한 설계로 실제 건설</li>
                  </ol>
                </div>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg">
              <h4 className="font-bold mb-3">80-20 법칙: AI와 인간의 역할 분담</h4>
              <div className="flex items-center gap-4 mb-3">
                <div className="flex-1 bg-indigo-300 dark:bg-indigo-700 rounded p-3 text-center">
                  <p className="font-bold">AI: 80%</p>
                  <p className="text-xs mt-1">반복적이고 정형화된 업무</p>
                </div>
                <div className="flex-1 bg-purple-300 dark:bg-purple-700 rounded p-3 text-center">
                  <p className="font-bold">인간: 20%</p>
                  <p className="text-xs mt-1">창의적이고 전략적인 업무</p>
                </div>
              </div>
              <p className="text-sm">
                AI가 전체 업무의 80%를 처리하고, 사람은 중요하고 창의적인 20%의 일에 집중하는 구조
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 한국 경쟁력 분석 */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <TrendingUp className="text-green-600" />
          한국의 Physical AI 경쟁력 분석
        </h2>

        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg border-l-4 border-green-500">
            <h3 className="font-bold mb-4 flex items-center gap-2">
              <TrendingUp className="text-green-600" />
              강점 (Strengths)
            </h3>
            <div className="space-y-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded">
                <h4 className="font-bold mb-2">🥇 로봇 밀도 세계 1위</h4>
                <div className="text-3xl font-bold text-green-600 mb-2">1,012대</div>
                <p className="text-sm">제조업 근로자 1만 명당 로봇 수 (2023년 기준)</p>
                <p className="text-xs mt-2 text-gray-600 dark:text-gray-400">세계 평균의 약 7배 수준</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded">
                <h4 className="font-bold mb-2">📊 풍부한 제조 데이터</h4>
                <ul className="text-sm space-y-1">
                  <li>• 반도체, 자동차, 전자 등 주력 산업</li>
                  <li>• 수십 년간 축적된 생산 데이터</li>
                  <li>• Physical AI 학습에 즉시 활용 가능</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg border-l-4 border-red-500">
            <h3 className="font-bold mb-4 flex items-center gap-2">
              <TrendingDown className="text-red-600" />
              약점 (Weaknesses)
            </h3>
            <div className="space-y-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded">
                <h4 className="font-bold mb-2">🔻 수요 중심 구조</h4>
                <ul className="text-sm space-y-1">
                  <li>• 로봇 도입은 많으나 자체 개발 역량 부족</li>
                  <li>• 핵심 부품·소프트웨어 수입 의존</li>
                  <li>• 공급 역량 (Supply-side) 취약</li>
                </ul>
              </div>
              <div className="bg-white dark:bg-gray-800 p-4 rounded">
                <h4 className="font-bold mb-2">📉 데이터 활용도 저하</h4>
                <ul className="text-sm space-y-1">
                  <li>• 제조 데이터 사일로 현상 (기업 간 단절)</li>
                  <li>• AI 학습용 데이터 표준화 미흡</li>
                  <li>• 데이터 공유 체계 부재</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 혁신 방안 */}
      <section className="my-8">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Shield className="text-indigo-600" />
          한국 제조업 혁신을 위한 Physical AI 전략
        </h2>

        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold flex-shrink-0">1</div>
              <div className="flex-1">
                <h3 className="font-bold mb-2">Physical AI 통합 전략 추진체계 마련</h3>
                <p className="text-sm mb-2">범부처 협력 체계 구축, 산·학·연 연계 강화, 장기 로드맵 수립</p>
                <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded text-xs">
                  <strong>Action Item:</strong> Physical AI 국가 컨트롤 타워 설립, 5개년 마스터플랜 수립
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold flex-shrink-0">2</div>
              <div className="flex-1">
                <h3 className="font-bold mb-2">핵심 난제 영역 돌파를 위한 R&D 강화</h3>
                <p className="text-sm mb-2">센서 융합, 실시간 제어, Sim2Real 기술 등 핵심 기술 개발에 집중 투자</p>
                <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded text-xs">
                  <strong>투자 분야:</strong> 자율 제어, 예지 보전, 디지털 트윈, 협동 로봇 기술
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white font-bold flex-shrink-0">3</div>
              <div className="flex-1">
                <h3 className="font-bold mb-2">개방형 제조 운영체제(OS) 및 대형 기계 모델 개발</h3>
                <p className="text-sm mb-2">표준화된 플랫폼으로 중소기업도 쉽게 Physical AI 도입 가능하도록 지원</p>
                <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded text-xs">
                  <strong>목표:</strong> 한국형 Manufacturing OS, Foundation Model for Robotics 개발
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-orange-500 rounded-full flex items-center justify-center text-white font-bold flex-shrink-0">4</div>
              <div className="flex-1">
                <h3 className="font-bold mb-2">Physical AI 도입 분야의 전략적 우선순위 설정</h3>
                <p className="text-sm mb-2">인력 부족, 위험 작업, 24시간 운영이 필요한 분야부터 단계적 도입</p>
                <div className="bg-orange-50 dark:bg-orange-900/20 p-3 rounded text-xs">
                  <strong>우선 적용 분야:</strong> 반도체 제조, 자동차 조립, 물류 자동화, 위험 물질 처리
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-cyan-500 rounded-full flex items-center justify-center text-white font-bold flex-shrink-0">5</div>
              <div className="flex-1">
                <h3 className="font-bold mb-2">Physical AI 통합 훈련 실증센터 구축 및 데이터 관리체계 강화</h3>
                <p className="text-sm mb-2">국가 차원의 테스트베드 구축, 데이터 표준화 및 공유 플랫폼 마련</p>
                <div className="bg-cyan-50 dark:bg-cyan-900/20 p-3 rounded text-xs">
                  <strong>인프라:</strong> Physical AI 종합 실증센터, 제조 데이터 허브, 클라우드 시뮬레이션 환경
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-indigo-500 rounded-full flex items-center justify-center text-white font-bold flex-shrink-0">6</div>
              <div className="flex-1">
                <h3 className="font-bold mb-2">Physical AI 역량을 공공 혁신에 우선 적용 확대</h3>
                <p className="text-sm mb-2">재난 대응, 인프라 점검, 안전 관리 등 공공 부문 선도 도입</p>
                <div className="bg-indigo-50 dark:bg-indigo-900/20 p-3 rounded text-xs">
                  <strong>적용 분야:</strong> 재난 구조 로봇, 시설물 안전 점검, 환경 모니터링, 스마트 시티
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center text-white font-bold flex-shrink-0">7</div>
              <div className="flex-1">
                <h3 className="font-bold mb-2">Physical AI 확산으로 야기될 위험 사전 고려</h3>
                <p className="text-sm mb-2">일자리 변화 대응, 안전 규제, 윤리적 가이드라인 마련</p>
                <div className="bg-red-50 dark:bg-red-900/20 p-3 rounded text-xs">
                  <strong>대응 전략:</strong> 재교육 프로그램, Safety-by-Design, AI 로봇 윤리 헌장
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 결론 */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 border-l-4 border-indigo-500 p-6 rounded-lg my-8">
        <h3 className="text-xl font-bold mb-3">🎯 결론: Physical AI는 선택이 아닌 필수</h3>
        <div className="space-y-3 text-sm">
          <p>
            한국 제조업은 인구 감소, 글로벌 경쟁 심화, 기술 패러다임 전환이라는
            <strong>3대 복합 위기</strong>에 직면해 있습니다.
          </p>
          <p>
            Physical AI는 이러한 위기를 극복하고 <strong>새로운 성장 동력</strong>을 확보할 수 있는
            유일한 해법입니다.
          </p>
          <p>
            세계 1위의 로봇 밀도와 풍부한 제조 데이터라는 강점을 활용하되,
            공급 역량 부족과 데이터 활용도 저하라는 약점을 보완해야 합니다.
          </p>
          <p className="font-bold mt-4">
            ✅ 지금 당장 Physical AI 전략을 수립하고 실행에 옮기는 기업만이
            미래 제조업의 승자가 될 것입니다.
          </p>
        </div>
      </div>
    </div>
  );
}