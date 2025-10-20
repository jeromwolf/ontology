'use client';

import {
  Factory, TrendingUp, Rocket, Globe, Brain
} from 'lucide-react';
import Link from 'next/link';
import References from '@/components/common/References';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      {/* 스마트팩토리란? - 핵심 설명 추가 */}
      <div className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-900/20 dark:to-gray-900/20 p-6 rounded-xl border border-slate-300 dark:border-slate-700">
        <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-4 flex items-center gap-2">
          🤔 잠깐, 스마트팩토리가 뭔가요?
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
              <Factory className="w-5 h-5 text-gray-600" />
              일반 공장
            </h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
              <li>• 사람이 직접 기계 조작</li>
              <li>• 종이로 생산 기록 관리</li>
              <li>• 고장나면 그때 수리</li>
              <li>• 월말에 생산량 집계</li>
              <li>• 불량품 나와도 나중에 발견</li>
            </ul>
          </div>
          
          <div className="bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-emerald-800/20 dark:to-teal-800/20 p-4 rounded-lg border border-emerald-200 dark:border-emerald-700">
            <h4 className="font-semibold text-emerald-900 dark:text-emerald-100 mb-3 flex items-center gap-2">
              <Brain className="w-5 h-5 text-emerald-600" />
              스마트팩토리
            </h4>
            <ul className="text-sm text-emerald-700 dark:text-emerald-300 space-y-2">
              <li>• 기계가 알아서 작동</li>
              <li>• 모든 데이터 자동 수집</li>
              <li>• 고장 나기 전에 미리 예측</li>
              <li>• 실시간으로 현황 확인</li>
              <li>• 불량품 즉시 발견 & 원인 분석</li>
            </ul>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
          <p className="text-gray-700 dark:text-gray-300 font-medium mb-2">
            💡 한 줄 정리: <span className="text-emerald-600 dark:text-emerald-400 font-bold">스마트팩토리 = 인공지능 + IoT + 빅데이터로 똑똑해진 공장</span>
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            마치 공장이 스마트폰처럼 똑똑해진 것! 알아서 일하고, 문제를 미리 알려주고, 
            최적의 방법을 찾아내는 '생각하는 공장'입니다.
          </p>
        </div>
        
        <div className="mt-4 flex items-center justify-center gap-3 text-sm text-gray-600 dark:text-gray-400">
          <span className="flex items-center gap-2">
            <Factory className="w-4 h-4" />
            단순 공장
          </span>
          <span className="text-xl">→</span>
          <span className="flex items-center gap-2 font-semibold text-emerald-600 dark:text-emerald-400">
            <Brain className="w-4 h-4" />
            똑똑한 공장
          </span>
        </div>
        
        {/* 간단한 실제 성과 미리보기 */}
        <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-700">
          <div className="text-xs text-amber-700 dark:text-amber-400 mb-2">💡 실제 도입 효과 미리보기</div>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="bg-white dark:bg-gray-800 p-2 rounded text-center">
              <div className="text-gray-600 dark:text-gray-400">검사 자동화</div>
              <div className="text-purple-600 dark:text-purple-400 font-bold">100배 빨라짐</div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-2 rounded text-center">
              <div className="text-gray-600 dark:text-gray-400">생산 라인 변경</div>
              <div className="text-orange-600 dark:text-orange-400 font-bold">144배 빨라짐</div>
            </div>
          </div>
          <div className="text-xs text-amber-600 dark:text-amber-500 mt-2 italic">
            ※ 자세한 성공 사례는 "글로벌 트렌드와 성공 사례" 챕터에서 확인하세요
          </div>
        </div>
      </div>

      {/* 스마트팩토리 생태계 맵 링크 */}
      <div className="mt-6 mb-8 p-5 bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-xl border border-emerald-200 dark:border-emerald-800">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-emerald-900 dark:text-emerald-200 mb-2 flex items-center gap-2">
              🗺️ 스마트팩토리 생태계 한눈에 보기
            </h3>
            <p className="text-sm text-emerald-700 dark:text-emerald-300">
              스마트팩토리를 구성하는 21개 핵심 요소들의 관계와 데이터 흐름을 인터랙티브하게 탐색해보세요.
            </p>
          </div>
          <Link
            href="/modules/smart-factory/simulators/smart-factory-ecosystem?from=/modules/smart-factory/why-smart-factory"
            className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition-colors shadow-sm hover:shadow-md"
          >
            <span>생태계 맵 보기</span>
            <span className="text-lg">→</span>
          </Link>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Factory className="w-6 h-6 text-slate-600" />
            제조업이 직면한 5대 위기
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 border-l-4 border-red-400 rounded">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">1. 인건비 상승 압박</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">중국 대비 한국 제조업 인건비 3.2배, 베트남 대비 8.5배</p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• 2022년 한국 제조업 시간당 임금: $25.9 (중국 $8.1, 베트남 $3.1)</li>
                <li>• 최저임금 인상률: 연평균 7.8% (2017-2023)</li>
                <li>• 단순 작업 자동화 시급성 증대</li>
              </ul>
            </div>
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 border-l-4 border-orange-400 rounded">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">2. 품질 요구 증가</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">제로 디펙트(불량 제로), 완전 추적성을 요구하는 글로벌 고객들</p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• 자동차: Six Sigma 품질 (99.99966% 수준)
                  <span className="text-blue-600 dark:text-blue-400"> = 100만개 중 3.4개만 불량 허용</span>
                </li>
                <li>• 반도체: PPM 단위 불량률 관리
                  <span className="text-blue-600 dark:text-blue-400"> = Parts Per Million, 백만분의 1 단위로 관리</span>
                </li>
                <li>• 의료기기: FDA 완전 추적성 의무화
                  <span className="text-blue-600 dark:text-blue-400"> = 미국 식약청, 모든 부품 이력 추적 필수</span>
                </li>
              </ul>
              
              {/* 용어 설명 박스 - 색상 변경 */}
              <div className="mt-3 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded">
                <h5 className="text-xs font-semibold text-amber-800 dark:text-amber-300 mb-2">💡 용어 쉽게 이해하기</h5>
                <div className="space-y-1 text-xs text-amber-700 dark:text-amber-400">
                  <div><strong>Six Sigma:</strong> 매우 엄격한 품질 기준. 콜라 100만병 중 3병만 맛이 이상해도 안됨</div>
                  <div><strong>PPM:</strong> "몇 ppm?" = "100만개 중 몇 개 불량?" 이라는 뜻</div>
                  <div><strong>FDA:</strong> 미국에서 약이나 의료기기 팔려면 여기 허가 필수</div>
                </div>
              </div>
            </div>
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 border-l-4 border-yellow-400 rounded">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">3. 맞춤화 수요 폭증</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">대량생산에서 다품종 소량생산으로 패러다임 전환</p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• Mass Customization: 개인화된 제품 대량 생산</li>
                <li>• 로트 사이즈 1까지 경제성 확보 필요</li>
                <li>• 빠른 제품 변경과 설정 전환 요구</li>
              </ul>
            </div>
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 border-l-4 border-green-400 rounded">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">4. 환경 규제 강화</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">탄소중립, 순환경제로 인한 제조 방식 혁신 필수</p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• EU 탄소국경세(CBAM) 2026년 본격 시행</li>
                <li>• RE100 요구: 재생에너지 100% 사용</li>
                <li>• 순환경제법: 재활용 소재 의무 사용</li>
              </ul>
            </div>
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 border-l-4 border-blue-400 rounded">
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">5. 숙련 인력 부족</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">고령화와 3D 업종 기피로 인한 인력난 심화</p>
              <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <li>• 제조업 구인난: 빈 일자리 34만 개 (2023)</li>
                <li>• 숙련공 은퇴: 베이비붐 세대 대량 은퇴</li>
                <li>• 기술 전수 공백: 암묵지 손실 위험</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Rocket className="w-6 h-6 text-slate-600" />
            스마트팩토리의 혁신 효과
          </h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded border">
              <div>
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">생산성 향상</h4>
                <p className="text-sm text-slate-600 dark:text-slate-400">자동화와 최적화</p>
              </div>
              <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">+30%</div>
            </div>
            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded border">
              <div>
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">불량률 감소</h4>
                <p className="text-sm text-slate-600 dark:text-slate-400">AI 품질 예측</p>
              </div>
              <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">-50%</div>
            </div>
            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded border">
              <div>
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">에너지 절약</h4>
                <p className="text-sm text-slate-600 dark:text-slate-400">지능형 에너지 관리</p>
              </div>
              <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">-20%</div>
            </div>
            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded border">
              <div>
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">리드타임 단축</h4>
                <p className="text-sm text-slate-600 dark:text-slate-400">실시간 스케줄링</p>
              </div>
              <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">-40%</div>
            </div>
            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded border">
              <div>
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">재고 감소</h4>
                <p className="text-sm text-slate-600 dark:text-slate-400">Just-in-Time 최적화</p>
              </div>
              <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">-25%</div>
            </div>
          </div>
        </div>
      </div>

      {/* 시뮬레이터 체험 섹션 */}
      <div className="mt-8 p-6 bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-xl border border-purple-200 dark:border-purple-800">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-purple-900 dark:text-purple-200 mb-2">
              🎮 스마트팩토리 직접 체험해보기
            </h3>
            <p className="text-sm text-purple-700 dark:text-purple-300">
              디지털 트윈 팩토리 시뮬레이터로 스마트팩토리의 핵심 개념을 실시간으로 체험해보세요.
            </p>
          </div>
          <Link
            href="/modules/smart-factory/simulators/digital-twin-factory?from=/modules/smart-factory/why-smart-factory"
            className="inline-flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
          >
            <span>시뮬레이터 체험</span>
            <span className="text-lg">→</span>
          </Link>
        </div>
      </div>

      <div className="bg-gray-50 dark:bg-gray-800 p-8 rounded-lg border border-gray-200 dark:border-gray-700">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-8 text-center">
          글로벌 제조업 혁신 동향
        </h3>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="w-16 h-16 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center mx-auto mb-4">
              <span className="text-lg font-bold text-red-700 dark:text-red-300">🇩🇪</span>
            </div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">독일 Industry 4.0</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">CPS 기반 제조혁신 정책으로 글로벌 리더십 확보</p>
            <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
              <li>• 2013년 국가 전략으로 시작</li>
              <li>• 플랫폼 인더스트리 4.0</li>
              <li>• 중소기업 지원 강화</li>
              <li>• RAMI 4.0 참조 모델</li>
            </ul>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center mx-auto mb-4">
              <span className="text-lg font-bold text-blue-700 dark:text-blue-300">🇺🇸</span>
            </div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">미국 Advanced Manufacturing</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">AI와 로봇공학 중심의 첨단 제조업 육성</p>
            <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
              <li>• Manufacturing USA 프로그램</li>
              <li>• 14개 제조혁신연구소</li>
              <li>• CHIPS Act 반도체 리쇼어링</li>
              <li>• AI 기반 제조혁신</li>
            </ul>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg flex items-center justify-center mx-auto mb-4">
              <span className="text-lg font-bold text-yellow-700 dark:text-yellow-300">🇨🇳</span>
            </div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">중국 Made in China 2025</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">제조업 디지털화로 제조강국 도약 전략</p>
            <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1">
              <li>• 10대 핵심 산업 집중 육성</li>
              <li>• 자주혁신 40%, 50%, 70% 목표</li>
              <li>• 스마트 제조 시범 프로젝트</li>
              <li>• 디지털 경제 통합</li>
            </ul>
          </div>
        </div>
      </div>

      <References
        sections={[
          {
            title: '📚 공식 문서 & 표준',
            icon: 'web' as const,
            color: 'border-emerald-500',
            items: [
              {
                title: 'Plattform Industrie 4.0 - Reference Architecture Model',
                link: 'https://www.plattform-i40.de/IP/Navigation/EN/Home/home.html',
                description: '독일 Industry 4.0 공식 표준 아키텍처 및 참조 모델'
              },
              {
                title: 'IEC 63339: Measurement framework for smart manufacturing',
                link: 'https://www.iec.ch/',
                description: 'IEC 스마트 제조 측정 프레임워크 국제 표준'
              },
              {
                title: 'ISO 23247: Digital Twin Framework for Manufacturing',
                link: 'https://www.iso.org/',
                description: '제조업 디지털 트윈 국제 표준 프레임워크'
              },
              {
                title: 'Industrial Internet Consortium (IIC) - Architecture Framework',
                link: 'https://www.iiconsortium.org/',
                description: 'IIoT 및 스마트팩토리 글로벌 아키텍처 가이드'
              }
            ]
          },
          {
            title: '📖 핵심 논문 & 연구',
            icon: 'paper' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Recommendations for implementing the strategic initiative INDUSTRIE 4.0',
                authors: 'Kagermann, H., Wahlster, W., Helbig, J.',
                year: '2013',
                description: 'Industry 4.0 개념을 정의한 원본 독일 정부 보고서'
              },
              {
                title: 'Industry 4.0: Building the digital enterprise',
                authors: 'PwC',
                year: '2016',
                description: '2,000개 기업 대상 Industry 4.0 글로벌 설문 연구'
              },
              {
                title: 'Smart Manufacturing: Past Research, Present Findings, and Future Directions',
                authors: 'Kusiak, A.',
                year: '2018',
                description: 'Manufacturing Engineering 학술지 - 스마트 제조 연구 동향'
              },
              {
                title: 'The Fourth Industrial Revolution',
                authors: 'Schwab, K.',
                year: '2016',
                description: '세계경제포럼 회장의 4차 산업혁명 개념서'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 리소스',
            icon: 'book' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'McKinsey - Industry 4.0: How to navigate digitization',
                link: 'https://www.mckinsey.com/capabilities/operations/our-insights',
                description: '맥킨지 제조업 디지털 전환 실전 가이드 및 사례 연구'
              },
              {
                title: 'Deloitte - Industry 4.0 and manufacturing ecosystems',
                link: 'https://www2.deloitte.com/insights',
                description: '딜로이트 제조업 생태계 전환 전략 보고서'
              },
              {
                title: 'Gartner Hype Cycle for Smart Manufacturing',
                link: 'https://www.gartner.com/',
                description: '스마트 제조 기술 성숙도 및 도입 로드맵 (매년 업데이트)'
              },
              {
                title: '중소벤처기업부 - 스마트공장 지원사업',
                link: 'https://www.smart-factory.kr/',
                description: '한국 스마트공장 지원 정책 및 성공 사례 (국문)'
              },
              {
                title: 'Boston Consulting Group - Embracing Industry 4.0',
                link: 'https://www.bcg.com/',
                description: 'BCG 제조업 혁신 전략 및 ROI 분석 도구'
              }
            ]
          }
        ]}
      />
    </div>
  );
}