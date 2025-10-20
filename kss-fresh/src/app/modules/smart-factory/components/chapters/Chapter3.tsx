'use client';

import {
  MapPin, Eye, BarChart3, Bot, Monitor, Smartphone, Clock, TestTube, HardDrive, AlertTriangle, Code
} from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      {/* 디지털 성숙도 레벨 */}
      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">📊 디지털 성숙도 평가 모델</h3>
        <div className="space-y-6">
          {[
            { level: 0, title: "수작업 (Manual)", desc: "종이 기반, 수작업 중심, 디지털화 0%", color: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400", features: ["종이 서류", "수동 기록", "경험 의존", "분산 데이터"] },
            { level: 1, title: "부분 디지털화 (Basic)", desc: "기본 ERP, 일부 자동화, 데이터 분산", color: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400", features: ["기본 ERP", "엑셀 관리", "부분 자동화", "사일로 시스템"] },
            { level: 2, title: "통합 디지털화 (Connected)", desc: "MES 연동, IoT 센서, 실시간 모니터링", color: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400", features: ["MES 도입", "센서 설치", "대시보드", "시스템 연동"] },
            { level: 3, title: "최적화 (Optimized)", desc: "AI 분석, 예측 시스템, 자동 최적화", color: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400", features: ["AI 분석", "예측 모델", "자동 최적화", "고급 분석"] },
            { level: 4, title: "예측형 (Predictive)", desc: "기계학습, 예측 유지보수, 자율 제어", color: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400", features: ["머신러닝", "예측 정비", "자율 제어", "딥러닝"] },
            { level: 5, title: "자율형 (Autonomous)", desc: "완전 자율, 자가 학습, 무인 공장", color: "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400", features: ["완전 자율", "자가 학습", "무인 운영", "AI 의사결정"] }
          ].map((level) => (
            <div key={level.level} className="flex items-start gap-4">
              <div className={`w-16 h-16 rounded-lg flex items-center justify-center font-bold text-lg ${level.color} flex-shrink-0`}>
                L{level.level}
              </div>
              <div className="flex-1">
                <h4 className="font-semibold text-gray-900 dark:text-white text-lg">{level.title}</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{level.desc}</p>
                <div className="flex flex-wrap gap-2">
                  {level.features.map((feature, idx) => (
                    <span key={idx} className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">
                      {feature}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 3단계 전환 로드맵 */}
      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">🗺️ 3단계 전환 로드맵</h3>
        <div className="grid lg:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="w-20 h-20 bg-blue-100 dark:bg-blue-900/30 rounded-xl flex items-center justify-center mx-auto mb-4">
              <Eye className="w-10 h-10 text-blue-600 dark:text-blue-400" />
            </div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">1단계: 가시화</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">현황 파악과 투명성 확보 (3-6개월)</p>
            <ul className="text-xs text-left text-slate-600 dark:text-slate-400 space-y-1">
              <li>• 생산 데이터 실시간 수집</li>
              <li>• 대시보드 구축</li>
              <li>• KPI 모니터링</li>
              <li>• 기본 MES 도입</li>
              <li>• 데이터 품질 확보</li>
              <li>• 작업자 교육</li>
              <li>• 프로세스 표준화</li>
              <li>• 기초 인프라 구축</li>
            </ul>
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
              <div className="text-xs text-blue-700 dark:text-blue-400 font-semibold">예상 효과</div>
              <div className="text-xs text-blue-600 dark:text-blue-400">OEE 10% 향상, 정보 투명성 확보</div>
            </div>
          </div>
          <div className="text-center">
            <div className="w-20 h-20 bg-green-100 dark:bg-green-900/30 rounded-xl flex items-center justify-center mx-auto mb-4">
              <BarChart3 className="w-10 h-10 text-green-600 dark:text-green-400" />
            </div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">2단계: 최적화</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">AI 기반 분석과 개선 (6-12개월)</p>
            <ul className="text-xs text-left text-slate-600 dark:text-slate-400 space-y-1">
              <li>• AI 품질 예측</li>
              <li>• 예측 유지보수</li>
              <li>• 생산 계획 최적화</li>
              <li>• 에너지 효율화</li>
              <li>• 공급망 연동</li>
              <li>• 고급 분석</li>
              <li>• 자동화 확대</li>
              <li>• 통합 플랫폼</li>
            </ul>
            <div className="mt-4 p-3 bg-green-50 dark:bg-green-900/20 rounded">
              <div className="text-xs text-green-700 dark:text-green-400 font-semibold">예상 효과</div>
              <div className="text-xs text-green-600 dark:text-green-400">생산성 25% 향상, 불량률 50% 감소</div>
            </div>
          </div>
          <div className="text-center">
            <div className="w-20 h-20 bg-purple-100 dark:bg-purple-900/30 rounded-xl flex items-center justify-center mx-auto mb-4">
              <Bot className="w-10 h-10 text-purple-600 dark:text-purple-400" />
            </div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">3단계: 자율화</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">자율 운영과 지능화 (12-24개월)</p>
            <ul className="text-xs text-left text-slate-600 dark:text-slate-400 space-y-1">
              <li>• 자율 생산 제어</li>
              <li>• 무인 물류 시스템</li>
              <li>• 완전 자동 품질관리</li>
              <li>• 디지털 트윈 완성</li>
              <li>• 생태계 연결</li>
              <li>• 자가 학습 시스템</li>
              <li>• 완전 통합 운영</li>
              <li>• 지속적 진화</li>
            </ul>
            <div className="mt-4 p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
              <div className="text-xs text-purple-700 dark:text-purple-400 font-semibold">예상 효과</div>
              <div className="text-xs text-purple-600 dark:text-purple-400">완전 최적화, 지속적 자체 개선</div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Win 전략 */}
      <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 p-6 rounded-lg">
        <h3 className="text-xl font-semibold text-green-900 dark:text-green-300 mb-4">⚡ Quick Win 전략 (즉시 효과)</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-green-800 dark:text-green-300 mb-3">단기간 성과 (1-3개월)</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-green-200 dark:border-green-800">
                <div className="flex items-center gap-2 mb-1">
                  <Monitor className="w-4 h-4 text-green-600" />
                  <h5 className="font-medium text-green-800 dark:text-green-300">생산 현황 실시간 대시보드</h5>
                </div>
                <p className="text-xs text-green-700 dark:text-green-400">투자: 300만원, 효과: 정보 투명성 +100%, 의사결정 속도 +50%</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-green-200 dark:border-green-800">
                <div className="flex items-center gap-2 mb-1">
                  <Smartphone className="w-4 h-4 text-green-600" />
                  <h5 className="font-medium text-green-800 dark:text-green-300">모바일 작업 지시서</h5>
                </div>
                <p className="text-xs text-green-700 dark:text-green-400">투자: 200만원, 효과: 종이 제거, 업데이트 실시간, 오류 -70%</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-green-200 dark:border-green-800">
                <div className="flex items-center gap-2 mb-1">
                  <BarChart3 className="w-4 h-4 text-green-600" />
                  <h5 className="font-medium text-green-800 dark:text-green-300">기본 OEE 측정 시스템</h5>
                </div>
                <p className="text-xs text-green-700 dark:text-green-400">투자: 500만원, 효과: 숨겨진 손실 발견, 가동률 +5%</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-green-200 dark:border-green-800">
                <div className="flex items-center gap-2 mb-1">
                  <Clock className="w-4 h-4 text-green-600" />
                  <h5 className="font-medium text-green-800 dark:text-green-300">자동 근태 관리</h5>
                </div>
                <p className="text-xs text-green-700 dark:text-green-400">투자: 100만원, 효과: 관리 시간 -80%, 정확도 +99%</p>
              </div>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-green-800 dark:text-green-300 mb-3">중기간 성과 (3-6개월)</h4>
            <div className="space-y-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-green-200 dark:border-green-800">
                <div className="flex items-center gap-2 mb-1">
                  <TestTube className="w-4 h-4 text-green-600" />
                  <h5 className="font-medium text-green-800 dark:text-green-300">품질 데이터 자동 수집</h5>
                </div>
                <p className="text-xs text-green-700 dark:text-green-400">투자: 800만원, 효과: 검사 시간 -60%, 품질 추적성 확보</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-green-200 dark:border-green-800">
                <div className="flex items-center gap-2 mb-1">
                  <HardDrive className="w-4 h-4 text-green-600" />
                  <h5 className="font-medium text-green-800 dark:text-green-300">재고 최적화 알림</h5>
                </div>
                <p className="text-xs text-green-700 dark:text-green-400">투자: 400만원, 효과: 재고 -20%, 결품 -90%</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-green-200 dark:border-green-800">
                <div className="flex items-center gap-2 mb-1">
                  <AlertTriangle className="w-4 h-4 text-green-600" />
                  <h5 className="font-medium text-green-800 dark:text-green-300">장비 이상 알림</h5>
                </div>
                <p className="text-xs text-green-700 dark:text-green-400">투자: 600만원, 효과: 다운타임 -30%, 대응 시간 -80%</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded border border-green-200 dark:border-green-800">
                <div className="flex items-center gap-2 mb-1">
                  <Code className="w-4 h-4 text-green-600" />
                  <h5 className="font-medium text-green-800 dark:text-green-300">전자 체크리스트</h5>
                </div>
                <p className="text-xs text-green-700 dark:text-green-400">투자: 300만원, 효과: 점검 누락 0%, 기록 자동화</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 투자 우선순위 매트릭스 */}
      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">🎯 투자 우선순위 결정 매트릭스</h3>
        
        {/* 매트릭스 시각화 */}
        <div className="mb-8">
          <div className="relative">
            {/* Y축 라벨 - 효과 */}
            <div className="absolute -left-16 top-1/2 -translate-y-1/2 -rotate-90 text-sm font-semibold text-gray-600 dark:text-gray-400">
              효과 →
            </div>
            
            {/* X축 라벨 - 리스크 */}
            <div className="absolute bottom-[-2rem] left-1/2 -translate-x-1/2 text-sm font-semibold text-gray-600 dark:text-gray-400">
              리스크 →
            </div>
            
            <div className="grid grid-cols-2 gap-1">
              {/* 높은 효과 + 낮은 리스크 (1사분면) */}
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-4 rounded-tl-lg border-2 border-blue-300 dark:border-blue-700 min-h-[200px]">
                <h4 className="font-bold text-blue-800 dark:text-blue-300 mb-2 text-sm">
                  ✅ 즉시 실행 (우선순위 1)
                </h4>
                <p className="text-xs text-blue-700 dark:text-blue-400 mb-2">
                  높은 효과 × 낮은 리스크
                </p>
                <ul className="text-xs text-blue-600 dark:text-blue-400 space-y-1">
                  <li>• 실시간 모니터링</li>
                  <li>• 기본 자동화</li>
                  <li>• 데이터 수집 인프라</li>
                  <li>• 전자 체크리스트</li>
                </ul>
              </div>
              
              {/* 높은 효과 + 높은 리스크 (2사분면) */}
              <div className="bg-gradient-to-br from-amber-50 to-amber-100 dark:from-amber-900/20 dark:to-amber-800/20 p-4 rounded-tr-lg border-2 border-amber-300 dark:border-amber-700 min-h-[200px]">
                <h4 className="font-bold text-amber-800 dark:text-amber-300 mb-2 text-sm">
                  🔍 신중 검토 (우선순위 2)
                </h4>
                <p className="text-xs text-amber-700 dark:text-amber-400 mb-2">
                  높은 효과 × 높은 리스크
                </p>
                <ul className="text-xs text-amber-600 dark:text-amber-400 space-y-1">
                  <li>• AI 예측 시스템</li>
                  <li>• 완전 자동화 라인</li>
                  <li>• 디지털 트윈</li>
                  <li>• 걸비 통합 플랫폼</li>
                </ul>
              </div>
              
              {/* 낮은 효과 + 낮은 리스크 (3사분면) */}
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900/20 dark:to-gray-800/20 p-4 rounded-bl-lg border-2 border-gray-300 dark:border-gray-700 min-h-[200px]">
                <h4 className="font-bold text-gray-800 dark:text-gray-300 mb-2 text-sm">
                  📈 점진적 개선 (우선순위 3)
                </h4>
                <p className="text-xs text-gray-700 dark:text-gray-400 mb-2">
                  낮은 효과 × 낮은 리스크
                </p>
                <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <li>• 문서 디지털화</li>
                  <li>• 기초 IoT 센서</li>
                  <li>• 작업자 교육</li>
                  <li>• 프로세스 표준화</li>
                </ul>
              </div>
              
              {/* 낮은 효과 + 높은 리스크 (4사분면) */}
              <div className="bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 p-4 rounded-br-lg border-2 border-red-300 dark:border-red-700 min-h-[200px]">
                <h4 className="font-bold text-red-800 dark:text-red-300 mb-2 text-sm">
                  ❌ 회피/연기 (우선순위 4)
                </h4>
                <p className="text-xs text-red-700 dark:text-red-400 mb-2">
                  낮은 효과 × 높은 리스크
                </p>
                <ul className="text-xs text-red-600 dark:text-red-400 space-y-1">
                  <li>• 검증되지 않은 신기술</li>
                  <li>• 대규모 재설계</li>
                  <li>• 현장 문화와 불일치</li>
                  <li>• 업체 의존도 높은 솔루션</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        
        {/* 추가 설명 */}
        <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-900/20 rounded-lg">
          <h4 className="font-semibold text-gray-900 dark:text-white mb-2">💡 매트릭스 활용 가이드</h4>
          <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>• <span className="font-semibold">우선순위 1 (즉시 실행):</span> 빠른 성과를 낼 수 있고 실패 리스크가 낮은 항목</li>
            <li>• <span className="font-semibold">우선순위 2 (신중 검토):</span> 효과는 크지만 철저한 준비와 파일럿 테스트 필수</li>
            <li>• <span className="font-semibold">우선순위 3 (점진적 개선):</span> 여유가 있을 때 차근차근 진행</li>
            <li>• <span className="font-semibold">우선순위 4 (회피/연기):</span> ROI가 불확실하고 실패 시 타격이 큰 항목</li>
          </ul>
        </div>
      </div>

      <References
        sections={[
          {
            title: '📚 공식 문서 & 프레임워크',
            icon: 'web' as const,
            color: 'border-emerald-500',
            items: [
              {
                title: 'CMMI - Capability Maturity Model Integration',
                link: 'https://cmmiinstitute.com/',
                description: 'CMMI Institute 공식 - 프로세스 성숙도 모델 및 평가 기준'
              },
              {
                title: 'ISA-95 - Enterprise-Control System Integration',
                link: 'https://www.isa.org/standards-and-publications/isa-standards/isa-standards-committees/isa95',
                description: 'ISA-95 국제 표준 - MES/ERP 통합 아키텍처'
              },
              {
                title: 'NIST Smart Manufacturing Systems Architecture',
                link: 'https://www.nist.gov/programs-projects/smart-manufacturing',
                description: '미국 NIST 스마트 제조 시스템 아키텍처 가이드'
              },
              {
                title: 'MES International - MES Best Practices',
                link: 'https://mesa.org/',
                description: 'MESA 국제 협회 - MES 구현 모범 사례 및 가이드라인'
              }
            ]
          },
          {
            title: '📖 핵심 논문 & 연구',
            icon: 'paper' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'A maturity model for assessing Industry 4.0 readiness and maturity',
                authors: 'Schumacher, A., Erol, S., Sihn, W.',
                year: '2016',
                description: 'Procedia CIRP - Industry 4.0 성숙도 모델 및 평가 방법론'
              },
              {
                title: 'Quick wins and long-term competitive advantage in Industry 4.0',
                authors: 'Müller, J. M., Kiel, D., Voigt, K. I.',
                year: '2018',
                description: 'Journal of Business Research - Quick Win 전략 및 실증 연구'
              },
              {
                title: 'Digital Transformation Initiative: Manufacturing Industry',
                authors: 'World Economic Forum',
                year: '2018',
                description: 'WEF 제조업 디지털 전환 로드맵 및 투자 우선순위'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 리소스',
            icon: 'book' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Acatech - Industry 4.0 Maturity Index',
                link: 'https://en.acatech.de/',
                description: '독일 Acatech Industry 4.0 성숙도 측정 도구'
              },
              {
                title: 'PwC - Industry 4.0 Readiness Assessment',
                link: 'https://www.pwc.com/',
                description: 'PwC Industry 4.0 준비도 자가 진단 도구 및 벤치마킹'
              },
              {
                title: 'Gartner - Digital Transformation Roadmap Tool',
                link: 'https://www.gartner.com/',
                description: '가트너 디지털 전환 로드맵 작성 도구 (유료)'
              },
              {
                title: 'MIT - Smart Factory Assessment Framework',
                link: 'https://sma.mit.edu/',
                description: 'MIT 스마트팩토리 평가 프레임워크 및 사례 연구'
              },
              {
                title: '한국생산성본부 - 스마트팩토리 수준 진단',
                link: 'https://www.kpc.or.kr/',
                description: '한국 스마트공장 수준 진단 도구 (국문)'
              }
            ]
          }
        ]}
      />
    </div>
  );
}