'use client';

import {
  Globe, Factory, Zap, Cog, Users, Cpu
} from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter2() {
  return (
    <div className="space-y-8">
      <div className="grid lg:grid-cols-2 gap-8">
        {/* 독일 지멘스 암베르크 공장 */}
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-12 bg-slate-100 dark:bg-slate-700 rounded-lg flex items-center justify-center">
              <Factory className="w-6 h-6 text-slate-600" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">독일 지멘스 암베르크</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">세계 최고 수준 디지털 팩토리</p>
            </div>
          </div>
          <div className="space-y-4">
            <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded">
              <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">핵심 성과</h4>
              <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                <li>• 자동화율: 75% (업계 평균 30%)</li>
                <li>• 품질 수준: 99.99885% (6시그마)</li>
                <li>• 생산성 향상: 8배 (1990년 대비)</li>
                <li>• 리드타임 단축: 50%</li>
                <li>• 작업자당 생산량: 연 1,500만 개 제품</li>
              </ul>
            </div>
            <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded">
              <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">핵심 기술</h4>
              <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                <li>• SIMATIC IT MES 시스템</li>
                <li>• 디지털 트윈 기반 생산 계획</li>
                <li>• RFID 기반 제품 추적</li>
                <li>• AI 기반 품질 예측</li>
                <li>• 1,200개 데이터 매트릭스 코드</li>
              </ul>
            </div>
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded border border-blue-200 dark:border-blue-800">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">핵심 인사이트</h4>
              <p className="text-sm text-blue-700 dark:text-blue-400">
                "데이터가 새로운 원자재다" - 완전한 데이터 투명성과 실시간 의사결정이 
                혁신의 핵심. 30년간의 점진적 발전이 현재의 성과 창출.
              </p>
            </div>
          </div>
        </div>

        {/* 미국 GE 브릴리언트 팩토리 */}
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-12 bg-slate-100 dark:bg-slate-700 rounded-lg flex items-center justify-center">
              <Zap className="w-6 h-6 text-slate-600" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">미국 GE 브릴리언트 팩토리</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">디지털 트윈 혁신 모델</p>
            </div>
          </div>
          <div className="space-y-4">
            <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded">
              <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">디지털 트윈 성과</h4>
              <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                <li>• 생산성 향상: 30%</li>
                <li>• 품질 개선: 25%</li>
                <li>• 재고 비용 절감: 20%</li>
                <li>• 신제품 출시 시간: 50% 단축</li>
                <li>• 설계 변경 시간: 75% 단축</li>
              </ul>
            </div>
            <div className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded">
              <h4 className="font-semibold text-slate-700 dark:text-slate-300 mb-2">Predix 플랫폼</h4>
              <ul className="text-sm text-slate-600 dark:text-slate-400 space-y-1">
                <li>• 실시간 장비 모니터링</li>
                <li>• 예측 유지보수</li>
                <li>• 생산 최적화 알고리즘</li>
                <li>• 에너지 효율 관리</li>
                <li>• 산업 앱스토어 운영</li>
              </ul>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded border border-green-200 dark:border-green-800">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">핵심 인사이트</h4>
              <p className="text-sm text-green-700 dark:text-green-400">
                디지털 트윈은 단순한 3D 모델이 아니라 실제 운영 데이터와 연동된 
                '살아있는 시뮬레이션'. 가상에서의 실험이 실제 비용을 절감.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* 일본 도요타 */}
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-slate-100 dark:bg-slate-700 rounded-lg flex items-center justify-center">
              <Cog className="w-5 h-5 text-slate-600" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white">도요타 TPS 2.0</h3>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">린 제조 + AI의 완벽한 결합</p>
          <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1 mb-4">
            <li>• JIT + AI 수요 예측</li>
            <li>• 칸반 시스템 디지털화</li>
            <li>• 카이젠 AI 지원</li>
            <li>• 품질 관리 자동화</li>
          </ul>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded border border-yellow-200 dark:border-yellow-800">
            <p className="text-xs text-yellow-700 dark:text-yellow-400">
              <strong>성과:</strong> 재고 30% 감소, 가동률 15% 향상
            </p>
          </div>
        </div>

        {/* 중국 하이얼 */}
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-slate-100 dark:bg-slate-700 rounded-lg flex items-center justify-center">
              <Users className="w-5 h-5 text-slate-600" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white">하이얼 렌단헤이</h3>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">개인화 대량생산 혁신</p>
          <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1 mb-4">
            <li>• 고객 맞춤형 생산</li>
            <li>• 자율경영체 운영</li>
            <li>• 플랫폼 생태계</li>
            <li>• 실시간 주문-생산 연동</li>
          </ul>
          <div className="bg-purple-50 dark:bg-purple-900/20 p-3 rounded border border-purple-200 dark:border-purple-800">
            <p className="text-xs text-purple-700 dark:text-purple-400">
              <strong>성과:</strong> 대량 맞춤화 생산, 로트 사이즈 1 달성
            </p>
          </div>
        </div>

        {/* 국내 삼성전자 */}
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-slate-100 dark:bg-slate-700 rounded-lg flex items-center justify-center">
              <Cpu className="w-5 h-5 text-slate-600" />
            </div>
            <h3 className="font-semibold text-gray-900 dark:text-white">삼성전자 화성</h3>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">반도체 스마트 팩토리</p>
          <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1 mb-4">
            <li>• 완전 자동화 생산</li>
            <li>• AI 불량 예측</li>
            <li>• 디지털 트윈 활용</li>
            <li>• 실시간 최적화</li>
          </ul>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded border border-blue-200 dark:border-blue-800">
            <p className="text-xs text-blue-700 dark:text-blue-400">
              <strong>성과:</strong> 불량률 90% 감소, 생산성 40% 향상
            </p>
          </div>
        </div>
      </div>

      {/* 실패 사례와 교훈 */}
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 p-6 rounded-lg">
        <h3 className="text-xl font-semibold text-red-900 dark:text-red-300 mb-4">⚠️ 실패 사례에서 배우는 교훈</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-red-800 dark:text-red-300 mb-2">주요 실패 요인</h4>
            <ul className="text-sm text-red-700 dark:text-red-400 space-y-1">
              <li>• 경영진의 의지 부족 (30%)</li>
              <li>• 직원 저항과 교육 부족 (25%)</li>
              <li>• 기술 과신과 점진적 접근 실패 (20%)</li>
              <li>• ROI 목표 설정 오류 (15%)</li>
              <li>• 레거시 시스템 통합 실패 (10%)</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-red-800 dark:text-red-300 mb-2">성공을 위한 핵심 요소</h4>
            <ul className="text-sm text-red-700 dark:text-red-400 space-y-1">
              <li>• 명확한 비즈니스 목표 설정</li>
              <li>• 점진적 단계별 접근</li>
              <li>• 전사적 변화 관리</li>
              <li>• 파일럿 프로젝트 선행</li>
              <li>• 지속적 모니터링과 개선</li>
            </ul>
          </div>
        </div>
        <div className="mt-4 p-4 bg-red-100 dark:bg-red-900/40 rounded border border-red-300 dark:border-red-700">
          <h5 className="font-semibold text-red-900 dark:text-red-300 mb-2">실제 실패 사례: A 중소기업</h5>
          <p className="text-sm text-red-800 dark:text-red-400">
            300억 투자 후 3년간 ROI 달성 실패. 원인: 기존 ERP와 연동 실패, 직원 교육 소홀, 
            과도한 기술 도입. 결국 단계적 재구축으로 2년 추가 소요.
          </p>
        </div>
      </div>

      <References
        sections={[
          {
            title: '📚 공식 문서 & 사례',
            icon: 'web' as const,
            color: 'border-emerald-500',
            items: [
              {
                title: 'Siemens Amberg Electronics Plant - Digital Enterprise',
                link: 'https://www.siemens.com/global/en/products/automation/topic-areas/digital-enterprise.html',
                description: '지멘스 암베르크 공장 공식 사례 및 디지털 팩토리 솔루션'
              },
              {
                title: 'GE Digital - Predix Platform Documentation',
                link: 'https://www.ge.com/digital/',
                description: 'GE Predix 산업용 IoT 플랫폼 공식 문서'
              },
              {
                title: 'Toyota Production System - Official Guide',
                link: 'https://global.toyota/en/company/vision-and-philosophy/production-system/',
                description: '도요타 생산 시스템(TPS) 공식 가이드 및 린 제조 철학'
              },
              {
                title: 'Haier COSMOPlat - Industrial Internet Platform',
                link: 'https://www.cosmoplat.com/',
                description: '하이얼 코스모플랫 대량 맞춤화 플랫폼 공식 사이트'
              }
            ]
          },
          {
            title: '📖 핵심 연구 & 분석',
            icon: 'paper' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Digital Twin: Enabling Technologies, Challenges and Open Research',
                authors: 'Grieves, M. & Vickers, J.',
                year: '2017',
                description: 'IEEE Access - 디지털 트윈 개념 및 구현 방법론'
              },
              {
                title: 'Smart Manufacturing: The Research and Application Perspectives',
                authors: 'Davis, J., Edgar, T. F.',
                year: '2015',
                description: 'Computers & Chemical Engineering - 스마트 제조 연구 동향'
              },
              {
                title: 'Benchmarking manufacturing sector 4.0',
                authors: 'Moeuf, A., et al.',
                year: '2018',
                description: 'Journal of Manufacturing Technology Management - 제조업 4.0 벤치마킹'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 리소스',
            icon: 'book' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'World Economic Forum - Fourth Industrial Revolution Center',
                link: 'https://www.weforum.org/centre-for-the-fourth-industrial-revolution/',
                description: '세계경제포럼 4차 산업혁명 센터 - 글로벌 사례 및 정책 연구'
              },
              {
                title: 'Deloitte - Digital Manufacturing Enterprise',
                link: 'https://www2.deloitte.com/insights',
                description: '딜로이트 디지털 제조 기업 전환 사례 연구 및 ROI 분석'
              },
              {
                title: 'MIT Technology Review - Smart Factory Reports',
                link: 'https://www.technologyreview.com/',
                description: 'MIT 기술 리뷰 - 스마트팩토리 최신 기술 및 트렌드'
              },
              {
                title: 'Industry Week - Manufacturing Excellence',
                link: 'https://www.industryweek.com/',
                description: '제조업 전문 매체 - 성공/실패 사례 분석'
              }
            ]
          }
        ]}
      />
    </div>
  );
}