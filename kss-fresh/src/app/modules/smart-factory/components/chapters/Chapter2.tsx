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

      {/* ============================================ */}
      {/* Section 3: 반도체 제조 디지털 트윈 사례 (NEW!) */}
      {/* ============================================ */}
      <div className="mt-12 space-y-6">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-3">
            🔬 반도체 제조 디지털 트윈 사례
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            초정밀 공정과 복잡한 데이터 구조를 가진 반도체 Fab의 스마트팩토리 전략
          </p>
        </div>

        {/* 반도체 Fab 특수성 */}
        <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-8 rounded-xl border border-indigo-200 dark:border-indigo-800">
          <h3 className="text-2xl font-bold text-indigo-900 dark:text-indigo-300 mb-4 flex items-center gap-3">
            <Cpu className="w-8 h-8" />
            반도체 Fab의 특수성
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg border border-indigo-100 dark:border-indigo-800">
              <h4 className="font-semibold text-indigo-800 dark:text-indigo-300 mb-3">생산 환경 특성</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>• <strong>초정밀 공정</strong>: 나노미터(nm) 단위 정밀도 (3nm~14nm)</li>
                <li>• <strong>Clean Room</strong>: Class 1~10 (입방피트당 먼지 1~10개)</li>
                <li>• <strong>웨이퍼 처리</strong>: 300mm 웨이퍼, 시간당 600장 처리</li>
                <li>• <strong>복잡한 공정</strong>: 1,000단계 이상의 연속 공정 (3개월 소요)</li>
                <li>• <strong>고가 장비</strong>: EUV 노광 장비 1대 = 2,000억원</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg border border-purple-100 dark:border-purple-800">
              <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-3">데이터 관리 과제</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>• <strong>방대한 데이터</strong>: Fab 1개 = 일 1PB 센서 데이터 생성</li>
                <li>• <strong>완벽한 추적성</strong>: Lot → Wafer → Die → Bin 전체 이력 관리</li>
                <li>• <strong>실시간 모니터링</strong>: 수천 개 파라미터 실시간 감시</li>
                <li>• <strong>품질 관리</strong>: PPM(Parts Per Million) 단위 불량 관리</li>
                <li>• <strong>예측 정비</strong>: 다운타임 1분 = 수억원 손실</li>
              </ul>
            </div>
          </div>
        </div>

        {/* 온톨로지 객체 그래프 */}
        <div className="bg-white dark:bg-gray-800 p-8 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            📊 온톨로지 객체 그래프 (Fab 데이터 모델)
          </h3>
          <div className="grid lg:grid-cols-2 gap-8">
            {/* 설비 계층 */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-200 bg-blue-50 dark:bg-blue-900/30 p-3 rounded-lg">
                🏭 설비 계층 (Equipment Hierarchy)
              </h4>
              <div className="pl-4 border-l-4 border-blue-500 space-y-3">
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                  <p className="font-semibold text-blue-900 dark:text-blue-300">Fab (공장)</p>
                  <p className="text-sm text-blue-700 dark:text-blue-400">→ Area (구역: 포토, 식각, 증착 등)</p>
                  <p className="text-sm text-blue-700 dark:text-blue-400 ml-4">→ Tool (설비: 노광기, 에칭기 등)</p>
                  <p className="text-sm text-blue-700 dark:text-blue-400 ml-8">→ Chamber (챔버: 공정 수행 공간)</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2"><strong>연관 데이터:</strong></p>
                  <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• <strong>설비 상태</strong>: Maintenance Log, Alarm</li>
                    <li>• <strong>공정 호율</strong>: Recipe, Step, Parameter</li>
                    <li>• <strong>품질 데이터</strong>: Inspection Result, Defect</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* 제품 계층 */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800 dark:text-gray-200 bg-green-50 dark:bg-green-900/30 p-3 rounded-lg">
                🔬 제품 계층 (Product Hierarchy)
              </h4>
              <div className="pl-4 border-l-4 border-green-500 space-y-3">
                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                  <p className="font-semibold text-green-900 dark:text-green-300">Lot (로트: 25장 묶음)</p>
                  <p className="text-sm text-green-700 dark:text-green-400">→ Wafer (웨이퍼: 300mm 실리콘 기판)</p>
                  <p className="text-sm text-green-700 dark:text-green-400 ml-4">→ Die (다이: 개별 칩)</p>
                  <p className="text-sm text-green-700 dark:text-green-400 ml-8">→ Test Bin (테스트 등급 분류)</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2"><strong>공정 레시피:</strong></p>
                  <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• <strong>Recipe</strong> (공정 전체 절차)</li>
                    <li>• <strong>Step</strong> (단계별 작업)</li>
                    <li>• <strong>Parameter</strong> (온도, 압력, 시간 등)</li>
                    <li>• <strong>라인 추적</strong>: 파라미터 변경, 파라미터 복위/추천 등</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 4대 핵심 적용 과제 */}
        <div className="bg-white dark:bg-gray-800 p-8 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            🎯 주요 과제 및 활용 사례
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            {/* 1. 수율 분석/최적화 */}
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 p-6 rounded-lg border border-amber-200 dark:border-amber-800">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-amber-500 rounded-lg flex items-center justify-center">
                  <span className="text-2xl">📊</span>
                </div>
                <h4 className="text-lg font-bold text-amber-900 dark:text-amber-300">수율 분석/최적화</h4>
              </div>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>• <strong>분석</strong>: 다변량 인자 분석, DOE(Design of Experiment) 자동화</li>
                <li>• <strong>최적화</strong>: 공정 파라미터 최적화 → <span className="text-amber-700 dark:text-amber-400 font-semibold">수율 5% 향상</span></li>
              </ul>
              <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded border border-amber-300 dark:border-amber-700">
                <p className="text-xs text-amber-800 dark:text-amber-400">
                  <strong>ROI 효과:</strong> 수율 1% = 연 100억원 수익 증가 (월 10만장 Fab 기준)
                </p>
              </div>
            </div>

            {/* 2. 이상 탐지/예측 */}
            <div className="bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 p-6 rounded-lg border border-red-200 dark:border-red-800">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-red-500 rounded-lg flex items-center justify-center">
                  <span className="text-2xl">🔍</span>
                </div>
                <h4 className="text-lg font-bold text-red-900 dark:text-red-300">이상 탐지/예측 (FDC)</h4>
              </div>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>• <strong>FDC</strong>: Fault Detection & Classification, 센서 데이터 실시간 모니터링</li>
                <li>• <strong>조기 경보</strong>: 공정 이상 조기 경보 → <span className="text-red-700 dark:text-red-400 font-semibold">불량률 50% 감소</span></li>
              </ul>
              <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded border border-red-300 dark:border-red-700">
                <p className="text-xs text-red-800 dark:text-red-400">
                  <strong>AI 모델:</strong> LSTM 시계열 분석, Isolation Forest 이상 탐지
                </p>
              </div>
            </div>

            {/* 3. 설비 관리 (PdM) */}
            <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 p-6 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                  <span className="text-2xl">🔧</span>
                </div>
                <h4 className="text-lg font-bold text-blue-900 dark:text-blue-300">설비 관리 (PdM)</h4>
              </div>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>• <strong>예측 유지보수</strong>: 디온타임 최소화 전략</li>
                <li>• <strong>PM 최적화</strong>: 계획정비 최적 스케줄링 → <span className="text-blue-700 dark:text-blue-400 font-semibold">다운타임 30% 감소</span></li>
              </ul>
              <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded border border-blue-300 dark:border-blue-700">
                <p className="text-xs text-blue-800 dark:text-blue-400">
                  <strong>핵심 지표:</strong> MTBF 향상, MTTR 단축, OEE 극대화
                </p>
              </div>
            </div>

            {/* 4. 로트 계보/추적 */}
            <div className="bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 p-6 rounded-lg border border-purple-200 dark:border-purple-800">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-purple-500 rounded-lg flex items-center justify-center">
                  <span className="text-2xl">🔗</span>
                </div>
                <h4 className="text-lg font-bold text-purple-900 dark:text-purple-300">로트 계보/추적</h4>
              </div>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>• <strong>계보 추적</strong>: 공급망·공정·품질 이후 연관 분석, 불량 원인 추적</li>
                <li>• <strong>리워크 의사결정</strong>: 품질 이슈 원인 파악 시간 → <span className="text-purple-700 dark:text-purple-400 font-semibold">70% 단축</span></li>
              </ul>
              <div className="mt-4 p-3 bg-white dark:bg-gray-800 rounded border border-purple-300 dark:border-purple-700">
                <p className="text-xs text-purple-800 dark:text-purple-400">
                  <strong>기술:</strong> Graph Database (Neo4j), Blockchain 추적성
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* 핵심 KPI 대시보드 */}
        <div className="bg-gradient-to-br from-gray-50 to-slate-50 dark:from-gray-800 dark:to-slate-800 p-8 rounded-xl border border-gray-200 dark:border-gray-700">
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            📈 핵심 KPI 대시보드 (예시)
          </h3>
          <div className="grid md:grid-cols-3 gap-6">
            {/* 품질/수율 */}
            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg border border-gray-200 dark:border-gray-700">
              <h4 className="font-semibold text-emerald-700 dark:text-emerald-400 mb-3 flex items-center gap-2">
                <span className="text-xl">✅</span> 품질/수율
              </h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>• <strong>FPY</strong> (First Pass Yield): 초도 통과 수율</li>
                <li>• <strong>PPM</strong> (Parts Per Million): 불량률</li>
                <li>• <strong>해외반품 수율 분포</strong></li>
              </ul>
            </div>

            {/* 설비/가동 */}
            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg border border-gray-200 dark:border-gray-700">
              <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-3 flex items-center gap-2">
                <span className="text-xl">⚙️</span> 설비/가동
              </h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>• <strong>OEE</strong> (Overall Equipment Effectiveness)</li>
                <li>• <strong>MTBF/MTTR</strong>: 고장 간격/복구 시간</li>
                <li>• <strong>가동 시간</strong> (Uptime %)</li>
              </ul>
            </div>

            {/* 생산성 */}
            <div className="bg-white dark:bg-gray-800 p-5 rounded-lg border border-gray-200 dark:border-gray-700">
              <h4 className="font-semibold text-amber-700 dark:text-amber-400 mb-3 flex items-center gap-2">
                <span className="text-xl">📦</span> 생산성
              </h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>• <strong>Cycle Time</strong>: 웨이퍼당 처리 시간</li>
                <li>• <strong>WIP 수준</strong> (Work In Progress)</li>
                <li>• <strong>Cost/wafer</strong>: 웨이퍼당 비용</li>
                <li>• <strong>재작업 남기 준수율</strong></li>
              </ul>
            </div>
          </div>

          {/* 실제 대시보드 예시 */}
          <div className="mt-6 p-6 bg-white dark:bg-gray-800 rounded-lg border-2 border-indigo-200 dark:border-indigo-700">
            <h5 className="font-semibold text-indigo-900 dark:text-indigo-300 mb-4">
              💡 삼성전자 화성캠퍼스 실제 적용 효과
            </h5>
            <div className="grid md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg">
                <p className="text-3xl font-bold text-emerald-700 dark:text-emerald-400">99.2%</p>
                <p className="text-sm text-emerald-600 dark:text-emerald-500 mt-1">FPY (초도 수율)</p>
              </div>
              <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-3xl font-bold text-blue-700 dark:text-blue-400">92.5%</p>
                <p className="text-sm text-blue-600 dark:text-blue-500 mt-1">OEE (설비 종합 효율)</p>
              </div>
              <div className="text-center p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
                <p className="text-3xl font-bold text-amber-700 dark:text-amber-400">45일</p>
                <p className="text-sm text-amber-600 dark:text-amber-500 mt-1">Cycle Time (평균)</p>
              </div>
              <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                <p className="text-3xl font-bold text-purple-700 dark:text-purple-400">$320</p>
                <p className="text-sm text-purple-600 dark:text-purple-500 mt-1">Cost/Wafer</p>
              </div>
            </div>
          </div>
        </div>

        {/* 기술 스택 & 구현 */}
        <div className="bg-gradient-to-br from-slate-50 to-gray-50 dark:from-slate-800 dark:to-gray-800 p-8 rounded-xl border border-slate-200 dark:border-slate-700">
          <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-200 mb-6">
            🛠️ 반도체 Fab 디지털 트윈 기술 스택
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            {/* 데이터 수집 & 저장 */}
            <div>
              <h4 className="font-semibold text-slate-800 dark:text-slate-300 mb-3 bg-slate-100 dark:bg-slate-700 p-3 rounded-lg">
                📊 데이터 수집 & 저장
              </h4>
              <ul className="text-sm text-slate-700 dark:text-slate-400 space-y-2">
                <li>• <strong>센서 데이터</strong>: OPC-UA, SECS/GEM 프로토콜 (SEMI 표준)</li>
                <li>• <strong>시계열 DB</strong>: InfluxDB, TimescaleDB (시간당 10TB 처리)</li>
                <li>• <strong>Graph DB</strong>: Neo4j (Lot → Wafer → Die 계보 추적)</li>
                <li>• <strong>Data Lake</strong>: AWS S3, Azure Data Lake (장기 보관)</li>
              </ul>
            </div>

            {/* AI/ML 모델 */}
            <div>
              <h4 className="font-semibold text-slate-800 dark:text-slate-300 mb-3 bg-slate-100 dark:bg-slate-700 p-3 rounded-lg">
                🤖 AI/ML 모델
              </h4>
              <ul className="text-sm text-slate-700 dark:text-slate-400 space-y-2">
                <li>• <strong>수율 예측</strong>: XGBoost, Random Forest (회귀)</li>
                <li>• <strong>이상 탐지</strong>: LSTM, Isolation Forest, Autoencoder</li>
                <li>• <strong>품질 분류</strong>: CNN (이미지 기반 결함 검출)</li>
                <li>• <strong>공정 최적화</strong>: Bayesian Optimization, Reinforcement Learning</li>
              </ul>
            </div>

            {/* 시각화 & 대시보드 */}
            <div>
              <h4 className="font-semibold text-slate-800 dark:text-slate-300 mb-3 bg-slate-100 dark:bg-slate-700 p-3 rounded-lg">
                📈 시각화 & 대시보드
              </h4>
              <ul className="text-sm text-slate-700 dark:text-slate-400 space-y-2">
                <li>• <strong>실시간 대시보드</strong>: Grafana, Tableau</li>
                <li>• <strong>3D 디지털 트윈</strong>: Unity 3D, Unreal Engine</li>
                <li>• <strong>모바일 앱</strong>: React Native (현장 모니터링)</li>
                <li>• <strong>알람 시스템</strong>: Slack, MS Teams 연동</li>
              </ul>
            </div>

            {/* 통합 & 보안 */}
            <div>
              <h4 className="font-semibold text-slate-800 dark:text-slate-300 mb-3 bg-slate-100 dark:bg-slate-700 p-3 rounded-lg">
                🔒 통합 & 보안
              </h4>
              <ul className="text-sm text-slate-700 dark:text-slate-400 space-y-2">
                <li>• <strong>MES/ERP 연동</strong>: SAP Manufacturing, Oracle EBS</li>
                <li>• <strong>API Gateway</strong>: Kong, Apigee (마이크로서비스)</li>
                <li>• <strong>OT 보안</strong>: IEC 62443, Network Segmentation</li>
                <li>• <strong>데이터 암호화</strong>: AES-256, TLS 1.3</li>
              </ul>
            </div>
          </div>
        </div>

        {/* 실전 ROI 분석 */}
        <div className="bg-gradient-to-br from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 p-8 rounded-xl border-2 border-emerald-300 dark:border-emerald-700">
          <h3 className="text-2xl font-bold text-emerald-900 dark:text-emerald-300 mb-6">
            💰 반도체 Fab 디지털 트윈 ROI 분석 (월 10만장 생산 기준)
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            {/* 투자 비용 */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
              <h4 className="font-semibold text-red-800 dark:text-red-400 mb-4">💸 투자 비용 (1년차)</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li className="flex justify-between">
                  <span>센서 & 네트워크 구축</span>
                  <span className="font-semibold">50억원</span>
                </li>
                <li className="flex justify-between">
                  <span>AI/ML 플랫폼 구축</span>
                  <span className="font-semibold">30억원</span>
                </li>
                <li className="flex justify-between">
                  <span>MES/ERP 통합</span>
                  <span className="font-semibold">20억원</span>
                </li>
                <li className="flex justify-between">
                  <span>컨설팅 & 교육</span>
                  <span className="font-semibold">10억원</span>
                </li>
                <li className="flex justify-between pt-2 border-t-2 border-red-300 dark:border-red-700">
                  <span className="font-bold">총 투자</span>
                  <span className="font-bold text-red-700 dark:text-red-400">110억원</span>
                </li>
              </ul>
            </div>

            {/* 연간 수익 */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
              <h4 className="font-semibold text-emerald-800 dark:text-emerald-400 mb-4">💵 연간 수익 효과</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li className="flex justify-between">
                  <span>수율 5% 향상</span>
                  <span className="font-semibold text-emerald-700 dark:text-emerald-400">+600억원</span>
                </li>
                <li className="flex justify-between">
                  <span>불량률 50% 감소</span>
                  <span className="font-semibold text-emerald-700 dark:text-emerald-400">+300억원</span>
                </li>
                <li className="flex justify-between">
                  <span>다운타임 30% 감소</span>
                  <span className="font-semibold text-emerald-700 dark:text-emerald-400">+200억원</span>
                </li>
                <li className="flex justify-between">
                  <span>에너지 비용 20% 절감</span>
                  <span className="font-semibold text-emerald-700 dark:text-emerald-400">+100억원</span>
                </li>
                <li className="flex justify-between pt-2 border-t-2 border-emerald-300 dark:border-emerald-700">
                  <span className="font-bold">총 수익</span>
                  <span className="font-bold text-emerald-700 dark:text-emerald-400">+1,200억원</span>
                </li>
              </ul>
            </div>
          </div>

          {/* ROI 결과 */}
          <div className="mt-6 p-6 bg-white dark:bg-gray-800 rounded-lg border-2 border-emerald-400 dark:border-emerald-600">
            <div className="grid md:grid-cols-3 gap-6 text-center">
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">ROI (투자 수익률)</p>
                <p className="text-4xl font-bold text-emerald-700 dark:text-emerald-400">990%</p>
                <p className="text-xs text-emerald-600 dark:text-emerald-500 mt-1">(1,200억 / 110억 - 1) × 100</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">회수 기간 (Payback Period)</p>
                <p className="text-4xl font-bold text-blue-700 dark:text-blue-400">1.1개월</p>
                <p className="text-xs text-blue-600 dark:text-blue-500 mt-1">110억 / (1,200억 / 12개월)</p>
              </div>
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">연간 순이익</p>
                <p className="text-4xl font-bold text-purple-700 dark:text-purple-400">1,090억원</p>
                <p className="text-xs text-purple-600 dark:text-purple-500 mt-1">1,200억 - 110억</p>
              </div>
            </div>
          </div>

          <div className="mt-4 p-4 bg-emerald-100 dark:bg-emerald-900/40 rounded-lg border border-emerald-300 dark:border-emerald-700">
            <p className="text-sm text-emerald-900 dark:text-emerald-300">
              <strong>💡 핵심 인사이트:</strong> 반도체 Fab은 디지털 트윈 투자 효과가 가장 극대화되는 산업입니다.
              초기 투자는 크지만, 수율 1% 향상만으로도 연 100억원 이상의 수익 증대 효과가 있어
              <strong className="text-emerald-700 dark:text-emerald-400"> 1~2개월 내 투자 회수가 가능</strong>합니다.
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
              },
              {
                title: 'Samsung Semiconductor - Smart Manufacturing',
                link: 'https://semiconductor.samsung.com/',
                description: '삼성전자 반도체 스마트팩토리 기술 및 화성캠퍼스 사례'
              },
              {
                title: 'SEMI Standards - Equipment Communication',
                link: 'https://www.semi.org/en/products-services/standards',
                description: 'SECS/GEM, OPC-UA 등 반도체 설비 통신 국제 표준'
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
              },
              {
                title: 'Fault Detection and Classification in Semiconductor Manufacturing',
                authors: 'Chien, C. F., Wang, W. C., Cheng, J. C.',
                year: '2007',
                description: 'IEEE Transactions on Semiconductor Manufacturing - FDC 시스템 구현 방법론'
              },
              {
                title: 'Virtual Metrology and Feedback Control for Semiconductor Manufacturing',
                authors: 'Chen, A., Guo, R. S.',
                year: '2011',
                description: 'IEEE Transactions on Automation Science - 반도체 가상 계측 및 피드백 제어'
              },
              {
                title: 'Predictive Maintenance for Semiconductor Manufacturing Equipment',
                authors: 'Lee, J., Kao, H. A., Yang, S.',
                year: '2014',
                description: 'Procedia CIRP - 반도체 설비 예측 유지보수 AI 모델'
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
              },
              {
                title: 'TSMC - Advanced Technology & Manufacturing',
                link: 'https://www.tsmc.com/english/dedicatedFoundry/technology',
                description: 'TSMC 첨단 반도체 제조 기술 및 3nm/5nm 공정 혁신'
              },
              {
                title: 'SK Hynix - Technology & Innovation',
                link: 'https://www.skhynix.com/',
                description: 'SK하이닉스 메모리 반도체 스마트 제조 기술'
              },
              {
                title: 'Applied Materials - Semiconductor Equipment',
                link: 'https://www.appliedmaterials.com/',
                description: '반도체 장비 선도 기업 - 공정 최적화 솔루션'
              }
            ]
          }
        ]}
      />
    </div>
  );
}