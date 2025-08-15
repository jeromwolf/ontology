'use client';

import { 
  Database, Cog, Network, Server, BarChart3
} from 'lucide-react';
import Link from 'next/link';

export default function Chapter11() {
  return (
    <div className="space-y-8">
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Cog className="w-6 h-6 text-slate-600" />
            MES 11대 핵심 기능
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-900/20 rounded border-l-4 border-blue-400">
              <div>
                <h4 className="font-semibold text-blue-800 dark:text-blue-300">작업지시 관리</h4>
                <p className="text-sm text-blue-600 dark:text-blue-400">Work Order Management</p>
              </div>
              <div className="text-xs bg-blue-100 dark:bg-blue-800 text-blue-700 dark:text-blue-300 px-2 py-1 rounded">WOM</div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded border-l-4 border-green-400">
              <div>
                <h4 className="font-semibold text-green-800 dark:text-green-300">자원 할당 및 상태</h4>
                <p className="text-sm text-green-600 dark:text-green-400">Resource Allocation & Status</p>
              </div>
              <div className="text-xs bg-green-100 dark:bg-green-800 text-green-700 dark:text-green-300 px-2 py-1 rounded">RAS</div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-purple-50 dark:bg-purple-900/20 rounded border-l-4 border-purple-400">
              <div>
                <h4 className="font-semibold text-purple-800 dark:text-purple-300">작업일정 관리</h4>
                <p className="text-sm text-purple-600 dark:text-purple-400">Operations Scheduling</p>
              </div>
              <div className="text-xs bg-purple-100 dark:bg-purple-800 text-purple-700 dark:text-purple-300 px-2 py-1 rounded">OSC</div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-red-50 dark:bg-red-900/20 rounded border-l-4 border-red-400">
              <div>
                <h4 className="font-semibold text-red-800 dark:text-red-300">문서 제어</h4>
                <p className="text-sm text-red-600 dark:text-red-400">Document Control</p>
              </div>
              <div className="text-xs bg-red-100 dark:bg-red-800 text-red-700 dark:text-red-300 px-2 py-1 rounded">DOC</div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded border-l-4 border-yellow-400">
              <div>
                <h4 className="font-semibold text-yellow-800 dark:text-yellow-300">데이터 수집/취득</h4>
                <p className="text-sm text-yellow-600 dark:text-yellow-400">Data Collection/Acquisition</p>
              </div>
              <div className="text-xs bg-yellow-100 dark:bg-yellow-800 text-yellow-700 dark:text-yellow-300 px-2 py-1 rounded">DCA</div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded border-l-4 border-indigo-400">
              <div>
                <h4 className="font-semibold text-indigo-800 dark:text-indigo-300">인력 관리</h4>
                <p className="text-sm text-indigo-600 dark:text-indigo-400">Labor Management</p>
              </div>
              <div className="text-xs bg-indigo-100 dark:bg-indigo-800 text-indigo-700 dark:text-indigo-300 px-2 py-1 rounded">LAB</div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Database className="w-6 h-6 text-slate-600" />
            MES 11대 기능 (계속)
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-teal-50 dark:bg-teal-900/20 rounded border-l-4 border-teal-400">
              <div>
                <h4 className="font-semibold text-teal-800 dark:text-teal-300">품질 관리</h4>
                <p className="text-sm text-teal-600 dark:text-teal-400">Quality Management</p>
              </div>
              <div className="text-xs bg-teal-100 dark:bg-teal-800 text-teal-700 dark:text-teal-300 px-2 py-1 rounded">QUA</div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-orange-50 dark:bg-orange-900/20 rounded border-l-4 border-orange-400">
              <div>
                <h4 className="font-semibold text-orange-800 dark:text-orange-300">공정 관리</h4>
                <p className="text-sm text-orange-600 dark:text-orange-400">Process Management</p>
              </div>
              <div className="text-xs bg-orange-100 dark:bg-orange-800 text-orange-700 dark:text-orange-300 px-2 py-1 rounded">PRO</div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-pink-50 dark:bg-pink-900/20 rounded border-l-4 border-pink-400">
              <div>
                <h4 className="font-semibold text-pink-800 dark:text-pink-300">유지보수 관리</h4>
                <p className="text-sm text-pink-600 dark:text-pink-400">Maintenance Management</p>
              </div>
              <div className="text-xs bg-pink-100 dark:bg-pink-800 text-pink-700 dark:text-pink-300 px-2 py-1 rounded">MAI</div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-cyan-50 dark:bg-cyan-900/20 rounded border-l-4 border-cyan-400">
              <div>
                <h4 className="font-semibold text-cyan-800 dark:text-cyan-300">제품 추적 관리</h4>
                <p className="text-sm text-cyan-600 dark:text-cyan-400">Product Tracking</p>
              </div>
              <div className="text-xs bg-cyan-100 dark:bg-cyan-800 text-cyan-700 dark:text-cyan-300 px-2 py-1 rounded">TRA</div>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/20 rounded border-l-4 border-gray-400">
              <div>
                <h4 className="font-semibold text-gray-800 dark:text-gray-300">성과 분석</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">Performance Analysis</p>
              </div>
              <div className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">PER</div>
            </div>
          </div>
        </div>
      </div>

      {/* 시뮬레이터 체험 섹션 */}
      <div className="mt-8 p-6 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl border border-indigo-200 dark:border-indigo-800">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-indigo-900 dark:text-indigo-200 mb-2">
              🎮 MES/ERP 통합 대시보드 체험
            </h3>
            <p className="text-sm text-indigo-700 dark:text-indigo-300">
              실시간 생산 현황과 경영 정보를 통합 관리하는 대시보드를 체험해보세요.
            </p>
          </div>
          <Link
            href="/modules/smart-factory/simulators/mes-erp-dashboard?from=/modules/smart-factory/mes-erp-integration"
            className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors"
          >
            <span>시뮬레이터 체험</span>
            <span className="text-lg">→</span>
          </Link>
        </div>
      </div>

      <div className="bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 p-8 rounded-xl border border-indigo-200 dark:border-indigo-800">
        <h3 className="text-2xl font-bold text-indigo-900 dark:text-indigo-200 mb-6 flex items-center gap-3">
          <Network className="w-8 h-8" />
          ERP-MES 통합 아키텍처
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-indigo-800/30 p-6 rounded-lg border border-indigo-300 dark:border-indigo-600">
            <h4 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">1층: ERP 시스템</h4>
            <ul className="space-y-2 text-sm text-indigo-700 dark:text-indigo-300">
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
                주문 관리 (Order Management)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
                생산 계획 (Production Planning)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
                자재 소요 계획 (MRP)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
                재무 회계 (Financial)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
                인사 관리 (HR)
              </li>
            </ul>
            <p className="text-xs text-indigo-500 dark:text-indigo-400 mt-3">
              전사 차원의 계획과 경영 정보 관리
            </p>
          </div>
          
          <div className="bg-white dark:bg-indigo-800/30 p-6 rounded-lg border border-indigo-300 dark:border-indigo-600">
            <h4 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">2층: 통합 미들웨어</h4>
            <ul className="space-y-2 text-sm text-indigo-700 dark:text-indigo-300">
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                ESB (Enterprise Service Bus)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                API Gateway
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                메시지 큐 (Message Queue)
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                데이터 매핑 엔진
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                트랜잭션 관리자
              </li>
            </ul>
            <p className="text-xs text-blue-500 dark:text-blue-400 mt-3">
              시스템 간 통신과 데이터 변환 처리
            </p>
          </div>
          
          <div className="bg-white dark:bg-indigo-800/30 p-6 rounded-lg border border-indigo-300 dark:border-indigo-600">
            <h4 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">3층: MES 시스템</h4>
            <ul className="space-y-2 text-sm text-indigo-700 dark:text-indigo-300">
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                실시간 생산 관리
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                작업장 스케줄링
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                품질 데이터 수집
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                장비 모니터링
              </li>
              <li className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                제품 추적성
              </li>
            </ul>
            <p className="text-xs text-green-500 dark:text-green-400 mt-3">
              제조 현장의 실행과 제어 담당
            </p>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <Server className="w-8 h-8 text-amber-600" />
          주요 ERP 벤더와 MES 연동
        </h3>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="w-20 h-20 bg-blue-500 rounded-lg flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-xl">SAP</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">SAP S/4HANA</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• BAPI/RFC 인터페이스</li>
              <li>• PI/PO 통합 플랫폼</li>
              <li>• Real-time data replication</li>
              <li>• PP/QM 모듈 연계</li>
            </ul>
            <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-xs text-blue-600 dark:text-blue-400">
              시장 점유율 26%
            </div>
          </div>
          
          <div className="text-center">
            <div className="w-20 h-20 bg-red-500 rounded-lg flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-lg">Oracle</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">Oracle Cloud ERP</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• REST/SOAP API</li>
              <li>• Oracle Integration Cloud</li>
              <li>• IoT Analytics 연동</li>
              <li>• SCM/Manufacturing</li>
            </ul>
            <div className="mt-3 p-2 bg-red-50 dark:bg-red-900/20 rounded text-xs text-red-600 dark:text-red-400">
              시장 점유율 15%
            </div>
          </div>
          
          <div className="text-center">
            <div className="w-20 h-20 bg-blue-600 rounded-lg flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-lg">MS</span>
            </div>
            <h4 className="font-bold text-gray-900 dark:text-white mb-2">Dynamics 365</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Power Platform</li>
              <li>• Azure Logic Apps</li>
              <li>• Mixed Reality 지원</li>
              <li>• AI Builder 통합</li>
            </ul>
            <div className="mt-3 p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-xs text-blue-600 dark:text-blue-400">
              시장 점유율 8%
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-8 rounded-xl border border-green-200 dark:border-green-800">
        <h3 className="text-2xl font-bold text-green-900 dark:text-green-200 mb-6 flex items-center gap-3">
          <BarChart3 className="w-8 h-8" />
          APS (Advanced Planning & Scheduling) 시스템
        </h3>
        <div className="grid lg:grid-cols-2 gap-8">
          <div className="space-y-4">
            <h4 className="text-xl font-semibold text-green-800 dark:text-green-200 mb-4">최적화 알고리즘</h4>
            <div className="space-y-3">
              <div className="p-4 bg-white dark:bg-green-800/30 rounded-lg border border-green-200 dark:border-green-600">
                <h5 className="font-semibold text-green-700 dark:text-green-300 mb-2">유전 알고리즘 (Genetic Algorithm)</h5>
                <p className="text-sm text-green-600 dark:text-green-400 mb-2">대규모 조합 최적화 문제 해결</p>
                <div className="text-xs text-green-500 dark:text-green-500">
                  • 선택(Selection) → 교배(Crossover) → 돌연변이(Mutation) → 진화
                </div>
              </div>
              
              <div className="p-4 bg-white dark:bg-green-800/30 rounded-lg border border-green-200 dark:border-green-600">
                <h5 className="font-semibold text-green-700 dark:text-green-300 mb-2">시뮬레이티드 어닐링</h5>
                <p className="text-sm text-green-600 dark:text-green-400 mb-2">지역 최적해 탈출 및 전역 최적해 탐색</p>
                <div className="text-xs text-green-500 dark:text-green-500">
                  • 온도 감소에 따른 점진적 최적화
                </div>
              </div>
              
              <div className="p-4 bg-white dark:bg-green-800/30 rounded-lg border border-green-200 dark:border-green-600">
                <h5 className="font-semibold text-green-700 dark:text-green-300 mb-2">타부 서치 (Tabu Search)</h5>
                <p className="text-sm text-green-600 dark:text-green-400 mb-2">금기 목록을 통한 순환 방지</p>
                <div className="text-xs text-green-500 dark:text-green-500">
                  • 이전 해 기억을 통한 지역 최적해 탈출
                </div>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <h4 className="text-xl font-semibold text-green-800 dark:text-green-200 mb-4">실제 적용 효과</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-white dark:bg-green-800/30 rounded-lg border border-green-200 dark:border-green-600">
                <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">-20%</div>
                <div className="text-sm text-green-700 dark:text-green-300">리드타임 단축</div>
              </div>
              <div className="text-center p-4 bg-white dark:bg-green-800/30 rounded-lg border border-green-200 dark:border-green-600">
                <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">+15%</div>
                <div className="text-sm text-green-700 dark:text-green-300">생산능력 향상</div>
              </div>
              <div className="text-center p-4 bg-white dark:bg-green-800/30 rounded-lg border border-green-200 dark:border-green-600">
                <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">-30%</div>
                <div className="text-sm text-green-700 dark:text-green-300">재고 감소</div>
              </div>
              <div className="text-center p-4 bg-white dark:bg-green-800/30 rounded-lg border border-green-200 dark:border-green-600">
                <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">+25%</div>
                <div className="text-sm text-green-700 dark:text-green-300">설비 가동률</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}