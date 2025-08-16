'use client';

import React from 'react';
import { Book, FileText, Calculator, AlertCircle, CheckCircle, XCircle, TrendingUp, Building } from 'lucide-react';

export default function Chapter36() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">GAAP vs IFRS 회계기준</h1>
      
      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Book className="w-8 h-8 text-blue-500" />
          회계기준 개요
        </h2>
        
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4">US GAAP</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              Generally Accepted Accounting Principles
            </p>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>미국 재무회계기준위원회(FASB) 제정</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>규칙 기반(Rule-based) 접근법</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>미국 상장기업 의무 적용</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-500">•</span>
                <span>상세하고 구체적인 지침 제공</span>
              </li>
            </ul>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="text-xl font-semibold mb-4">IFRS</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              International Financial Reporting Standards
            </p>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <span>국제회계기준위원회(IASB) 제정</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <span>원칙 기반(Principle-based) 접근법</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <span>140개국 이상 채택</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-500">•</span>
                <span>전문가 판단 중시</span>
              </li>
            </ul>
          </div>
        </div>
        
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            왜 차이를 알아야 하는가?
          </h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>• 동일 기업도 회계기준에 따라 재무성과가 다르게 표시됨</li>
            <li>• 글로벌 기업 비교 분석 시 조정 필요</li>
            <li>• M&A 및 투자 의사결정에 직접적 영향</li>
            <li>• 회계 조작 가능성 파악에 도움</li>
          </ul>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Calculator className="w-8 h-8 text-purple-500" />
          주요 차이점 10가지
        </h2>
        
        <div className="space-y-6">
          {/* 1. 재고자산 평가 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">1. 재고자산 평가방법</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2 text-blue-600">US GAAP</h4>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li><CheckCircle className="inline w-4 h-4 text-green-500 mr-1" />FIFO (선입선출)</li>
                  <li><CheckCircle className="inline w-4 h-4 text-green-500 mr-1" />LIFO (후입선출)</li>
                  <li><CheckCircle className="inline w-4 h-4 text-green-500 mr-1" />가중평균법</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2 text-green-600">IFRS</h4>
                <ul className="space-y-1 text-gray-700 dark:text-gray-300">
                  <li><CheckCircle className="inline w-4 h-4 text-green-500 mr-1" />FIFO (선입선출)</li>
                  <li><XCircle className="inline w-4 h-4 text-red-500 mr-1" />LIFO 금지</li>
                  <li><CheckCircle className="inline w-4 h-4 text-green-500 mr-1" />가중평균법</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-sm">
                <strong>투자 영향:</strong> LIFO 사용 기업은 인플레이션 시기에 이익이 낮게 보고됨
              </p>
            </div>
          </div>

          {/* 2. 개발비 처리 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">2. 연구개발비 처리</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2 text-blue-600">US GAAP</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  모든 R&D 비용은 발생 시점에 즉시 비용 처리
                </p>
              </div>
              <div>
                <h4 className="font-medium mb-2 text-green-600">IFRS</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  연구비: 비용 처리<br/>
                  개발비: 조건 충족 시 자산화 가능
                </p>
              </div>
            </div>
            <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-sm">
                <strong>투자 영향:</strong> IFRS 적용 기업이 단기 수익성이 높게 나타날 수 있음
              </p>
            </div>
          </div>

          {/* 3. 자산 재평가 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">3. 유형자산 재평가</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2 text-blue-600">US GAAP</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  역사적 원가 모델만 허용<br/>
                  (재평가 금지)
                </p>
              </div>
              <div>
                <h4 className="font-medium mb-2 text-green-600">IFRS</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  원가 모델 또는 재평가 모델 선택 가능<br/>
                  (공정가치로 재평가 허용)
                </p>
              </div>
            </div>
            <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-sm">
                <strong>투자 영향:</strong> IFRS 기업의 자산가치가 시장가치를 반영할 수 있음
              </p>
            </div>
          </div>

          {/* 4. 손상차손 환입 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">4. 손상차손 환입</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2 text-blue-600">US GAAP</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  손상차손 인식 후 환입 금지<br/>
                  (영업권 및 기타 자산 모두)
                </p>
              </div>
              <div>
                <h4 className="font-medium mb-2 text-green-600">IFRS</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  영업권 제외한 자산의 손상차손 환입 가능<br/>
                  (조건 충족 시)
                </p>
              </div>
            </div>
            <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-sm">
                <strong>투자 영향:</strong> 경기 회복 시 IFRS 기업의 이익 변동성이 더 클 수 있음
              </p>
            </div>
          </div>

          {/* 5. 리스 회계 */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">5. 리스 회계 (최신 기준)</h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2 text-blue-600">US GAAP (ASC 842)</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  운용리스: 사용권자산과 리스부채 인식<br/>
                  손익계산서: 정액법 비용 인식
                </p>
              </div>
              <div>
                <h4 className="font-medium mb-2 text-green-600">IFRS (IFRS 16)</h4>
                <p className="text-gray-700 dark:text-gray-300">
                  모든 리스: 사용권자산과 리스부채 인식<br/>
                  손익계산서: 이자비용과 감가상각비 분리
                </p>
              </div>
            </div>
            <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-700 rounded">
              <p className="text-sm">
                <strong>투자 영향:</strong> 부채비율과 EBITDA 계산에 차이 발생
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <FileText className="w-8 h-8 text-green-500" />
          실제 기업 사례
        </h2>
        
        <div className="bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">Daimler AG 사례 (2007년)</h3>
          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <p className="font-medium mb-2">회계기준 전환 영향</p>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• US GAAP 순이익: €3,985 million</li>
                <li>• IFRS 순이익: €4,048 million</li>
                <li>• 차이: €63 million (1.6% 증가)</li>
              </ul>
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded p-4">
              <p className="font-medium mb-2">주요 조정 항목</p>
              <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                <li>• 개발비 자산화: +€167 million</li>
                <li>• 연금 회계: -€89 million</li>
                <li>• 파생상품 평가: -€15 million</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <TrendingUp className="w-8 h-8 text-orange-500" />
          투자 분석 시 조정 방법
        </h2>
        
        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4">재무비율 분석 조정</h3>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left p-2">재무비율</th>
                    <th className="text-left p-2">조정 필요 항목</th>
                    <th className="text-left p-2">영향도</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-100 dark:border-gray-800">
                    <td className="p-2 font-medium">P/E Ratio</td>
                    <td className="p-2">R&D 자산화, 손상차손 환입</td>
                    <td className="p-2">
                      <span className="text-orange-600">중간</span>
                    </td>
                  </tr>
                  <tr className="border-b border-gray-100 dark:border-gray-800">
                    <td className="p-2 font-medium">부채비율</td>
                    <td className="p-2">리스부채, 연금부채</td>
                    <td className="p-2">
                      <span className="text-red-600">높음</span>
                    </td>
                  </tr>
                  <tr className="border-b border-gray-100 dark:border-gray-800">
                    <td className="p-2 font-medium">ROA</td>
                    <td className="p-2">자산 재평가, 개발비 자산화</td>
                    <td className="p-2">
                      <span className="text-orange-600">중간</span>
                    </td>
                  </tr>
                  <tr className="border-b border-gray-100 dark:border-gray-800">
                    <td className="p-2 font-medium">영업이익률</td>
                    <td className="p-2">재고자산 평가, R&D 처리</td>
                    <td className="p-2">
                      <span className="text-yellow-600">낮음</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">실무 조정 프로세스</h3>
            <ol className="list-decimal list-inside space-y-3 text-gray-700 dark:text-gray-300">
              <li>
                <strong>재무제표 주석 확인</strong>
                <p className="ml-6 text-sm mt-1">회계정책 변경사항, 중요한 회계추정 확인</p>
              </li>
              <li>
                <strong>조정표(Reconciliation) 검토</strong>
                <p className="ml-6 text-sm mt-1">듀얼 리스팅 기업의 GAAP/IFRS 조정표 분석</p>
              </li>
              <li>
                <strong>산업별 특성 고려</strong>
                <p className="ml-6 text-sm mt-1">제약(R&D), 부동산(재평가), 리테일(리스) 등</p>
              </li>
              <li>
                <strong>시계열 일관성 확보</strong>
                <p className="ml-6 text-sm mt-1">회계기준 변경 시점 전후 비교가능성 조정</p>
              </li>
            </ol>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
          <Building className="w-8 h-8 text-purple-500" />
          주요 시장별 회계기준
        </h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-2">🇺🇸 미국</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">US GAAP 의무 적용</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-2">🇪🇺 유럽연합</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">IFRS 의무 적용</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-2">🇯🇵 일본</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">J-GAAP (IFRS 선택 가능)</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-2">🇨🇳 중국</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">CAS (IFRS 유사)</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-2">🇰🇷 한국</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">K-IFRS (IFRS 기반)</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold mb-2">🇬🇧 영국</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300">UK-adopted IFRS</p>
          </div>
        </div>
      </section>

      <div className="bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg p-8 text-white">
        <h2 className="text-2xl font-bold mb-4">해외기업 분석 핵심 체크리스트</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-3">📋 필수 확인사항</h3>
            <ul className="space-y-2">
              <li>✓ 적용 회계기준 확인 (10-K, 20-F)</li>
              <li>✓ 중요 회계정책 변경사항</li>
              <li>✓ 비GAAP 지표 조정내역</li>
              <li>✓ 감사의견 및 핵심감사사항</li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-3">⚠️ 주의 항목</h3>
            <ul className="space-y-2">
              <li>✓ 일회성 손익 항목</li>
              <li>✓ 우발부채 및 약정사항</li>
              <li>✓ 관계사 거래</li>
              <li>✓ 보고부문 변경</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}