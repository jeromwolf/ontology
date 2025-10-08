'use client';

import { useState } from 'react';
import {
  Brain, Database, TrendingUp, GitBranch, Target,
  ArrowRight, CheckCircle, AlertCircle, Info,
  BarChart3, LineChart, PieChart, Activity,
  Users, Briefcase, GraduationCap, Code2,
  ChevronRight, Play, FileText, Lightbulb
} from 'lucide-react';
import References from '@/components/common/References';

interface ChapterProps {
  onComplete?: () => void
}

export default function Chapter1({ onComplete }: ChapterProps) {
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedRole, setSelectedRole] = useState('scientist')

  return (
    <div className="space-y-8">
      {/* 챕터 헤더 */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">데이터 사이언스 개요</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          데이터 사이언티스트의 역할과 워크플로우를 이해하고 시작하기
        </p>
      </div>

      {/* 학습 목표 */}
      <div className="bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Target className="text-emerald-600" />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">데이터 사이언스의 정의와 범위</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">현대 기업에서의 역할과 가치</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">데이터 사이언스 프로세스</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">문제 정의부터 배포까지</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">필수 스킬과 도구</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">통계, 프로그래밍, 머신러닝</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">커리어 패스 이해</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">성장 경로와 기회</p>
            </div>
          </div>
        </div>
      </div>

      {/* 1. 데이터 사이언스란? */}
      <section>
        <h2 className="text-3xl font-bold mb-6">1. 데이터 사이언스란 무엇인가?</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Brain className="text-emerald-500" />
            정의와 핵심 개념
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <strong>데이터 사이언스(Data Science)</strong>는 구조화된 데이터와 비구조화된 데이터에서 
            지식과 인사이트를 추출하는 다학제적 분야입니다. 통계학, 수학, 컴퓨터 과학, 
            도메인 지식을 결합하여 복잡한 비즈니스 문제를 해결합니다.
          </p>
          
          <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg">
            <p className="text-sm font-medium mb-2">Conway의 데이터 사이언스 벤다이어그램:</p>
            <div className="grid grid-cols-3 gap-4 mt-4">
              <div className="text-center">
                <div className="w-24 h-24 mx-auto bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                  <span className="text-blue-600 font-semibold">수학/통계</span>
                </div>
              </div>
              <div className="text-center">
                <div className="w-24 h-24 mx-auto bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center">
                  <span className="text-green-600 font-semibold">프로그래밍</span>
                </div>
              </div>
              <div className="text-center">
                <div className="w-24 h-24 mx-auto bg-purple-100 dark:bg-purple-900/30 rounded-full flex items-center justify-center">
                  <span className="text-purple-600 font-semibold">도메인 지식</span>
                </div>
              </div>
            </div>
            <p className="text-center mt-4 text-sm text-gray-600 dark:text-gray-400">
              세 영역의 교집합이 데이터 사이언스
            </p>
          </div>
        </div>

        {/* 데이터 사이언스의 가치 */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
            <h4 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <TrendingUp className="text-blue-600" />
              비즈니스 가치
            </h4>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>데이터 기반 의사결정 지원</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>고객 행동 예측 및 개인화</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>운영 효율성 최적화</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>새로운 수익 모델 창출</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>리스크 관리 및 사기 탐지</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
            <h4 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Lightbulb className="text-purple-600" />
              실제 적용 사례
            </h4>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-purple-600 mt-1">•</span>
                <span>넷플릭스: 콘텐츠 추천 시스템</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-600 mt-1">•</span>
                <span>우버: 동적 가격 책정</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-600 mt-1">•</span>
                <span>아마존: 수요 예측 및 재고 관리</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-600 mt-1">•</span>
                <span>구글: 검색 알고리즘 최적화</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-purple-600 mt-1">•</span>
                <span>금융: 신용 평가 및 사기 탐지</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 2. 데이터 사이언스 프로세스 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">2. 데이터 사이언스 프로세스</h2>
        
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-6 rounded-xl mb-6">
          <h3 className="text-xl font-semibold mb-4">CRISP-DM (Cross-Industry Standard Process for Data Mining)</h3>
          
          <div className="space-y-4">
            {[
              {
                step: "1. 비즈니스 이해 (Business Understanding)",
                tasks: ["문제 정의", "목표 설정", "성공 지표 정의", "제약사항 파악"],
                icon: <Briefcase className="text-indigo-600" />
              },
              {
                step: "2. 데이터 이해 (Data Understanding)",
                tasks: ["데이터 수집", "탐색적 분석", "데이터 품질 평가", "가설 수립"],
                icon: <Database className="text-purple-600" />
              },
              {
                step: "3. 데이터 준비 (Data Preparation)",
                tasks: ["데이터 정제", "특성 선택", "변환 및 인코딩", "데이터 통합"],
                icon: <GitBranch className="text-blue-600" />
              },
              {
                step: "4. 모델링 (Modeling)",
                tasks: ["알고리즘 선택", "모델 학습", "하이퍼파라미터 튜닝", "교차 검증"],
                icon: <Brain className="text-green-600" />
              },
              {
                step: "5. 평가 (Evaluation)",
                tasks: ["성능 측정", "비즈니스 목표 검증", "모델 해석", "개선점 도출"],
                icon: <BarChart3 className="text-orange-600" />
              },
              {
                step: "6. 배포 (Deployment)",
                tasks: ["모델 배포", "모니터링 설정", "문서화", "유지보수 계획"],
                icon: <Activity className="text-red-600" />
              }
            ].map((phase, idx) => (
              <div key={idx} className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="flex items-start gap-3">
                  <div className="mt-1">{phase.icon}</div>
                  <div className="flex-1">
                    <h4 className="font-semibold text-lg mb-2">{phase.step}</h4>
                    <div className="flex flex-wrap gap-2">
                      {phase.tasks.map((task, tidx) => (
                        <span key={tidx} className="text-sm bg-gray-100 dark:bg-gray-700 px-3 py-1 rounded-full">
                          {task}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 반복적 프로세스 강조 */}
        <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg mb-6">
          <div className="flex items-start gap-3">
            <AlertCircle className="text-yellow-600 mt-1" />
            <div>
              <p className="font-semibold text-yellow-800 dark:text-yellow-400">중요: 반복적 프로세스</p>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                데이터 사이언스는 선형적이지 않습니다. 각 단계에서 이전 단계로 돌아가 
                개선하는 것이 일반적이며, 이러한 반복을 통해 더 나은 결과를 얻을 수 있습니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 3. 데이터 사이언티스트 vs 관련 직군 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">3. 데이터 관련 직군 비교</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <div className="flex gap-2 mb-4">
            {['scientist', 'analyst', 'engineer', 'mlengineer'].map((role) => (
              <button
                key={role}
                onClick={() => setSelectedRole(role)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedRole === role
                    ? 'bg-emerald-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                {role === 'scientist' && '데이터 사이언티스트'}
                {role === 'analyst' && '데이터 분석가'}
                {role === 'engineer' && '데이터 엔지니어'}
                {role === 'mlengineer' && 'ML 엔지니어'}
              </button>
            ))}
          </div>

          {selectedRole === 'scientist' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-emerald-600 dark:text-emerald-400">데이터 사이언티스트</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">주요 업무</h4>
                  <ul className="space-y-1 text-sm">
                    <li>• 예측 모델 개발</li>
                    <li>• A/B 테스트 설계 및 분석</li>
                    <li>• 비즈니스 문제 해결</li>
                    <li>• 인사이트 도출 및 제안</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">필수 스킬</h4>
                  <ul className="space-y-1 text-sm">
                    <li>• Python/R 프로그래밍</li>
                    <li>• 통계학 및 머신러닝</li>
                    <li>• SQL 및 데이터 분석</li>
                    <li>• 비즈니스 커뮤니케이션</li>
                  </ul>
                </div>
              </div>
              <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg">
                <p className="text-sm"><strong>연봉 범위:</strong> 6,000 - 12,000만원 (경력 3-5년 기준)</p>
                <p className="text-sm mt-1"><strong>성장 경로:</strong> Senior DS → Lead DS → Chief Data Officer</p>
              </div>
            </div>
          )}

          {selectedRole === 'analyst' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-blue-600 dark:text-blue-400">데이터 분석가</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">주요 업무</h4>
                  <ul className="space-y-1 text-sm">
                    <li>• 비즈니스 리포트 작성</li>
                    <li>• 대시보드 개발</li>
                    <li>• KPI 모니터링</li>
                    <li>• 데이터 시각화</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">필수 스킬</h4>
                  <ul className="space-y-1 text-sm">
                    <li>• SQL 마스터</li>
                    <li>• Excel/Spreadsheet</li>
                    <li>• Tableau/PowerBI</li>
                    <li>• 기초 통계</li>
                  </ul>
                </div>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                <p className="text-sm"><strong>연봉 범위:</strong> 4,000 - 8,000만원 (경력 3-5년 기준)</p>
                <p className="text-sm mt-1"><strong>성장 경로:</strong> Senior Analyst → Analytics Manager → Head of Analytics</p>
              </div>
            </div>
          )}

          {selectedRole === 'engineer' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-purple-600 dark:text-purple-400">데이터 엔지니어</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">주요 업무</h4>
                  <ul className="space-y-1 text-sm">
                    <li>• 데이터 파이프라인 구축</li>
                    <li>• ETL/ELT 프로세스 개발</li>
                    <li>• 데이터 웨어하우스 관리</li>
                    <li>• 인프라 최적화</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">필수 스킬</h4>
                  <ul className="space-y-1 text-sm">
                    <li>• Python/Scala</li>
                    <li>• Apache Spark/Airflow</li>
                    <li>• 클라우드 플랫폼</li>
                    <li>• 분산 시스템</li>
                  </ul>
                </div>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                <p className="text-sm"><strong>연봉 범위:</strong> 5,000 - 10,000만원 (경력 3-5년 기준)</p>
                <p className="text-sm mt-1"><strong>성장 경로:</strong> Senior DE → Lead DE → Data Architect</p>
              </div>
            </div>
          )}

          {selectedRole === 'mlengineer' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400">ML 엔지니어</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">주요 업무</h4>
                  <ul className="space-y-1 text-sm">
                    <li>• 모델 프로덕션 배포</li>
                    <li>• ML 파이프라인 구축</li>
                    <li>• 모델 성능 모니터링</li>
                    <li>• A/B 테스트 인프라</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">필수 스킬</h4>
                  <ul className="space-y-1 text-sm">
                    <li>• Python/Java</li>
                    <li>• Docker/Kubernetes</li>
                    <li>• MLOps 도구</li>
                    <li>• 클라우드 ML 서비스</li>
                  </ul>
                </div>
              </div>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg">
                <p className="text-sm"><strong>연봉 범위:</strong> 7,000 - 13,000만원 (경력 3-5년 기준)</p>
                <p className="text-sm mt-1"><strong>성장 경로:</strong> Senior MLE → Staff MLE → ML Platform Lead</p>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 4. 필수 스킬과 도구 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">4. 데이터 사이언티스트 필수 스킬</h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
          <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Code2 className="text-red-500" />
              프로그래밍
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span>Python (NumPy, Pandas)</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span>R (선택적)</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span>SQL (필수)</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-red-400" />
                <span>Git/GitHub</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <BarChart3 className="text-blue-500" />
              통계 & 수학
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span>기술통계 & 추론통계</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span>확률론</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span>선형대수</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-blue-400" />
                <span>최적화 이론</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Brain className="text-green-500" />
              머신러닝
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span>지도학습 알고리즘</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span>비지도학습</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span>딥러닝 기초</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-green-400" />
                <span>모델 평가</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <LineChart className="text-purple-500" />
              시각화
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span>Matplotlib/Seaborn</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span>Plotly</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span>Tableau/PowerBI</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-purple-400" />
                <span>대시보드 설계</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-yellow-50 to-amber-50 dark:from-yellow-900/20 dark:to-amber-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Database className="text-yellow-600" />
              데이터 처리
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-yellow-500" />
                <span>빅데이터 도구</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-yellow-500" />
                <span>클라우드 플랫폼</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-yellow-500" />
                <span>데이터베이스</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-yellow-500" />
                <span>API 활용</span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 p-6 rounded-xl">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Users className="text-teal-500" />
              소프트 스킬
            </h3>
            <ul className="space-y-2">
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span>비즈니스 이해</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span>스토리텔링</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span>프레젠테이션</span>
              </li>
              <li className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-teal-400" />
                <span>팀 협업</span>
              </li>
            </ul>
          </div>
        </div>

        {/* 학습 로드맵 */}
        <div className="bg-gray-900 rounded-xl p-6">
          <h3 className="text-white font-semibold mb-4">추천 학습 순서</h3>
          <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
            <code className="text-sm text-gray-300">{`# 데이터 사이언티스트 학습 로드맵

## 1단계: 기초 (3-6개월)
- Python 프로그래밍 기초
- 통계학 기초 (기술통계, 확률분포)
- SQL 마스터
- 데이터 시각화 (Matplotlib, Seaborn)

## 2단계: 핵심 (6-12개월)
- 머신러닝 알고리즘 이해
- Scikit-learn 활용
- 데이터 전처리 및 특성 공학
- 모델 평가 및 검증

## 3단계: 심화 (12-18개월)
- 딥러닝 기초 (TensorFlow/PyTorch)
- 빅데이터 처리 (Spark)
- A/B 테스트 및 인과추론
- 클라우드 플랫폼 활용

## 4단계: 전문화 (18개월+)
- 도메인 특화 (금융, 헬스케어, 이커머스 등)
- MLOps 및 모델 배포
- 리더십 및 프로젝트 관리
- 최신 기술 트렌드 (LLM, GenAI 등)`}</code>
          </pre>
        </div>
      </section>

      {/* 5. 첫 프로젝트 시작하기 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">5. 첫 번째 데이터 사이언스 프로젝트</h2>
        
        <div className="bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 p-6 rounded-xl mb-6">
          <h3 className="text-xl font-semibold mb-4">프로젝트: 타이타닉 생존자 예측</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Kaggle의 가장 유명한 입문 프로젝트로, 타이타닉 승객 정보를 기반으로 
            생존 여부를 예측하는 이진 분류 문제입니다.
          </p>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">학습 목표:</h4>
            <ul className="space-y-1 text-sm">
              <li>✓ 데이터 탐색 및 시각화</li>
              <li>✓ 결측치 처리 방법</li>
              <li>✓ 특성 공학 (Feature Engineering)</li>
              <li>✓ 여러 알고리즘 비교</li>
              <li>✓ 모델 평가 및 개선</li>
            </ul>
          </div>
        </div>

        <div className="bg-gray-900 rounded-xl p-6">
          <h4 className="text-white font-semibold mb-3">시작 코드</h4>
          <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
            <code className="text-sm text-gray-300">{`import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 데이터 로드
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 데이터 탐색
print(f"훈련 데이터 크기: {train_df.shape}")
print(f"테스트 데이터 크기: {test_df.shape}")
print("\\n데이터 정보:")
print(train_df.info())

# 3. 생존율 확인
survival_rate = train_df['Survived'].value_counts(normalize=True)
print(f"\\n전체 생존율: {survival_rate[1]:.2%}")

# 4. 시각화
plt.figure(figsize=(12, 5))

# 성별에 따른 생존율
plt.subplot(1, 3, 1)
sns.barplot(data=train_df, x='Sex', y='Survived')
plt.title('Gender vs Survival')

# 클래스별 생존율
plt.subplot(1, 3, 2)
sns.barplot(data=train_df, x='Pclass', y='Survived')
plt.title('Class vs Survival')

# 나이 분포
plt.subplot(1, 3, 3)
survived = train_df[train_df['Survived']==1]['Age'].dropna()
not_survived = train_df[train_df['Survived']==0]['Age'].dropna()
plt.hist([survived, not_survived], label=['Survived', 'Not Survived'], 
         bins=20, alpha=0.7)
plt.xlabel('Age')
plt.legend()
plt.title('Age Distribution')

plt.tight_layout()
plt.show()

# 5. 데이터 전처리
def preprocess_data(df):
    # 결측치 처리
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # 새로운 특성 생성
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 범주형 변수 인코딩
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    return df

# 전처리 적용
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# 6. 모델 학습
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone']
X = train_df[features]
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 랜덤 포레스트 모델
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 7. 모델 평가
y_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"\\n검증 정확도: {accuracy:.4f}")

# 특성 중요도
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\n특성 중요도:")
print(feature_importance)`}</code>
          </pre>
        </div>
      </section>

      {/* 실습 프로젝트 */}
      <section className="mt-12">
        <div className="bg-gradient-to-r from-emerald-600 to-green-600 text-white p-8 rounded-xl">
          <h2 className="text-2xl font-bold mb-4">🚀 데이터 사이언스 여정을 시작하세요!</h2>
          <p className="mb-6">
            이제 기초를 배웠으니, 실제 프로젝트를 시작해봅시다. 
            Kaggle에서 'Titanic' 대회에 참가하고, 첫 submission을 만들어보세요.
          </p>
          <div className="flex gap-4">
            <button 
              onClick={onComplete}
              className="bg-white text-emerald-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              챕터 완료하기
            </button>
            <button className="bg-emerald-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-emerald-400 transition-colors">
              Kaggle 시작하기
            </button>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Essential Reading',
            icon: 'paper',
            color: 'border-emerald-500',
            items: [
              {
                title: 'Data Science for Business',
                authors: 'Foster Provost, Tom Fawcett',
                year: '2013',
                description: '비즈니스 관점에서의 데이터 사이언스 - 의사결정자를 위한 필독서',
                link: 'https://www.oreilly.com/library/view/data-science-for/9781449374273/'
              },
              {
                title: 'The Art of Data Science',
                authors: 'Roger D. Peng, Elizabeth Matsui',
                year: '2015',
                description: '데이터 분석의 실제 프로세스 - 무료 온라인 제공',
                link: 'https://bookdown.org/rdpeng/artofdatascience/'
              },
              {
                title: 'Python for Data Analysis',
                authors: 'Wes McKinney',
                year: '2022',
                description: 'Pandas 창시자가 쓴 데이터 분석 바이블 (3rd Edition)',
                link: 'https://wesmckinney.com/book/'
              },
              {
                title: 'Doing Data Science',
                authors: 'Cathy O\'Neil, Rachel Schutt',
                year: '2013',
                description: '컬럼비아 대학 데이터 사이언스 강의 기반',
                link: 'https://www.oreilly.com/library/view/doing-data-science/9781449363871/'
              }
            ]
          },
          {
            title: 'Data Science Process & Methodology',
            icon: 'paper',
            color: 'border-blue-500',
            items: [
              {
                title: 'CRISP-DM Methodology',
                authors: 'Chapman et al.',
                year: '2000',
                description: '업계 표준 데이터 마이닝 프로세스 (Cross-Industry Standard Process)',
                link: 'https://www.datascience-pm.com/crisp-dm-2/'
              },
              {
                title: 'Reproducible Research in Computational Science',
                authors: 'Roger D. Peng',
                year: '2011',
                description: '재현 가능한 연구의 중요성 (Science)',
                link: 'https://www.science.org/doi/10.1126/science.1213847'
              },
              {
                title: 'Data Science: An Introduction',
                authors: 'Jeffrey Stanton',
                year: '2013',
                description: '데이터 사이언스 입문 - 무료 온라인 교재',
                link: 'https://surface.syr.edu/istpub/165/'
              }
            ]
          },
          {
            title: 'Tools & Platforms',
            icon: 'web',
            color: 'border-purple-500',
            items: [
              {
                title: 'Kaggle Learn',
                description: '무료 인터랙티브 데이터 사이언스 강의 플랫폼',
                link: 'https://www.kaggle.com/learn'
              },
              {
                title: 'Jupyter Project',
                description: '데이터 사이언티스트의 필수 노트북 환경',
                link: 'https://jupyter.org/'
              },
              {
                title: 'Anaconda Distribution',
                description: '데이터 사이언스를 위한 파이썬 배포판',
                link: 'https://www.anaconda.com/'
              },
              {
                title: 'Google Colab',
                description: '무료 GPU를 제공하는 클라우드 Jupyter 환경',
                link: 'https://colab.research.google.com/'
              }
            ]
          },
          {
            title: 'Community & Resources',
            icon: 'web',
            color: 'border-green-500',
            items: [
              {
                title: 'Towards Data Science',
                description: 'Medium의 데이터 사이언스 전문 퍼블리케이션',
                link: 'https://towardsdatascience.com/'
              },
              {
                title: 'Data Science Stack Exchange',
                description: '데이터 사이언스 Q&A 커뮤니티',
                link: 'https://datascience.stackexchange.com/'
              },
              {
                title: 'KDnuggets',
                description: '데이터 사이언스 뉴스 및 튜토리얼',
                link: 'https://www.kdnuggets.com/'
              },
              {
                title: 'DataCamp Community',
                description: '데이터 사이언스 학습 커뮤니티',
                link: 'https://www.datacamp.com/community'
              }
            ]
          }
        ]}
      />
    </div>
  )
}