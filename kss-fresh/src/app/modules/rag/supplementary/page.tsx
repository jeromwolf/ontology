'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  CheckCircle2, Award, BookOpen, Shield, DollarSign, AlertTriangle, 
  Clock, ArrowLeft, ChevronRight, TrendingUp, BarChart3, Lock
} from 'lucide-react'

export default function SupplementaryPage() {
  const [completedModules, setCompletedModules] = useState<string[]>([])
  const [completedChecklist, setCompletedChecklist] = useState<string[]>([])

  const modules = [
    {
      id: 'evaluation',
      title: 'RAG 평가 및 품질 관리',
      description: 'RAGAS 프레임워크와 A/B 테스팅으로 RAG 성능을 정량적으로 측정',
      icon: <BarChart3 className="text-blue-600" size={24} />,
      color: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-700',
      duration: '2시간',
      topics: [
        'RAGAS 프레임워크 마스터',
        'A/B 테스팅 프레임워크',
        '성능 메트릭 설계',
        '실시간 평가 대시보드'
      ],
      keyFeatures: [
        'Faithfulness, Relevancy 등 7가지 핵심 메트릭',
        '통계적 유의성 검증 (T-test, Cohen\'s d)',
        'Streamlit 기반 실시간 모니터링',
        '약점 진단 및 개선 제안 자동화'
      ]
    },
    {
      id: 'security',
      title: '보안 및 프라이버시',
      description: '민감 정보 처리와 프롬프트 인젝션 방어로 안전한 RAG 시스템 구축',
      icon: <Shield className="text-red-600" size={24} />,
      color: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-700',
      duration: '2시간',
      topics: [
        '민감 정보(PII) 마스킹',
        '프롬프트 인젝션 방어',
        '접근 권한 관리',
        '감사 로그 시스템'
      ],
      keyFeatures: [
        '이메일, 전화번호, 주민번호 자동 탐지',
        '13가지 인젝션 패턴 실시간 차단',
        'RBAC 기반 문서 접근 제어',
        'GDPR, HIPAA 컴플라이언스 지원'
      ]
    },
    {
      id: 'optimization',
      title: '비용 최적화 전략',
      description: '스마트 캐싱과 모델 선택으로 운영 비용을 최대 80% 절감',
      icon: <DollarSign className="text-green-600" size={24} />,
      color: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-700',
      duration: '2시간',
      topics: [
        '스마트 캐싱 시스템',
        '비용 기반 모델 선택',
        '배치 처리 최적화',
        'TCO 분석 프레임워크'
      ],
      keyFeatures: [
        '3계층 캐싱 (메모리, Redis, 디스크)',
        '쿼리 복잡도별 자동 모델 선택',
        'API 호출 최소화 배치 처리',
        '온프레미스 vs 클라우드 TCO 비교'
      ]
    },
    {
      id: 'resilience',
      title: '실패 처리 및 복구',
      description: '4단계 폴백과 서킷브레이커로 99.9% 가동률 달성',
      icon: <AlertTriangle className="text-orange-600" size={24} />,
      color: 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-700',
      duration: '2시간',
      topics: [
        'Graceful Degradation',
        '다단계 폴백 체인',
        '서킷 브레이커 패턴',
        '종합 모니터링 시스템'
      ],
      keyFeatures: [
        '4단계 폴백: Primary → Cache → Simplified → Static',
        '실패 임계값 자동 감지 (5회 실패시 차단)',
        'Prometheus + Grafana 통합 모니터링',
        '자동 알림 및 복구 프로세스'
      ]
    }
  ]

  const checklist = [
    'RAGAS 평가 시스템을 구현하고 모든 메트릭이 0.8 이상 달성',
    '민감정보 마스킹 시스템 구축 및 5개 이상 PII 패턴 차단',
    '프롬프트 인젝션 방어 시스템으로 위험 점수 3점 이상 차단',
    '3계층 캐싱으로 API 호출 50% 이상 절감',
    '비용 모니터링 대시보드 구축 및 월 $1000 이하 운영',
    '4단계 폴백 시스템으로 장애 상황에서도 응답 제공',
    '서킷 브레이커로 장애 전파 차단 및 자동 복구',
    'Prometheus 메트릭 수집 및 Grafana 대시보드 구축',
    'A/B 테스팅으로 새 기능의 통계적 유의성 검증',
    'Production Readiness Checklist 7개 항목 모두 완료'
  ]

  const toggleModule = (moduleId: string) => {
    setCompletedModules(prev => 
      prev.includes(moduleId) 
        ? prev.filter(id => id !== moduleId)
        : [...prev, moduleId]
    )
  }

  const toggleChecklistItem = (item: string) => {
    setCompletedChecklist(prev => 
      prev.includes(item) 
        ? prev.filter(i => i !== item)
        : [...prev, item]
    )
  }

  const getProgress = () => {
    return (completedModules.length / modules.length) * 100
  }

  const getChecklistProgress = () => {
    return (completedChecklist.length / checklist.length) * 100
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-500 to-orange-600 rounded-2xl p-8 text-white">
        <Link
          href="/modules/rag"
          className="inline-flex items-center gap-2 text-amber-100 hover:text-white mb-6 transition-colors"
        >
          <ArrowLeft size={20} />
          RAG 모듈로 돌아가기
        </Link>
        
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
            <Award size={32} />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Step 4: 보충 과정</h1>
            <p className="text-amber-100 text-lg">실무 필수 요소</p>
          </div>
        </div>
        
        <p className="text-amber-100 mb-6">
          RAG 시스템을 실제 프로덕션 환경에서 운영하기 위해 필요한 모든 실무 지식을 학습합니다. 
          평가, 보안, 비용, 모니터링 등 간과하기 쉽지만 중요한 요소들을 마스터하세요.
        </p>

        {/* Progress Overview */}
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-amber-100">모듈 진행률</span>
              <span className="font-bold">{Math.round(getProgress())}%</span>
            </div>
            <div className="w-full bg-white/20 rounded-full h-3">
              <div 
                className="bg-white h-3 rounded-full transition-all duration-500"
                style={{ width: `${getProgress()}%` }}
              />
            </div>
          </div>
          
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-amber-100">체크리스트 진행률</span>
              <span className="font-bold">{Math.round(getChecklistProgress())}%</span>
            </div>
            <div className="w-full bg-white/20 rounded-full h-3">
              <div 
                className="bg-white h-3 rounded-full transition-all duration-500"
                style={{ width: `${getChecklistProgress()}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Prerequisites */}
      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-2xl p-6 border border-blue-200 dark:border-blue-700">
        <div className="flex items-center gap-3 mb-4">
          <Lock className="text-blue-600" size={20} />
          <h3 className="font-bold text-blue-800 dark:text-blue-200">선수 과목</h3>
        </div>
        <p className="text-blue-700 dark:text-blue-300 mb-4">
          보충 과정을 시작하기 전에 다음 과정들을 완료해야 합니다:
        </p>
        <div className="flex flex-wrap gap-3">
          <Link href="/modules/rag/beginner" className="px-3 py-1 bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200 rounded-full text-sm font-medium border border-green-200 dark:border-green-700 hover:bg-green-200 dark:hover:bg-green-900/30 transition-colors">
            ✓ Step 1: 초급
          </Link>
          <Link href="/modules/rag/intermediate" className="px-3 py-1 bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-200 rounded-full text-sm font-medium border border-blue-200 dark:border-blue-700 hover:bg-blue-200 dark:hover:bg-blue-900/30 transition-colors">
            ✓ Step 2: 중급
          </Link>
          <Link href="/modules/rag/advanced" className="px-3 py-1 bg-purple-100 dark:bg-purple-900/20 text-purple-800 dark:text-purple-200 rounded-full text-sm font-medium border border-purple-200 dark:border-purple-700 hover:bg-purple-200 dark:hover:bg-purple-900/30 transition-colors">
            ✓ Step 3: 고급
          </Link>
        </div>
      </div>

      {/* Course Overview */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">과정 개요</h2>
        
        <div className="grid md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto bg-amber-100 dark:bg-amber-900/20 rounded-xl flex items-center justify-center mb-4">
              <Clock className="text-amber-600" size={24} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">학습 시간</h3>
            <p className="text-gray-600 dark:text-gray-400">총 8시간</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 mx-auto bg-red-100 dark:bg-red-900/20 rounded-xl flex items-center justify-center mb-4">
              <Shield className="text-red-600" size={24} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">핵심 영역</h3>
            <p className="text-gray-600 dark:text-gray-400">보안 & 품질</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 mx-auto bg-green-100 dark:bg-green-900/20 rounded-xl flex items-center justify-center mb-4">
              <TrendingUp className="text-green-600" size={24} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">목표</h3>
            <p className="text-gray-600 dark:text-gray-400">프로덕션 준비</p>
          </div>
          
          <div className="text-center">
            <div className="w-16 h-16 mx-auto bg-purple-100 dark:bg-purple-900/20 rounded-xl flex items-center justify-center mb-4">
              <Award className="text-purple-600" size={24} />
            </div>
            <h3 className="font-bold text-gray-900 dark:text-white mb-2">완료 후</h3>
            <p className="text-gray-600 dark:text-gray-400">RAG 전문가</p>
          </div>
        </div>
      </div>

      {/* Detailed Modules */}
      <div className="space-y-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">상세 모듈</h2>
        
        {modules.map((module, index) => {
          const isCompleted = completedModules.includes(module.id)
          
          return (
            <div key={module.id} className={`rounded-2xl p-8 shadow-sm border transition-all duration-200 ${module.color} ${isCompleted ? 'ring-2 ring-green-500' : ''}`}>
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                    isCompleted 
                      ? 'bg-green-500 text-white' 
                      : 'bg-white dark:bg-gray-800 shadow-sm'
                  }`}>
                    {isCompleted ? <CheckCircle2 size={24} /> : module.icon}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-1">
                      Module {index + 1}: {module.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400">{module.description}</p>
                  </div>
                </div>
                
                <div className="text-right">
                  <span className="inline-block px-3 py-1 bg-white dark:bg-gray-800 rounded-full text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                    {module.duration}
                  </span>
                  <button
                    onClick={() => toggleModule(module.id)}
                    className={`block px-4 py-2 rounded-lg font-medium transition-all ${
                      isCompleted
                        ? 'bg-green-500 text-white'
                        : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 border border-gray-200 dark:border-gray-600'
                    }`}
                  >
                    {isCompleted ? '완료됨' : '완료 표시'}
                  </button>
                </div>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">학습 내용</h4>
                  <ul className="space-y-2">
                    {module.topics.map((topic, i) => (
                      <li key={i} className="text-sm text-gray-600 dark:text-gray-400 flex items-start gap-2">
                        <CheckCircle2 size={14} className="text-green-500 mt-0.5 flex-shrink-0" />
                        <span>{topic}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">핵심 기능</h4>
                  <ul className="space-y-2">
                    {module.keyFeatures.map((feature, i) => (
                      <li key={i} className="text-sm text-gray-600 dark:text-gray-400 flex items-start gap-2">
                        <span className="w-2 h-2 bg-amber-500 rounded-full mt-2 flex-shrink-0"></span>
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Production Readiness Checklist */}
      <div className="bg-amber-50 dark:bg-amber-900/20 rounded-2xl p-8">
        <h3 className="text-xl font-bold text-amber-800 dark:text-amber-200 mb-6 flex items-center gap-2">
          <CheckCircle2 size={24} />
          프로덕션 준비 체크리스트
        </h3>
        
        <p className="text-amber-700 dark:text-amber-300 mb-6">
          아래 모든 항목을 완료하면 실제 프로덕션 환경에서 RAG 시스템을 안전하고 효율적으로 운영할 수 있습니다.
        </p>
        
        <div className="grid md:grid-cols-2 gap-4">
          {checklist.map((item, i) => {
            const isChecked = completedChecklist.includes(item)
            
            return (
              <label key={i} className="flex items-start gap-3 cursor-pointer p-3 rounded-lg hover:bg-amber-100 dark:hover:bg-amber-900/30 transition-colors">
                <input 
                  type="checkbox" 
                  checked={isChecked}
                  onChange={() => toggleChecklistItem(item)}
                  className="mt-0.5 text-amber-500 rounded"
                />
                <span className={`text-sm ${isChecked ? 'text-amber-800 dark:text-amber-200 line-through' : 'text-gray-700 dark:text-gray-300'}`}>
                  {item}
                </span>
              </label>
            )
          })}
        </div>
        
        {getChecklistProgress() === 100 && (
          <div className="mt-6 p-4 bg-gradient-to-r from-green-500 to-emerald-600 rounded-lg text-white">
            <p className="font-bold mb-2">🎉 축하합니다! RAG 전문가가 되셨습니다!</p>
            <p className="text-green-100 mb-4">
              프로덕션 레벨의 RAG 시스템을 구축하고 운영할 수 있는 모든 지식을 갖추셨습니다. 
              이제 실제 프로젝트에 적용해보세요!
            </p>
            <div className="flex flex-wrap gap-3">
              <button className="px-4 py-2 bg-white text-green-600 rounded-lg font-medium hover:bg-green-50 transition-colors">
                인증서 받기
              </button>
              <button className="px-4 py-2 bg-green-400 text-white rounded-lg font-medium hover:bg-green-300 transition-colors">
                포트폴리오 프로젝트 시작
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/advanced"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            고급 과정으로
          </Link>
          
          <Link
            href="/modules/rag"
            className="inline-flex items-center gap-2 bg-amber-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-amber-600 transition-colors"
          >
            RAG 메인으로
            <ChevronRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}