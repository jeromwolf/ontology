'use client'

import { useState } from 'react'
import { Search, AlertTriangle, CheckCircle, XCircle, Clock, FileText } from 'lucide-react'

interface VulnerabilityCheck {
  id: string
  category: string
  name: string
  description: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  status: 'checking' | 'pass' | 'fail' | 'warning'
  details?: string
  recommendation?: string
}

interface AuditReport {
  overallScore: number
  totalChecks: number
  passed: number
  failed: number
  warnings: number
  riskLevel: 'low' | 'medium' | 'high' | 'critical'
  categories: Record<string, { passed: number; failed: number; warnings: number }>
}

export default function SecurityAuditTool() {
  const [isAuditing, setIsAuditing] = useState(false)
  const [selectedSystem, setSelectedSystem] = useState<'image-classifier' | 'nlp-model' | 'recommendation'>('image-classifier')
  const [auditProgress, setAuditProgress] = useState(0)
  const [vulnerabilities, setVulnerabilities] = useState<VulnerabilityCheck[]>([])
  const [auditReport, setAuditReport] = useState<AuditReport | null>(null)

  const systemProfiles = {
    'image-classifier': {
      name: '이미지 분류 시스템',
      description: 'CNN 기반 이미지 분류 모델',
      checks: [
        {
          id: 'adv-robustness',
          category: '견고성',
          name: '적대적 공격 내성',
          description: '적대적 예제에 대한 모델의 견고성 평가',
          severity: 'high' as const
        },
        {
          id: 'input-validation',
          category: '입력 검증',
          name: '입력 데이터 검증',
          description: '악의적 입력 데이터 필터링 메커니즘',
          severity: 'medium' as const
        },
        {
          id: 'model-encryption',
          category: '모델 보안',
          name: '모델 암호화',
          description: '저장된 모델 파일의 암호화 상태',
          severity: 'medium' as const
        },
        {
          id: 'api-security',
          category: 'API 보안',
          name: 'API 인증/인가',
          description: 'API 접근 제어 및 인증 메커니즘',
          severity: 'high' as const
        },
        {
          id: 'data-poisoning',
          category: '데이터 무결성',
          name: '데이터 중독 방어',
          description: '훈련 데이터 무결성 검증',
          severity: 'critical' as const
        },
        {
          id: 'model-extraction',
          category: '모델 보호',
          name: '모델 추출 방어',
          description: '모델 가중치 유출 방지 메커니즘',
          severity: 'high' as const
        }
      ]
    },
    'nlp-model': {
      name: 'NLP 감성 분석 시스템',
      description: 'BERT 기반 텍스트 분석 모델',
      checks: [
        {
          id: 'prompt-injection',
          category: '입력 보안',
          name: '프롬프트 인젝션 방어',
          description: '악의적 프롬프트 주입 공격 방어',
          severity: 'critical' as const
        },
        {
          id: 'text-adversarial',
          category: '견고성',
          name: '텍스트 적대적 공격',
          description: '단어 치환 공격에 대한 내성',
          severity: 'high' as const
        },
        {
          id: 'bias-detection',
          category: '공정성',
          name: '편향성 검사',
          description: '모델 출력의 편향성 평가',
          severity: 'medium' as const
        },
        {
          id: 'toxic-content',
          category: '콘텐츠 안전',
          name: '독성 콘텐츠 필터링',
          description: '유해 콘텐츠 생성 방지',
          severity: 'high' as const
        }
      ]
    },
    'recommendation': {
      name: '추천 시스템',
      description: '협업 필터링 기반 추천 엔진',
      checks: [
        {
          id: 'fake-profiles',
          category: '데이터 무결성',
          name: '가짜 프로필 탐지',
          description: '시빌 공격 및 가짜 사용자 탐지',
          severity: 'high' as const
        },
        {
          id: 'privacy-inference',
          category: '프라이버시',
          name: '사용자 프라이버시',
          description: '사용자 선호도 추론 공격 방어',
          severity: 'critical' as const
        },
        {
          id: 'recommendation-bias',
          category: '공정성',
          name: '추천 편향성',
          description: '필터 버블 및 편향된 추천 방지',
          severity: 'medium' as const
        }
      ]
    }
  }

  const runSecurityAudit = async () => {
    setIsAuditing(true)
    setAuditProgress(0)
    setVulnerabilities([])
    setAuditReport(null)

    const checks = systemProfiles[selectedSystem].checks
    const results: VulnerabilityCheck[] = []

    for (let i = 0; i < checks.length; i++) {
      const check = checks[i]
      
      setAuditProgress(((i + 1) / checks.length) * 100)
      
      const checkingResult: VulnerabilityCheck = {
        ...check,
        status: 'checking'
      }
      setVulnerabilities([...results, checkingResult])
      
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      const severityWeights = { critical: 0.7, high: 0.5, medium: 0.3, low: 0.1 }
      const failureProbability = severityWeights[check.severity]
      const random = Math.random()
      
      let status: VulnerabilityCheck['status']
      let details: string
      let recommendation: string
      
      if (random < failureProbability) {
        status = 'fail'
        details = '보안 취약점이 발견되었습니다.'
        recommendation = '즉시 보안 조치를 취하세요.'
      } else if (random < failureProbability + 0.2) {
        status = 'warning'
        details = '일부 보안 위험이 감지되었습니다.'
        recommendation = '추가 검토가 필요합니다.'
      } else {
        status = 'pass'
        details = '검사를 통과했습니다.'
        recommendation = '현재 보안 수준을 유지하세요.'
      }
      
      const result: VulnerabilityCheck = {
        ...check,
        status,
        details,
        recommendation
      }
      
      results.push(result)
      setVulnerabilities([...results])
    }

    // 감사 보고서 생성
    const passed = results.filter(r => r.status === 'pass').length
    const failed = results.filter(r => r.status === 'fail').length
    const warnings = results.filter(r => r.status === 'warning').length
    
    const overallScore = Math.round((passed / results.length) * 100)
    
    let riskLevel: AuditReport['riskLevel']
    if (failed === 0 && warnings <= 1) riskLevel = 'low'
    else if (failed <= 2 && warnings <= 3) riskLevel = 'medium'
    else if (failed <= 4) riskLevel = 'high'
    else riskLevel = 'critical'

    const categories: Record<string, { passed: number; failed: number; warnings: number }> = {}
    results.forEach(result => {
      if (!categories[result.category]) {
        categories[result.category] = { passed: 0, failed: 0, warnings: 0 }
      }
      if (result.status === 'pass') categories[result.category].passed++
      else if (result.status === 'fail') categories[result.category].failed++
      else if (result.status === 'warning') categories[result.category].warnings++
    })

    const report: AuditReport = {
      overallScore,
      totalChecks: results.length,
      passed,
      failed,
      warnings,
      riskLevel,
      categories
    }

    setAuditReport(report)
    setIsAuditing(false)
  }

  const resetAudit = () => {
    setIsAuditing(false)
    setAuditProgress(0)
    setVulnerabilities([])
    setAuditReport(null)
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-700 bg-red-100 dark:text-red-300 dark:bg-red-900/30'
      case 'high': return 'text-orange-700 bg-orange-100 dark:text-orange-300 dark:bg-orange-900/30'
      case 'medium': return 'text-yellow-700 bg-yellow-100 dark:text-yellow-300 dark:bg-yellow-900/30'
      case 'low': return 'text-green-700 bg-green-100 dark:text-green-300 dark:bg-green-900/30'
      default: return 'text-gray-700 bg-gray-100 dark:text-gray-300 dark:bg-gray-900/30'
    }
  }

  const getStatusIcon = (status: VulnerabilityCheck['status']) => {
    switch (status) {
      case 'checking': return <Clock className="w-5 h-5 text-blue-600 animate-spin" />
      case 'pass': return <CheckCircle className="w-5 h-5 text-green-600" />
      case 'fail': return <XCircle className="w-5 h-5 text-red-600" />
      case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-600" />
    }
  }

  return (
    <div className="space-y-6">
      {/* 시스템 선택 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">감사 대상 시스템</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(systemProfiles).map(([id, profile]) => (
            <button
              key={id}
              onClick={() => setSelectedSystem(id as any)}
              className={`p-4 text-left rounded-lg border-2 transition-colors ${
                selectedSystem === id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-600 hover:border-blue-300'
              }`}
            >
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                {profile.name}
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                {profile.description}
              </p>
              <div className="text-xs text-blue-600 dark:text-blue-400">
                {profile.checks.length}개 보안 검사 항목
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 감사 실행 */}
      <div className="flex gap-4">
        <button
          onClick={runSecurityAudit}
          disabled={isAuditing}
          className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Search className="w-5 h-5" />
          {isAuditing ? '보안 감사 진행 중...' : '보안 감사 시작'}
        </button>
        
        <button
          onClick={resetAudit}
          className="flex items-center gap-2 px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          초기화
        </button>
      </div>

      {/* 진행 상황 */}
      {isAuditing && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">감사 진행 상황</h3>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {auditProgress.toFixed(0)}% 완료
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
            <div
              className="bg-blue-600 h-3 rounded-full transition-all duration-300"
              style={{ width: `${auditProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* 감사 결과 */}
      {vulnerabilities.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">보안 검사 결과</h3>
          
          <div className="space-y-4">
            {vulnerabilities.map((vuln, index) => (
              <div key={index} className="border border-gray-200 dark:border-gray-600 rounded-lg p-4">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      {getStatusIcon(vuln.status)}
                      <h4 className="font-medium text-gray-900 dark:text-white">
                        {vuln.name}
                      </h4>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(vuln.severity)}`}>
                        {vuln.severity === 'critical' ? '심각' :
                         vuln.severity === 'high' ? '높음' :
                         vuln.severity === 'medium' ? '보통' : '낮음'}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      {vuln.description}
                    </p>
                    {vuln.details && (
                      <p className={`text-sm mb-2 ${
                        vuln.status === 'fail' ? 'text-red-600 dark:text-red-400' :
                        vuln.status === 'warning' ? 'text-yellow-600 dark:text-yellow-400' :
                        'text-green-600 dark:text-green-400'
                      }`}>
                        {vuln.details}
                      </p>
                    )}
                    {vuln.recommendation && vuln.status !== 'pass' && (
                      <p className="text-sm text-blue-600 dark:text-blue-400">
                        <strong>권장사항:</strong> {vuln.recommendation}
                      </p>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                    {vuln.category}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 감사 보고서 */}
      {auditReport && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              종합 보안 감사 보고서
            </h3>
            <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
              <FileText className="w-4 h-4" />
              보고서 다운로드
            </button>
          </div>

          {/* 전체 점수 */}
          <div className="text-center mb-8">
            <div className={`text-6xl font-bold mb-2 ${
              auditReport.riskLevel === 'low' ? 'text-green-600' :
              auditReport.riskLevel === 'medium' ? 'text-yellow-600' :
              auditReport.riskLevel === 'high' ? 'text-orange-600' : 'text-red-600'
            }`}>
              {auditReport.overallScore}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              보안 점수 (100점 만점)
            </div>
            <div className={`inline-block px-4 py-2 rounded-lg font-medium ${
              auditReport.riskLevel === 'low' ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300' :
              auditReport.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300' :
              auditReport.riskLevel === 'high' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300' :
              'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
            }`}>
              {auditReport.riskLevel === 'low' ? '낮은 위험' :
               auditReport.riskLevel === 'medium' ? '보통 위험' :
               auditReport.riskLevel === 'high' ? '높은 위험' : '심각한 위험'}
            </div>
          </div>

          {/* 통계 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
                {auditReport.totalChecks}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">총 검사 항목</div>
            </div>
            <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="text-2xl font-bold text-green-600 mb-1">
                {auditReport.passed}
              </div>
              <div className="text-sm text-green-700 dark:text-green-300">통과</div>
            </div>
            <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <div className="text-2xl font-bold text-yellow-600 mb-1">
                {auditReport.warnings}
              </div>
              <div className="text-sm text-yellow-700 dark:text-yellow-300">경고</div>
            </div>
            <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <div className="text-2xl font-bold text-red-600 mb-1">
                {auditReport.failed}
              </div>
              <div className="text-sm text-red-700 dark:text-red-300">실패</div>
            </div>
          </div>

          {/* 권장사항 */}
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-3">
              우선순위 보안 조치
            </h4>
            <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-2">
              {auditReport.failed > 0 && (
                <li>• 실패한 {auditReport.failed}개 항목에 대한 즉시 보안 조치 필요</li>
              )}
              {auditReport.warnings > 0 && (
                <li>• {auditReport.warnings}개 경고 항목에 대한 모니터링 강화</li>
              )}
              <li>• 정기적인 보안 감사 실시 (월 1회 권장)</li>
              <li>• 보안 교육 및 인식 제고 프로그램 운영</li>
            </ul>
          </div>
        </div>
      )}

      {/* 설명 */}
      <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
          AI 보안 감사 도구 소개
        </h3>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          이 도구는 AI 시스템의 포괄적인 보안 평가를 수행합니다. 
          적대적 공격, 프라이버시 침해, 모델 추출, 데이터 중독 등 다양한 보안 위협에 대한 
          시스템의 취약점을 체계적으로 분석하고 구체적인 보안 강화 방안을 제시합니다.
        </p>
      </div>
    </div>
  )
}