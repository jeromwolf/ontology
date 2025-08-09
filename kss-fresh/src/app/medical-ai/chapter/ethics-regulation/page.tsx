'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Scale,
  Shield,
  Users,
  Lock,
  FileText,
  AlertTriangle,
  Globe,
  ChevronRight,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  Brain,
  Heart,
  Eye,
  UserCheck,
  Gavel,
  BookOpen
} from 'lucide-react'

export default function EthicsRegulationPage() {
  const [activeSection, setActiveSection] = useState('overview')

  const sections = [
    { id: 'overview', title: '개요', icon: Scale },
    { id: 'principles', title: '윤리 원칙', icon: Heart },
    { id: 'bias', title: '편향과 공정성', icon: Users },
    { id: 'regulations', title: '규제 프레임워크', icon: Gavel },
    { id: 'implementation', title: '실무 가이드', icon: FileText }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/medical-ai"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>목록으로</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700"></div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                Chapter 8: 윤리와 규제
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 rounded-full text-sm font-medium">
                필수
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid lg:grid-cols-4 gap-8">
          {/* Sidebar Navigation */}
          <aside className="lg:col-span-1">
            <div className="sticky top-24 space-y-2">
              {sections.map((section) => {
                const Icon = section.icon
                return (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                      activeSection === section.id
                        ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg'
                        : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="font-medium">{section.title}</span>
                  </button>
                )
              })}
            </div>
          </aside>

          {/* Main Content */}
          <main className="lg:col-span-3 space-y-8">
            {/* Overview Section */}
            {activeSection === 'overview' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    Medical AI의 윤리와 규제
                  </h2>
                  
                  <div className="prose prose-lg dark:prose-invert max-w-none">
                    <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                      의료 AI의 윤리와 규제는 환자 안전, 프라이버시 보호, 공정성 확보를 위한 
                      필수적인 프레임워크입니다. AI 기술이 의료 현장에 안전하고 책임감 있게 
                      적용되도록 보장합니다.
                    </p>

                    <div className="grid md:grid-cols-2 gap-6 my-8">
                      <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
                        <Scale className="w-10 h-10 text-indigo-600 dark:text-indigo-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          핵심 윤리 이슈
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-indigo-500 mt-0.5" />
                            <span>알고리즘 편향</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-indigo-500 mt-0.5" />
                            <span>설명 가능성</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-indigo-500 mt-0.5" />
                            <span>책임 소재</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-indigo-500 mt-0.5" />
                            <span>환자 동의</span>
                          </li>
                        </ul>
                      </div>

                      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                        <Gavel className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          주요 규제
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>FDA 의료기기 규제</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>CE 마크 (유럽)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>MFDS 인허가 (한국)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>데이터 보호법</span>
                          </li>
                        </ul>
                      </div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 my-8">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        글로벌 동향
                      </h3>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="text-center">
                          <div className="text-3xl font-bold text-indigo-600 dark:text-indigo-400">193</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">WHO 회원국</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">67%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">AI 규제 도입</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">2025</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">EU AI Act 시행</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Principles Section */}
            {activeSection === 'principles' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    의료 AI 윤리 원칙
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Beneficence */}
                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        선행 (Beneficence)
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        환자의 이익을 최우선으로 하며, AI 시스템이 의료 서비스 향상에 기여해야 함
                      </p>
                      <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                        <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                          <li>• 치료 결과 개선</li>
                          <li>• 의료 접근성 향상</li>
                          <li>• 의료진 업무 효율화</li>
                          <li>• 의료 비용 절감</li>
                        </ul>
                      </div>
                    </div>

                    {/* Non-maleficence */}
                    <div className="border-l-4 border-red-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        무해 (Non-maleficence)
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        "해를 끼치지 말라"는 원칙으로, AI 시스템의 위험을 최소화
                      </p>
                      <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                        <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                          <li>• 안전성 검증</li>
                          <li>• 오진 방지</li>
                          <li>• 부작용 모니터링</li>
                          <li>• 시스템 오류 대응</li>
                        </ul>
                      </div>
                    </div>

                    {/* Autonomy */}
                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        자율성 (Autonomy)
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        환자의 자기결정권과 정보에 입각한 동의 보장
                      </p>
                      <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                        <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                          <li>• 충분한 정보 제공</li>
                          <li>• AI 사용 고지</li>
                          <li>• 옵트아웃 권리</li>
                          <li>• 의사결정 참여</li>
                        </ul>
                      </div>
                    </div>

                    {/* Justice */}
                    <div className="border-l-4 border-purple-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        정의 (Justice)
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        공정한 의료 서비스 분배와 차별 없는 AI 적용
                      </p>
                      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                        <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                          <li>• 건강 형평성</li>
                          <li>• 접근성 보장</li>
                          <li>• 편향 제거</li>
                          <li>• 취약 계층 보호</li>
                        </ul>
                      </div>
                    </div>

                    {/* Additional Principles */}
                    <div className="bg-gradient-to-r from-indigo-100 to-purple-100 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        AI 특화 윤리 원칙
                      </h3>
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-semibold mb-2 flex items-center gap-2">
                            <Eye className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                            투명성 (Transparency)
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            AI 시스템의 작동 방식과 의사결정 과정을 이해할 수 있어야 함
                          </p>
                        </div>
                        <div>
                          <h4 className="font-semibold mb-2 flex items-center gap-2">
                            <UserCheck className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                            책임성 (Accountability)
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            AI 시스템의 결정에 대한 명확한 책임 체계 수립
                          </p>
                        </div>
                        <div>
                          <h4 className="font-semibold mb-2 flex items-center gap-2">
                            <Lock className="w-5 h-5 text-green-600 dark:text-green-400" />
                            프라이버시 (Privacy)
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            환자 데이터의 기밀성과 보안 유지
                          </p>
                        </div>
                        <div>
                          <h4 className="font-semibold mb-2 flex items-center gap-2">
                            <Shield className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                            안전성 (Safety)
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            AI 시스템의 신뢰성과 안정성 보장
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Bias Section */}
            {activeSection === 'bias' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    편향과 공정성
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Types of Bias */}
                    <div className="bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        의료 AI의 편향 유형
                      </h3>
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-semibold mb-3">데이터 편향</h4>
                          <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                            <li className="flex items-start gap-2">
                              <AlertCircle className="w-4 h-4 text-red-500 mt-0.5" />
                              <span>인종/민족 불균형</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <AlertCircle className="w-4 h-4 text-red-500 mt-0.5" />
                              <span>성별 편향</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <AlertCircle className="w-4 h-4 text-red-500 mt-0.5" />
                              <span>연령대 편중</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <AlertCircle className="w-4 h-4 text-red-500 mt-0.5" />
                              <span>지역적 편향</span>
                            </li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-semibold mb-3">알고리즘 편향</h4>
                          <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                            <li className="flex items-start gap-2">
                              <AlertTriangle className="w-4 h-4 text-orange-500 mt-0.5" />
                              <span>레이블링 오류</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <AlertTriangle className="w-4 h-4 text-orange-500 mt-0.5" />
                              <span>특징 선택 편향</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <AlertTriangle className="w-4 h-4 text-orange-500 mt-0.5" />
                              <span>평가 지표 편향</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <AlertTriangle className="w-4 h-4 text-orange-500 mt-0.5" />
                              <span>과적합/과소적합</span>
                            </li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Fairness Metrics */}
                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        공정성 평가 지표
                      </h3>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python">
{`# Fairness Evaluation Framework
import numpy as np
from sklearn.metrics import confusion_matrix

class FairnessEvaluator:
    def __init__(self, predictions, labels, sensitive_attributes):
        self.predictions = predictions
        self.labels = labels
        self.sensitive_attrs = sensitive_attributes
        
    def demographic_parity(self):
        """인구통계학적 평등성"""
        groups = np.unique(self.sensitive_attrs)
        positive_rates = {}
        
        for group in groups:
            mask = self.sensitive_attrs == group
            positive_rate = np.mean(self.predictions[mask])
            positive_rates[group] = positive_rate
        
        # Calculate disparity
        max_rate = max(positive_rates.values())
        min_rate = min(positive_rates.values())
        disparity = max_rate - min_rate
        
        return {
            'positive_rates': positive_rates,
            'disparity': disparity,
            'fair': disparity < 0.1  # Threshold
        }
    
    def equalized_odds(self):
        """동등 기회"""
        groups = np.unique(self.sensitive_attrs)
        tpr_dict = {}  # True Positive Rate
        fpr_dict = {}  # False Positive Rate
        
        for group in groups:
            mask = self.sensitive_attrs == group
            tn, fp, fn, tp = confusion_matrix(
                self.labels[mask], 
                self.predictions[mask]
            ).ravel()
            
            tpr_dict[group] = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_dict[group] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_disparity = max(tpr_dict.values()) - min(tpr_dict.values())
        fpr_disparity = max(fpr_dict.values()) - min(fpr_dict.values())
        
        return {
            'tpr_by_group': tpr_dict,
            'fpr_by_group': fpr_dict,
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'fair': tpr_disparity < 0.1 and fpr_disparity < 0.1
        }
    
    def individual_fairness(self):
        """개인 공정성"""
        # Similar individuals should receive similar predictions
        similarity_threshold = 0.9
        prediction_threshold = 0.1
        
        violations = 0
        total_pairs = 0
        
        for i in range(len(self.predictions)):
            for j in range(i+1, len(self.predictions)):
                similarity = self.calculate_similarity(i, j)
                if similarity > similarity_threshold:
                    pred_diff = abs(self.predictions[i] - self.predictions[j])
                    if pred_diff > prediction_threshold:
                        violations += 1
                    total_pairs += 1
        
        return {
            'violation_rate': violations / total_pairs if total_pairs > 0 else 0,
            'fair': violations / total_pairs < 0.05 if total_pairs > 0 else True
        }`}
                          </code>
                        </pre>
                      </div>
                    </div>

                    {/* Mitigation Strategies */}
                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        편향 완화 전략
                      </h3>
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">Pre-processing</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 데이터 리샘플링</li>
                            <li>• 합성 데이터 생성</li>
                            <li>• 특징 변환</li>
                            <li>• 데이터 증강</li>
                          </ul>
                        </div>
                        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">In-processing</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 공정성 제약 추가</li>
                            <li>• 적대적 디바이싱</li>
                            <li>• 다목적 최적화</li>
                            <li>• 정규화 기법</li>
                          </ul>
                        </div>
                        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">Post-processing</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 임계값 최적화</li>
                            <li>• 보정 (Calibration)</li>
                            <li>• 결과 조정</li>
                            <li>• 앙상블 기법</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Case Studies */}
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        편향 사례와 교훈
                      </h3>
                      <div className="space-y-4">
                        <div className="border-l-4 border-red-400 pl-4">
                          <h4 className="font-semibold">피부암 진단 AI</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            백인 환자 데이터로만 학습하여 흑인 환자에서 정확도 40% 하락
                          </p>
                        </div>
                        <div className="border-l-4 border-blue-400 pl-4">
                          <h4 className="font-semibold">폐렴 예측 모델</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            특정 병원 장비의 마커를 학습하여 다른 병원에서 성능 저하
                          </p>
                        </div>
                        <div className="border-l-4 border-green-400 pl-4">
                          <h4 className="font-semibold">심장질환 위험 평가</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            여성 환자 데이터 부족으로 여성의 심장질환 과소평가
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Regulations Section */}
            {activeSection === 'regulations' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    규제 프레임워크
                  </h2>
                  
                  <div className="space-y-8">
                    {/* FDA Regulation */}
                    <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <Globe className="w-10 h-10 text-blue-600 dark:text-blue-400" />
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                            FDA (미국)
                          </h3>
                          <p className="text-gray-600 dark:text-gray-300 mb-4">
                            Software as a Medical Device (SaMD) 규제 프레임워크
                          </p>
                          <div className="grid md:grid-cols-2 gap-4">
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                              <h4 className="font-semibold mb-2">분류 체계</h4>
                              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• Class I: 저위험 (일반 관리)</li>
                                <li>• Class II: 중위험 (510(k))</li>
                                <li>• Class III: 고위험 (PMA)</li>
                              </ul>
                            </div>
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                              <h4 className="font-semibold mb-2">AI/ML 특별 경로</h4>
                              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• Pre-Cert Program</li>
                                <li>• Continuous Learning</li>
                                <li>• Change Control Plan</li>
                              </ul>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* EU MDR */}
                    <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <Globe className="w-10 h-10 text-purple-600 dark:text-purple-400" />
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                            MDR/IVDR (유럽)
                          </h3>
                          <p className="text-gray-600 dark:text-gray-300 mb-4">
                            Medical Device Regulation & AI Act
                          </p>
                          <div className="grid md:grid-cols-2 gap-4">
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                              <h4 className="font-semibold mb-2">MDR 요구사항</h4>
                              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• CE 마크 획득</li>
                                <li>• 임상 평가</li>
                                <li>• 시판 후 감시</li>
                                <li>• UDI 시스템</li>
                              </ul>
                            </div>
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                              <h4 className="font-semibold mb-2">AI Act (2025)</h4>
                              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• 고위험 AI 분류</li>
                                <li>• 적합성 평가</li>
                                <li>• 투명성 의무</li>
                                <li>• 인간 감독</li>
                              </ul>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Korean Regulation */}
                    <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <Globe className="w-10 h-10 text-green-600 dark:text-green-400" />
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                            MFDS (한국)
                          </h3>
                          <p className="text-gray-600 dark:text-gray-300 mb-4">
                            AI 기반 의료기기 허가·심사 가이드라인
                          </p>
                          <div className="grid md:grid-cols-3 gap-3">
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                              <h4 className="font-semibold text-sm mb-1">등급 분류</h4>
                              <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• 1등급: 신고</li>
                                <li>• 2-4등급: 허가</li>
                              </ul>
                            </div>
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                              <h4 className="font-semibold text-sm mb-1">임상시험</h4>
                              <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• 탐색 임상</li>
                                <li>• 확증 임상</li>
                              </ul>
                            </div>
                            <div className="bg-white dark:bg-gray-800 rounded-lg p-3">
                              <h4 className="font-semibold text-sm mb-1">혁신의료기기</h4>
                              <ul className="text-xs space-y-1 text-gray-600 dark:text-gray-400">
                                <li>• 우선 심사</li>
                                <li>• 단계별 허가</li>
                              </ul>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Compliance Timeline */}
                    <div className="border-l-4 border-indigo-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        규제 승인 프로세스
                      </h3>
                      <div className="space-y-4">
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900 rounded-full flex items-center justify-center text-indigo-600 dark:text-indigo-400 font-bold">
                            1
                          </div>
                          <div>
                            <h4 className="font-semibold">제품 분류 결정</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">위험도 평가 및 규제 경로 선택</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900 rounded-full flex items-center justify-center text-indigo-600 dark:text-indigo-400 font-bold">
                            2
                          </div>
                          <div>
                            <h4 className="font-semibold">기술 문서 준비</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">설계, 검증, 임상 데이터</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900 rounded-full flex items-center justify-center text-indigo-600 dark:text-indigo-400 font-bold">
                            3
                          </div>
                          <div>
                            <h4 className="font-semibold">심사 제출</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">규제 기관 제출 및 보완</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900 rounded-full flex items-center justify-center text-indigo-600 dark:text-indigo-400 font-bold">
                            4
                          </div>
                          <div>
                            <h4 className="font-semibold">승인 및 모니터링</h4>
                            <p className="text-sm text-gray-600 dark:text-gray-400">시판 후 감시 및 업데이트</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Implementation Section */}
            {activeSection === 'implementation' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    실무 구현 가이드
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Ethics Committee */}
                    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        AI 윤리위원회 구성
                      </h3>
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                          <Users className="w-8 h-8 text-indigo-600 dark:text-indigo-400 mb-2" />
                          <h4 className="font-semibold mb-2">구성원</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 의료진</li>
                            <li>• AI 전문가</li>
                            <li>• 윤리학자</li>
                            <li>• 법률 전문가</li>
                            <li>• 환자 대표</li>
                          </ul>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                          <FileText className="w-8 h-8 text-blue-600 dark:text-blue-400 mb-2" />
                          <h4 className="font-semibold mb-2">역할</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 정책 수립</li>
                            <li>• 프로젝트 심사</li>
                            <li>• 위험 평가</li>
                            <li>• 사고 조사</li>
                            <li>• 교육 제공</li>
                          </ul>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                          <BookOpen className="w-8 h-8 text-green-600 dark:text-green-400 mb-2" />
                          <h4 className="font-semibold mb-2">활동</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 월례 회의</li>
                            <li>• 가이드라인 개발</li>
                            <li>• 감사 실시</li>
                            <li>• 보고서 발행</li>
                            <li>• 이해관계자 소통</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Implementation Checklist */}
                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        구현 체크리스트
                      </h3>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                        <div className="space-y-4">
                          <div>
                            <h4 className="font-semibold mb-2">개발 단계</h4>
                            <ul className="space-y-2 text-sm">
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>윤리적 영향 평가 실시</span>
                              </li>
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>다양한 데이터셋 확보</span>
                              </li>
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>편향 테스트 수행</span>
                              </li>
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>설명 가능성 구현</span>
                              </li>
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-semibold mb-2">검증 단계</h4>
                            <ul className="space-y-2 text-sm">
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>임상 검증 프로토콜</span>
                              </li>
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>안전성 테스트</span>
                              </li>
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>규제 요구사항 확인</span>
                              </li>
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>문서화 완료</span>
                              </li>
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-semibold mb-2">배포 단계</h4>
                            <ul className="space-y-2 text-sm">
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>사용자 교육 프로그램</span>
                              </li>
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>모니터링 시스템 구축</span>
                              </li>
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>피드백 채널 운영</span>
                              </li>
                              <li className="flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-green-500" />
                                <span>정기 감사 계획</span>
                              </li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Best Practices */}
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        모범 사례
                      </h3>
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-semibold mb-3 flex items-center gap-2">
                            <Brain className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                            기술적 모범 사례
                          </h4>
                          <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                            <li>• 지속적 모델 모니터링</li>
                            <li>• A/B 테스트 실시</li>
                            <li>• 버전 관리 시스템</li>
                            <li>• 롤백 메커니즘 구현</li>
                            <li>• 성능 지표 추적</li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-semibold mb-3 flex items-center gap-2">
                            <Heart className="w-5 h-5 text-red-600 dark:text-red-400" />
                            조직적 모범 사례
                          </h4>
                          <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                            <li>• 다학제 팀 구성</li>
                            <li>• 정기 윤리 교육</li>
                            <li>• 투명한 커뮤니케이션</li>
                            <li>• 환자 참여 프로그램</li>
                            <li>• 지속적 개선 문화</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Future Directions */}
                    <div className="bg-gradient-to-r from-green-100 to-emerald-100 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        미래 방향
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                        <div>
                          <div className="text-2xl font-bold text-green-600 dark:text-green-400">2025</div>
                          <div className="text-sm text-gray-500">글로벌 표준 확립</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">85%</div>
                          <div className="text-sm text-gray-500">XAI 도입률</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">100+</div>
                          <div className="text-sm text-gray-500">윤리 가이드라인</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">24/7</div>
                          <div className="text-sm text-gray-500">실시간 감시</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Navigation */}
            <div className="flex justify-between items-center pt-8">
              <Link
                href="/medical-ai/chapter/medical-data"
                className="flex items-center gap-2 px-6 py-3 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                이전 챕터
              </Link>
              <Link
                href="/medical-ai"
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:shadow-lg transition-all"
              >
                과정 완료
                <CheckCircle className="w-5 h-5" />
              </Link>
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}