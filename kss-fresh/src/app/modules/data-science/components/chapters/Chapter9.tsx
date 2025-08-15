'use client';

import React, { useState } from 'react';
import { BookOpen, FlaskConical, Calculator, BarChart3, Lightbulb, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface Chapter9Props {
  onComplete?: () => void
}

export default function Chapter9({ onComplete }: Chapter9Props) {
  const [activeTab, setActiveTab] = useState('theory')

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-primary">Chapter 9: A/B 테스팅</h1>
        <p className="text-xl text-muted-foreground">
          데이터 기반 의사결정을 위한 실험 설계와 분석 방법을 학습합니다
        </p>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 px-6 py-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <FlaskConical className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            A/B 테스팅이란?
          </h2>
        </div>
        <div className="p-6 space-y-4">
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">정의</h3>
            <p className="text-gray-600 dark:text-gray-400">
              A/B 테스팅은 두 가지 이상의 버전을 비교하여 어느 것이 더 나은 성과를 
              내는지 통계적으로 검증하는 실험 방법입니다. 주로 웹사이트, 앱, 마케팅 
              캠페인 등의 개선에 사용됩니다.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
              <h4 className="font-semibold mb-2">활용 분야</h4>
              <ul className="space-y-2 list-disc list-inside text-sm">
                <li>웹사이트 UI/UX 최적화</li>
                <li>이메일 마케팅 캠페인</li>
                <li>제품 기능 테스트</li>
                <li>가격 정책 실험</li>
                <li>추천 알고리즘 개선</li>
              </ul>
            </div>
            
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
              <h4 className="font-semibold mb-2">핵심 지표</h4>
              <ul className="space-y-2 list-disc list-inside text-sm">
                <li>전환율 (Conversion Rate)</li>
                <li>클릭률 (CTR)</li>
                <li>평균 주문 금액 (AOV)</li>
                <li>사용자 체류 시간</li>
                <li>이탈률 (Bounce Rate)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* 탭 네비게이션 */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex border-b border-gray-200 dark:border-gray-700">
          <button
            onClick={() => setActiveTab('theory')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === 'theory'
                ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
            }`}
          >
            이론
          </button>
          <button
            onClick={() => setActiveTab('statistics')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === 'statistics'
                ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
            }`}
          >
            통계
          </button>
          <button
            onClick={() => setActiveTab('practice')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === 'practice'
                ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
            }`}
          >
            실습
          </button>
          <button
            onClick={() => setActiveTab('pitfalls')}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === 'pitfalls'
                ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-b-2 border-blue-600'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
            }`}
          >
            주의사항
          </button>
        </div>

        <div className="p-6">
          {activeTab === 'theory' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                A/B 테스트 프로세스
              </h3>
              <div className="space-y-4">
                <div className="border-l-4 border-blue-500 pl-4">
                  <h4 className="font-semibold">1. 가설 설정</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    명확하고 측정 가능한 가설을 수립합니다.
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-900 p-3 rounded mt-2">
                    <p className="text-sm font-mono">
                      예시: "버튼 색상을 파란색에서 녹색으로 변경하면 클릭률이 10% 증가할 것이다"
                    </p>
                  </div>
                </div>
                
                <div className="border-l-4 border-purple-500 pl-4">
                  <h4 className="font-semibold">2. 실험 설계</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    샘플 크기, 실험 기간, 분할 방법 등을 결정합니다.
                  </p>
                  <ul className="text-sm mt-2 space-y-1 list-disc list-inside">
                    <li>무작위 분할 (Random Split)</li>
                    <li>층화 추출 (Stratified Sampling)</li>
                    <li>시간대별 분할 (Time-based Split)</li>
                  </ul>
                </div>
                
                <div className="border-l-4 border-green-500 pl-4">
                  <h4 className="font-semibold">3. 실험 실행</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    충분한 데이터가 수집될 때까지 실험을 진행합니다.
                  </p>
                  <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-3 mt-2">
                    <p className="text-sm">
                      최소 실험 기간: 2주 (주중/주말 패턴 포함)
                    </p>
                  </div>
                </div>
                
                <div className="border-l-4 border-gray-500 pl-4">
                  <h4 className="font-semibold">4. 결과 분석</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    통계적 유의성을 검증하고 비즈니스 영향을 평가합니다.
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'statistics' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold flex items-center gap-2">
                <Calculator className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                통계적 개념
              </h3>
              <div className="space-y-4">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <h4 className="font-semibold mb-2">샘플 크기 계산</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    실험에 필요한 최소 샘플 크기를 계산합니다.
                  </p>
                  <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded text-xs overflow-x-auto">
{`import statsmodels.stats.power as smp

# 파라미터 설정
baseline_rate = 0.10  # 현재 전환율 10%
minimum_detectable_effect = 0.02  # 최소 감지 효과 2%p
alpha = 0.05  # 유의수준
power = 0.80  # 검정력

# 샘플 크기 계산
effect_size = minimum_detectable_effect / baseline_rate
sample_size = smp.zt_ind_solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    ratio=1.0,
    alternative='two-sided'
)

print(f"그룹당 필요 샘플 크기: {int(sample_size):,}")`}</pre>
                </div>
                
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                  <h4 className="font-semibold mb-2">통계적 유의성 검정</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    p-value와 신뢰구간을 계산하여 결과의 유의성을 판단합니다.
                  </p>
                  <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded text-xs overflow-x-auto">
{`from scipy import stats
import numpy as np

# A/B 그룹 데이터
conversions_A = 120  # A그룹 전환 수
visitors_A = 1000    # A그룹 방문자 수
conversions_B = 150  # B그룹 전환 수
visitors_B = 1000    # B그룹 방문자 수

# 전환율 계산
rate_A = conversions_A / visitors_A
rate_B = conversions_B / visitors_B

# 이항 검정
z_score, p_value = stats.proportions_ztest(
    [conversions_A, conversions_B],
    [visitors_A, visitors_B]
)

# 신뢰구간 계산
se_A = np.sqrt(rate_A * (1 - rate_A) / visitors_A)
se_B = np.sqrt(rate_B * (1 - rate_B) / visitors_B)
ci_A = stats.norm.interval(0.95, loc=rate_A, scale=se_A)
ci_B = stats.norm.interval(0.95, loc=rate_B, scale=se_B)

print(f"A그룹 전환율: {rate_A:.2%} (95% CI: {ci_A[0]:.2%} - {ci_A[1]:.2%})")
print(f"B그룹 전환율: {rate_B:.2%} (95% CI: {ci_B[0]:.2%} - {ci_B[1]:.2%})")
print(f"p-value: {p_value:.4f}")
print(f"통계적 유의성: {'있음' if p_value < 0.05 else '없음'}")`}</pre>
                </div>

                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h4 className="font-semibold mb-2">검정력 분석</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    실험이 실제 효과를 감지할 수 있는 능력을 평가합니다.
                  </p>
                  <div className="grid grid-cols-2 gap-4 mt-3">
                    <div className="bg-white dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700">
                      <p className="text-xs font-semibold">제1종 오류 (α)</p>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        효과가 없는데 있다고 판단할 확률 (일반적으로 5%)
                      </p>
                    </div>
                    <div className="bg-white dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700">
                      <p className="text-xs font-semibold">제2종 오류 (β)</p>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        효과가 있는데 없다고 판단할 확률 (일반적으로 20%)
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'practice' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-green-600 dark:text-green-400" />
                실전 A/B 테스트 구현
              </h3>
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">1. 실험 설정 및 사용자 할당</h4>
                  <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`import hashlib
import pandas as pd
from datetime import datetime, timedelta

class ABTest:
    def __init__(self, test_name, control_name='control', variant_name='variant', 
                 traffic_split=0.5):
        self.test_name = test_name
        self.control_name = control_name
        self.variant_name = variant_name
        self.traffic_split = traffic_split
        self.results = {'control': [], 'variant': []}
    
    def assign_variant(self, user_id):
        """사용자를 A 또는 B 그룹에 할당"""
        # 해시를 사용하여 일관된 할당 보장
        hash_value = int(hashlib.md5(
            f"{user_id}_{self.test_name}".encode()
        ).hexdigest(), 16)
        
        # 트래픽 분할에 따라 그룹 할당
        if (hash_value % 100) / 100 < self.traffic_split:
            return self.variant_name
        return self.control_name
    
    def track_conversion(self, user_id, converted):
        """전환 이벤트 추적"""
        variant = self.assign_variant(user_id)
        self.results[variant].append({
            'user_id': user_id,
            'converted': converted,
            'timestamp': datetime.now()
        })

# 실험 생성
button_test = ABTest(
    test_name='green_button_test',
    control_name='blue_button',
    variant_name='green_button',
    traffic_split=0.5
)`}</pre>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">2. 실시간 모니터링 대시보드</h4>
                  <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`class ABTestDashboard:
    def __init__(self, test):
        self.test = test
    
    def calculate_metrics(self):
        """실시간 지표 계산"""
        metrics = {}
        
        for variant in ['control', 'variant']:
            data = pd.DataFrame(self.test.results[variant])
            if len(data) > 0:
                total_users = len(data)
                conversions = data['converted'].sum()
                conversion_rate = conversions / total_users
                
                # 신뢰구간 계산
                se = np.sqrt(conversion_rate * (1 - conversion_rate) / total_users)
                ci_lower = conversion_rate - 1.96 * se
                ci_upper = conversion_rate + 1.96 * se
                
                metrics[variant] = {
                    'users': total_users,
                    'conversions': conversions,
                    'rate': conversion_rate,
                    'ci': (ci_lower, ci_upper)
                }
            else:
                metrics[variant] = {
                    'users': 0,
                    'conversions': 0,
                    'rate': 0,
                    'ci': (0, 0)
                }
        
        # 상대적 개선도 계산
        if metrics['control']['rate'] > 0:
            lift = ((metrics['variant']['rate'] - metrics['control']['rate']) / 
                   metrics['control']['rate'] * 100)
        else:
            lift = 0
        
        metrics['lift'] = lift
        return metrics`}</pre>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">3. 세그먼트별 분석</h4>
                  <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg text-sm overflow-x-auto">
{`def segment_analysis(test_results, user_segments):
    """사용자 세그먼트별 효과 분석"""
    segment_metrics = {}
    
    for segment_name, user_ids in user_segments.items():
        segment_data = {
            'control': [],
            'variant': []
        }
        
        # 세그먼트별 데이터 필터링
        for variant in ['control', 'variant']:
            for result in test_results[variant]:
                if result['user_id'] in user_ids:
                    segment_data[variant].append(result)
        
        # 세그먼트별 지표 계산
        control_rate = (sum(r['converted'] for r in segment_data['control']) / 
                       len(segment_data['control']) if segment_data['control'] else 0)
        variant_rate = (sum(r['converted'] for r in segment_data['variant']) / 
                       len(segment_data['variant']) if segment_data['variant'] else 0)
        
        segment_metrics[segment_name] = {
            'control_rate': control_rate,
            'variant_rate': variant_rate,
            'lift': ((variant_rate - control_rate) / control_rate * 100 
                    if control_rate > 0 else 0)
        }
    
    return segment_metrics`}</pre>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'pitfalls' && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400" />
                A/B 테스팅의 함정과 주의사항
              </h3>
              <div className="space-y-4">
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                  <div className="flex gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0" />
                    <div>
                      <strong className="block mb-1">1. 조기 종료 (Peeking Problem)</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        실험 중간에 결과를 확인하고 조기에 종료하면 잘못된 결론을 내릴 수 있습니다.
                        사전에 정한 샘플 크기나 기간을 준수해야 합니다.
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                  <div className="flex gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0" />
                    <div>
                      <strong className="block mb-1">2. 다중 비교 문제 (Multiple Comparisons)</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        여러 지표를 동시에 테스트하면 우연히 유의한 결과가 나올 확률이 증가합니다.
                        Bonferroni 보정 등을 사용하여 유의수준을 조정해야 합니다.
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                  <div className="flex gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0" />
                    <div>
                      <strong className="block mb-1">3. 샘플 선택 편향 (Selection Bias)</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        무작위 할당이 제대로 이루어지지 않으면 그룹 간 차이가 실험 효과가 아닌
                        원래 특성 차이일 수 있습니다.
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                  <div className="flex gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0" />
                    <div>
                      <strong className="block mb-1">4. 신규성 효과 (Novelty Effect)</strong>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        새로운 변경사항이 단기적으로는 좋은 결과를 보이지만, 시간이 지나면
                        효과가 사라질 수 있습니다. 충분한 기간 동안 실험해야 합니다.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400" />
                    모범 사례
                  </h4>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start gap-2">
                      <span className="text-green-600 dark:text-green-400 mt-0.5">✓</span>
                      <span>실험 전 샘플 크기와 기간을 미리 계산하고 준수</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-600 dark:text-green-400 mt-0.5">✓</span>
                      <span>A/A 테스트로 실험 환경의 신뢰성 검증</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-600 dark:text-green-400 mt-0.5">✓</span>
                      <span>주요 지표와 보조 지표를 함께 모니터링</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-600 dark:text-green-400 mt-0.5">✓</span>
                      <span>실험 결과를 문서화하고 조직 내 공유</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-600 dark:text-green-400 mt-0.5">✓</span>
                      <span>세그먼트별 분석으로 이질적 효과 확인</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 flex gap-3">
        <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
        <div>
          <strong>고급 기법:</strong> 더 정교한 실험을 위한 방법들:
          <ul className="mt-2 space-y-1 list-disc list-inside">
            <li>다변량 테스트 (Multivariate Testing): 여러 요소를 동시에 테스트</li>
            <li>밴딧 알고리즘 (Bandit Algorithm): 실시간으로 트래픽을 조정</li>
            <li>베이지안 A/B 테스트: 사전 정보를 활용한 확률적 접근</li>
            <li>CUPED: 공변량을 활용한 분산 감소 기법</li>
          </ul>
        </div>
      </div>

      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/10 dark:to-indigo-900/10 rounded-xl p-6 border-2 border-blue-200 dark:border-blue-800">
        <h3 className="text-xl font-semibold mb-2">실습 프로젝트: E-commerce A/B 테스트 플랫폼</h3>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          실제 온라인 쇼핑몰의 A/B 테스트를 설계하고 분석하는 플랫폼을 구축해봅시다
        </p>
        <div className="space-y-4">
          <div className="space-y-3">
            <h4 className="font-semibold">프로젝트 기능:</h4>
            <ol className="space-y-2 list-decimal list-inside">
              <li>실험 생성 및 관리 인터페이스</li>
              <li>사용자 할당 및 추적 시스템</li>
              <li>실시간 지표 모니터링 대시보드</li>
              <li>통계적 유의성 자동 계산</li>
              <li>세그먼트별 분석 리포트</li>
              <li>실험 결과 아카이브 및 학습</li>
            </ol>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <h5 className="font-semibold text-sm mb-2">실험 예시</h5>
              <div className="text-xs space-y-2">
                <p><strong>가설:</strong> 무료 배송 임계값을 5만원에서 3만원으로 낮추면 주문 전환율이 증가할 것이다</p>
                <p><strong>대조군:</strong> 5만원 이상 무료배송</p>
                <p><strong>실험군:</strong> 3만원 이상 무료배송</p>
                <p><strong>주요 지표:</strong> 주문 전환율</p>
                <p><strong>보조 지표:</strong> 평균 주문 금액, 장바구니 이탈률</p>
              </div>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <h5 className="font-semibold text-sm mb-2">예상 결과</h5>
              <div className="text-xs space-y-1">
                <p>• 전환율: 3.2% → 3.8% (+18.8%)</p>
                <p>• 평균 주문 금액: 72,000원 → 65,000원 (-9.7%)</p>
                <p>• p-value: 0.023 (유의함)</p>
                <p>• 필요 샘플: 그룹당 15,000명</p>
                <p>• 예상 수익 영향: +12.3%</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center pt-8">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          A/B 테스팅은 데이터 기반 의사결정의 핵심 도구입니다
        </p>
        {onComplete && (
          <button
            onClick={onComplete}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            다음 챕터로
          </button>
        )}
      </div>
    </div>
  )
}