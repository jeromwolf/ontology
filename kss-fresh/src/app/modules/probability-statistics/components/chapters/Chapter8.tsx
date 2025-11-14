'use client';

import { useState } from 'react';
import { Zap, FlaskConical } from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter8() {
  const [regularizationStrength, setRegularizationStrength] = useState(0.1)

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold mb-4">머신러닝을 위한 통계</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          머신러닝의 이론적 기반이 되는 통계적 개념들을 살펴봅니다.
          모델의 성능 평가, 과적합 방지, 검증 방법 등을 다룹니다.
        </p>
      </div>

      <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Zap className="text-violet-500" />
          편향-분산 트레이드오프
        </h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
          <div className="text-center mb-4">
            <p className="text-lg font-mono">
              총 오차 = 편향² + 분산 + 노이즈
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-violet-100 dark:bg-violet-900/30 p-4 rounded-lg">
              <h4 className="font-semibold text-violet-700 dark:text-violet-300 mb-2">
                높은 편향 (Underfitting)
              </h4>
              <ul className="text-sm space-y-1">
                <li>• 모델이 너무 단순함</li>
                <li>• 훈련/테스트 오차 모두 높음</li>
                <li>• 중요한 패턴을 놓침</li>
              </ul>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                해결: 복잡한 모델, 더 많은 특성
              </p>
            </div>
            
            <div className="bg-purple-100 dark:bg-purple-900/30 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">
                높은 분산 (Overfitting)
              </h4>
              <ul className="text-sm space-y-1">
                <li>• 모델이 너무 복잡함</li>
                <li>• 훈련 오차는 낮지만 테스트 오차 높음</li>
                <li>• 노이즈까지 학습</li>
              </ul>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                해결: 정규화, 더 많은 데이터
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">교차 검증 (Cross-Validation)</h3>
        
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
              K-Fold Cross-Validation
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              데이터를 K개 폴드로 나누어 K번 학습/검증
            </p>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map(fold => (
                <div 
                  key={fold} 
                  className={`flex-1 h-8 flex items-center justify-center text-xs font-semibold rounded ${
                    fold === 3 ? 'bg-blue-500 text-white' : 'bg-gray-300 dark:bg-gray-600'
                  }`}
                >
                  {fold === 3 ? 'Test' : 'Train'}
                </div>
              ))}
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
                LOOCV
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                Leave-One-Out: n개 샘플에 대해 n번 검증
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                작은 데이터셋에 유용
              </p>
            </div>
            
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-600 dark:text-blue-400 mb-2">
                Stratified K-Fold
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                각 폴드의 클래스 비율 유지
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                불균형 데이터에 필수
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">정규화 (Regularization)</h3>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg">
          <div className="mb-4">
            <label className="block text-sm font-semibold mb-2">
              정규화 강도 (λ): {regularizationStrength.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={regularizationStrength}
              onChange={(e) => setRegularizationStrength(Number(e.target.value))}
              className="w-full"
            />
          </div>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-1">L1 (Lasso)</h4>
              <p className="text-xs">계수를 0으로 만들어 변수 선택</p>
              <p className="text-xs font-mono mt-1">Σ|βᵢ|</p>
            </div>
            
            <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-1">L2 (Ridge)</h4>
              <p className="text-xs">계수를 축소하여 과적합 방지</p>
              <p className="text-xs font-mono mt-1">Σβᵢ²</p>
            </div>
            
            <div className="bg-green-100 dark:bg-green-900/30 p-3 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-1">Elastic Net</h4>
              <p className="text-xs">L1과 L2의 장점 결합</p>
              <p className="text-xs font-mono mt-1">αΣ|βᵢ| + (1-α)Σβᵢ²</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">성능 평가 지표</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-3">분류 지표</h4>
            <div className="space-y-2">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">정확도 (Accuracy)</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">전체 예측 중 맞춘 비율</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">정밀도 (Precision)</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">양성 예측 중 실제 양성 비율</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">재현율 (Recall)</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">실제 양성 중 양성 예측 비율</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">F1 Score</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">정밀도와 재현율의 조화평균</p>
              </div>
            </div>
          </div>
          
          <div>
            <h4 className="font-semibold text-orange-600 dark:text-orange-400 mb-3">회귀 지표</h4>
            <div className="space-y-2">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">MAE</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">평균 절대 오차</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">MSE/RMSE</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">평균 제곱 오차</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">R² Score</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">결정 계수</p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-sm">MAPE</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">평균 절대 백분율 오차</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4">특성 공학과 선택</h3>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">
              특성 공학
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 다항 특성 생성</li>
              <li>• 교호작용 항</li>
              <li>• 로그/지수 변환</li>
              <li>• 구간화 (Binning)</li>
              <li>• 원-핫 인코딩</li>
            </ul>
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2">
              특성 선택
            </h4>
            <ul className="text-sm space-y-1">
              <li>• 필터 방법 (상관계수, χ²)</li>
              <li>• 래퍼 방법 (RFE)</li>
              <li>• 임베디드 방법 (Lasso)</li>
              <li>• 순열 중요도</li>
              <li>• SHAP 값</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-yellow-50 dark:bg-yellow-900/20 p-6 rounded-xl">
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <FlaskConical className="text-yellow-500" />
          A/B 테스팅과 실험 설계
        </h3>

        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <h4 className="font-semibold mb-2">표본 크기 계산</h4>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              최소 탐지 효과(MDE), 검정력, 유의수준을 고려
            </p>
            <div className="bg-gray-100 dark:bg-gray-700 p-2 rounded text-xs">
              n = 2 × (Z_α/2 + Z_β)² × σ² / δ²
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">
                다중 검정 보정
              </h4>
              <ul className="text-sm space-y-1">
                <li>• Bonferroni 보정</li>
                <li>• Benjamini-Hochberg</li>
                <li>• False Discovery Rate</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <h4 className="font-semibold text-yellow-600 dark:text-yellow-400 mb-2">
                실험 설계 원칙
              </h4>
              <ul className="text-sm space-y-1">
                <li>• 무작위 배정</li>
                <li>• 통제 변수</li>
                <li>• 실험 기간 설정</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <References
        sections={[
          {
            title: '📚 온라인 강의 & 리소스',
            icon: 'web' as const,
            color: 'border-orange-500',
            items: [
              {
                title: 'Khan Academy Statistics and Probability',
                url: 'https://www.khanacademy.org/math/statistics-probability',
                description: '무료 확률통계 강의 - 기초부터 가설검정, 회귀분석까지 완벽 커버 (2024)',
                year: 2024
              },
              {
                title: 'MIT OCW 18.05 - Introduction to Probability and Statistics',
                url: 'https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/',
                description: 'MIT 확률통계 강의 - 엄밀한 수학적 접근과 실전 응용 (2024)',
                year: 2024
              },
              {
                title: 'StatQuest with Josh Starmer',
                url: 'https://www.youtube.com/c/joshstarmer',
                description: 'YouTube 통계 강의 - 복잡한 개념을 재미있게 설명 (2024)',
                year: 2024
              },
              {
                title: 'Coursera - Statistics with R Specialization',
                url: 'https://www.coursera.org/specializations/statistics',
                description: 'Duke 대학 통계학 특화 과정 - R 프로그래밍 포함 (2024)',
                year: 2024
              }
            ]
          },
          {
            title: '📖 핵심 교재',
            icon: 'research' as const,
            color: 'border-red-500',
            items: [
              {
                title: 'Introduction to Probability (Blitzstein & Hwang)',
                url: 'https://projects.iq.harvard.edu/stat110/home',
                description: 'Harvard 확률론 명강의 교재 - 직관과 엄밀함의 완벽한 조화 (2판, 2019)',
                year: 2019
              },
              {
                title: 'All of Statistics (Larry Wasserman)',
                url: 'https://www.stat.cmu.edu/~larry/all-of-statistics/',
                description: '통계학 핵심 총정리 - 머신러닝을 위한 통계적 배경 완벽 제공 (2004)',
                year: 2004
              },
              {
                title: 'Statistical Inference (Casella & Berger)',
                url: 'https://www.cengage.com/c/statistical-inference-2e-casella',
                description: '통계적 추론 고급 교재 - 대학원 수준 엄밀한 이론 (2판, 2002)',
                year: 2002
              },
              {
                title: 'The Elements of Statistical Learning (Hastie et al.)',
                url: 'https://hastie.su.domains/ElemStatLearn/',
                description: '통계학습 바이블 - 머신러닝 이론의 통계적 기초 무료 공개 (2판, 2009)',
                year: 2009
              }
            ]
          },
          {
            title: '🛠️ 실전 도구',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'R & RStudio',
                url: 'https://posit.co/download/rstudio-desktop/',
                description: '통계 분석 표준 도구 - 강력한 시각화 및 패키지 생태계 (2024)',
                year: 2024
              },
              {
                title: 'Python statsmodels',
                url: 'https://www.statsmodels.org/',
                description: 'Python 통계 모델링 라이브러리 - 회귀, 시계열, 가설검정 (2024)',
                year: 2024
              },
              {
                title: 'SPSS',
                url: 'https://www.ibm.com/products/spss-statistics',
                description: 'IBM 통계 분석 소프트웨어 - GUI 기반 직관적 분석 (2024)',
                year: 2024
              },
              {
                title: 'SAS',
                url: 'https://www.sas.com/en_us/software/stat.html',
                description: '엔터프라이즈 통계 분석 플랫폼 - 대규모 데이터 처리 (2024)',
                year: 2024
              },
              {
                title: 'Stata',
                url: 'https://www.stata.com/',
                description: '경제학/사회과학 통계 도구 - 패널 데이터 분석 특화 (2024)',
                year: 2024
              }
            ]
          }
        ]}
      />
    </div>
  )
}