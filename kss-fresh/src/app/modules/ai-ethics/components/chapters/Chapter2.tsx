import React from 'react';
import { Scale, TrendingUp, AlertCircle, Users, BarChart } from 'lucide-react';
import References from '../References';

export default function Chapter2() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-6 text-gray-900 dark:text-white">편향과 공정성</h1>

      <div className="bg-gradient-to-r from-rose-100 to-pink-100 dark:from-rose-900/30 dark:to-pink-900/30 p-6 rounded-lg mb-8">
        <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
          AI 시스템의 편향은 훈련 데이터, 알고리즘 설계, 사회적 맥락이 복합적으로 작용한 결과입니다.
          공정성(Fairness)은 단순히 "똑같이 대우"하는 것이 아니라, 역사적·구조적 불평등을 고려한
          정의로운 결과를 의미합니다.
        </p>
      </div>

      {/* 공정성의 3대 정의 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Scale className="w-8 h-8 text-rose-600" />
          공정성의 3대 수학적 정의
        </h2>

        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-blue-500">
            <h3 className="text-2xl font-bold mb-3 text-gray-900 dark:text-white">1. Statistical Parity (통계적 동등성)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              모든 그룹이 동일한 비율로 긍정적 결과를 받아야 함
            </p>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded mb-4">
              <p className="font-mono text-sm text-gray-800 dark:text-gray-200">
                P(Ŷ = 1 | A = 0) = P(Ŷ = 1 | A = 1)
              </p>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                Ŷ: 예측 결과, A: 보호 속성 (예: 성별, 인종)
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">예시: 대출 승인</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>남성 그룹 대출 승인율: 60%</li>
                <li>여성 그룹 대출 승인율: 60% ← Statistical Parity 만족</li>
              </ul>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2 italic">
                문제점: 실제 신용도 차이를 무시할 수 있음
              </p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-green-500">
            <h3 className="text-2xl font-bold mb-3 text-gray-900 dark:text-white">2. Equal Opportunity (기회 평등)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              실제 긍정 케이스(Y=1)에서 모든 그룹의 True Positive Rate이 동일해야 함
            </p>

            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded mb-4">
              <p className="font-mono text-sm text-gray-800 dark:text-gray-200">
                P(Ŷ = 1 | Y = 1, A = 0) = P(Ŷ = 1 | Y = 1, A = 1)
              </p>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                Y: 실제 값 (Ground Truth)
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">예시: 채용 AI</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>실제 자격 있는 남성 후보 중 합격: 80%</li>
                <li>실제 자격 있는 여성 후보 중 합격: 80% ← Equal Opportunity 만족</li>
              </ul>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2 italic">
                장점: 자격 있는 사람에게 공평한 기회 보장
              </p>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-purple-500">
            <h3 className="text-2xl font-bold mb-3 text-gray-900 dark:text-white">3. Predictive Parity (예측 동등성)</h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              긍정 예측(Ŷ=1)의 정확도(Precision)가 모든 그룹에서 동일해야 함
            </p>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded mb-4">
              <p className="font-mono text-sm text-gray-800 dark:text-gray-200">
                P(Y = 1 | Ŷ = 1, A = 0) = P(Y = 1 | Ŷ = 1, A = 1)
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">예시: 재범 예측 (COMPAS)</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>백인 그룹: 고위험 예측 중 실제 재범률 63%</li>
                <li>흑인 그룹: 고위험 예측 중 실제 재범률 63% ← Predictive Parity 만족</li>
              </ul>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-2 italic">
                중요: 세 가지 정의를 동시에 만족하는 것은 수학적으로 불가능 (Impossibility Theorem)
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* COMPAS 알고리즘 사례 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <AlertCircle className="w-8 h-8 text-red-600" />
          COMPAS 알고리즘: 편향의 고전적 사례
        </h2>

        <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg mb-6">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">사건 개요 (ProPublica 조사, 2016)</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            미국 법원에서 사용하는 재범 위험 예측 알고리즘 COMPAS가 흑인에게 체계적으로 불리한 결과를 생성.
            ProPublica의 분석 결과, 알고리즘이 "Predictive Parity"는 만족하지만 "Equal Opportunity"를 위반함이 밝혀짐.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">흑인 피고인</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>False Positive Rate (무고한데 고위험): <strong className="text-red-600">44.9%</strong></li>
                <li>False Negative Rate (위험한데 저위험): 28.0%</li>
                <li>실제 재범하지 않았는데 고위험 판정받은 비율이 백인의 2배</li>
              </ul>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <h4 className="font-bold text-gray-900 dark:text-white mb-2">백인 피고인</h4>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>False Positive Rate: <strong className="text-green-600">23.5%</strong></li>
                <li>False Negative Rate (위험한데 저위험): 47.7%</li>
                <li>실제 재범했는데 저위험 판정받은 비율이 흑인의 1.7배</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
          <p className="text-sm font-semibold text-gray-900 dark:text-white mb-2">핵심 교훈:</p>
          <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>알고리즘이 "객관적"이라는 주장은 위험 (데이터 자체에 편향 존재)</li>
            <li>여러 공정성 지표를 동시에 만족하는 것은 불가능할 수 있음</li>
            <li>역사적 차별이 데이터에 반영되어 AI가 이를 학습하고 강화</li>
            <li>High-stakes 결정(형사사법)에서 AI 보조 도구는 신중히 사용해야 함</li>
          </ul>
        </div>
      </section>

      {/* 실전 코드: Fairlearn & IBM AIF360 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">실전 코드: 편향 탐지 및 완화</h2>

        <div className="mb-6">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">1. Fairlearn - Microsoft 공정성 라이브러리</h3>
          <div className="bg-gray-900 dark:bg-gray-950 p-6 rounded-lg overflow-x-auto">
            <pre className="text-sm text-gray-100">
              <code>{`from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression
import pandas as pd

# 예시: 대출 승인 데이터
# X: 특성 (소득, 부채 등), y: 승인 여부 (0/1), sensitive: 성별
X_train, X_test = ...  # 훈련/테스트 데이터
y_train, y_test = ...  # 레이블
sensitive_train = ...  # 보호 속성 (예: ['male', 'female', ...])
sensitive_test = ...

# 1단계: 기본 모델 학습
base_model = LogisticRegression()
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)

# 2단계: 공정성 지표 측정
metrics = {
    'selection_rate': selection_rate,
    'false_positive_rate': false_positive_rate,
    'accuracy': lambda y_true, y_pred: (y_true == y_pred).mean()
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred_base,
    sensitive_features=sensitive_test
)

print("=== 기본 모델 공정성 지표 ===")
print(metric_frame.by_group)
# 출력 예시:
#                selection_rate  false_positive_rate  accuracy
# sensitive_features
# female                   0.45                 0.28      0.82
# male                     0.62                 0.15      0.85
# → 남성 그룹이 여성보다 17%p 높은 승인율 (편향 존재!)

# 3단계: 편향 완화 (Demographic Parity 제약)
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(),
    constraints=DemographicParity()  # Statistical Parity 강제
)

mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
y_pred_mitigated = mitigator.predict(X_test)

# 4단계: 완화 후 공정성 재측정
metric_frame_mitigated = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred_mitigated,
    sensitive_features=sensitive_test
)

print("\\n=== 편향 완화 후 ===")
print(metric_frame_mitigated.by_group)
# 출력 예시:
#                selection_rate  false_positive_rate  accuracy
# sensitive_features
# female                   0.53                 0.22      0.80
# male                     0.54                 0.21      0.81
# → 승인율 차이 1%p로 감소! (정확도 약간 감소 trade-off 존재)

# 5단계: 공정성-정확도 Trade-off 시각화
import matplotlib.pyplot as plt

disparities = []
accuracies = []

for constraint_weight in [0.01, 0.05, 0.1, 0.5, 1.0]:
    mitigator = ExponentiatedGradient(
        estimator=LogisticRegression(),
        constraints=DemographicParity(difference_bound=constraint_weight)
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    y_pred = mitigator.predict(X_test)

    mf = MetricFrame(
        metrics={'selection_rate': selection_rate},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_test
    )
    disparity = mf.by_group['selection_rate'].max() - mf.by_group['selection_rate'].min()
    accuracy = (y_test == y_pred).mean()

    disparities.append(disparity)
    accuracies.append(accuracy)

plt.plot(disparities, accuracies, 'o-')
plt.xlabel('Selection Rate Disparity (차이)')
plt.ylabel('Overall Accuracy')
plt.title('Fairness-Accuracy Trade-off')
plt.show()`}</code>
            </pre>
          </div>
        </div>

        <div className="mb-6">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">2. IBM AI Fairness 360 - 고급 편향 완화</h3>
          <div className="bg-gray-900 dark:bg-gray-950 p-6 rounded-lg overflow-x-auto">
            <pre className="text-sm text-gray-100">
              <code>{`from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

# 1단계: 데이터 편향 분석 (Pre-processing)
dataset = BinaryLabelDataset(
    df=df,
    label_names=['approved'],
    protected_attribute_names=['gender']
)

metric = BinaryLabelDatasetMetric(
    dataset,
    unprivileged_groups=[{'gender': 0}],  # 여성
    privileged_groups=[{'gender': 1}]      # 남성
)

print("=== 데이터 편향 지표 ===")
print(f"Mean Difference: {metric.mean_difference():.3f}")
# 양수: privileged 그룹이 더 많은 긍정 레이블
print(f"Disparate Impact: {metric.disparate_impact():.3f}")
# 1.0에서 멀수록 편향 심함 (0.8 미만이면 법적 문제 가능)

# 2단계: 데이터 재가중치 (Reweighing)
# 소수 그룹 샘플에 더 큰 가중치 부여
RW = Reweighing(
    unprivileged_groups=[{'gender': 0}],
    privileged_groups=[{'gender': 1}]
)
dataset_transformed = RW.fit_transform(dataset)

metric_transformed = BinaryLabelDatasetMetric(
    dataset_transformed,
    unprivileged_groups=[{'gender': 0}],
    privileged_groups=[{'gender': 1}]
)
print(f"\\n재가중치 후 Disparate Impact: {metric_transformed.disparate_impact():.3f}")
# → 1.0에 가까워짐 (편향 감소)

# 3단계: In-processing 편향 완화 (학습 중 공정성 통합)
prejudice_remover = PrejudiceRemover(
    sensitive_attr='gender',
    eta=1.0  # 공정성 정규화 강도 (높을수록 공정성 우선)
)
prejudice_remover.fit(dataset_transformed)

# 4단계: 예측 및 공정성 평가
dataset_pred = prejudice_remover.predict(dataset_test)

classification_metric = ClassificationMetric(
    dataset_test,
    dataset_pred,
    unprivileged_groups=[{'gender': 0}],
    privileged_groups=[{'gender': 1}]
)

print("\\n=== 최종 공정성 지표 ===")
print(f"Equal Opportunity Difference: {classification_metric.equal_opportunity_difference():.3f}")
# 0에 가까울수록 공정 (TPR 차이)
print(f"Average Odds Difference: {classification_metric.average_odds_difference():.3f}")
# 0에 가까울수록 공정 (TPR + FPR 차이)
print(f"Disparate Impact: {classification_metric.disparate_impact():.3f}")
print(f"Statistical Parity Difference: {classification_metric.statistical_parity_difference():.3f}")

# 5단계: 공정성 대시보드 (실시간 모니터링)
from aif360.explainers import MetricTextExplainer

explainer = MetricTextExplainer(classification_metric)
print("\\n" + explainer.explain())`}</code>
            </pre>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
          <p className="text-sm font-semibold text-gray-900 dark:text-white mb-2">도구 선택 가이드:</p>
          <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li><strong>Fairlearn</strong>: 빠른 적용, scikit-learn 통합, Microsoft 지원</li>
            <li><strong>IBM AIF360</strong>: 70+ 편향 지표, Pre/In/Post-processing 알고리즘, 법률 준수</li>
            <li><strong>Google What-If Tool</strong>: 시각적 탐색, TensorFlow/Keras 모델</li>
          </ul>
        </div>
      </section>

      {/* 성별/인종 편향 완화 기법 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">편향 완화 3단계 전략</h2>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3 text-gray-900 dark:text-white">Pre-processing</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              데이터 수집·준비 단계에서 편향 제거
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li><strong>Reweighing</strong>: 소수 그룹 샘플 가중치 증가</li>
              <li><strong>Sampling</strong>: 균형 잡힌 데이터셋 구성</li>
              <li><strong>Data Augmentation</strong>: 부족한 그룹 데이터 생성</li>
            </ul>
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3 text-gray-900 dark:text-white">In-processing</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              모델 학습 시 공정성 제약 추가
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li><strong>Adversarial Debiasing</strong>: GAN 기반 편향 제거</li>
              <li><strong>Prejudice Remover</strong>: 정규화 손실 함수</li>
              <li><strong>Constraint Optimization</strong>: 공정성 제약 조건</li>
            </ul>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg">
            <h3 className="text-lg font-bold mb-3 text-gray-900 dark:text-white">Post-processing</h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              예측 결과 후처리로 공정성 달성
            </p>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
              <li><strong>Threshold Optimization</strong>: 그룹별 임계값 조정</li>
              <li><strong>Equalized Odds</strong>: TPR/FPR 균등화</li>
              <li><strong>Calibration</strong>: 그룹별 확률 보정</li>
            </ul>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 공정성 지표 & 프레임워크',
            icon: 'docs' as const,
            color: 'border-rose-500',
            items: [
              {
                title: 'Fairness Definitions Explained (NIST)',
                url: 'https://pages.nist.gov/privacy_collaborative_research/fairness/definitions.html',
                description: '21개 공정성 정의 비교 분석 (NIST 미국 표준기술연구소)'
              },
              {
                title: 'Fairness Indicators (Google)',
                url: 'https://www.tensorflow.org/responsible_ai/fairness_indicators/guide',
                description: 'TensorFlow 기반 공정성 평가 도구'
              },
              {
                title: 'Aequitas Toolkit',
                url: 'http://aequitas.dssg.io/',
                description: '감사·편향 평가 오픈소스 (시카고대 DSSG)'
              },
              {
                title: 'EU AI Act Fairness Requirements',
                url: 'https://artificialintelligenceact.eu/',
                description: 'EU AI Act의 고위험 시스템 공정성 요구사항 (2024.08 발효)'
              }
            ]
          },
          {
            title: '📖 핵심 연구 논문',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Machine Bias (ProPublica, 2016)',
                url: 'https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing',
                description: 'COMPAS 알고리즘 편향 조사 - AI 공정성 논의의 시발점'
              },
              {
                title: 'Fairness and Machine Learning (Book)',
                url: 'https://fairmlbook.org/',
                description: 'Barocas, Hardt, Narayanan의 공정성 교과서 (무료 공개)'
              },
              {
                title: 'Fairness Constraints: Mechanisms for Fair Classification',
                url: 'https://arxiv.org/abs/1507.05259',
                description: 'Microsoft Research - 공정성 제약 최적화 방법론'
              },
              {
                title: 'Inherent Trade-Offs in Algorithmic Fairness',
                url: 'https://arxiv.org/abs/1609.05807',
                description: '공정성 정의 간 수학적 불가능성 증명 (Impossibility Theorem)'
              },
              {
                title: 'Gender Shades: Intersectional Accuracy Disparities',
                url: 'http://gendershades.org/',
                description: 'MIT - 얼굴 인식 AI의 성별·인종 편향 연구 (Joy Buolamwini)'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Fairlearn Documentation',
                url: 'https://fairlearn.org/',
                description: 'Microsoft Fairlearn 공식 문서 및 튜토리얼'
              },
              {
                title: 'IBM AI Fairness 360',
                url: 'https://aif360.mybluemix.net/',
                description: '70+ 공정성 지표, 10+ 완화 알고리즘'
              },
              {
                title: 'Google What-If Tool',
                url: 'https://pair-code.github.io/what-if-tool/',
                description: 'TensorBoard 통합 공정성 시각화 도구'
              },
              {
                title: 'LinkedIn Fairness Toolkit (LiFT)',
                url: 'https://github.com/linkedin/lift',
                description: 'Spark 기반 대규모 공정성 측정 (LinkedIn 오픈소스)'
              },
              {
                title: 'Fairness Gym',
                url: 'https://github.com/google/ml-fairness-gym',
                description: 'Google - 장기적 공정성 영향 시뮬레이션'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
