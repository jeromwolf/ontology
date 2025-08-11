'use client'

import { useState } from 'react'
import { 
  Brain, Target, TrendingUp, GitBranch, BarChart3,
  CheckCircle, AlertCircle, Info, Activity,
  ChevronRight, Play, Zap, FileText, Settings
} from 'lucide-react'

interface ChapterProps {
  onComplete?: () => void
}

export default function Chapter4({ onComplete }: ChapterProps) {
  const [activeAlgorithm, setActiveAlgorithm] = useState('logistic')
  const [showMetrics, setShowMetrics] = useState(false)

  return (
    <div className="space-y-8">
      {/* 챕터 헤더 */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">지도학습 - 분류와 회귀</h1>
        <p className="text-xl text-gray-600 dark:text-gray-400">
          로지스틱 회귀, SVM, 랜덤 포레스트, XGBoost로 예측 모델 만들기
        </p>
      </div>

      {/* 학습 목표 */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Target className="text-blue-600" />
          학습 목표
        </h2>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">분류와 회귀 문제 구분</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">언제 어떤 접근법을 사용할지</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">주요 알고리즘 이해</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">작동 원리와 장단점</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">모델 평가와 선택</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">적절한 평가 지표 활용</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="text-green-500 mt-1 flex-shrink-0" />
            <div>
              <p className="font-semibold">하이퍼파라미터 튜닝</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">GridSearch, RandomSearch, Bayesian</p>
            </div>
          </div>
        </div>
      </div>

      {/* 1. 지도학습 개요 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">1. 지도학습이란?</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Brain className="text-blue-500" />
            지도학습의 핵심 개념
          </h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <strong>지도학습(Supervised Learning)</strong>은 입력(X)과 정답(y)이 쌍으로 주어진 데이터로부터 
            입력과 출력 간의 관계를 학습하는 머신러닝 방법입니다. 새로운 입력에 대해 출력을 예측하는 것이 목표입니다.
          </p>
          
          <div className="grid md:grid-cols-2 gap-6 mt-6">
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-blue-700 dark:text-blue-400 mb-3">🎯 분류 (Classification)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                이산적인 클래스를 예측하는 문제
              </p>
              <ul className="space-y-1 text-sm">
                <li>• 이메일 스팸 필터링</li>
                <li>• 질병 진단</li>
                <li>• 고객 이탈 예측</li>
                <li>• 이미지 분류</li>
              </ul>
            </div>
            
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-green-700 dark:text-green-400 mb-3">📈 회귀 (Regression)</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                연속적인 수치를 예측하는 문제
              </p>
              <ul className="space-y-1 text-sm">
                <li>• 주택 가격 예측</li>
                <li>• 매출 예측</li>
                <li>• 온도 예측</li>
                <li>• 주가 예측</li>
              </ul>
            </div>
          </div>
        </div>

        {/* 학습 과정 */}
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
          <h3 className="text-xl font-semibold mb-4">지도학습 워크플로우</h3>
          
          <div className="space-y-3">
            {[
              { step: "데이터 수집", desc: "레이블된 훈련 데이터 준비" },
              { step: "데이터 전처리", desc: "정제, 정규화, 특성 추출" },
              { step: "훈련/검증 분할", desc: "보통 70:30 또는 80:20 비율" },
              { step: "모델 선택", desc: "문제에 적합한 알고리즘 선택" },
              { step: "모델 학습", desc: "훈련 데이터로 파라미터 최적화" },
              { step: "모델 평가", desc: "검증 세트로 성능 측정" },
              { step: "하이퍼파라미터 튜닝", desc: "최적의 설정 찾기" },
              { step: "최종 평가", desc: "테스트 세트로 일반화 성능 확인" }
            ].map((item, idx) => (
              <div key={idx} className="flex items-center gap-3">
                <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold text-sm">
                  {idx + 1}
                </div>
                <div>
                  <span className="font-semibold">{item.step}:</span> {item.desc}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 2. 분류 알고리즘 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">2. 주요 분류 알고리즘</h2>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 mb-6">
          <div className="flex gap-2 mb-4 flex-wrap">
            {['logistic', 'svm', 'tree', 'rf', 'xgboost'].map((algo) => (
              <button
                key={algo}
                onClick={() => setActiveAlgorithm(algo)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  activeAlgorithm === algo
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                {algo === 'logistic' && '로지스틱 회귀'}
                {algo === 'svm' && 'SVM'}
                {algo === 'tree' && '결정 트리'}
                {algo === 'rf' && '랜덤 포레스트'}
                {algo === 'xgboost' && 'XGBoost'}
              </button>
            ))}
          </div>

          {activeAlgorithm === 'logistic' && (
            <div>
              <h3 className="text-lg font-semibold mb-3 text-blue-600 dark:text-blue-400">로지스틱 회귀 (Logistic Regression)</h3>
              
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div>
                  <h4 className="font-semibold mb-2">원리</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    선형 회귀에 시그모이드 함수를 적용하여 0과 1 사이의 확률값으로 변환
                  </p>
                  <div className="mt-2 p-3 bg-gray-50 dark:bg-gray-900/50 rounded">
                    <p className="font-mono text-sm">p = 1 / (1 + e^(-z))</p>
                    <p className="font-mono text-sm">z = β₀ + β₁x₁ + ... + βₙxₙ</p>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">특징</h4>
                  <ul className="space-y-1 text-sm">
                    <li>✓ 선형 결정 경계</li>
                    <li>✓ 확률적 해석 가능</li>
                    <li>✓ 특성 중요도 파악 용이</li>
                    <li>✓ 계산 효율적</li>
                  </ul>
                </div>
              </div>
              
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-sm text-gray-300 overflow-x-auto">
{`from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 데이터 준비
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링 (로지스틱 회귀는 스케일에 민감)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
log_reg = LogisticRegression(
    penalty='l2',           # 규제 종류 (l1, l2, elasticnet)
    C=1.0,                 # 규제 강도 (작을수록 강한 규제)
    solver='lbfgs',        # 최적화 알고리즘
    max_iter=1000,
    random_state=42
)
log_reg.fit(X_train_scaled, y_train)

# 예측
y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\\n", classification_report(y_test, y_pred))

# 계수 확인 (특성 중요도)
coefficients = pd.DataFrame({
    'feature': feature_names,
    'coefficient': log_reg.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)`}</pre>
              </div>
            </div>
          )}

          {activeAlgorithm === 'svm' && (
            <div>
              <h3 className="text-lg font-semibold mb-3 text-purple-600 dark:text-purple-400">서포트 벡터 머신 (SVM)</h3>
              
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div>
                  <h4 className="font-semibold mb-2">원리</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    클래스 간 마진을 최대화하는 초평면을 찾는 알고리즘. 커널 트릭으로 비선형 분류 가능
                  </p>
                  <div className="mt-2">
                    <p className="text-sm font-semibold">주요 커널:</p>
                    <ul className="text-sm space-y-1">
                      <li>• Linear: K(x,y) = x·y</li>
                      <li>• RBF: K(x,y) = exp(-γ||x-y||²)</li>
                      <li>• Polynomial: K(x,y) = (γx·y + r)^d</li>
                    </ul>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">특징</h4>
                  <ul className="space-y-1 text-sm">
                    <li>✓ 고차원 데이터에 효과적</li>
                    <li>✓ 비선형 분류 가능</li>
                    <li>✓ 이상치에 강건</li>
                    <li>✗ 대용량 데이터에 느림</li>
                    <li>✗ 확률 추정 어려움</li>
                  </ul>
                </div>
              </div>
              
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-sm text-gray-300 overflow-x-auto">
{`from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# SVM은 스케일링 필수!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 기본 SVM
svm = SVC(
    kernel='rbf',          # 커널 종류
    C=1.0,                 # 규제 파라미터
    gamma='scale',         # RBF 커널 계수
    probability=True,      # 확률 추정 활성화
    random_state=42
)

# GridSearch로 최적 파라미터 찾기
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(
    svm, param_grid, 
    cv=5,                  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,            # 모든 CPU 코어 사용
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# 최적 모델
best_svm = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# 예측 및 평가
y_pred = best_svm.predict(X_test_scaled)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")`}</pre>
              </div>
            </div>
          )}

          {activeAlgorithm === 'tree' && (
            <div>
              <h3 className="text-lg font-semibold mb-3 text-green-600 dark:text-green-400">결정 트리 (Decision Tree)</h3>
              
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div>
                  <h4 className="font-semibold mb-2">원리</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    특성을 기준으로 데이터를 재귀적으로 분할하여 트리 구조를 만드는 알고리즘
                  </p>
                  <div className="mt-2">
                    <p className="text-sm font-semibold">분할 기준:</p>
                    <ul className="text-sm space-y-1">
                      <li>• Gini: 불순도 측정</li>
                      <li>• Entropy: 정보 이득</li>
                    </ul>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">특징</h4>
                  <ul className="space-y-1 text-sm">
                    <li>✓ 해석 가능한 모델</li>
                    <li>✓ 비선형 관계 포착</li>
                    <li>✓ 전처리 불필요</li>
                    <li>✗ 과적합 경향</li>
                    <li>✗ 불안정성</li>
                  </ul>
                </div>
              </div>
              
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-sm text-gray-300 overflow-x-auto">
{`from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 결정 트리 모델
dt = DecisionTreeClassifier(
    criterion='gini',       # 분할 기준
    max_depth=5,           # 최대 깊이 제한
    min_samples_split=20,  # 분할 최소 샘플 수
    min_samples_leaf=10,   # 리프 노드 최소 샘플 수
    max_features='sqrt',   # 분할 시 고려할 특성 수
    random_state=42
)

dt.fit(X_train, y_train)

# 특성 중요도
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# 트리 시각화
plt.figure(figsize=(20, 10))
plot_tree(dt, 
          feature_names=feature_names,
          class_names=['Class 0', 'Class 1'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()

# 과적합 방지를 위한 가지치기
# cost_complexity_pruning
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]

# 각 alpha 값에 대한 정확도
train_scores = []
val_scores = []

for ccp_alpha in ccp_alphas:
    dt_pruned = DecisionTreeClassifier(
        random_state=42, 
        ccp_alpha=ccp_alpha
    )
    dt_pruned.fit(X_train, y_train)
    train_scores.append(dt_pruned.score(X_train, y_train))
    val_scores.append(dt_pruned.score(X_val, y_val))

# 최적 alpha 선택
best_alpha = ccp_alphas[np.argmax(val_scores)]`}</pre>
              </div>
            </div>
          )}

          {activeAlgorithm === 'rf' && (
            <div>
              <h3 className="text-lg font-semibold mb-3 text-orange-600 dark:text-orange-400">랜덤 포레스트 (Random Forest)</h3>
              
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div>
                  <h4 className="font-semibold mb-2">원리</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    여러 결정 트리를 학습하고 그 예측을 평균(회귀) 또는 투표(분류)로 결합하는 앙상블 방법
                  </p>
                  <div className="mt-2">
                    <p className="text-sm font-semibold">핵심 기법:</p>
                    <ul className="text-sm space-y-1">
                      <li>• Bagging: 부트스트랩 샘플링</li>
                      <li>• Random Subspace: 특성 무작위 선택</li>
                    </ul>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">특징</h4>
                  <ul className="space-y-1 text-sm">
                    <li>✓ 과적합에 강건</li>
                    <li>✓ 높은 정확도</li>
                    <li>✓ 병렬 처리 가능</li>
                    <li>✓ 특성 중요도 제공</li>
                    <li>✗ 해석 어려움</li>
                  </ul>
                </div>
              </div>
              
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-sm text-gray-300 overflow-x-auto">
{`from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# 랜덤 포레스트 모델
rf = RandomForestClassifier(
    n_estimators=100,      # 트리 개수
    max_depth=None,        # 트리 최대 깊이
    min_samples_split=2,   # 노드 분할 최소 샘플
    min_samples_leaf=1,    # 리프 노드 최소 샘플
    max_features='sqrt',   # 각 분할에서 고려할 특성 수
    bootstrap=True,        # 부트스트랩 샘플링
    oob_score=True,        # OOB 점수 계산
    n_jobs=-1,            # 병렬 처리
    random_state=42
)

# RandomizedSearchCV로 하이퍼파라미터 튜닝
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    rf, param_dist,
    n_iter=20,            # 시도할 조합 수
    cv=5,
    scoring='roc_auc',    # AUC-ROC 점수 사용
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

# 최적 모델
best_rf = random_search.best_estimator_
print(f"Best parameters: {random_search.best_params_}")
print(f"OOB Score: {best_rf.oob_score_:.4f}")

# 특성 중요도 시각화
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(15), 
            x='importance', y='feature')
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.show()

# 개별 트리 예측의 분산 확인
tree_predictions = np.array([tree.predict_proba(X_test)[:, 1] 
                            for tree in best_rf.estimators_])
prediction_std = np.std(tree_predictions, axis=0)
print(f"Average prediction uncertainty: {np.mean(prediction_std):.4f}")`}</pre>
              </div>
            </div>
          )}

          {activeAlgorithm === 'xgboost' && (
            <div>
              <h3 className="text-lg font-semibold mb-3 text-red-600 dark:text-red-400">XGBoost</h3>
              
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div>
                  <h4 className="font-semibold mb-2">원리</h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    그래디언트 부스팅의 최적화된 구현. 순차적으로 약한 학습기를 추가하여 이전 모델의 오차를 보정
                  </p>
                  <div className="mt-2">
                    <p className="text-sm font-semibold">핵심 특징:</p>
                    <ul className="text-sm space-y-1">
                      <li>• 정규화로 과적합 방지</li>
                      <li>• 병렬 처리 최적화</li>
                      <li>• 결측치 자동 처리</li>
                    </ul>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">장단점</h4>
                  <ul className="space-y-1 text-sm">
                    <li>✓ 최고 수준의 성능</li>
                    <li>✓ 빠른 학습 속도</li>
                    <li>✓ 다양한 목적함수</li>
                    <li>✓ 특성 중요도</li>
                    <li>✗ 많은 하이퍼파라미터</li>
                  </ul>
                </div>
              </div>
              
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-sm text-gray-300 overflow-x-auto">
{`import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import optuna  # Bayesian optimization

# XGBoost DMatrix 형식으로 변환
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# 기본 파라미터
base_params = {
    'objective': 'binary:logistic',  # 이진 분류
    'eval_metric': 'auc',           # 평가 지표
    'tree_method': 'hist',          # 히스토그램 기반 (빠름)
    'device': 'cuda',               # GPU 사용 (가능한 경우)
    'random_state': 42
}

# Optuna를 사용한 베이지안 최적화
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }
    
    # Cross-validation
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
        
        model = xgb.XGBClassifier(**params, **base_params)
        model.fit(
            X_cv_train, y_cv_train,
            eval_set=[(X_cv_val, y_cv_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        cv_scores.append(model.best_score)
    
    return np.mean(cv_scores)

# Optuna 스터디 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, n_jobs=-1)

# 최적 파라미터로 최종 모델 학습
best_params = study.best_params
xgb_model = xgb.XGBClassifier(**best_params, **base_params)

# 조기 종료와 함께 학습
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=True
)

# SHAP으로 모델 해석
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)`}</pre>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 3. 회귀 알고리즘 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">3. 회귀 알고리즘</h2>
        
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl mb-6">
          <h3 className="text-xl font-semibold mb-4">회귀 모델 종합 비교</h3>
          
          <div className="bg-gray-900 rounded-lg p-4">
            <pre className="text-sm text-gray-300 overflow-x-auto">
{`from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# 여러 회귀 모델 비교
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0)
}

results = []

for name, model in models.items():
    # 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 평가
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })

# 결과 정리
results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
print(results_df)

# 잔차 분석
best_model = models['Random Forest']  # 예시
y_pred_best = best_model.predict(X_test)
residuals = y_test - y_pred_best

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 실제 vs 예측
axes[0, 0].scatter(y_test, y_pred_best, alpha=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title('Actual vs Predicted')

# 2. 잔차 플롯
axes[0, 1].scatter(y_pred_best, residuals, alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')

# 3. 잔차 히스토그램
axes[1, 0].hist(residuals, bins=30, edgecolor='black')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residual Distribution')

# 4. Q-Q 플롯
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()`}</pre>
          </div>
        </div>
      </section>

      {/* 4. 모델 평가 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">4. 모델 평가 지표</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* 분류 지표 */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 text-blue-600 dark:text-blue-400">분류 평가 지표</h3>
            
            <div className="space-y-3">
              <div>
                <h4 className="font-semibold text-sm">정확도 (Accuracy)</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">전체 중 올바르게 예측한 비율</p>
                <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  (TP + TN) / (TP + TN + FP + FN)
                </code>
              </div>
              
              <div>
                <h4 className="font-semibold text-sm">정밀도 (Precision)</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">양성 예측 중 실제 양성 비율</p>
                <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  TP / (TP + FP)
                </code>
              </div>
              
              <div>
                <h4 className="font-semibold text-sm">재현율 (Recall)</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">실제 양성 중 올바르게 예측한 비율</p>
                <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  TP / (TP + FN)
                </code>
              </div>
              
              <div>
                <h4 className="font-semibold text-sm">F1 Score</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">정밀도와 재현율의 조화평균</p>
                <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  2 × (Precision × Recall) / (Precision + Recall)
                </code>
              </div>
              
              <div>
                <h4 className="font-semibold text-sm">AUC-ROC</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">ROC 곡선 아래 면적 (0.5~1.0)</p>
              </div>
            </div>
          </div>

          {/* 회귀 지표 */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 text-green-600 dark:text-green-400">회귀 평가 지표</h3>
            
            <div className="space-y-3">
              <div>
                <h4 className="font-semibold text-sm">MAE (Mean Absolute Error)</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">예측 오차의 절댓값 평균</p>
                <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  Σ|y - ŷ| / n
                </code>
              </div>
              
              <div>
                <h4 className="font-semibold text-sm">MSE (Mean Squared Error)</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">예측 오차의 제곱 평균</p>
                <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  Σ(y - ŷ)² / n
                </code>
              </div>
              
              <div>
                <h4 className="font-semibold text-sm">RMSE (Root MSE)</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">MSE의 제곱근, 원래 단위</p>
                <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  √(MSE)
                </code>
              </div>
              
              <div>
                <h4 className="font-semibold text-sm">R² (결정계수)</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">모델이 설명하는 분산 비율</p>
                <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  1 - (SS_res / SS_tot)
                </code>
              </div>
              
              <div>
                <h4 className="font-semibold text-sm">MAPE</h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">평균 절대 백분율 오차</p>
                <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                  Σ|y - ŷ|/|y| × 100 / n
                </code>
              </div>
            </div>
          </div>
        </div>

        {/* 평가 코드 예시 */}
        <div className="bg-gray-900 rounded-xl p-6 mt-6">
          <h3 className="text-white font-semibold mb-4">종합 평가 코드</h3>
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className="mb-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            {showMetrics ? '숨기기' : '평가 코드 보기'}
          </button>
          
          {showMetrics && (
            <pre className="bg-gray-800 p-4 rounded-lg overflow-x-auto">
              <code className="text-sm text-gray-300">{`from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# 1. 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 2. 분류 리포트
print(classification_report(y_test, y_pred))

# 3. ROC 곡선
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 4. Precision-Recall 곡선
precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.show()

# 5. 임계값 조정
thresholds = np.arange(0.1, 1.0, 0.1)
scores = []

for threshold in thresholds:
    y_pred_thresh = (y_proba[:, 1] >= threshold).astype(int)
    score = {
        'threshold': threshold,
        'precision': precision_score(y_test, y_pred_thresh),
        'recall': recall_score(y_test, y_pred_thresh),
        'f1': f1_score(y_test, y_pred_thresh)
    }
    scores.append(score)

scores_df = pd.DataFrame(scores)
scores_df.plot(x='threshold', y=['precision', 'recall', 'f1'])
plt.title('Performance Metrics by Threshold')
plt.show()`}</code>
            </pre>
          )}
        </div>
      </section>

      {/* 5. 실전 팁 */}
      <section>
        <h2 className="text-3xl font-bold mb-6">5. 실전 팁과 체크리스트</h2>
        
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl">
          <h3 className="text-xl font-semibold mb-4">모델 선택 가이드</h3>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">데이터 특성별 추천</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-purple-400 mt-0.5" />
                  <span><strong>선형 관계:</strong> 로지스틱/선형 회귀</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-purple-400 mt-0.5" />
                  <span><strong>비선형 관계:</strong> SVM(RBF), 트리 기반</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-purple-400 mt-0.5" />
                  <span><strong>고차원 데이터:</strong> SVM, 규제된 회귀</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-purple-400 mt-0.5" />
                  <span><strong>대용량 데이터:</strong> SGD, XGBoost</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-purple-400 mt-0.5" />
                  <span><strong>해석 필요:</strong> 선형 모델, 결정 트리</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">성능 개선 체크리스트</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>데이터 품질 검증</span>
                </li>
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>특성 엔지니어링</span>
                </li>
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>교차 검증 수행</span>
                </li>
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>하이퍼파라미터 튜닝</span>
                </li>
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>앙상블 방법 시도</span>
                </li>
                <li className="flex items-center gap-2">
                  <input type="checkbox" className="rounded" />
                  <span>과적합 확인</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 프로젝트 */}
      <section className="mt-12">
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-8 rounded-xl">
          <h2 className="text-2xl font-bold mb-4">🎯 실전 프로젝트: 고객 이탈 예측</h2>
          <p className="mb-6">
            배운 알고리즘들을 활용해 실제 비즈니스 문제를 해결해보세요. 
            통신사 고객 데이터를 사용해 이탈 가능성이 높은 고객을 예측하는 모델을 만들어봅시다.
          </p>
          <div className="flex gap-4">
            <button 
              onClick={onComplete}
              className="bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              챕터 완료하기
            </button>
            <button className="bg-blue-500 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-400 transition-colors">
              프로젝트 시작하기
            </button>
          </div>
        </div>
      </section>
    </div>
  )
}