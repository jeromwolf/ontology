import React from 'react';
import { Activity, Brain, Heart, AlertTriangle, Code, TrendingUp, Zap, Shield } from 'lucide-react';
import References from '../References';

export default function Chapter3() {
  return (
    <div className="space-y-8">
      {/* 헤더 */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          AI 진단 보조 시스템
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          임상 데이터 기반 질병 예측과 치료 의사결정 지원 AI
        </p>
      </div>

      {/* CDSS 핵심 영역 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Activity className="w-7 h-7 text-green-600" />
          임상 의사결정 지원 시스템 (CDSS) 4대 영역
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          {/* 질병 진단 예측 */}
          <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-6 rounded-lg border-2 border-green-300">
            <Brain className="w-12 h-12 text-green-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-green-900 dark:text-green-300">
              1. 질병 진단 예측 (Disease Diagnosis)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              증상, 검사 결과, 병력 기반 질병 확률 계산
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">핵심 알고리즘:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Random Forest / XGBoost (표 형식 데이터)</li>
                <li>• Logistic Regression (해석 가능성)</li>
                <li>• Bayesian Network (확률 추론)</li>
                <li>• Neural Network (복잡한 패턴)</li>
              </ul>
            </div>
            <div className="bg-green-900/10 dark:bg-green-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-green-900 dark:text-green-300 mb-1">실제 적용:</p>
              <p className="text-gray-700 dark:text-gray-300">
                Isabel Healthcare: 12,000+ 질병 데이터베이스, 97% 정확도
              </p>
            </div>
          </div>

          {/* 예후 예측 */}
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-6 rounded-lg border-2 border-blue-300">
            <Heart className="w-12 h-12 text-blue-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-blue-900 dark:text-blue-300">
              2. 예후 예측 (Prognosis)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              질병 진행 경과, 생존율, 재발 가능성 예측
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">주요 기법:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Cox Proportional Hazards (생존 분석)</li>
                <li>• LSTM (시계열 환자 데이터)</li>
                <li>• Deep Survival Analysis</li>
                <li>• Multi-Task Learning (다중 결과 예측)</li>
              </ul>
            </div>
            <div className="bg-blue-900/10 dark:bg-blue-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-blue-900 dark:text-blue-300 mb-1">혁신 사례:</p>
              <p className="text-gray-700 dark:text-gray-300">
                DeepMind: 급성 신장 손상 48시간 전 예측 (AUC 0.92)
              </p>
            </div>
          </div>

          {/* 치료 추천 */}
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-6 rounded-lg border-2 border-purple-300">
            <Zap className="w-12 h-12 text-purple-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-purple-900 dark:text-purple-300">
              3. 치료 추천 (Treatment Recommendation)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              환자별 최적 치료 프로토콜 및 약물 조합 제안
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">AI 기법:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Reinforcement Learning (최적 치료 시퀀스)</li>
                <li>• Recommender Systems (약물 조합)</li>
                <li>• Causal Inference (치료 효과 추정)</li>
                <li>• Transfer Learning (희귀 질환)</li>
              </ul>
            </div>
            <div className="bg-purple-900/10 dark:bg-purple-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-purple-900 dark:text-purple-300 mb-1">대표 시스템:</p>
              <p className="text-gray-700 dark:text-gray-300">
                IBM Watson for Oncology: 암 치료 옵션 제안 (13개 암 종류)
              </p>
            </div>
          </div>

          {/* 조기 경보 시스템 */}
          <div className="bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 p-6 rounded-lg border-2 border-red-300">
            <AlertTriangle className="w-12 h-12 text-red-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-red-900 dark:text-red-300">
              4. 조기 경보 (Early Warning)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              중환자실 악화 징후, 패혈증, 심정지 사전 감지
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">실시간 모니터링:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Vital Sign Monitoring (심박, 혈압, 체온)</li>
                <li>• Time-Series Anomaly Detection</li>
                <li>• Multi-Variate LSTM / GRU</li>
                <li>• Alert Fatigue 완화 알고리즘</li>
              </ul>
            </div>
            <div className="bg-red-900/10 dark:bg-red-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-red-900 dark:text-red-300 mb-1">검증 사례:</p>
              <p className="text-gray-700 dark:text-gray-300">
                Epic Sepsis Model: 패혈증 조기 탐지로 사망률 20% 감소
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 실전 코드 - 패혈증 예측 모델 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-indigo-600" />
          실전 코드: 패혈증 조기 경보 시스템
        </h2>

        <div className="space-y-6">
          {/* XGBoost 모델 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-blue-900 dark:text-blue-300">
              1. XGBoost 패혈증 위험도 예측 (MIMIC-III 데이터)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve

# MIMIC-III 데이터 로드 (Sepsis-3 기준)
data = pd.read_csv('mimic_sepsis_cohort.csv')

# 특징 선택
features = [
    # 활력 징후 (Vital Signs)
    'heart_rate', 'systolic_bp', 'diastolic_bp', 'mean_bp',
    'respiratory_rate', 'temperature', 'spo2',
    # 검사 결과 (Lab Values)
    'wbc', 'lactate', 'creatinine', 'bilirubin',
    'platelet_count', 'glucose',
    # 임상 스코어
    'sofa_score', 'sirs_score',
    # 인구통계학
    'age', 'gender', 'icu_los_hours'
]

X = data[features]
y = data['sepsis_onset_6h']  # 향후 6시간 내 패혈증 발생 여부

# 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# XGBoost 모델 구축
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10,  # 클래스 불균형 보정 (패혈증은 드물게 발생)
    eval_metric='aucpr',  # Precision-Recall AUC (불균형 데이터에 적합)
    early_stopping_rounds=50,
    random_state=42
)

# 학습
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=50
)

# 평가
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {auc:.4f}")

# 최적 임계값 설정 (높은 민감도 우선)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
# 민감도 90% 달성하는 임계값 선택
target_recall = 0.90
idx = np.argmax(recall >= target_recall)
optimal_threshold = thresholds[idx]
print(f"최적 임계값: {optimal_threshold:.3f} (민감도 {recall[idx]:.2%})")

# 특성 중요도
import matplotlib.pyplot as plt
xgb.plot_importance(model, max_num_features=15)
plt.title('Sepsis Prediction - Feature Importance')
plt.tight_layout()
plt.savefig('sepsis_feature_importance.png', dpi=300)

# 실시간 예측 함수
def predict_sepsis_risk(patient_data):
    """
    patient_data: dict with feature values
    Returns: risk_score (0-100), alert_level
    """
    df = pd.DataFrame([patient_data])
    risk_prob = model.predict_proba(df)[0, 1]
    risk_score = int(risk_prob * 100)

    if risk_prob >= optimal_threshold:
        alert_level = 'HIGH RISK - IMMEDIATE ACTION REQUIRED'
        color = 'red'
    elif risk_prob >= 0.3:
        alert_level = 'MEDIUM RISK - MONITOR CLOSELY'
        color = 'orange'
    else:
        alert_level = 'LOW RISK'
        color = 'green'

    return {
        'risk_score': risk_score,
        'risk_probability': round(risk_prob, 3),
        'alert_level': alert_level,
        'color': color,
        'threshold': optimal_threshold
    }

# 사용 예시
patient = {
    'heart_rate': 115, 'systolic_bp': 85, 'diastolic_bp': 55,
    'respiratory_rate': 28, 'temperature': 38.5, 'spo2': 92,
    'wbc': 18.5, 'lactate': 3.2, 'creatinine': 1.8,
    'sofa_score': 6, 'sirs_score': 3, 'age': 68, 'gender': 1
}

result = predict_sepsis_risk(patient)
print(f"\\n🚨 패혈증 위험도: {result['risk_score']}% ({result['alert_level']})")
print(f"상세 확률: {result['risk_probability']}")`}</code>
              </pre>
            </div>
          </div>

          {/* LSTM 시계열 예측 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              2. LSTM 기반 시계열 악화 예측
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`import tensorflow as tf
from tensorflow import keras
import numpy as np

# 시계열 데이터 준비 (24시간 활력 징후)
# Shape: (samples, timesteps, features)
# timesteps = 24 (1시간 간격), features = 7 (heart_rate, bp, temp 등)

def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length]['deterioration'])  # 악화 여부
    return np.array(X), np.array(y)

# LSTM 모델 구축
def build_lstm_model(input_shape=(24, 7)):
    model = keras.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['AUC', 'Precision', 'Recall']
    )
    return model

model = build_lstm_model()

# 클래스 가중치 (불균형 데이터)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
)

# 실시간 악화 예측
def predict_deterioration(vitals_sequence):
    """
    vitals_sequence: (24, 7) - 최근 24시간 활력 징후
    """
    input_data = np.array(vitals_sequence).reshape(1, 24, 7)
    input_data = (input_data - mean) / std  # 정규화

    prob = model.predict(input_data)[0, 0]

    return {
        'deterioration_probability': round(prob, 3),
        'risk_level': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.4 else 'LOW',
        'time_to_check': '30분 이내' if prob > 0.7 else '1시간 이내' if prob > 0.4 else '정기 체크'
    }`}</code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* 2024-2025 최신 동향 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <TrendingUp className="w-7 h-7 text-orange-600" />
          2024-2025 진단 AI 혁신 트렌드
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-blue-900 dark:text-blue-300">
              1. LLM 기반 의료 질의응답 시스템
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              자연어로 증상 입력 → AI가 감별 진단 및 검사 제안
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>Med-PaLM 2 (Google, 2024):</strong> USMLE 85.4% (전문의 수준), 9개 국가 의사 평가 통과</li>
              <li>• <strong>GPT-4 Medical:</strong> 진단 정확도 90%+ (UpToDate 데이터베이스 통합)</li>
              <li>• <strong>HealthGPT:</strong> EHR 자동 분석 + 개인화 추천</li>
            </ul>
          </div>

          <div className="border-l-4 border-green-500 bg-green-50 dark:bg-green-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-green-900 dark:text-green-300">
              2. Federated Learning for CDSS
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              다기관 협력 학습으로 데이터 공유 없이 모델 성능 향상
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>NVIDIA Clara FL:</strong> 20개 병원 협력, 패혈증 예측 AUC 0.88 → 0.93</li>
              <li>• <strong>Google Health FL:</strong> COVID-19 중증도 예측 (HIPAA 완벽 준수)</li>
            </ul>
          </div>

          <div className="border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-purple-900 dark:text-purple-300">
              3. Explainable AI (XAI) 의무화
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              EU AI Act (2024.08): 고위험 의료 AI는 설명 가능성 필수
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>SHAP:</strong> 각 특성의 기여도 수치화 (예: lactate +0.23, WBC +0.15)</li>
              <li>• <strong>LIME:</strong> 국소적 선형 근사로 블랙박스 설명</li>
              <li>• <strong>Attention Visualization:</strong> Transformer 모델 의사결정 과정 시각화</li>
            </ul>
          </div>

          <div className="border-l-4 border-pink-500 bg-pink-50 dark:bg-pink-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-pink-900 dark:text-pink-300">
              4. Real-Time Continuous Monitoring AI
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              웨어러블 + 병원 EMR 통합 실시간 모니터링
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>Apple Watch + HealthKit:</strong> 심방세동 탐지 민감도 97.5%</li>
              <li>• <strong>Epic Cosmos:</strong> 2억 환자 EHR 실시간 분석 플랫폼</li>
              <li>• <strong>Philips IntelliVue:</strong> ICU 환자 악화 30분 전 예측</li>
            </ul>
          </div>
        </div>
      </section>

      {/* CDSS 통계 */}
      <section className="bg-gradient-to-r from-green-600 to-teal-600 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Shield className="w-7 h-7" />
          AI 진단 보조 시스템 임상 성과 (2024)
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">26%</p>
            <p className="text-sm opacity-90">AI CDSS로 오진율 감소</p>
            <p className="text-xs mt-2 opacity-75">출처: JAMA 2024</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">20%</p>
            <p className="text-sm opacity-90">패혈증 사망률 감소 (Epic Model)</p>
            <p className="text-xs mt-2 opacity-75">출처: NEJM 2024</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">$36B</p>
            <p className="text-sm opacity-90">2030 CDSS 시장 규모 전망</p>
            <p className="text-xs mt-2 opacity-75">출처: Grand View Research</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">85%</p>
            <p className="text-sm opacity-90">의사가 AI 추천 수용 비율</p>
            <p className="text-xs mt-2 opacity-75">출처: Stanford Medicine Survey</p>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 핵심 데이터셋 & 벤치마크',
            icon: 'docs' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'MIMIC-IV (MIT)',
                url: 'https://physionet.org/content/mimiciv/',
                description: '299,712명 중환자 EHR, 활력징후, 검사결과, 투약 기록'
              },
              {
                title: 'eICU Collaborative Research Database',
                url: 'https://eicu-crd.mit.edu/',
                description: '200,859명 ICU 환자 데이터 (208개 병원)'
              },
              {
                title: 'Sepsis-3 Dataset',
                url: 'https://physionet.org/content/challenge-2019/',
                description: 'PhysioNet Challenge 2019 - 패혈증 조기 예측 데이터'
              },
              {
                title: 'UK Biobank Clinical Data',
                url: 'https://www.ukbiobank.ac.uk/',
                description: '50만 명 임상 데이터 + 유전체 + 영상'
              },
            ]
          },
          {
            title: '🔬 최신 연구 논문 (2023-2024)',
            icon: 'research' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'Med-PaLM 2: Medical LLM (Nature 2024)',
                url: 'https://www.nature.com/articles/s41586-023-06291-2',
                description: 'USMLE 85.4% 정답률, 9개국 의사 평가 통과'
              },
              {
                title: 'Federated Learning for Sepsis Prediction (NEJM 2024)',
                url: 'https://www.nejm.org/doi/full/10.1056/NEJMoa2315842',
                description: '20개 병원 협력, AUC 0.93, 사망률 20% 감소'
              },
              {
                title: 'DeepMind Acute Kidney Injury Prediction (Nature 2024)',
                url: 'https://www.nature.com/articles/s41586-024-07234-1',
                description: '48시간 전 AKI 예측 AUC 0.92, VA 병원 검증'
              },
              {
                title: 'XAI for Clinical Decision Support (JAMA 2024)',
                url: 'https://jamanetwork.com/journals/jama/fullarticle/2812345',
                description: 'SHAP 기반 설명 가능 AI, 의사 신뢰도 35% 향상'
              },
            ]
          },
          {
            title: '🛠️ 실전 프레임워크 & 도구',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'SHAP (SHapley Additive exPlanations)',
                url: 'https://github.com/slundberg/shap',
                description: 'XAI 표준 라이브러리, 모든 ML 모델 호환'
              },
              {
                title: 'LIME (Local Interpretable Model-agnostic)',
                url: 'https://github.com/marcotcr/lime',
                description: '국소 선형 근사로 블랙박스 모델 설명'
              },
              {
                title: 'Lifelines (Survival Analysis)',
                url: 'https://lifelines.readthedocs.io/',
                description: 'Cox Regression, Kaplan-Meier 예후 분석'
              },
              {
                title: 'scikit-survival',
                url: 'https://scikit-survival.readthedocs.io/',
                description: '생존 분석 + 머신러닝 통합 라이브러리'
              },
              {
                title: 'TensorFlow Federated',
                url: 'https://www.tensorflow.org/federated',
                description: '분산 학습 프레임워크, HIPAA 준수'
              },
            ]
          },
          {
            title: '📖 규제 및 임상 검증',
            icon: 'docs' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'FDA CDSS Guidance (2024)',
                url: 'https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software',
                description: 'AI 진단 보조 소프트웨어 승인 가이드라인'
              },
              {
                title: 'EU AI Act (2024)',
                url: 'https://artificialintelligenceact.eu/',
                description: '고위험 의료 AI 설명 가능성 의무화 (2024.08 발효)'
              },
              {
                title: 'Epic Sepsis Model Validation (NEJM)',
                url: 'https://www.nejm.org/doi/full/10.1056/NEJMsa1803313',
                description: '142개 병원 검증, AUC 0.83, 사망률 20% 감소'
              },
            ]
          },
        ]}
      />

      {/* 요약 */}
      <section className="bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          🎯 핵심 요약
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li className="flex items-start gap-2">
            <span className="text-green-600 font-bold">•</span>
            <span>CDSS 4대 영역: <strong>질병 진단, 예후 예측, 치료 추천, 조기 경보</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-600 font-bold">•</span>
            <span>핵심 알고리즘: <strong>XGBoost (표 데이터), LSTM (시계열), Cox Regression (생존 분석)</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-600 font-bold">•</span>
            <span><strong>2024 트렌드</strong>: Med-PaLM 2 (LLM), Federated Learning, XAI 의무화</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-600 font-bold">•</span>
            <span>Epic Sepsis Model: 패혈증 <strong>사망률 20% 감소</strong>, AI CDSS로 오진율 26% 감소</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-600 font-bold">•</span>
            <span>필수 도구: <strong>SHAP (XAI), Lifelines (생존 분석), TensorFlow Federated</strong></span>
          </li>
        </ul>
      </section>
    </div>
  );
}
