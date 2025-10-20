import React from 'react';
import { Heart, Brain, Microscope, Activity, TrendingUp, Code, Database, AlertCircle } from 'lucide-react';
import References from '../References';

export default function Chapter1() {
  return (
    <div className="space-y-8">
      {/* 헤더 */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Medical AI 기초
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          의료 AI의 역사부터 최신 동향까지, 헬스케어 혁신의 모든 것
        </p>
      </div>

      {/* Medical AI 4대 핵심 영역 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Heart className="w-7 h-7 text-pink-600" />
          Medical AI의 4대 핵심 영역
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          {/* 의료 영상 분석 */}
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-6 rounded-lg border-2 border-blue-300">
            <Microscope className="w-12 h-12 text-blue-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-blue-900 dark:text-blue-300">
              1. 의료 영상 분석 (Medical Imaging)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              X-ray, CT, MRI 영상에서 질병 패턴 자동 탐지
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">핵심 기술:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• CNN (ResNet, DenseNet, EfficientNet)</li>
                <li>• U-Net / Mask R-CNN (영역 분할)</li>
                <li>• Vision Transformer (ViT)</li>
                <li>• Transfer Learning (ImageNet 사전학습)</li>
              </ul>
            </div>
            <div className="bg-blue-900/10 dark:bg-blue-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-blue-900 dark:text-blue-300 mb-1">FDA 승인 사례:</p>
              <p className="text-gray-700 dark:text-gray-300">
                Aidoc (뇌출혈 탐지), Zebra Medical (흉부 X-ray 분석) - 민감도 95%+
              </p>
            </div>
          </div>

          {/* 진단 보조 시스템 */}
          <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-6 rounded-lg border-2 border-green-300">
            <Activity className="w-12 h-12 text-green-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-green-900 dark:text-green-300">
              2. 진단 보조 시스템 (Clinical Decision Support)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              증상, 검사 결과 기반 질병 확률 예측 및 치료 제안
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">핵심 알고리즘:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Random Forest / XGBoost (표 형식 데이터)</li>
                <li>• LSTM / Transformer (시계열 생체신호)</li>
                <li>• Bayesian Network (확률 추론)</li>
                <li>• Ensemble Learning (다중 모델 결합)</li>
              </ul>
            </div>
            <div className="bg-green-900/10 dark:bg-green-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-green-900 dark:text-green-300 mb-1">실제 적용:</p>
              <p className="text-gray-700 dark:text-gray-300">
                IBM Watson Health (암 치료 추천), Epic Sepsis Model (패혈증 조기 경보)
              </p>
            </div>
          </div>

          {/* AI 신약 개발 */}
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-6 rounded-lg border-2 border-purple-300">
            <Brain className="w-12 h-12 text-purple-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-purple-900 dark:text-purple-300">
              3. AI 신약 개발 (Drug Discovery)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              신약 후보 물질 탐색, 합성, 임상시험 설계 자동화
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">핵심 도구:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• GAN / VAE (분자 구조 생성)</li>
                <li>• Graph Neural Network (약물-단백질 상호작용)</li>
                <li>• AlphaFold 2 (단백질 구조 예측)</li>
                <li>• Reinforcement Learning (최적 합성 경로)</li>
              </ul>
            </div>
            <div className="bg-purple-900/10 dark:bg-purple-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-purple-900 dark:text-purple-300 mb-1">혁신 사례:</p>
              <p className="text-gray-700 dark:text-gray-300">
                Insilico Medicine (특발성 폐섬유증 치료제 18개월 만에 임상 진입)
              </p>
            </div>
          </div>

          {/* 정밀 의료 */}
          <div className="bg-gradient-to-br from-pink-50 to-pink-100 dark:from-pink-900/20 dark:to-pink-800/20 p-6 rounded-lg border-2 border-pink-300">
            <Database className="w-12 h-12 text-pink-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-pink-900 dark:text-pink-300">
              4. 정밀 의료 (Precision Medicine)
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              유전체, 라이프스타일 데이터 기반 개인화 치료 설계
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">데이터 소스:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Genomic Data (WGS, WES, RNA-seq)</li>
                <li>• Proteomic / Metabolomic Data</li>
                <li>• EHR (Electronic Health Records)</li>
                <li>• Wearable Device Data (Apple Watch, Fitbit)</li>
              </ul>
            </div>
            <div className="bg-pink-900/10 dark:bg-pink-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-pink-900 dark:text-pink-300 mb-1">대표 프로젝트:</p>
              <p className="text-gray-700 dark:text-gray-300">
                NIH All of Us (100만 명 유전체 데이터), UK Biobank (50만 명)
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 실무 코드 예제 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-indigo-600" />
          실무 코드 예제
        </h2>

        <div className="space-y-6">
          {/* X-ray 분류 모델 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-blue-900 dark:text-blue-300">
              1. 흉부 X-ray 폐렴 분류 (PyTorch + ResNet50)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 사전학습된 ResNet50 로드 (ImageNet weights)
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 이진 분류: Normal vs Pneumonia

# 체크포인트 로드 (실제 환경)
checkpoint = torch.load('pneumonia_resnet50_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 추론 함수
def predict_pneumonia(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    labels = ['Normal', 'Pneumonia']
    confidence = probabilities[0][prediction].item() * 100

    return {
        'prediction': labels[prediction],
        'confidence': f'{confidence:.2f}%',
        'probabilities': {
            'Normal': f'{probabilities[0][0].item()*100:.2f}%',
            'Pneumonia': f'{probabilities[0][1].item()*100:.2f}%'
        }
    }

# 사용 예시
result = predict_pneumonia('chest_xray_sample.jpg')
print(f"진단: {result['prediction']} (확신도: {result['confidence']})")
# 출력: 진단: Pneumonia (확신도: 94.73%)`}</code>
              </pre>
            </div>
          </div>

          {/* ECG 이상 탐지 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              2. ECG 부정맥 탐지 (TensorFlow + 1D CNN)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1D CNN 모델 구축 (MIT-BIH Arrhythmia Dataset 기준)
def build_ecg_model(input_shape=(5000, 1)):
    model = keras.Sequential([
        keras.layers.Conv1D(64, kernel_size=7, activation='relu',
                            input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),

        keras.layers.Conv1D(128, kernel_size=5, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),

        keras.layers.Conv1D(256, kernel_size=3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling1D(),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation='softmax')  # 5개 부정맥 타입
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 실시간 ECG 신호 처리
def process_ecg_signal(ecg_signal):
    """
    ecg_signal: (5000,) - 5초 분량 ECG (sampling rate 1000Hz)
    """
    # 정규화
    signal_normalized = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    # 모델 입력 형태로 변환
    input_signal = signal_normalized.reshape(1, 5000, 1)

    # 예측
    model = keras.models.load_model('ecg_arrhythmia_model.h5')
    predictions = model.predict(input_signal)

    arrhythmia_types = ['Normal', 'Atrial Fibrillation', 'Ventricular Tachycardia',
                        'Premature Ventricular Contraction', 'Sinus Bradycardia']

    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100

    # 위험도 평가
    if predicted_class == 0:
        risk_level = 'Low'
    elif predicted_class in [1, 2]:
        risk_level = 'High (Emergency)'
    else:
        risk_level = 'Medium (Monitor)'

    return {
        'arrhythmia': arrhythmia_types[predicted_class],
        'confidence': f'{confidence:.2f}%',
        'risk_level': risk_level,
        'heart_rate': calculate_heart_rate(ecg_signal)
    }

def calculate_heart_rate(ecg_signal):
    # R-peak 탐지 (간단한 예제)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(ecg_signal, distance=200, height=0.5)
    hr = (len(peaks) / 5) * 60  # 5초 데이터 → 분당 심박수
    return int(hr)

# 사용 예시
ecg_data = np.load('patient_ecg_sample.npy')  # (5000,)
result = process_ecg_signal(ecg_data)
print(f"진단: {result['arrhythmia']} ({result['confidence']})")
print(f"위험도: {result['risk_level']}, 심박수: {result['heart_rate']} bpm")`}</code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* 2024-2025 Medical AI 최신 동향 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <TrendingUp className="w-7 h-7 text-orange-600" />
          2024-2025 Medical AI 혁신 동향
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-blue-900 dark:text-blue-300 flex items-center gap-2">
              1. Foundation Models for Medicine 🧠
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              대규모 의료 데이터로 학습한 범용 의료 AI 모델 등장
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
              <p className="text-sm font-semibold mb-2 text-blue-800 dark:text-blue-300">주요 모델:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• <strong>Med-PaLM 2 (Google, 2024.05)</strong>: USMLE 85.4% 정답률 (전문의 수준)</li>
                <li>• <strong>BioGPT-Large</strong>: PubMed 1,500만 논문 학습, 의학 질의응답 SOTA</li>
                <li>• <strong>GatorTron (UF Health)</strong>: 900억 파라미터, EHR 특화 LLM</li>
                <li>• <strong>ClinicalBERT</strong>: MIMIC-III 데이터셋 기반, 임상 노트 분석</li>
              </ul>
            </div>
          </div>

          <div className="border-l-4 border-green-500 bg-green-50 dark:bg-green-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-green-900 dark:text-green-300">
              2. Multimodal AI for Diagnosis
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              영상, 텍스트, 유전체 데이터를 통합 분석하는 멀티모달 AI
            </p>
            <div className="grid md:grid-cols-2 gap-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-green-700 dark:text-green-400 mb-1">PLIP (PathLang)</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  병리 이미지 + 진단 텍스트 결합, OpenAI CLIP 아키텍처
                </p>
              </div>
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <p className="font-semibold text-green-700 dark:text-green-400 mb-1">MedCLIP</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  의료 영상-텍스트 매칭, Zero-shot 질병 분류 가능
                </p>
              </div>
            </div>
          </div>

          <div className="border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-purple-900 dark:text-purple-300">
              3. Federated Learning in Healthcare
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              환자 데이터를 병원 밖으로 내보내지 않고 협력 학습
            </p>
            <ul className="text-sm space-y-2 text-gray-600 dark:text-gray-400 ml-4">
              <li>
                <span className="font-semibold">• NVIDIA Clara FL:</span> 20개 병원 협력, 뇌종양 분할 정확도 5% 향상
              </li>
              <li>
                <span className="font-semibold">• Google Health FL:</span> COVID-19 예측 모델, HIPAA 완벽 준수
              </li>
              <li>
                <span className="font-semibold">• Owkin (프랑스):</span> 연합학습 기반 항암제 반응 예측 (Nature Medicine, 2024)
              </li>
            </ul>
          </div>

          <div className="border-l-4 border-pink-500 bg-pink-50 dark:bg-pink-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-pink-900 dark:text-pink-300">
              4. Real-World Evidence (RWE) AI
            </h3>
            <p className="text-gray-700 dark:text-gray-300 mb-3">
              실제 진료 데이터로 치료 효과를 입증하는 AI 플랫폼
            </p>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg">
              <p className="text-sm font-semibold mb-2">활용 사례:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Flatiron Health: 280만 암 환자 데이터 분석 → FDA 승인 근거</li>
                <li>• Tempus: 유전체 + 임상 데이터 통합, 개인화 치료 추천</li>
                <li>• TriNetX: 3억 5천만 환자 EHR 네트워크, 임상시험 디자인 최적화</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 글로벌 Medical AI 통계 */}
      <section className="bg-gradient-to-r from-pink-600 to-red-600 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Activity className="w-7 h-7" />
          2024-2025 Medical AI 시장 현황
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">$188B</p>
            <p className="text-sm opacity-90">2030 Medical AI 시장 규모 전망</p>
            <p className="text-xs mt-2 opacity-75">출처: Grand View Research, 2024</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">37.5%</p>
            <p className="text-sm opacity-90">연평균 성장률 (CAGR 2024-2030)</p>
            <p className="text-xs mt-2 opacity-75">출처: MarketsandMarkets, 2024</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">520+</p>
            <p className="text-sm opacity-90">FDA 승인 AI/ML 의료기기 (2024.09 기준)</p>
            <p className="text-xs mt-2 opacity-75">출처: FDA Digital Health Center</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">86%</p>
            <p className="text-sm opacity-90">AI 도입 고려 중인 병원 비율</p>
            <p className="text-xs mt-2 opacity-75">출처: Deloitte Healthcare Survey 2024</p>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-3xl font-bold mb-1">15-30%</p>
            <p className="text-sm">AI 진단 보조로 오진율 감소</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-3xl font-bold mb-1">40%</p>
            <p className="text-sm">AI 신약 개발 시간 단축 (10년 → 6년)</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-3xl font-bold mb-1">$150B</p>
            <p className="text-sm">AI로 절감 가능한 연간 의료비 (미국)</p>
          </div>
        </div>
      </section>

      {/* Medical AI 윤리 및 도전과제 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <AlertCircle className="w-7 h-7 text-yellow-600" />
          Medical AI의 도전과제와 해결 방안
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-bold text-lg mb-3 text-red-900 dark:text-red-300">주요 도전과제</h3>
            <div className="space-y-3">
              {[
                {
                  challenge: '데이터 편향 (Data Bias)',
                  detail: '특정 인종/성별 과소 대표로 인한 성능 저하',
                  example: 'MIT 연구: 흑인 환자 대상 피부암 진단 정확도 20%p 낮음'
                },
                {
                  challenge: '설명 가능성 (Explainability)',
                  detail: '블랙박스 AI의 의사결정 근거 불명확',
                  example: 'EU AI Act: 고위험 의료 AI는 설명 가능해야 함 (2024.08 발효)'
                },
                {
                  challenge: '개인정보 보호 (Privacy)',
                  detail: 'HIPAA, GDPR 준수 및 익명화 한계',
                  example: '2024 영국 NHS 데이터 유출 사건: 150만 환자 정보 노출'
                },
                {
                  challenge: '규제 불확실성',
                  detail: 'AI 의료기기 승인 기준 국가별 상이',
                  example: 'FDA SaMD: 소프트웨어 업데이트마다 재승인 필요 논란'
                },
              ].map((item, idx) => (
                <div key={idx} className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-l-4 border-red-500">
                  <p className="font-semibold text-red-900 dark:text-red-300 mb-1">{item.challenge}</p>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">{item.detail}</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 bg-white dark:bg-gray-800 p-2 rounded">
                    📌 {item.example}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">해결 방안</h3>
            <div className="space-y-3">
              {[
                {
                  solution: 'Fairness-Aware Learning',
                  method: '편향 완화 알고리즘 적용',
                  tools: 'IBM AI Fairness 360, Google What-If Tool'
                },
                {
                  solution: 'Explainable AI (XAI)',
                  method: 'SHAP, Grad-CAM, LIME으로 예측 근거 시각화',
                  tools: 'SHAP (SHapley Additive exPlanations), Captum (PyTorch)'
                },
                {
                  solution: 'Differential Privacy',
                  method: '데이터에 노이즈 추가하여 개인 식별 방지',
                  tools: 'TensorFlow Privacy, OpenDP'
                },
                {
                  solution: 'Continuous Validation',
                  method: '배포 후 실시간 성능 모니터링 및 재학습',
                  tools: 'MLflow, Evidently AI, Fiddler AI'
                },
              ].map((item, idx) => (
                <div key={idx} className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
                  <p className="font-semibold text-green-900 dark:text-green-300 mb-1">{item.solution}</p>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">{item.method}</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 bg-white dark:bg-gray-800 p-2 rounded">
                    🛠️ {item.tools}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 Medical AI 필수 데이터셋',
            icon: 'docs' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'MIMIC-IV (MIT)',
                url: 'https://physionet.org/content/mimiciv/',
                description: '299,712명 중환자 EHR 데이터, 연구 목적 무료'
              },
              {
                title: 'ChestX-ray14 (NIH)',
                url: 'https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community',
                description: '112,120장 흉부 X-ray, 14개 질병 라벨'
              },
              {
                title: 'The Cancer Genome Atlas (TCGA)',
                url: 'https://www.cancer.gov/tcga',
                description: '33개 암 종류, 유전체+임상 데이터 통합'
              },
              {
                title: 'UK Biobank',
                url: 'https://www.ukbiobank.ac.uk/',
                description: '50만 명 유전체+영상+라이프스타일 데이터'
              },
            ]
          },
          {
            title: '🔬 핵심 연구 논문 (2023-2024)',
            icon: 'research' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'Med-PaLM 2 (Google, Nature 2024)',
                url: 'https://www.nature.com/articles/s41586-023-06291-2',
                description: '의학 시험 85.4% 정답률, 임상의 수준 달성'
              },
              {
                title: 'AlphaFold 3 (DeepMind, Science 2024)',
                url: 'https://www.science.org/doi/10.1126/science.adl2528',
                description: '단백질-약물 복합체 구조 예측 정확도 90%+'
              },
              {
                title: 'Federated Learning for Healthcare (Nature Medicine 2024)',
                url: 'https://www.nature.com/articles/s41591-024-02756-2',
                description: 'Owkin의 20개 병원 협력 연구, HIPAA 준수 학습'
              },
              {
                title: 'AI Bias in Medical Imaging (Radiology 2024)',
                url: 'https://pubs.rsna.org/doi/10.1148/radiol.230679',
                description: '인종/성별 편향 완화 방법론 제시'
              },
            ]
          },
          {
            title: '🛠️ 실전 프레임워크 & 도구',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'MONAI (Medical Open Network for AI)',
                url: 'https://monai.io/',
                description: 'PyTorch 기반 의료 영상 딥러닝 라이브러리 (NVIDIA 후원)'
              },
              {
                title: 'TensorFlow Federated',
                url: 'https://www.tensorflow.org/federated',
                description: '분산 학습 프레임워크, 의료 데이터 보호에 최적'
              },
              {
                title: 'SHAP for Healthcare',
                url: 'https://github.com/slundberg/shap',
                description: 'AI 의사결정 설명 도구, FDA 승인 심사에 필수'
              },
              {
                title: 'NVIDIA Clara',
                url: 'https://www.nvidia.com/en-us/clara/',
                description: 'AI 의료 영상 분석 플랫폼 (Holoscan SDK 포함)'
              },
              {
                title: 'DeepChem',
                url: 'https://deepchem.io/',
                description: '화학/신약 개발 전용 딥러닝 라이브러리'
              },
            ]
          },
          {
            title: '📖 규제 및 가이드라인',
            icon: 'docs' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'FDA Software as a Medical Device (SaMD)',
                url: 'https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd',
                description: 'AI 의료기기 승인 절차 및 기준'
              },
              {
                title: 'EU AI Act (2024)',
                url: 'https://artificialintelligenceact.eu/',
                description: '고위험 의료 AI 규제 (설명 가능성, 투명성 필수)'
              },
              {
                title: 'WHO Guidance on AI for Health (2024)',
                url: 'https://www.who.int/publications/i/item/9789240029200',
                description: 'AI 의료 윤리 가이드라인 및 거버넌스 프레임워크'
              },
              {
                title: 'HIPAA & AI Compliance',
                url: 'https://www.hhs.gov/hipaa/for-professionals/privacy/index.html',
                description: '미국 의료정보 보호법, AI 적용 시 고려사항'
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
            <span className="text-pink-600 font-bold">•</span>
            <span>Medical AI 4대 영역: <strong>의료 영상 분석, 진단 보조, 신약 개발, 정밀 의료</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-pink-600 font-bold">•</span>
            <span><strong>2024 핵심 트렌드</strong>: Foundation Models (Med-PaLM 2), Multimodal AI, Federated Learning</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-pink-600 font-bold">•</span>
            <span><strong>시장 규모 $188B (2030)</strong>, FDA 승인 AI 의료기기 520+ 개 (2024.09 기준)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-pink-600 font-bold">•</span>
            <span><strong>주요 도전과제</strong>: 데이터 편향, 설명 가능성, 개인정보 보호, 규제 불확실성</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-pink-600 font-bold">•</span>
            <span><strong>필수 도구</strong>: MONAI (의료 영상), SHAP (XAI), TensorFlow Federated (프라이버시)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-pink-600 font-bold">•</span>
            <span>AI로 오진율 15-30% 감소, 신약 개발 시간 40% 단축 가능</span>
          </li>
        </ul>
      </section>
    </div>
  );
}
