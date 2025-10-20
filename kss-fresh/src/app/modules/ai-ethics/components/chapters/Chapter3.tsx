import React from 'react';
import { Eye, Lightbulb, Code, Image as ImageIcon, Brain } from 'lucide-react';
import References from '../References';

export default function Chapter3() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-6 text-gray-900 dark:text-white">투명성과 설명가능성</h1>

      <div className="bg-gradient-to-r from-rose-100 to-pink-100 dark:from-rose-900/30 dark:to-pink-900/30 p-6 rounded-lg mb-8">
        <p className="text-lg text-gray-800 dark:text-gray-200 leading-relaxed">
          AI의 "블랙박스" 문제는 신뢰성과 책임성의 핵심 장애물입니다.
          Explainable AI (XAI)는 복잡한 모델의 의사결정을 인간이 이해할 수 있게 만들어,
          EU AI Act 등 규제 준수와 사용자 신뢰 구축을 가능하게 합니다.
        </p>
      </div>

      {/* XAI 4대 기법 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Eye className="w-8 h-8 text-rose-600" />
          XAI 핵심 기법 4종
        </h2>

        <div className="space-y-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-blue-500">
            <div className="flex items-center gap-3 mb-3">
              <Code className="w-6 h-6 text-blue-600" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">1. SHAP (SHapley Additive exPlanations)</h3>
            </div>

            <p className="text-gray-700 dark:text-gray-300 mb-4">
              게임 이론의 Shapley Value를 기반으로 각 특성이 예측에 기여한 정도를 계산.
              Model-agnostic하여 모든 ML 모델에 적용 가능.
            </p>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded mb-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">수학적 정의</h4>
              <p className="font-mono text-sm text-gray-800 dark:text-gray-200 mb-2">
                φ<sub>i</sub> = Σ (|S|! * (n - |S| - 1)! / n!) * [f(S ∪ {'{i}'}) - f(S)]
              </p>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>φ<sub>i</sub>: 특성 i의 SHAP value (기여도)</li>
                <li>S: 특성 부분집합</li>
                <li>f(S): 부분집합 S로만 예측한 결과</li>
              </ul>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">장점</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>유일한 완전 이론적 근거 (Shapley Value 공리)</li>
                <li>Global + Local 설명 모두 제공</li>
                <li>특성 간 상호작용 포착 가능</li>
              </ul>
              <p className="font-semibold text-gray-900 dark:text-white mt-3 mb-2">단점</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>계산 복잡도 높음 (O(2^n))</li>
                <li>특성이 많으면 근사값만 계산 가능</li>
              </ul>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-green-500">
            <div className="flex items-center gap-3 mb-3">
              <Lightbulb className="w-6 h-6 text-green-600" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">2. LIME (Local Interpretable Model-agnostic Explanations)</h3>
            </div>

            <p className="text-gray-700 dark:text-gray-300 mb-4">
              복잡한 모델을 "국지적으로" 단순한 선형 모델로 근사하여 설명.
              "왜 이 이미지가 고양이로 분류되었나?"에 대한 직관적 답변 제공.
            </p>

            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded mb-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">작동 원리</h4>
              <ol className="list-decimal list-inside text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <li>설명할 샘플 x 주변에 perturbation 생성 (약간 변형된 샘플들)</li>
                <li>각 perturbation에 대해 블랙박스 모델 예측값 얻기</li>
                <li>x에 가까운 샘플에 높은 가중치 부여</li>
                <li>가중치 기반 선형 회귀로 국지 모델 학습</li>
                <li>선형 모델의 계수 = 특성 중요도</li>
              </ol>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">활용 사례</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>이미지 분류: 어떤 픽셀 영역이 결정에 중요했나?</li>
                <li>텍스트 분류: 어떤 단어가 감정 분석에 영향?</li>
                <li>추천 시스템: 왜 이 상품을 추천했나?</li>
              </ul>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-purple-500">
            <div className="flex items-center gap-3 mb-3">
              <ImageIcon className="w-6 h-6 text-purple-600" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">3. Grad-CAM (Gradient-weighted Class Activation Mapping)</h3>
            </div>

            <p className="text-gray-700 dark:text-gray-300 mb-4">
              CNN 기반 이미지 모델의 시각적 설명. 어떤 이미지 영역이 분류 결정에 중요한지 히트맵으로 표시.
            </p>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded mb-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">수학적 원리</h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                마지막 Convolution Layer의 활성화 맵에 Gradient 가중치 적용:
              </p>
              <p className="font-mono text-sm text-gray-800 dark:text-gray-200 mb-2">
                L<sub>Grad-CAM</sub> = ReLU(Σ α<sub>k</sub> * A<sub>k</sub>)
              </p>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>α<sub>k</sub>: 클래스 c에 대한 특성 맵 k의 중요도 (Gradient 평균)</li>
                <li>A<sub>k</sub>: k번째 특성 맵 활성화</li>
                <li>ReLU: 양수 기여도만 시각화 (음수는 억제 효과)</li>
              </ul>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">실전 활용</p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>의료 영상 진단: 어디가 암 조직인가?</li>
                <li>자율주행: 어떤 물체를 보행자로 판단했나?</li>
                <li>안면 인식: 눈/코/입 중 어디를 봤나?</li>
              </ul>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border-l-4 border-orange-500">
            <div className="flex items-center gap-3 mb-3">
              <Brain className="w-6 h-6 text-orange-600" />
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">4. Attention Visualization (Transformer 계열)</h3>
            </div>

            <p className="text-gray-700 dark:text-gray-300 mb-4">
              GPT, BERT 등 Transformer 모델의 Attention 가중치를 시각화하여
              모델이 어떤 토큰에 집중했는지 파악.
            </p>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded mb-4">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Self-Attention 메커니즘</h4>
              <p className="font-mono text-sm text-gray-800 dark:text-gray-200 mb-2">
                Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) * V
              </p>
              <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>Q: Query (현재 토큰이 찾는 정보)</li>
                <li>K: Key (다른 토큰이 제공하는 정보)</li>
                <li>V: Value (실제 전달할 정보)</li>
                <li>Attention Score: 각 토큰 쌍의 관련성 (0~1)</li>
              </ul>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded">
              <p className="font-semibold text-gray-900 dark:text-white mb-2">해석 예시</p>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                문장: "The cat sat on the mat because it was tired."
              </p>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>"it"이 "cat"에 높은 attention (대명사 해소)</li>
                <li>"tired"가 "sat"에 attention (인과 관계)</li>
                <li>Multi-Head Attention: 각 헤드가 다른 패턴 학습</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* EU AI Act 설명가능성 요구사항 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">EU AI Act 설명가능성 요구사항 (2024.08 발효)</h2>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg mb-6">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">고위험 AI 시스템 의무사항</h3>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            EU AI Act는 고위험 시스템(의료, 법률, 채용, 신용 평가 등)에 대해 "적절한 수준의 해석가능성"을 요구합니다.
          </p>

          <div className="space-y-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Article 13: Transparency and Information</h4>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>AI 시스템 사용 사실을 명확히 고지</li>
                <li>의사결정 논리(decision logic)를 이해할 수 있게 설명</li>
                <li>인간 감독(human oversight) 방법 명시</li>
                <li>시스템 성능·한계·위험 공개</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Article 14: Human Oversight</h4>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>AI 출력을 완전히 이해할 수 있어야 함</li>
                <li>결과를 무시하거나 수정할 수 있는 권한</li>
                <li>시스템 중단 또는 중지 능력</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">실무 적용 기준</h4>
              <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
                <li>SHAP/LIME 등 XAI 기법 최소 1개 적용</li>
                <li>Model Card 또는 Datasheet 작성</li>
                <li>사용자 대상 설명 UI 제공 (예: "왜 이 결정이?")</li>
                <li>정기 감사 및 재학습 로그 유지</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
          <p className="text-sm font-semibold text-gray-900 dark:text-white mb-2">⚠️ 미준수 시 처벌</p>
          <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1">
            <li>최대 €3,500만 또는 전 세계 매출의 7% (둘 중 높은 금액)</li>
            <li>EU 시장 진입 금지</li>
            <li>소송 시 입증 책임 전환 (기업이 무해함을 증명해야 함)</li>
          </ul>
        </div>
      </section>

      {/* 실전 코드 */}
      <section className="mb-12">
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">실전 코드: SHAP & LIME & Grad-CAM</h2>

        <div className="mb-6">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">1. SHAP Values 계산 (XGBoost 예시)</h3>
          <div className="bg-gray-900 dark:bg-gray-950 p-6 rounded-lg overflow-x-auto">
            <pre className="text-sm text-gray-100">
              <code>{`import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# 1단계: 모델 학습
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# 2단계: SHAP Explainer 생성
explainer = shap.TreeExplainer(model)  # Tree 모델용 (빠름)
# explainer = shap.KernelExplainer(model.predict_proba, X_train)  # Model-agnostic (느림)

shap_values = explainer.shap_values(X_test)

# 3단계: 개별 예측 설명 (Local Explanation)
sample_idx = 0
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    X_test.iloc[sample_idx],
    matplotlib=True
)
# → 빨간색: 예측 증가 기여, 파란색: 감소 기여

# 4단계: 특성 중요도 (Global Explanation)
shap.summary_plot(shap_values, X_test, plot_type="bar")
# → 전체 샘플에서 평균 |SHAP value| 높은 순서

# 5단계: 특성 간 상호작용 (Dependence Plot)
shap.dependence_plot(
    "income",  # 관심 특성
    shap_values,
    X_test,
    interaction_index="age"  # 상호작용 특성
)
# → income이 높을수록 SHAP value 어떻게 변하나? (age로 색상 구분)

# 6단계: 워터폴 차트 (단일 예측 상세 분해)
shap.plots.waterfall(shap_values[sample_idx])
# → 기준값(expected_value)에서 각 특성이 예측값까지 어떻게 기여했는지

# 7단계: Decision Plot (여러 샘플 비교)
shap.decision_plot(
    explainer.expected_value,
    shap_values[:20],  # 20개 샘플
    X_test.iloc[:20]
)
# → 각 샘플이 특성별로 어떻게 다른 결정을 내렸는지 시각화`}</code>
            </pre>
          </div>
        </div>

        <div className="mb-6">
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">2. LIME 이미지 설명</h3>
          <div className="bg-gray-900 dark:bg-gray-950 p-6 rounded-lg overflow-x-auto">
            <pre className="text-sm text-gray-100">
              <code>{`from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

# 예시: CNN 이미지 분류 모델
def predict_fn(images):
    """모델 예측 래퍼 (확률 반환)"""
    return model.predict(images)

# 1단계: LIME Explainer 생성
explainer = lime_image.LimeImageExplainer()

# 2단계: 이미지 설명 생성
image = load_image('cat.jpg')  # (224, 224, 3)
explanation = explainer.explain_instance(
    image,
    predict_fn,
    top_labels=5,        # 상위 5개 클래스 설명
    hide_color=0,        # 가린 영역을 검정색으로
    num_samples=1000     # Perturbation 샘플 수 (많을수록 정확하지만 느림)
)

# 3단계: 긍정 기여 영역 시각화 (예측을 강화한 영역)
temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=True,
    num_features=10,     # 상위 10개 superpixel만 표시
    hide_rest=False
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title('LIME Explanation (Positive Contributions)')
plt.show()

# 4단계: 긍정+부정 기여 비교
temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=False,
    num_features=10,
    hide_rest=False
)
# → 녹색: 예측 강화, 빨간색: 예측 약화

# 5단계: 텍스트 설명
from lime.lime_text import LimeTextExplainer

text_explainer = LimeTextExplainer(class_names=['negative', 'positive'])
text_explanation = text_explainer.explain_instance(
    "This movie is absolutely fantastic!",
    classifier_fn=sentiment_model.predict_proba,
    num_features=6
)

print(text_explanation.as_list())
# 출력: [('fantastic', 0.42), ('absolutely', 0.28), ...]`}</code>
            </pre>
          </div>
        </div>

        <div>
          <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">3. Grad-CAM 시각화 (PyTorch)</h3>
          <div className="bg-gray-900 dark:bg-gray-950 p-6 rounded-lg overflow-x-auto">
            <pre className="text-sm text-gray-100">
              <code>{`import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np

# 1단계: 사전학습 모델 로드
model = models.resnet50(pretrained=True)
model.eval()

# 2단계: Grad-CAM 계산 함수
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook 등록
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()

        # Grad-CAM 계산
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)  # ReLU
        heatmap /= torch.max(heatmap)  # Normalize

        return heatmap.cpu().numpy()

# 3단계: 적용
grad_cam = GradCAM(model, model.layer4[-1])  # ResNet 마지막 Conv layer

image = load_and_preprocess_image('dog.jpg')
heatmap = grad_cam(image)

# 4단계: 히트맵 시각화
heatmap_resized = cv2.resize(heatmap, (224, 224))
heatmap_colored = cv2.applyColorMap(
    np.uint8(255 * heatmap_resized),
    cv2.COLORMAP_JET
)

# Original + Heatmap 합성
original_image = cv2.imread('dog.jpg')
original_image = cv2.resize(original_image, (224, 224))
superimposed = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)

cv2.imshow('Grad-CAM', superimposed)
cv2.waitKey(0)`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 XAI 프레임워크 & 도구',
            icon: 'tools' as const,
            color: 'border-rose-500',
            items: [
              {
                title: 'SHAP Documentation',
                url: 'https://shap.readthedocs.io/',
                description: 'Official SHAP library (Lundberg & Lee, 2017) - 가장 널리 사용되는 XAI 도구'
              },
              {
                title: 'LIME GitHub',
                url: 'https://github.com/marcotcr/lime',
                description: 'Marco Tulio Ribeiro의 LIME 구현 - 이미지/텍스트/테이블 데이터 지원'
              },
              {
                title: 'Grad-CAM PyTorch',
                url: 'https://github.com/jacobgil/pytorch-grad-cam',
                description: 'Grad-CAM, Grad-CAM++, Score-CAM 등 다양한 CAM 변형 구현'
              },
              {
                title: 'InterpretML (Microsoft)',
                url: 'https://interpret.ml/',
                description: 'Glassbox 모델 (EBM) + SHAP/LIME 통합 - 정확도와 해석가능성 동시 추구'
              },
              {
                title: 'Captum (PyTorch)',
                url: 'https://captum.ai/',
                description: 'Facebook의 PyTorch 전용 XAI 라이브러리 - 15+ 알고리즘'
              }
            ]
          },
          {
            title: '📖 핵심 연구 논문',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'A Unified Approach to Interpreting Model Predictions (SHAP)',
                url: 'https://arxiv.org/abs/1705.07874',
                description: 'NeurIPS 2017 Best Paper - Shapley Value를 ML에 적용'
              },
              {
                title: '"Why Should I Trust You?": Explaining the Predictions (LIME)',
                url: 'https://arxiv.org/abs/1602.04938',
                description: 'KDD 2016 - 국지적 선형 근사 방법론'
              },
              {
                title: 'Grad-CAM: Visual Explanations from Deep Networks',
                url: 'https://arxiv.org/abs/1610.02391',
                description: 'ICCV 2017 - CNN 시각화의 표준'
              },
              {
                title: 'Attention is Not Explanation',
                url: 'https://arxiv.org/abs/1902.10186',
                description: 'ACL 2019 - Attention 가중치가 진정한 설명인가? 비판적 고찰'
              },
              {
                title: 'The (Un)reliability of saliency methods',
                url: 'https://arxiv.org/abs/1711.00867',
                description: 'NeurIPS 2017 - XAI 기법들의 신뢰성 검증'
              }
            ]
          },
          {
            title: '🛠️ 실전 리소스',
            icon: 'docs' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'EU AI Act Official Text',
                url: 'https://artificialintelligenceact.eu/',
                description: 'EU AI Act 전문 및 해설 (2024.08 발효)'
              },
              {
                title: 'Model Cards Toolkit (Google)',
                url: 'https://github.com/tensorflow/model-card-toolkit',
                description: 'ML 모델 투명성 문서화 자동 생성 도구'
              },
              {
                title: 'What-If Tool (Google PAIR)',
                url: 'https://pair-code.github.io/what-if-tool/',
                description: 'TensorBoard 통합 인터랙티브 모델 분석 도구'
              },
              {
                title: 'AI Explainability 360 (IBM)',
                url: 'https://aix360.mybluemix.net/',
                description: '8개 알고리즘 (SHAP, LIME, ProfWeight 등) + 튜토리얼'
              },
              {
                title: 'DALEX (R/Python)',
                url: 'https://dalex.drwhy.ai/',
                description: 'Descriptive mAchine Learning EXplanations - 통합 XAI 패키지'
              }
            ]
          }
        ]}
      />
    </div>
  );
}
