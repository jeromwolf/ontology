'use client';

import { useState } from 'react';
import { 
  BookOpen, 
  Code, 
  Lightbulb, 
  CheckCircle, 
  AlertCircle,
  Terminal,
  Copy,
  Check,
  PlayCircle,
  ExternalLink
} from 'lucide-react';

interface ChapterContentProps {
  chapterId: string;
}

export default function ChapterContent({ chapterId }: ChapterContentProps) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const renderChapterContent = () => {
    switch (chapterId) {
      case 'cv-basics':
        return (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-4">디지털 이미지의 이해</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                디지털 이미지는 픽셀(pixel)이라는 작은 점들의 2차원 배열로 구성됩니다. 
                각 픽셀은 색상 정보를 담고 있으며, 이러한 픽셀들이 모여 우리가 볼 수 있는 이미지를 형성합니다.
              </p>

              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6 mb-6">
                <div className="flex items-start gap-3">
                  <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">핵심 개념</h3>
                    <ul className="space-y-2 text-blue-800 dark:text-blue-200">
                      <li>• 해상도: 이미지의 가로 × 세로 픽셀 수</li>
                      <li>• 비트 깊이: 각 픽셀이 표현할 수 있는 색상의 수</li>
                      <li>• 채널: 색상을 구성하는 개별 요소 (RGB, HSV 등)</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mb-6">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - OpenCV로 이미지 읽기</span>
                  </div>
                  <button
                    onClick={() => copyToClipboard(`import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기
img = cv2.imread('image.jpg')
# BGR을 RGB로 변환 (OpenCV는 기본적으로 BGR 사용)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 이미지 정보 출력
print(f"이미지 크기: {img.shape}")
print(f"데이터 타입: {img.dtype}")
print(f"픽셀 값 범위: {img.min()} - {img.max()}")

# 이미지 표시
plt.imshow(img_rgb)
plt.axis('off')
plt.show()`, 'cv-basics-1')}
                    className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                  >
                    {copiedCode === 'cv-basics-1' ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                    )}
                  </button>
                </div>
                <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
                  <code>{`import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기
img = cv2.imread('image.jpg')
# BGR을 RGB로 변환 (OpenCV는 기본적으로 BGR 사용)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 이미지 정보 출력
print(f"이미지 크기: {img.shape}")
print(f"데이터 타입: {img.dtype}")
print(f"픽셀 값 범위: {img.min()} - {img.max()}")

# 이미지 표시
plt.imshow(img_rgb)
plt.axis('off')
plt.show()`}</code>
                </pre>
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-bold mb-4">색상 공간</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                색상 공간은 색을 수치적으로 표현하는 방법입니다. 각 색상 공간은 특정 용도에 최적화되어 있습니다.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                  <h3 className="font-semibold text-lg mb-3 text-teal-600 dark:text-teal-400">RGB 색상 공간</h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-3">
                    빨강(Red), 초록(Green), 파랑(Blue)의 조합으로 색을 표현
                  </p>
                  <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                    <li>• 디스플레이 장치에 최적화</li>
                    <li>• 각 채널 0-255 범위</li>
                    <li>• 직관적이지만 조명 변화에 민감</li>
                  </ul>
                </div>
                <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                  <h3 className="font-semibold text-lg mb-3 text-teal-600 dark:text-teal-400">HSV 색상 공간</h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-3">
                    색상(Hue), 채도(Saturation), 명도(Value)로 표현
                  </p>
                  <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                    <li>• 색상 기반 필터링에 유용</li>
                    <li>• 조명 변화에 강건</li>
                    <li>• 인간의 색 인식과 유사</li>
                  </ul>
                </div>
              </div>
            </section>

            <section className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">실습 과제</h3>
                  <ul className="space-y-2 text-green-800 dark:text-green-200">
                    <li>1. 이미지를 읽고 각 색상 채널을 분리하여 표시해보세요</li>
                    <li>2. RGB 이미지를 HSV로 변환하고 특정 색상만 추출해보세요</li>
                    <li>3. 히스토그램을 그려 이미지의 밝기 분포를 분석해보세요</li>
                  </ul>
                </div>
              </div>
            </section>
          </div>
        );

      case 'image-processing':
        return (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-4">공간 도메인 필터링</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                공간 도메인 필터링은 이미지의 픽셀 값을 직접 조작하여 다양한 효과를 만드는 기법입니다.
                커널(kernel) 또는 마스크를 사용하여 이미지에 컨볼루션 연산을 수행합니다.
              </p>

              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mb-6">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - 가우시안 블러 적용</span>
                  </div>
                  <button
                    onClick={() => copyToClipboard(`import cv2
import numpy as np

# 가우시안 블러 - 노이즈 제거
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 엣지 검출을 위한 소벨 필터
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
edges = np.sqrt(sobelx**2 + sobely**2)

# 샤프닝 필터
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])
sharpened = cv2.filter2D(img, -1, kernel)`, 'image-proc-1')}
                    className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                  >
                    {copiedCode === 'image-proc-1' ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                    )}
                  </button>
                </div>
                <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
                  <code>{`import cv2
import numpy as np

# 가우시안 블러 - 노이즈 제거
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 엣지 검출을 위한 소벨 필터
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
edges = np.sqrt(sobelx**2 + sobely**2)

# 샤프닝 필터
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])
sharpened = cv2.filter2D(img, -1, kernel)`}</code>
                </pre>
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-bold mb-4">주파수 도메인 처리</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                푸리에 변환을 사용하여 이미지를 주파수 도메인으로 변환하면, 
                주기적인 패턴이나 노이즈를 효과적으로 처리할 수 있습니다.
              </p>

              <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-yellow-900 dark:text-yellow-100 mb-2">주의사항</h3>
                    <p className="text-yellow-800 dark:text-yellow-200">
                      주파수 도메인 처리는 계산량이 많으므로, 큰 이미지의 경우 처리 시간이 오래 걸릴 수 있습니다.
                      FFT(Fast Fourier Transform)를 사용하여 계산 효율성을 높일 수 있습니다.
                    </p>
                  </div>
                </div>
              </div>
            </section>
          </div>
        );

      case 'feature-detection':
        return (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-4">특징점 검출 알고리즘</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                특징점(feature points)은 이미지에서 독특하고 식별 가능한 지점들입니다. 
                이러한 특징점들은 이미지 매칭, 객체 인식, 3D 재구성 등에 활용됩니다.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">SIFT</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    Scale-Invariant Feature Transform. 크기와 회전에 불변한 특징점 검출
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">SURF</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    Speeded-Up Robust Features. SIFT보다 빠른 특징점 검출
                  </p>
                </div>
                <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                  <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">ORB</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    Oriented FAST and Rotated BRIEF. 실시간 처리에 적합한 빠른 알고리즘
                  </p>
                </div>
              </div>
            </section>
          </div>
        );

      case 'deep-learning-vision':
        return (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-4">CNN 아키텍처의 진화</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Convolutional Neural Networks(CNN)는 컴퓨터 비전 분야에 혁명을 일으켰습니다.
                LeNet부터 최신 EfficientNet까지, CNN 아키텍처는 계속 발전하고 있습니다.
              </p>

              <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 rounded-lg p-6 mb-6">
                <h3 className="text-lg font-semibold mb-4 text-teal-900 dark:text-teal-100">주요 CNN 아키텍처</h3>
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
                    <span className="font-medium">AlexNet (2012)</span> - ImageNet 대회 우승, 딥러닝 시대 개막
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
                    <span className="font-medium">VGGNet (2014)</span> - 단순하고 깊은 구조의 효과 증명
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
                    <span className="font-medium">ResNet (2015)</span> - Skip connection으로 매우 깊은 네트워크 학습
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
                    <span className="font-medium">EfficientNet (2019)</span> - 효율적인 스케일링으로 높은 성능
                  </div>
                </div>
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-bold mb-4">Vision Transformer</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Transformer 아키텍처를 컴퓨터 비전에 적용한 Vision Transformer(ViT)는 
                CNN의 대안으로 주목받고 있습니다. 이미지를 패치로 나누어 시퀀스로 처리합니다.
              </p>

              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - Vision Transformer 사용</span>
                  </div>
                  <button
                    onClick={() => copyToClipboard(`from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# 사전 학습된 ViT 모델 로드
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 이미지 전처리
image = Image.open('sample.jpg')
inputs = processor(images=image, return_tensors="pt")

# 예측
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print(f"예측된 클래스: {model.config.id2label[predicted_class]}")`, 'dl-vision-1')}
                    className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                  >
                    {copiedCode === 'dl-vision-1' ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                    )}
                  </button>
                </div>
                <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
                  <code>{`from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# 사전 학습된 ViT 모델 로드
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 이미지 전처리
image = Image.open('sample.jpg')
inputs = processor(images=image, return_tensors="pt")

# 예측
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print(f"예측된 클래스: {model.config.id2label[predicted_class]}")`}</code>
                </pre>
              </div>
            </section>
          </div>
        );

      case '2d-to-3d':
        return (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-4">스테레오 비전</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                스테레오 비전은 두 개 이상의 카메라를 사용하여 깊이 정보를 추출하는 기술입니다.
                인간의 양안 시차와 같은 원리로 3D 정보를 복원합니다.
              </p>

              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6 mb-6">
                <div className="flex items-start gap-3">
                  <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">핵심 원리</h3>
                    <ul className="space-y-2 text-blue-800 dark:text-blue-200">
                      <li>• 삼각측량: 두 카메라의 시차를 이용한 거리 계산</li>
                      <li>• 에피폴라 기하학: 스테레오 매칭의 기하학적 제약</li>
                      <li>• 디스패리티 맵: 픽셀별 깊이 정보를 담은 이미지</li>
                    </ul>
                  </div>
                </div>
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-bold mb-4">단일 이미지 깊이 추정</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                최신 딥러닝 기술을 활용하면 단일 이미지에서도 깊이 정보를 추정할 수 있습니다.
                MiDaS, DPT 등의 모델이 대표적입니다.
              </p>

              <div className="flex items-center gap-3 p-4 bg-teal-50 dark:bg-teal-900/20 rounded-lg mb-6">
                <PlayCircle className="w-6 h-6 text-teal-600 dark:text-teal-400" />
                <p className="text-teal-800 dark:text-teal-200">
                  2D to 3D Converter 시뮬레이터에서 실시간으로 깊이 추정을 체험해보세요!
                </p>
              </div>
            </section>
          </div>
        );

      case 'object-detection-tracking':
        return (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-4">실시간 객체 검출</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                객체 검출은 이미지에서 특정 객체의 위치와 클래스를 동시에 찾는 작업입니다.
                YOLO, Faster R-CNN 등 다양한 알고리즘이 개발되어 있습니다.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                  <h3 className="font-semibold text-lg mb-3 text-teal-600 dark:text-teal-400">YOLO 계열</h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-3">
                    You Only Look Once - 한 번의 추론으로 객체 검출
                  </p>
                  <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                    <li>✓ 매우 빠른 처리 속도</li>
                    <li>✓ 실시간 처리 가능</li>
                    <li>✓ 작은 객체 검출에 약함</li>
                  </ul>
                </div>
                <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                  <h3 className="font-semibold text-lg mb-3 text-teal-600 dark:text-teal-400">R-CNN 계열</h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-3">
                    Region-based CNN - 영역 기반 검출
                  </p>
                  <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                    <li>✓ 높은 정확도</li>
                    <li>✓ 작은 객체도 잘 검출</li>
                    <li>✓ 상대적으로 느림</li>
                  </ul>
                </div>
              </div>

              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - YOLOv8 사용 예제</span>
                  </div>
                  <button
                    onClick={() => copyToClipboard(`from ultralytics import YOLO
import cv2

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # nano 버전

# 이미지에서 객체 검출
results = model('image.jpg')

# 결과 시각화
for r in results:
    boxes = r.boxes
    for box in boxes:
        # 바운딩 박스 좌표
        x1, y1, x2, y2 = box.xyxy[0]
        # 클래스와 신뢰도
        cls = int(box.cls)
        conf = float(box.conf)
        
        # 바운딩 박스 그리기
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f'{model.names[cls]} {conf:.2f}', 
                    (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)`, 'object-det-1')}
                    className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                  >
                    {copiedCode === 'object-det-1' ? (
                      <Check className="w-4 h-4 text-green-600" />
                    ) : (
                      <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                    )}
                  </button>
                </div>
                <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
                  <code>{`from ultralytics import YOLO
import cv2

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # nano 버전

# 이미지에서 객체 검출
results = model('image.jpg')

# 결과 시각화
for r in results:
    boxes = r.boxes
    for box in boxes:
        # 바운딩 박스 좌표
        x1, y1, x2, y2 = box.xyxy[0]
        # 클래스와 신뢰도
        cls = int(box.cls)
        conf = float(box.conf)
        
        # 바운딩 박스 그리기
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f'{model.names[cls]} {conf:.2f}', 
                    (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)`}</code>
                </pre>
              </div>
            </section>
          </div>
        );

      case 'face-recognition':
        return (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-4">얼굴 인식 파이프라인</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                얼굴 인식은 여러 단계로 구성된 복잡한 프로세스입니다. 
                검출, 정렬, 특징 추출, 매칭의 단계를 거칩니다.
              </p>

              <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 rounded-lg p-6 mb-6">
                <h3 className="text-lg font-semibold mb-4 text-teal-900 dark:text-teal-100">처리 단계</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">1</div>
                    <p className="font-medium">얼굴 검출</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">이미지에서 얼굴 영역 찾기</p>
                  </div>
                  <div className="text-center">
                    <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">2</div>
                    <p className="font-medium">얼굴 정렬</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">랜드마크 기반 정규화</p>
                  </div>
                  <div className="text-center">
                    <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">3</div>
                    <p className="font-medium">특징 추출</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">얼굴 임베딩 벡터 생성</p>
                  </div>
                  <div className="text-center">
                    <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">4</div>
                    <p className="font-medium">신원 확인</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">벡터 유사도 비교</p>
                  </div>
                </div>
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-bold mb-4">감정 인식</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                얼굴 표정으로부터 감정을 인식하는 기술은 HCI, 마케팅, 의료 등 다양한 분야에서 활용됩니다.
              </p>

              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                <h3 className="font-semibold mb-3">7가지 기본 감정</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😊 행복</div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😢 슬픔</div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😠 분노</div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😱 두려움</div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😲 놀람</div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">🤢 혐오</div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😐 중립</div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">🤔 기타</div>
                </div>
              </div>
            </section>
          </div>
        );

      case 'real-time-applications':
        return (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-4">증강 현실 (AR)</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                증강 현실은 실제 환경에 가상의 객체를 겹쳐서 보여주는 기술입니다.
                컴퓨터 비전은 AR의 핵심 기술로, 환경 인식과 추적을 담당합니다.
              </p>

              <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-6 mb-6">
                <h3 className="font-semibold text-purple-900 dark:text-purple-100 mb-3">AR의 핵심 기술</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-purple-800 dark:text-purple-200 mb-2">SLAM</h4>
                    <p className="text-sm text-purple-700 dark:text-purple-300">
                      Simultaneous Localization and Mapping - 동시에 위치를 추정하고 지도를 생성
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium text-purple-800 dark:text-purple-200 mb-2">마커 추적</h4>
                    <p className="text-sm text-purple-700 dark:text-purple-300">
                      ArUco, QR 코드 등의 마커를 인식하여 3D 콘텐츠 배치
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-bold mb-4">자율주행 비전</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                자율주행 차량의 '눈' 역할을 하는 컴퓨터 비전은 도로, 차선, 신호등, 
                보행자, 다른 차량 등을 실시간으로 인식하고 추적합니다.
              </p>

              <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
                <h3 className="font-semibold mb-4">자율주행 비전 시스템 구성</h3>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                      <span className="text-teal-600 dark:text-teal-400 font-bold">1</span>
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">다중 센서 융합</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        카메라, LiDAR, 레이더 데이터를 통합하여 정확한 환경 인식
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                      <span className="text-teal-600 dark:text-teal-400 font-bold">2</span>
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">실시간 객체 검출</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        차량, 보행자, 자전거, 신호등 등을 밀리초 단위로 검출
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 bg-teal-100 dark:bg-teal-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                      <span className="text-teal-600 dark:text-teal-400 font-bold">3</span>
                    </div>
                    <div>
                      <h4 className="font-medium mb-1">차선 및 도로 인식</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        세그멘테이션을 통한 주행 가능 영역 파악
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section>
              <h2 className="text-2xl font-bold mb-4">의료 영상 분석</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                딥러닝 기반 의료 영상 분석은 질병의 조기 진단과 정확한 판독에 기여하고 있습니다.
              </p>

              <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">적용 분야</h3>
                    <ul className="space-y-2 text-green-800 dark:text-green-200">
                      <li>• X-ray, CT, MRI 영상에서 종양 검출</li>
                      <li>• 망막 사진을 통한 당뇨병성 망막병증 진단</li>
                      <li>• 피부 사진을 통한 피부암 스크리닝</li>
                      <li>• 병리 슬라이드 이미지 분석</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="flex items-center gap-3">
                  <ExternalLink className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  <p className="text-blue-800 dark:text-blue-200">
                    각 시뮬레이터에서 이러한 기술들을 직접 체험하고 실습해보세요!
                  </p>
                </div>
              </div>
            </section>
          </div>
        );

      default:
        return (
          <div className="text-center py-12">
            <p className="text-gray-500 dark:text-gray-400">챕터 콘텐츠를 불러올 수 없습니다.</p>
          </div>
        );
    }
  };

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      {renderChapterContent()}
    </div>
  );
}