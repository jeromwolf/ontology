'use client';

import { useState } from 'react';
import { 
  Lightbulb, 
  CheckCircle,
  Terminal,
  Copy,
  Check
} from 'lucide-react';

export default function Chapter1() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

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
}