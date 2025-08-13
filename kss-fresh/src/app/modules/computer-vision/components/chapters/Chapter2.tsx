'use client';

import { useState } from 'react';
import { 
  AlertCircle,
  Terminal,
  Copy,
  Check
} from 'lucide-react';

export default function Chapter2() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

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
}