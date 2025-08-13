'use client';

import { useState } from 'react';
import { 
  Terminal,
  Copy,
  Check
} from 'lucide-react';

export default function Chapter6() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

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
}