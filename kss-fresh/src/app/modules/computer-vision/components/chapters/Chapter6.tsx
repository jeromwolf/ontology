'use client';

import { useState } from 'react';
import {
  Terminal,
  Copy,
  Check
} from 'lucide-react';
import References from '@/components/common/References';

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

      <References
        sections={[
          {
            title: 'YOLO Series',
            icon: 'paper' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'You Only Look Once (YOLOv1)',
                authors: 'Joseph Redmon, et al.',
                year: '2016',
                description: 'Single-shot 객체 검출의 시작 - 실시간 처리 혁명',
                link: 'https://arxiv.org/abs/1506.02640'
              },
              {
                title: 'YOLOv3: An Incremental Improvement',
                authors: 'Joseph Redmon, Ali Farhadi',
                year: '2018',
                description: '다중 스케일 예측 - 작은 객체 검출 개선',
                link: 'https://arxiv.org/abs/1804.02767'
              },
              {
                title: 'YOLOv7: Trainable Bag-of-Freebies',
                authors: 'Chien-Yao Wang, et al.',
                year: '2022',
                description: '추가 비용 없는 정확도 향상 - SOTA 속도/정확도 트레이드오프',
                link: 'https://arxiv.org/abs/2207.02696'
              },
              {
                title: 'YOLOv8',
                authors: 'Ultralytics',
                year: '2023',
                description: '최신 YOLO - 사용하기 쉬운 Python API',
                link: 'https://docs.ultralytics.com/'
              }
            ]
          },
          {
            title: 'R-CNN Family',
            icon: 'paper' as const,
            color: 'border-cyan-500',
            items: [
              {
                title: 'Rich Feature Hierarchies (R-CNN)',
                authors: 'Ross Girshick, et al.',
                year: '2014',
                description: 'Region-based CNN - 객체 검출의 패러다임 전환',
                link: 'https://arxiv.org/abs/1311.2524'
              },
              {
                title: 'Fast R-CNN',
                authors: 'Ross Girshick',
                year: '2015',
                description: 'R-CNN 9배 가속 - RoI Pooling 도입',
                link: 'https://arxiv.org/abs/1504.08083'
              },
              {
                title: 'Faster R-CNN',
                authors: 'Shaoqing Ren, et al.',
                year: '2015',
                description: 'Region Proposal Network - End-to-end 학습',
                link: 'https://arxiv.org/abs/1506.01497'
              },
              {
                title: 'Mask R-CNN',
                authors: 'Kaiming He, et al.',
                year: '2017',
                description: '인스턴스 세그멘테이션 - 객체별 픽셀 마스크',
                link: 'https://arxiv.org/abs/1703.06870'
              }
            ]
          },
          {
            title: 'Modern Detectors',
            icon: 'paper' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Focal Loss for Dense Object Detection (RetinaNet)',
                authors: 'Tsung-Yi Lin, et al.',
                year: '2017',
                description: 'Focal Loss로 클래스 불균형 해결 - One-stage SOTA',
                link: 'https://arxiv.org/abs/1708.02002'
              },
              {
                title: 'End-to-End Object Detection with Transformers (DETR)',
                authors: 'Nicolas Carion, et al.',
                year: '2020',
                description: 'Transformer 기반 검출 - NMS 불필요',
                link: 'https://arxiv.org/abs/2005.12872'
              },
              {
                title: 'EfficientDet',
                authors: 'Mingxing Tan, et al.',
                year: '2020',
                description: 'Compound Scaling - 효율적인 객체 검출',
                link: 'https://arxiv.org/abs/1911.09070'
              }
            ]
          },
          {
            title: 'Datasets & Benchmarks',
            icon: 'paper' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'Microsoft COCO Dataset',
                authors: 'Tsung-Yi Lin, et al.',
                year: '2014',
                description: '80 클래스, 330K 이미지 - 객체 검출의 표준 벤치마크',
                link: 'https://cocodataset.org/'
              },
              {
                title: 'PASCAL VOC Challenge',
                authors: 'Mark Everingham, et al.',
                year: '2010',
                description: '20 클래스 - 객체 검출/세그멘테이션 벤치마크',
                link: 'http://host.robots.ox.ac.uk/pascal/VOC/'
              }
            ]
          },
          {
            title: 'Tools & Frameworks',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Ultralytics YOLOv8',
                authors: 'Ultralytics',
                year: '2024',
                description: 'YOLO 공식 구현 - Detection, Segmentation, Pose',
                link: 'https://github.com/ultralytics/ultralytics'
              },
              {
                title: 'Detectron2',
                authors: 'Facebook AI Research',
                year: '2024',
                description: 'Meta의 객체 검출 라이브러리 - Mask R-CNN, RetinaNet',
                link: 'https://github.com/facebookresearch/detectron2'
              },
              {
                title: 'MMDetection',
                authors: 'OpenMMLab',
                year: '2024',
                description: '50+ 검출 모델 - PyTorch 기반 통합 툴박스',
                link: 'https://github.com/open-mmlab/mmdetection'
              },
              {
                title: 'Roboflow',
                authors: 'Roboflow',
                year: '2024',
                description: 'CV 데이터셋 관리 - 라벨링, 증강, 배포',
                link: 'https://roboflow.com/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}