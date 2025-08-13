'use client'

import React from 'react'

export default function Chapter3() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h2>컴퓨터 비전과 인식</h2>
      
      <h3>1. 실시간 객체 탐지</h3>
      <p>
        Physical AI가 현실 세계와 상호작용하기 위해서는 주변 환경을 정확하게 인식해야 합니다.
      </p>

      <div className="bg-teal-50 dark:bg-teal-900/20 p-6 rounded-lg my-6">
        <h4 className="font-semibold mb-3">YOLO 실시간 탐지</h4>
        <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`import cv2
import torch

class YOLODetector:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = 0.45  # 신뢰도 임계값
        
    def detect(self, frame):
        results = self.model(frame)
        detections = results.pandas().xyxy[0]
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), \\
                             int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, detections`}
        </pre>
      </div>
    </div>
  )
}