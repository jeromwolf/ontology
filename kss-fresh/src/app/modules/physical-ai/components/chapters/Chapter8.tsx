'use client';

import React from 'react';

export default function Chapter8() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h2>Physical AI 응용 사례</h2>
      
      <h3>1. 산업용 로봇</h3>
      <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-lg my-6">
        <h4 className="font-semibold mb-3">협동 로봇 (Cobot) 시스템</h4>
        <ul className="space-y-2">
          <li>• <strong>Universal Robots</strong>: 힘 제어 기반 안전 협업</li>
          <li>• <strong>ABB YuMi</strong>: 듀얼 암 정밀 조립</li>
          <li>• <strong>KUKA LBR iiwa</strong>: 7축 민감 로봇</li>
        </ul>
      </div>

      <h3>2. 자율주행 시스템</h3>
      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg my-6">
        <h4 className="font-semibold mb-3">Tesla FSD (Full Self-Driving)</h4>
        <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`# Tesla의 비전 기반 접근
class TeslaFSD:
    def __init__(self):
        self.cameras = [Camera() for _ in range(8)]
        self.neural_net = HydraNet()  # 다중 작업 신경망
        
    def process_frame(self):
        # 8개 카메라에서 동시 입력
        images = [cam.capture() for cam in self.cameras]
        
        # 단일 신경망으로 다중 작업 처리
        outputs = self.neural_net.predict(images)
        
        return {
            'objects': outputs['detection'],
            'lanes': outputs['lane_detection'],
            'depth': outputs['depth_estimation'],
            'motion': outputs['optical_flow']
        }`}
        </pre>
      </div>

      <h3>3. 드론과 UAV</h3>
      <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg my-6">
        <h4 className="font-semibold mb-3">자율 비행 제어</h4>
        <ul className="space-y-2">
          <li>• <strong>DJI</strong>: 장애물 회피와 자동 귀환</li>
          <li>• <strong>Skydio</strong>: AI 기반 자율 추적</li>
          <li>• <strong>Wing (Google)</strong>: 배송 드론 네비게이션</li>
        </ul>
      </div>

      <h3>4. 휴머노이드 로봇</h3>
      <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg my-6">
        <h4 className="font-semibold mb-3">Boston Dynamics Atlas</h4>
        <p className="mb-3">
          전신 제어와 동적 균형을 통한 파쿠르, 백플립 등 고난도 동작 수행
        </p>
        <ul className="space-y-2">
          <li>• <strong>모델 예측 제어</strong>: 100Hz 전신 궤적 최적화</li>
          <li>• <strong>접촉 감지</strong>: 발과 손의 힘 센서</li>
          <li>• <strong>시각 인식</strong>: 지형 매핑과 장애물 감지</li>
        </ul>
      </div>
    </div>
  )
}