'use client';

import React from 'react';

export default function Chapter5() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h2>디지털 트윈과 시뮬레이션</h2>
      
      <h3>1. 디지털 트윈 개념</h3>
      <p>
        디지털 트윈은 물리적 시스템의 실시간 디지털 복제본으로, 
        Physical AI의 핵심 기술입니다.
      </p>

      <div className="bg-cyan-50 dark:bg-cyan-900/20 p-6 rounded-lg my-6">
        <h4 className="font-semibold mb-3">디지털 트윈 구현</h4>
        <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`class DigitalTwin:
    def __init__(self, physical_system):
        self.physical_state = physical_system.get_state()
        self.simulation_model = self.create_model()
        self.sync_interval = 0.1  # 100ms
        
    def synchronize(self):
        # 물리 시스템에서 센서 데이터 수집
        sensor_data = self.collect_sensor_data()
        
        # 디지털 모델 업데이트
        self.update_model(sensor_data)
        
        # 예측 및 최적화
        predictions = self.predict_future_states()
        optimizations = self.optimize_performance()
        
        return predictions, optimizations`}
        </pre>
      </div>
    </div>
  )
}