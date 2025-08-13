'use client'

import React from 'react'

export default function Chapter6() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h2>센서 융합과 상태 추정</h2>
      
      <h3>1. 칼만 필터</h3>
      <p>
        여러 센서의 노이즈가 있는 측정값을 융합하여 정확한 상태를 추정합니다.
      </p>

      <div className="bg-pink-50 dark:bg-pink-900/20 p-6 rounded-lg my-6">
        <h4 className="font-semibold mb-3">확장 칼만 필터 (EKF)</h4>
        <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`class ExtendedKalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x  # 상태 벡터 차원
        self.dim_z = dim_z  # 측정 벡터 차원
        
        self.x = np.zeros(dim_x)  # 상태 추정값
        self.P = np.eye(dim_x)     # 오차 공분산
        self.Q = np.eye(dim_x) * 0.1  # 프로세스 노이즈
        self.R = np.eye(dim_z) * 1.0  # 측정 노이즈
        
    def predict(self, f, F_jacobian):
        # 예측 단계
        self.x = f(self.x)  # 비선형 상태 전이
        F = F_jacobian(self.x)  # 야코비안
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z, h, H_jacobian):
        # 업데이트 단계
        y = z - h(self.x)  # 혁신
        H = H_jacobian(self.x)  # 측정 야코비안
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)  # 칼만 게인
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ H) @ self.P`}
        </pre>
      </div>
    </div>
  )
}