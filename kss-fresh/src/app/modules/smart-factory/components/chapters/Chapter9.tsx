'use client';

import {
  Activity, Wrench, TestTube, Brain, AlertTriangle, Code
} from 'lucide-react';
import CodeEditor from '../CodeEditor';
import Link from 'next/link';
import References from '@/components/common/References';

export default function Chapter9() {
  return (
    <div className="space-y-8">
      <div className="grid lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <Wrench className="w-6 h-6 text-slate-600" />
            유지보수 전략 비교
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-red-50 dark:bg-red-900/20 border-l-4 border-red-400 rounded">
              <h4 className="font-semibold text-red-800 dark:text-red-300 mb-2">사후보전 (Reactive Maintenance)</h4>
              <p className="text-sm text-red-700 dark:text-red-400 mb-2">고장 발생 후 수리하는 전통적 방식</p>
              <ul className="text-xs text-red-600 dark:text-red-400 space-y-1">
                <li>• 예상치 못한 다운타임: 평균 8-12시간</li>
                <li>• 생산 중단 손실: 시간당 $50,000-200,000</li>
                <li>• 응급 부품 비용: 정상 가격의 3-5배</li>
              </ul>
            </div>
            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-400 rounded">
              <h4 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-2">예방보전 (Preventive Maintenance)</h4>
              <p className="text-sm text-yellow-700 dark:text-yellow-400 mb-2">일정 주기마다 정기 점검 및 교체</p>
              <ul className="text-xs text-yellow-600 dark:text-yellow-400 space-y-1">
                <li>• 계획된 다운타임으로 생산 영향 최소화</li>
                <li>• 과도한 예방 정비로 인한 비용 증가</li>
                <li>• 수명이 남은 부품도 교체하는 낭비</li>
              </ul>
            </div>
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-400 rounded">
              <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">예측보전 (Predictive Maintenance)</h4>
              <p className="text-sm text-blue-700 dark:text-blue-400 mb-2">AI 예측 기반 최적 정비 시점 결정</p>
              <ul className="text-xs text-blue-600 dark:text-blue-400 space-y-1">
                <li>• 다운타임 70% 감소</li>
                <li>• 정비 비용 25% 절약</li>
                <li>• 부품 수명 30% 연장</li>
              </ul>
            </div>
            <div className="p-4 bg-green-50 dark:bg-green-900/20 border-l-4 border-green-400 rounded">
              <h4 className="font-semibold text-green-800 dark:text-green-300 mb-2">CBM (Condition-Based Maintenance)</h4>
              <p className="text-sm text-green-700 dark:text-green-400 mb-2">실시간 상태 모니터링 기반 즉시 대응</p>
              <ul className="text-xs text-green-600 dark:text-green-400 space-y-1">
                <li>• 24/7 연속 모니터링</li>
                <li>• 이상 징후 즉시 알람</li>
                <li>• 최적 정비 작업 자동 스케줄링</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
            <TestTube className="w-6 h-6 text-slate-600" />
            센서 기반 상태 모니터링
          </h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded border">
              <div>
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">진동 센서</h4>
                <p className="text-sm text-slate-600 dark:text-slate-400">베어링, 기어박스 이상 탐지</p>
              </div>
              <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">FFT</div>
            </div>
            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded border">
              <div>
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">온도 센서</h4>
                <p className="text-sm text-slate-600 dark:text-slate-400">과열, 냉각 시스템 문제</p>
              </div>
              <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">IR</div>
            </div>
            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded border">
              <div>
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">전류 센서</h4>
                <p className="text-sm text-slate-600 dark:text-slate-400">모터 부하, 전기적 이상</p>
              </div>
              <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">MCSA</div>
            </div>
            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded border">
              <div>
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">소음 센서</h4>
                <p className="text-sm text-slate-600 dark:text-slate-400">비정상 작동음 감지</p>
              </div>
              <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">dB</div>
            </div>
            <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded border">
              <div>
                <h4 className="font-semibold text-slate-800 dark:text-slate-200">오일 분석</h4>
                <p className="text-sm text-slate-600 dark:text-slate-400">윤활유 상태, 마모 입자</p>
              </div>
              <div className="text-2xl font-bold text-slate-700 dark:text-slate-300">PPM</div>
            </div>
          </div>
        </div>
      </div>

      {/* 시뮬레이터 체험 섹션 */}
      <div className="mt-8 p-6 bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-xl border border-emerald-200 dark:border-emerald-800">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-emerald-900 dark:text-emerald-200 mb-2">
            🎮 예측 유지보수 시뮬레이터 체험
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white dark:bg-emerald-800/20 p-4 rounded-lg border border-emerald-200 dark:border-emerald-700">
              <h4 className="font-medium text-emerald-800 dark:text-emerald-300 mb-2">예측 정비 실험실</h4>
              <p className="text-sm text-emerald-700 dark:text-emerald-400 mb-3">
                실시간 장비 데이터 분석과 고장 예측 시스템을 체험해보세요.
              </p>
              <Link
                href="/modules/smart-factory/simulators/predictive-maintenance-lab?from=/modules/smart-factory/predictive-maintenance"
                className="inline-flex items-center gap-2 px-3 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg transition-colors text-sm"
              >
                <span>시뮬레이터 열기</span>
                <span>→</span>
              </Link>
            </div>
            <div className="bg-white dark:bg-teal-800/20 p-4 rounded-lg border border-teal-200 dark:border-teal-700">
              <h4 className="font-medium text-teal-800 dark:text-teal-300 mb-2">베어링 고장 예측</h4>
              <p className="text-sm text-teal-700 dark:text-teal-400 mb-3">
                진동 데이터 기반 베어링 수명 예측 AI 시스템을 체험해보세요.
              </p>
              <Link
                href="/modules/smart-factory/simulators/bearing-failure-prediction?from=/modules/smart-factory/predictive-maintenance"
                className="inline-flex items-center gap-2 px-3 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg transition-colors text-sm"
              >
                <span>시뮬레이터 열기</span>
                <span>→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-8 rounded-xl border border-blue-200 dark:border-blue-800">
        <h3 className="text-2xl font-bold text-blue-900 dark:text-blue-200 mb-6 flex items-center gap-3">
          <Brain className="w-8 h-8" />
          RUL (Remaining Useful Life) 예측 모델
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-white dark:bg-blue-800/30 p-6 rounded-lg border border-blue-300 dark:border-blue-600">
            <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-3">통계적 방법</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">Weibull 분포</span>
                <span className="font-mono text-blue-600 dark:text-blue-400">β, η</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">지수 분포</span>
                <span className="font-mono text-blue-600 dark:text-blue-400">λ</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">감마 분포</span>
                <span className="font-mono text-blue-600 dark:text-blue-400">α, β</span>
              </div>
            </div>
            <p className="text-xs text-blue-600 dark:text-blue-400 mt-3">
              고전적 신뢰성 이론 기반, 단순하지만 정확도 제한
            </p>
          </div>
          
          <div className="bg-white dark:bg-blue-800/30 p-6 rounded-lg border border-blue-300 dark:border-blue-600">
            <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-3">물리 기반 모델</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">피로 균열 성장</span>
                <span className="font-mono text-blue-600 dark:text-blue-400">da/dN</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">마모 모델</span>
                <span className="font-mono text-blue-600 dark:text-blue-400">Archard</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">부식 모델</span>
                <span className="font-mono text-blue-600 dark:text-blue-400">Tafel</span>
              </div>
            </div>
            <p className="text-xs text-blue-600 dark:text-blue-400 mt-3">
              물리적 원리 기반, 높은 정확도하지만 복잡한 모델링
            </p>
          </div>
          
          <div className="bg-white dark:bg-blue-800/30 p-6 rounded-lg border border-blue-300 dark:border-blue-600">
            <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-3">AI/ML 모델</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">LSTM</span>
                <span className="font-mono text-blue-600 dark:text-blue-400">시계열</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">Random Forest</span>
                <span className="font-mono text-blue-600 dark:text-blue-400">앙상블</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-700 dark:text-blue-300">CNN</span>
                <span className="font-mono text-blue-600 dark:text-blue-400">신호처리</span>
              </div>
            </div>
            <p className="text-xs text-blue-600 dark:text-blue-400 mt-3">
              빅데이터 기반 학습, 복잡한 패턴 인식 가능
            </p>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 p-8 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-3">
          <AlertTriangle className="w-8 h-8 text-amber-600" />
          고장 모드 분석 (FMEA)
        </h3>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">장비</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">고장 모드</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">원인</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">영향</th>
                <th className="text-center py-3 px-4 font-semibold text-gray-900 dark:text-white">RPN</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">예측 방법</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              <tr>
                <td className="py-3 px-4 text-gray-900 dark:text-white font-medium">베어링</td>
                <td className="py-3 px-4 text-gray-700 dark:text-gray-300">내륜 박리</td>
                <td className="py-3 px-4 text-gray-600 dark:text-gray-400">윤활 부족, 과부하</td>
                <td className="py-3 px-4 text-red-600 dark:text-red-400">회전축 손상</td>
                <td className="py-3 px-4 text-center font-bold text-red-700 dark:text-red-300">320</td>
                <td className="py-3 px-4 text-blue-600 dark:text-blue-400">진동 FFT 분석</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-gray-900 dark:text-white font-medium">기어박스</td>
                <td className="py-3 px-4 text-gray-700 dark:text-gray-300">치면 피팅</td>
                <td className="py-3 px-4 text-gray-600 dark:text-gray-400">표면 거칠기, 오염</td>
                <td className="py-3 px-4 text-orange-600 dark:text-orange-400">효율 저하</td>
                <td className="py-3 px-4 text-center font-bold text-orange-700 dark:text-orange-300">240</td>
                <td className="py-3 px-4 text-blue-600 dark:text-blue-400">오일 입자 분석</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-gray-900 dark:text-white font-medium">모터</td>
                <td className="py-3 px-4 text-gray-700 dark:text-gray-300">권선 절연파괴</td>
                <td className="py-3 px-4 text-gray-600 dark:text-gray-400">과열, 습기</td>
                <td className="py-3 px-4 text-red-600 dark:text-red-400">모터 소손</td>
                <td className="py-3 px-4 text-center font-bold text-red-700 dark:text-red-300">300</td>
                <td className="py-3 px-4 text-blue-600 dark:text-blue-400">절연저항 측정</td>
              </tr>
              <tr>
                <td className="py-3 px-4 text-gray-900 dark:text-white font-medium">펌프</td>
                <td className="py-3 px-4 text-gray-700 dark:text-gray-300">임펠러 마모</td>
                <td className="py-3 px-4 text-gray-600 dark:text-gray-400">캐비테이션, 이물질</td>
                <td className="py-3 px-4 text-yellow-600 dark:text-yellow-400">유량 감소</td>
                <td className="py-3 px-4 text-center font-bold text-yellow-700 dark:text-yellow-300">160</td>
                <td className="py-3 px-4 text-blue-600 dark:text-blue-400">압력/유량 모니터링</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="mt-4 text-xs text-gray-500 dark:text-gray-400">
          <p><strong>RPN (Risk Priority Number)</strong> = 발생도 × 심각도 × 검출도 (각 1-10점)</p>
          <p>200 이상: 즉시 대응 필요, 100-199: 주의 관찰, 100 미만: 정기 점검</p>
        </div>
      </div>

      <div className="bg-green-50 dark:bg-green-900/20 p-8 rounded-xl border border-green-200 dark:border-green-800">
        <h3 className="text-2xl font-bold text-green-900 dark:text-green-200 mb-6 flex items-center gap-3">
          <Code className="w-8 h-8" />
          실습: 베어링 고장 예측 모델 개발
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">1단계: 데이터 수집 및 전처리</h4>
            <CodeEditor 
              code={`import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 진동 데이터 로드
data = pd.read_csv('bearing_vibration.csv')
features = ['rms', 'peak', 'kurtosis', 'skewness']

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# 레이블 생성 (RUL 기준)
y = data['remaining_life']`}
              language="python"
              title="1단계: 데이터 수집 및 전처리"
              filename="data_preprocessing.py"
              maxHeight="300px"
            />
          </div>
          
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">2단계: LSTM 모델 구축</h4>
            <CodeEditor 
              code={`from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, return_sequences=True, 
         input_shape=(sequence_length, n_features)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)  # RUL 예측
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100)`}
              language="python"
              title="2단계: LSTM 모델 구축"
              filename="lstm_model.py"
              maxHeight="300px"
            />
          </div>
          
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">3단계: 알람 시스템</h4>
            <CodeEditor 
              code={`def check_bearing_health(sensor_data):
    rul_prediction = model.predict(sensor_data)
    
    if rul_prediction < 24:  # 24시간 미만
        send_alert("CRITICAL", "즉시 교체 필요")
    elif rul_prediction < 168:  # 1주일 미만
        send_alert("WARNING", "정비 계획 수립")
    
    return rul_prediction`}
              language="python"
              title="3단계: 알람 시스템"
              filename="alert_system.py"
              maxHeight="250px"
            />
          </div>
          
          <div className="bg-white dark:bg-green-800/30 p-6 rounded-lg">
            <h4 className="font-bold text-green-800 dark:text-green-200 mb-4">4단계: 최적화 스케줄링</h4>
            <CodeEditor 
              code={`from ortools.linear_solver import pywraplp

def optimize_maintenance_schedule():
    solver = pywraplp.Solver.CreateSolver('SCIP')
    
    # 결정 변수: 각 장비별 정비 시점
    maintenance_time = {}
    for equipment in equipment_list:
        maintenance_time[equipment] = solver.IntVar(
            0, planning_horizon, f'maint_{equipment}')
    
    # 제약 조건: 생산 계획과 충돌 방지
    # 목적 함수: 정비 비용 최소화
    solver.Minimize(total_maintenance_cost)`}
              language="python"
              title="4단계: 최적화 스케줄링"
              filename="maintenance_scheduling.py"
              maxHeight="300px"
            />
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-green-100 dark:bg-green-800/50 rounded-lg">
          <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">🎯 기대 효과</h4>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="font-semibold text-green-700 dark:text-green-300">다운타임 감소</span>
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">-70%</div>
            </div>
            <div>
              <span className="font-semibold text-green-700 dark:text-green-300">정비비용 절약</span>
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">-25%</div>
            </div>
            <div>
              <span className="font-semibold text-green-700 dark:text-green-300">부품수명 연장</span>
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">+30%</div>
            </div>
          </div>
        </div>
      </div>

      <References
        sections={[
          {
            title: '📚 공식 표준 & 문서',
            icon: 'web' as const,
            color: 'border-emerald-500',
            items: [
              {
                title: 'ISO 13381 - Condition Monitoring and Diagnostics of Machines',
                url: 'https://www.iso.org/standard/51436.html',
                description: '기계 상태 모니터링 및 진단을 위한 국제 표준 - 예지 정비 방법론 가이드'
              },
              {
                title: 'NASA Prognostics Center of Excellence',
                url: 'https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/',
                description: 'NASA의 예측 정비 및 시스템 건강성 관리 연구 센터 - RUL 예측 알고리즘'
              },
              {
                title: 'ISO 14224 - Reliability & Maintenance Data',
                url: 'https://www.iso.org/standard/64520.html',
                description: '석유, 천연가스, 석유화학 산업의 신뢰성 및 정비 데이터 수집 표준'
              },
              {
                title: 'Microsoft Azure Predictive Maintenance Guide',
                url: 'https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-automated-ml-for-ml-models',
                description: 'Azure ML을 활용한 예측 정비 시스템 구축 가이드 - 실전 템플릿 제공'
              },
              {
                title: 'IEC 61508 - Functional Safety for Industrial Systems',
                url: 'https://www.iec.ch/functionalsafety/',
                description: '산업 시스템의 기능 안전성 표준 - 예측 정비와 안전성 통합 가이드'
              }
            ]
          },
          {
            title: '🔬 핵심 논문 & 연구',
            icon: 'research' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'LSTM Networks for Remaining Useful Life Prediction (IEEE, 2019)',
                url: 'https://ieeexplore.ieee.org/document/8642543',
                description: 'LSTM 기반 잔존 수명(RUL) 예측 모델 - CMAPSS 데이터셋 검증 (95.2% 정확도)'
              },
              {
                title: 'Deep Learning for Predictive Maintenance (Nature Machine Intelligence, 2021)',
                url: 'https://www.nature.com/articles/s42256-021-00349-1',
                description: 'CNN과 LSTM 하이브리드 모델을 활용한 베어링 고장 예측 연구'
              },
              {
                title: 'PHM Society Data Challenge - Bearing Fault Diagnosis',
                url: 'https://www.phmsociety.org/competition/phm/09',
                description: 'PHM Society의 베어링 고장 진단 챌린지 데이터셋 및 우수 알고리즘 공개'
              },
              {
                title: 'Transfer Learning for Cross-Machine Failure Prediction (Mechanical Systems and Signal Processing, 2020)',
                url: 'https://www.sciencedirect.com/science/article/abs/pii/S0888327020302387',
                description: '전이 학습을 활용한 다양한 장비 간 고장 예측 모델 적용 연구'
              }
            ]
          },
          {
            title: '🛠️ 실전 도구 & 플랫폼',
            icon: 'tools' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'AWS Lookout for Equipment',
                url: 'https://aws.amazon.com/lookout-for-equipment/',
                description: 'AWS의 산업 장비 이상 탐지 서비스 - 센서 데이터 기반 자동 ML 모델 생성'
              },
              {
                title: 'SKF Enlight AI - Bearing Condition Monitoring',
                url: 'https://www.skf.com/us/products/condition-monitoring-systems/skf-enlight-ai',
                description: '베어링 전문업체 SKF의 AI 예측 정비 플랫폼 - 진동 분석 및 수명 예측'
              },
              {
                title: 'C3 AI Predictive Maintenance',
                url: 'https://c3.ai/products/c3-ai-predictive-maintenance/',
                description: '엔터프라이즈급 예측 정비 솔루션 - IoT 통합 및 대규모 장비 관리'
              },
              {
                title: 'PyODDS - Python Outlier Detection Library',
                url: 'https://github.com/datamllab/pyodds',
                description: '시계열 이상 탐지를 위한 오픈소스 파이썬 라이브러리 - 10+ 알고리즘 제공'
              },
              {
                title: 'PyCaret - AutoML for Predictive Maintenance',
                url: 'https://pycaret.org/',
                description: '예측 정비 모델 개발을 위한 Low-Code ML 라이브러리 - 자동 하이퍼파라미터 튜닝'
              }
            ]
          }
        ]}
      />
    </div>
  );
}