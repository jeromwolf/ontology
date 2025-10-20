'use client'

import React from 'react'
import ChapterNavigation from '../ChapterNavigation'

export default function Chapter8() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-gradient-to-r from-orange-600 to-red-600 rounded-2xl p-8 mb-8 text-white">
        <h1 className="text-4xl font-bold mb-4">Chapter 8: 협동 로봇 (Collaborative Robots)</h1>
        <p className="text-xl text-white/90">
          사람과 함께 일하는 안전한 코봇 기술
        </p>
      </div>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        {/* 1. Cobot Overview */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            1. 협동 로봇이란?
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              정의
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              <strong>협동 로봇(Collaborative Robot, Cobot)</strong>은 안전 울타리 없이
              사람과 같은 작업 공간에서 직접 협력할 수 있도록 설계된 로봇입니다.
              내장된 안전 센서와 힘 제한 기능으로 사람과의 충돌 시 위험을 최소화합니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            산업용 로봇 vs 협동 로봇
          </h3>

          <table className="min-w-full border border-blue-300 dark:border-blue-700 mt-3 mb-6">
            <thead className="bg-blue-100 dark:bg-blue-900/50">
              <tr>
                <th className="px-4 py-2 border-b text-left">특징</th>
                <th className="px-4 py-2 border-b text-left">산업용 로봇</th>
                <th className="px-4 py-2 border-b text-left">협동 로봇</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">안전 울타리</td>
                <td className="px-4 py-2">필수</td>
                <td className="px-4 py-2">불필요 (상황에 따라 선택)</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">동작 속도</td>
                <td className="px-4 py-2">빠름 (최대 2-3 m/s)</td>
                <td className="px-4 py-2">느림 (최대 0.5-1 m/s)</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">힘/토크 제한</td>
                <td className="px-4 py-2">없음</td>
                <td className="px-4 py-2">있음 (ISO/TS 15066 준수)</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">가반 하중</td>
                <td className="px-4 py-2">높음 (수백 kg)</td>
                <td className="px-4 py-2">낮음 (3-35 kg)</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">프로그래밍</td>
                <td className="px-4 py-2">전문가 필요</td>
                <td className="px-4 py-2">직관적 (Hand Guiding)</td>
              </tr>
              <tr>
                <td className="px-4 py-2 font-semibold">유연성</td>
                <td className="px-4 py-2">고정 설치</td>
                <td className="px-4 py-2">이동 가능</td>
              </tr>
            </tbody>
          </table>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border-l-4 border-green-500">
            <h3 className="text-xl font-semibold text-green-700 dark:text-green-300 mb-3">
              주요 제조사 및 제품
            </h3>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>Universal Robots</strong>: UR3e, UR5e, UR10e, UR16e, UR20, UR30</li>
              <li><strong>FANUC</strong>: CR-7iA, CR-15iA (초록색 외관)</li>
              <li><strong>ABB</strong>: YuMi (듀얼 암), GoFa, SWIFTI</li>
              <li><strong>KUKA</strong>: LBR iiwa (7-DOF, 민감한 토크 센싱)</li>
              <li><strong>Doosan Robotics</strong>: M0609, M1013, H2017 (한국)</li>
            </ul>
          </div>
        </section>

        {/* 2. Safety Standards */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            2. 안전 기준 (ISO/TS 15066)
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              ISO/TS 15066이란?
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              협동 로봇의 안전을 규정하는 국제 표준입니다. ISO 10218-1/2 (산업용 로봇 안전)의
              보완 기술 사양으로, 인간-로봇 협업 시나리오에 대한 구체적 요구사항을 제시합니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            4가지 협업 운전 모드
          </h3>

          <div className="space-y-4 mb-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border-l-4 border-purple-500">
              <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">
                1. Safety-Rated Monitored Stop (안전 등급 모니터링 정지)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                작업자가 협업 작업 공간에 진입하면 로봇이 즉시 정지합니다.
              </p>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li><strong>특징</strong>: 로봇은 정지 상태에서만 안전</li>
                <li><strong>센서</strong>: 라이트 커튼, 레이저 스캐너</li>
                <li><strong>응용</strong>: 로봇이 부품 고정, 작업자가 조립</li>
              </ul>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border-l-4 border-green-500">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">
                2. Hand Guiding (핸드 가이딩)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                작업자가 로봇을 직접 손으로 잡고 움직여 프로그래밍합니다.
              </p>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li><strong>특징</strong>: Teach Pendant 없이 직관적 프로그래밍</li>
                <li><strong>센서</strong>: 6축 Force/Torque 센서</li>
                <li><strong>응용</strong>: 복잡한 경로 티칭, 미세 조정</li>
              </ul>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 border-l-4 border-orange-500">
              <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-2">
                3. Speed and Separation Monitoring (속도 및 거리 모니터링)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                작업자와 로봇 사이의 거리에 따라 로봇 속도를 자동 조절합니다.
              </p>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li><strong>특징</strong>: 가까워질수록 로봇 속도 감소</li>
                <li><strong>센서</strong>: 3D 비전, LiDAR, Time-of-Flight 카메라</li>
                <li><strong>응용</strong>: 작업자가 간헐적으로 접근하는 작업</li>
              </ul>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border-l-4 border-red-500">
              <h4 className="font-semibold text-red-700 dark:text-red-300 mb-2">
                4. Power and Force Limiting (출력 및 힘 제한)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                로봇이 작업자와 접촉해도 안전한 수준으로 힘과 압력을 제한합니다.
              </p>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li><strong>특징</strong>: 가장 유연한 협업 모드</li>
                <li><strong>센서</strong>: 관절 토크 센서, 표면 압력 센서</li>
                <li><strong>응용</strong>: 지속적인 인간-로봇 물리적 상호작용</li>
              </ul>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            생체역학적 한계값 (Biomechanical Limits)
          </h3>

          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-6 mb-6">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              ISO/TS 15066은 신체 부위별로 허용 가능한 압력과 힘의 한계값을 정의합니다.
            </p>

            <table className="min-w-full border border-blue-300 dark:border-blue-700 mt-3">
              <thead className="bg-blue-100 dark:bg-blue-900/50">
                <tr>
                  <th className="px-4 py-2 border-b text-left">신체 부위</th>
                  <th className="px-4 py-2 border-b text-left">최대 압력 (N/cm²)</th>
                  <th className="px-4 py-2 border-b text-left">최대 힘 (N)</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                <tr className="border-b">
                  <td className="px-4 py-2 font-semibold">두개골 (Skull)</td>
                  <td className="px-4 py-2">130</td>
                  <td className="px-4 py-2">-</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-2 font-semibold">이마 (Forehead)</td>
                  <td className="px-4 py-2">130</td>
                  <td className="px-4 py-2">-</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-2 font-semibold">가슴 (Chest)</td>
                  <td className="px-4 py-2">110</td>
                  <td className="px-4 py-2">-</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-2 font-semibold">손바닥 (Palm)</td>
                  <td className="px-4 py-2">140</td>
                  <td className="px-4 py-2">-</td>
                </tr>
                <tr>
                  <td className="px-4 py-2 font-semibold">손가락 (Finger)</td>
                  <td className="px-4 py-2">-</td>
                  <td className="px-4 py-2">140</td>
                </tr>
              </tbody>
            </table>

            <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
              * 준정적(Quasi-static) 접촉 기준. 동적 충돌 시 더 낮은 값 적용.
            </p>
          </div>
        </section>

        {/* 3. Force/Torque Control */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            3. Force/Torque 제어
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              중요성
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              협동 로봇의 핵심 기술로, 로봇이 환경 및 사람과의 상호작용 시 가하는 힘을 정밀하게
              제어합니다. 이를 통해 안전성과 작업 품질을 동시에 보장합니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            3.1 Force/Torque 센서
          </h3>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 mb-6 border-l-4 border-purple-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              로봇의 엔드 이펙터 또는 각 관절에 장착되어 외부 힘을 측정합니다.
            </p>

            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-3">
              센서 종류
            </h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li>
                <strong>6축 F/T 센서</strong>: 3축 힘(Fx, Fy, Fz) + 3축 토크(Tx, Ty, Tz)
                <ul className="list-disc list-inside ml-6 mt-1 text-sm">
                  <li>위치: 엔드 이펙터와 손목 사이</li>
                  <li>대표 제품: ATI Gamma, OnRobot HEX-E</li>
                </ul>
              </li>
              <li>
                <strong>관절 토크 센서</strong>: 각 관절의 토크 측정
                <ul className="list-disc list-inside ml-6 mt-1 text-sm">
                  <li>위치: 모터와 감속기 사이</li>
                  <li>장점: 외부 센서 불필요, 컴팩트</li>
                </ul>
              </li>
            </ul>

            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-3">
              측정 원리
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              스트레인 게이지(Strain Gauge) 또는 정전용량 센서를 사용하여 변형을 측정하고,
              이를 힘/토크로 변환합니다. 일반적으로 0.01-1N 정도의 분해능을 가집니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            3.2 임피던스 제어 (Impedance Control)
          </h3>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 mb-6 border-l-4 border-green-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              로봇의 동적 특성을 질량-스프링-댐퍼 시스템으로 모델링하여 제어하는 방법입니다.
            </p>

            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-4">
              <p className="text-sm font-mono text-gray-800 dark:text-gray-200 mb-2">
                임피던스 제어 수식:
              </p>
              <p className="text-center text-lg font-semibold text-green-700 dark:text-green-300 mb-2">
                M(ẍ - ẍ<sub>d</sub>) + B(ẋ - ẋ<sub>d</sub>) + K(x - x<sub>d</sub>) = F<sub>ext</sub>
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
                • M: 질량 (Inertia)<br />
                • B: 감쇠 (Damping)<br />
                • K: 강성 (Stiffness)<br />
                • F<sub>ext</sub>: 외부 힘<br />
                • x, x<sub>d</sub>: 현재/목표 위치
              </p>
            </div>

            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3">
              파라미터 조정
            </h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>높은 K (강성)</strong>: 로봇이 단단함, 정확한 위치 추종</li>
              <li><strong>낮은 K (유연성)</strong>: 로봇이 부드러움, 힘에 순응</li>
              <li><strong>높은 B (감쇠)</strong>: 진동 억제, 안정성 향상</li>
              <li><strong>낮은 B</strong>: 빠른 응답, 에너지 효율적</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            3.3 충돌 감지 및 대응
          </h3>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 border-l-4 border-orange-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              예상치 못한 접촉을 실시간으로 감지하고 즉시 대응하는 시스템입니다.
            </p>

            <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-3">
              충돌 감지 알고리즘
            </h4>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-4">
              <pre className="text-sm overflow-x-auto">
                <code className="text-orange-600 dark:text-orange-400">{`# 외부 토크 추정
τ_ext = τ_measured - τ_model - τ_friction

# 충돌 임계값 비교
if |τ_ext| > τ_threshold:
    # 충돌 감지!
    trigger_emergency_stop()
    retract_to_safe_position()`}</code>
              </pre>
            </div>

            <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-3">
              대응 전략
            </h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>긴급 정지 (Emergency Stop)</strong>: 즉시 모든 움직임 중단</li>
              <li><strong>후퇴 (Retraction)</strong>: 접촉 방향 반대로 일정 거리 이동</li>
              <li><strong>힘 제한 (Force Limiting)</strong>: 힘을 안전 한계값 이하로 제한</li>
              <li><strong>순응 (Compliance)</strong>: 외부 힘에 따라 부드럽게 움직임</li>
            </ul>
          </div>
        </section>

        {/* 4. Human-Robot Interaction */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            4. 인간-로봇 상호작용 (HRI)
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              정의
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              <strong>Human-Robot Interaction (HRI)</strong>은 사람과 로봇 간의 효과적이고
              안전한 상호작용을 연구하는 분야입니다. 물리적, 인지적, 감정적 측면을 모두 포함합니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            4.1 인터페이스 종류
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800">
              <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">
                물리적 인터페이스
              </h4>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>Hand Guiding</li>
                <li>Force/Torque 피드백</li>
                <li>햅틱 디바이스</li>
              </ul>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border border-green-200 dark:border-green-800">
              <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">
                시각적 인터페이스
              </h4>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>Teach Pendant (터치스크린)</li>
                <li>AR/VR 시각화</li>
                <li>LED 상태 표시등</li>
              </ul>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 border border-orange-200 dark:border-orange-800">
              <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-2">
                음성 인터페이스
              </h4>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>음성 명령 인식</li>
                <li>음성 피드백</li>
                <li>자연어 처리 (NLP)</li>
              </ul>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border border-red-200 dark:border-red-800">
              <h4 className="font-semibold text-red-700 dark:text-red-300 mb-2">
                제스처 인터페이스
              </h4>
              <ul className="text-sm list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                <li>손 제스처 인식</li>
                <li>시선 추적 (Eye Gaze)</li>
                <li>자세 인식 (Pose Estimation)</li>
              </ul>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            4.2 의도 예측 (Intention Recognition)
          </h3>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 mb-6 border-l-4 border-purple-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              작업자의 다음 행동을 예측하여 로봇이 선제적으로 대응하는 기술입니다.
            </p>

            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-3">
              주요 접근법
            </h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li>
                <strong>시선 추적</strong>: 작업자가 보는 물체가 다음 목표
              </li>
              <li>
                <strong>손 궤적 예측</strong>: 손의 이동 방향과 속도로 의도 추론
              </li>
              <li>
                <strong>컨텍스트 기반</strong>: 작업 흐름에서 다음 단계 예측
              </li>
              <li>
                <strong>딥러닝 모델</strong>: LSTM, Transformer로 시계열 행동 학습
              </li>
            </ul>

            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-3">
              응용 예시
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              작업자가 볼트를 잡으려 손을 뻗으면, 로봇은 다음 작업을 위해 너트를 미리 가져와
              대기합니다. 이를 통해 작업 효율이 크게 향상됩니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            4.3 적응형 행동 (Adaptive Behavior)
          </h3>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border-l-4 border-green-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              로봇이 작업자의 선호도와 작업 스타일을 학습하여 행동을 조정합니다.
            </p>

            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3">
              학습 요소
            </h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li><strong>작업 속도</strong>: 숙련도에 맞춰 로봇 속도 조절</li>
              <li><strong>위치 선호</strong>: 작업자가 선호하는 부품 전달 위치 학습</li>
              <li><strong>타이밍</strong>: 작업자의 리듬에 맞춰 동작 시작</li>
              <li><strong>피로도 감지</strong>: 반응 시간 증가 시 휴식 제안</li>
            </ul>

            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3">
              학습 알고리즘
            </h4>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              강화학습(Reinforcement Learning), 모방학습(Imitation Learning),
              온라인 학습(Online Learning)을 활용하여 실시간으로 작업자에게 적응합니다.
            </p>
          </div>
        </section>

        {/* 5. Applications */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            5. 협동 로봇 응용 분야
          </h2>

          <div className="space-y-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 border-l-4 border-blue-500">
              <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
                1. 제조업 (Manufacturing)
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>조립</strong>: 나사 체결, 부품 끼워 맞춤 (자동차, 전자제품)</li>
                <li><strong>품질 검사</strong>: 비전 시스템과 결합하여 불량 검출</li>
                <li><strong>포장</strong>: 제품을 상자에 담고 밀봉</li>
                <li><strong>머신 텐딩</strong>: CNC 기계에 원자재 투입/제품 회수</li>
              </ul>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 border-l-4 border-purple-500">
              <h3 className="text-xl font-semibold text-purple-700 dark:text-purple-300 mb-3">
                2. 의료 (Healthcare)
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>수술 보조</strong>: 집도의에게 기구 전달, 조직 고정</li>
                <li><strong>재활 치료</strong>: 환자의 관절 운동 보조</li>
                <li><strong>약품 조제</strong>: 정확한 용량으로 약품 분배</li>
                <li><strong>검체 처리</strong>: 혈액 샘플 자동 분류 및 분석</li>
              </ul>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border-l-4 border-green-500">
              <h3 className="text-xl font-semibold text-green-700 dark:text-green-300 mb-3">
                3. 물류 (Logistics)
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>팔레타이징</strong>: 상자를 팔레트에 정렬하여 적재</li>
                <li><strong>피킹</strong>: 창고에서 주문 상품 선택</li>
                <li><strong>분류</strong>: 소포를 목적지별로 분류</li>
                <li><strong>검수</strong>: 바코드 스캔 및 재고 확인</li>
              </ul>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 border-l-4 border-orange-500">
              <h3 className="text-xl font-semibold text-orange-700 dark:text-orange-300 mb-3">
                4. 서비스 (Service)
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>음식 서빙</strong>: 레스토랑에서 음식 배달 (Servi, Bear Robotics)</li>
                <li><strong>청소</strong>: 상업 공간 자동 청소</li>
                <li><strong>안내</strong>: 공항, 쇼핑몰에서 길 안내 및 정보 제공</li>
                <li><strong>교육</strong>: 어린이 교육용 대화형 로봇</li>
              </ul>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-6 border-l-4 border-red-500">
              <h3 className="text-xl font-semibold text-red-700 dark:text-red-300 mb-3">
                5. 농업 (Agriculture)
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>수확</strong>: 과일, 채소 수확 (소프트 그리퍼 활용)</li>
                <li><strong>정밀 농업</strong>: 작물별 물, 비료 정밀 공급</li>
                <li><strong>가축 관리</strong>: 자동 급식, 건강 모니터링</li>
                <li><strong>온실 관리</strong>: 온도, 습도 자동 조절</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 6. Future Trends */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            6. 미래 트렌드
          </h2>

          <div className="space-y-4">
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border-l-4 border-purple-500">
              <h3 className="text-lg font-semibold text-purple-700 dark:text-purple-300 mb-2">
                AI 통합 협동 로봇
              </h3>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                GPT-4, Claude 등 대규모 언어 모델(LLM)과 통합되어 자연어 명령을 이해하고,
                복잡한 작업을 자율적으로 계획합니다. 예: "책상을 정리해줘" → 물체 인식,
                분류, 정리 전략 수립, 실행.
              </p>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border-l-4 border-green-500">
              <h3 className="text-lg font-semibold text-green-700 dark:text-green-300 mb-2">
                모바일 매니퓰레이터
              </h3>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                이동형 로봇(AMR)과 협동 매니퓰레이터를 결합한 형태. 넓은 작업 공간을 자유롭게
                이동하며 다양한 작업 수행. 예: Boston Dynamics Stretch, Fetch Robotics.
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 border-l-4 border-orange-500">
              <h3 className="text-lg font-semibold text-orange-700 dark:text-orange-300 mb-2">
                감정 인식 로봇
              </h3>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                작업자의 표정, 목소리 톤, 생체 신호를 분석하여 감정 상태를 파악하고,
                이에 맞춰 행동을 조정합니다. 스트레스가 감지되면 작업 속도를 늦추거나
                휴식을 권장합니다.
              </p>
            </div>

            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4 border-l-4 border-red-500">
              <h3 className="text-lg font-semibold text-red-700 dark:text-red-300 mb-2">
                5G/6G 기반 원격 협동 로봇
              </h3>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                초저지연 5G/6G 네트워크를 활용하여 전문가가 원격지에서 로봇을 실시간으로
                제어합니다. 원격 수술, 위험 지역 작업 등에 활용 가능.
              </p>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border-l-4 border-blue-500">
              <h3 className="text-lg font-semibold text-blue-700 dark:text-blue-300 mb-2">
                소프트 로보틱스 융합
              </h3>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                유연한 재료와 공압 액추에이터를 사용한 소프트 협동 로봇이 등장합니다.
                사람에게 더욱 안전하며, 불규칙한 물체 조작에 유리합니다.
                예: Soft Robotics Inc., Pneubotics.
              </p>
            </div>
          </div>
        </section>

        {/* 7. Summary */}
        <section className="mb-12 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-8 border border-orange-200 dark:border-orange-800">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            핵심 요약
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                1. 협동 로봇의 특징
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li>안전 울타리 없이 사람과 협업 가능</li>
                <li>힘/토크 제한으로 충돌 시 안전</li>
                <li>직관적 프로그래밍 (Hand Guiding)</li>
                <li>유연한 배치 및 재배치</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                2. ISO/TS 15066 안전 기준
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li>4가지 협업 운전 모드 정의</li>
                <li>신체 부위별 생체역학적 한계값 제시</li>
                <li>속도, 힘, 압력 제한 요구</li>
                <li>위험 평가 및 안전 설계 가이드</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                3. Force/Torque 제어
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li>6축 F/T 센서 또는 관절 토크 센서 활용</li>
                <li>임피던스 제어로 유연한 상호작용</li>
                <li>충돌 감지 및 즉각적 대응</li>
                <li>힘 제한으로 안전성 보장</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                4. 인간-로봇 상호작용
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li>물리적, 시각적, 음성, 제스처 인터페이스</li>
                <li>의도 예측으로 선제적 대응</li>
                <li>적응형 행동으로 작업자에 최적화</li>
                <li>자연스럽고 효율적인 협업</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                5. 응용 분야
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li>제조업: 조립, 검사, 포장, 머신 텐딩</li>
                <li>의료: 수술 보조, 재활, 약품 조제</li>
                <li>물류: 팔레타이징, 피킹, 분류</li>
                <li>서비스: 음식 서빙, 청소, 안내</li>
                <li>농업: 수확, 정밀 농업, 가축 관리</li>
              </ul>
            </div>
          </div>

          <div className="mt-8 bg-orange-100 dark:bg-orange-900/30 rounded-lg p-6 border-l-4 border-orange-500">
            <h3 className="text-xl font-semibold text-orange-700 dark:text-orange-300 mb-3">
              Robotics and Manipulation 모듈 완료!
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              8개 챕터를 모두 마쳤습니다. 로봇 공학의 수학적 기초부터 최신 협동 로봇 기술까지
              종합적으로 학습했습니다. 이제 여러분은:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li>로봇의 <strong>운동학과 동역학</strong>을 이해하고 계산할 수 있습니다</li>
              <li><strong>경로 계획 알고리즘</strong>을 선택하고 적용할 수 있습니다</li>
              <li><strong>그래스프 계획</strong>과 조작 전략을 설계할 수 있습니다</li>
              <li><strong>ROS2와 MoveIt2</strong>로 실제 로봇을 제어할 수 있습니다</li>
              <li><strong>협동 로봇의 안전 기준</strong>을 준수하는 시스템을 구축할 수 있습니다</li>
            </ul>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mt-4">
              다음 단계로 실제 로봇 프로젝트에 도전해보세요!
            </p>
          </div>
        </section>

        {/* Chapter Navigation */}
        <ChapterNavigation
          currentChapter={8}
          totalChapters={8}
          moduleSlug="robotics-manipulation"
        />
      </div>
    </div>
  )
}
