'use client'

import React from 'react'
import ChapterNavigation from '../ChapterNavigation'

export default function Chapter6() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-gradient-to-r from-orange-600 to-red-600 rounded-2xl p-8 mb-8 text-white">
        <h1 className="text-4xl font-bold mb-4">Chapter 6: 그리핑과 조작 (Grasping & Manipulation)</h1>
        <p className="text-xl text-white/90">
          로봇 그리퍼 설계와 물체 조작 전략
        </p>
      </div>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        {/* 1. Overview */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            1. 그리핑과 조작이란?
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              정의
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              <strong>그리핑(Grasping)</strong>은 로봇이 물체를 안정적으로 파지하는 행위이며,
              <strong>조작(Manipulation)</strong>은 파지된 물체를 원하는 위치와 자세로 이동시키거나
              물체의 상태를 변경하는 행위입니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            그리핑 vs 조작
          </h3>

          <table className="min-w-full border border-blue-300 dark:border-blue-700 mt-3 mb-6">
            <thead className="bg-blue-100 dark:bg-blue-900/50">
              <tr>
                <th className="px-4 py-2 border-b text-left">구분</th>
                <th className="px-4 py-2 border-b text-left">그리핑 (Grasping)</th>
                <th className="px-4 py-2 border-b text-left">조작 (Manipulation)</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">목표</td>
                <td className="px-4 py-2">물체의 안정적 파지</td>
                <td className="px-4 py-2">물체의 위치/자세 변경</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">핵심</td>
                <td className="px-4 py-2">접촉점 선택, 힘 제어</td>
                <td className="px-4 py-2">경로 계획, 동역학 제어</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">고려사항</td>
                <td className="px-4 py-2">Force Closure, 마찰력</td>
                <td className="px-4 py-2">충돌 회피, 속도/가속도 제한</td>
              </tr>
              <tr>
                <td className="px-4 py-2 font-semibold">예시</td>
                <td className="px-4 py-2">병 집기, 나사 잡기</td>
                <td className="px-4 py-2">병 따기, 나사 조이기</td>
              </tr>
            </tbody>
          </table>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 border-l-4 border-green-500">
            <h3 className="text-xl font-semibold text-green-700 dark:text-green-300 mb-3">
              실제 응용 사례
            </h3>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>픽앤플레이스(Pick-and-Place)</strong>: 창고 자동화, 조립 라인</li>
              <li><strong>빈 피킹(Bin Picking)</strong>: 무작위 물체 분류 및 정리</li>
              <li><strong>인-핸드 조작(In-Hand Manipulation)</strong>: 물체를 손안에서 재배치</li>
              <li><strong>정밀 조립</strong>: 전자제품 조립, 나사 체결</li>
            </ul>
          </div>
        </section>

        {/* 2. Gripper Types */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            2. 그리퍼의 종류
          </h2>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            2.1 평행 그리퍼 (Parallel Jaw Gripper)
          </h3>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              두 개의 손가락이 평행하게 움직이며 물체를 양쪽에서 잡는 방식입니다.
            </p>
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">장점:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 mb-3">
              <li>구조가 간단하고 제어가 용이</li>
              <li>규칙적인 형태의 물체에 효과적</li>
              <li>높은 정밀도와 반복성</li>
            </ul>
            <h4 className="font-semibold text-blue-700 dark:text-blue-300 mb-2">단점:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
              <li>불규칙한 형태의 물체 파지 어려움</li>
              <li>제한된 자유도 (보통 1-DOF)</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            2.2 진공 그리퍼 (Vacuum Gripper)
          </h3>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 mb-6 border-l-4 border-purple-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              진공 흡착을 이용하여 물체의 표면을 빨아당겨 파지하는 방식입니다.
            </p>
            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">장점:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 mb-3">
              <li>평평한 표면의 물체에 매우 효과적</li>
              <li>빠른 파지/해제 속도</li>
              <li>물체에 손상을 주지 않음</li>
            </ul>
            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">단점:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 mb-3">
              <li>다공성 또는 곡면 물체에는 부적합</li>
              <li>진공 펌프 필요 (에너지 소비)</li>
            </ul>
            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">응용:</h4>
            <p className="text-gray-700 dark:text-gray-300">
              반도체 웨이퍼, 유리판, 판지 상자, 플라스틱 시트 등
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            2.3 소프트 그리퍼 (Soft Gripper)
          </h3>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 mb-6 border-l-4 border-green-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              유연한 재료(실리콘, 고무 등)로 만들어져 물체의 형태에 순응하며 파지하는 방식입니다.
            </p>
            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">장점:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 mb-3">
              <li>불규칙한 형태의 물체 파지 가능</li>
              <li>충격 흡수로 물체 보호</li>
              <li>정밀한 힘 제어 불필요</li>
            </ul>
            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">단점:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 mb-3">
              <li>정밀도가 낮음</li>
              <li>무거운 물체 파지 어려움</li>
            </ul>
            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-2">응용:</h4>
            <p className="text-gray-700 dark:text-gray-300">
              과일/채소 수확, 식품 가공, 의료 기기, 협동 로봇
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            2.4 다지 그리퍼 (Multi-Finger Gripper)
          </h3>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 border-l-4 border-orange-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              3개 이상의 손가락을 가진 그리퍼로, 인간의 손과 유사한 기능을 제공합니다.
            </p>
            <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-2">장점:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 mb-3">
              <li>복잡한 형태의 물체 파지 가능</li>
              <li>In-Hand Manipulation 가능</li>
              <li>높은 적응성 및 유연성</li>
            </ul>
            <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-2">단점:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 mb-3">
              <li>복잡한 제어 알고리즘 필요</li>
              <li>높은 비용</li>
              <li>센서 통합 어려움</li>
            </ul>
            <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-2">대표 사례:</h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
              <li><strong>Barrett Hand</strong>: 3-finger, 4-DOF</li>
              <li><strong>Shadow Hand</strong>: 5-finger, 24-DOF (인간 손 모사)</li>
              <li><strong>Allegro Hand</strong>: 4-finger, 16-DOF</li>
            </ul>
          </div>
        </section>

        {/* 3. Force Closure */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            3. Force Closure (힘 폐쇄)
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              정의
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              <strong>Force Closure</strong>는 그리퍼의 접촉점들이 물체에 가하는 힘이
              모든 방향의 외력과 토크를 상쇄할 수 있는 상태를 의미합니다.
              즉, 중력이나 외부 힘에도 물체가 떨어지지 않고 안정적으로 파지된 상태입니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            Force Closure의 조건
          </h3>

          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-6 mb-6">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              2D 평면에서 Force Closure를 달성하려면:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li><strong>최소 3개의 접촉점</strong>이 필요</li>
              <li>접촉점의 법선 벡터들이 <strong>원점(물체의 중심)을 포함하는 다면체</strong>를 형성해야 함</li>
              <li>각 접촉점에서 <strong>마찰 원뿔(Friction Cone)</strong> 내의 힘을 가할 수 있어야 함</li>
            </ul>

            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              3D 공간에서는:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>최소 4개의 접촉점</strong>이 필요 (일반적으로)</li>
              <li>7개의 자유도를 구속해야 함 (3 translation + 3 rotation + 1 redundancy)</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            마찰 원뿔 (Friction Cone)
          </h3>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 mb-6 border-l-4 border-purple-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              접촉점에서 가할 수 있는 힘의 범위를 나타내는 원뿔 형태의 영역입니다.
            </p>

            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-4">
              <p className="text-sm font-mono text-gray-800 dark:text-gray-200 mb-2">
                마찰 계수 μ에 의해 결정되는 원뿔 각도:
              </p>
              <p className="text-center text-lg font-semibold text-purple-700 dark:text-purple-300">
                θ = arctan(μ)
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
                • μ = 0.5 → θ ≈ 26.6°<br />
                • μ = 1.0 → θ = 45°
              </p>
            </div>

            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              접촉점에서 가해지는 힘 <strong>f</strong>는 법선 방향 <strong>n</strong>에 대해
              θ 이내의 각도를 유지해야 물체가 미끄러지지 않습니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            Form Closure vs Force Closure
          </h3>

          <table className="min-w-full border border-blue-300 dark:border-blue-700 mt-3">
            <thead className="bg-blue-100 dark:bg-blue-900/50">
              <tr>
                <th className="px-4 py-2 border-b text-left">구분</th>
                <th className="px-4 py-2 border-b text-left">Form Closure</th>
                <th className="px-4 py-2 border-b text-left">Force Closure</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">정의</td>
                <td className="px-4 py-2">기하학적 구속으로 고정</td>
                <td className="px-4 py-2">힘의 균형으로 고정</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">마찰 의존성</td>
                <td className="px-4 py-2">마찰 불필요</td>
                <td className="px-4 py-2">마찰 필요</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">예시</td>
                <td className="px-4 py-2">공을 양손으로 감싸기</td>
                <td className="px-4 py-2">평평한 책을 양손으로 누르기</td>
              </tr>
              <tr>
                <td className="px-4 py-2 font-semibold">안정성</td>
                <td className="px-4 py-2">매우 안정적 (수동적)</td>
                <td className="px-4 py-2">능동적 힘 제어 필요</td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* 4. Grasp Planning */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            4. 그래스프 계획 (Grasp Planning)
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              목표
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              물체의 형상, 무게, 재질 등을 고려하여 최적의 그리핑 위치와 자세를 찾는 과정입니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            4.1 분석적 방법 (Analytical Approach)
          </h3>

          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-6 mb-6">
            <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
              1. 접촉점 샘플링
            </h4>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              물체 표면에서 가능한 접촉점 조합을 생성합니다.
            </p>

            <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
              2. Force Closure 검증
            </h4>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              각 접촉점 조합이 Force Closure 조건을 만족하는지 확인합니다.
            </p>

            <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">
              3. 품질 평가
            </h4>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              그래스프의 안정성, 견고성 등을 평가하는 품질 메트릭을 계산합니다:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>Ferrari-Canny Metric</strong>: 최소 레버리지 기반</li>
              <li><strong>Volume Metric</strong>: 가능한 렌치 공간의 부피</li>
              <li><strong>Grasp Wrench Space (GWS)</strong>: 가할 수 있는 힘/토크의 집합</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            4.2 데이터 기반 방법 (Data-Driven Approach)
          </h3>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 mb-6 border-l-4 border-green-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              딥러닝을 활용하여 RGB-D 이미지에서 직접 그래스프 후보를 예측합니다.
            </p>

            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3">
              주요 방법론
            </h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li><strong>GraspNet</strong>: CNN 기반 그래스프 포즈 예측</li>
              <li><strong>Dex-Net</strong>: 대규모 합성 데이터셋 학습</li>
              <li><strong>PointNet++</strong>: 포인트 클라우드 직접 처리</li>
              <li><strong>6-DOF GraspNet</strong>: 6자유도 그래스프 예측</li>
            </ul>

            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3">
              장점
            </h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li>실시간 처리 가능 (추론 속도 &lt; 100ms)</li>
              <li>미지의 물체에도 일반화</li>
              <li>복잡한 기하학적 계산 불필요</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            4.3 Python 구현 예제: 간단한 Grasp Quality 계산
          </h3>

          <div className="bg-gray-900 rounded-lg p-6 mb-6">
            <pre className="text-green-400 text-sm overflow-x-auto">
              <code>{`import numpy as np
from scipy.spatial import ConvexHull

def compute_grasp_quality(contact_points, normals, friction_coeff=0.5):
    """
    그래스프 품질 계산 (Simplified Ferrari-Canny Metric)

    Parameters:
    -----------
    contact_points : np.ndarray, shape (n, 3)
        접촉점의 3D 좌표
    normals : np.ndarray, shape (n, 3)
        각 접촉점의 법선 벡터
    friction_coeff : float
        마찰 계수 (μ)

    Returns:
    --------
    quality : float
        그래스프 품질 (높을수록 좋음)
    """
    n_contacts = len(contact_points)

    # 1. 마찰 원뿔 근사 (4개의 선형화된 벡터)
    wrench_space = []

    for i in range(n_contacts):
        p = contact_points[i]
        n = normals[i]

        # 접선 방향 2개 생성
        if abs(n[0]) < 0.9:
            t1 = np.cross(n, [1, 0, 0])
        else:
            t1 = np.cross(n, [0, 1, 0])
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        # 마찰 원뿔의 4개 모서리
        for sign1 in [-1, 1]:
            for sign2 in [-1, 1]:
                f = n + friction_coeff * (sign1 * t1 + sign2 * t2)
                f = f / np.linalg.norm(f)

                # 힘 벡터
                force = f

                # 토크 벡터
                torque = np.cross(p, f)

                # 렌치 (Wrench) = [force, torque]
                wrench = np.concatenate([force, torque])
                wrench_space.append(wrench)

    wrench_space = np.array(wrench_space)

    # 2. Grasp Wrench Space의 원점으로부터 최소 거리 계산
    try:
        hull = ConvexHull(wrench_space)

        # 원점이 convex hull 내부에 있는지 확인
        # 간단한 근사: hull의 중심으로부터 거리
        center = np.mean(wrench_space, axis=0)
        quality = 1.0 / (np.linalg.norm(center) + 1e-6)

        return quality
    except:
        # Convex Hull 생성 실패 (degenerate case)
        return 0.0


# 예제: 3-finger grasp
contact_points = np.array([
    [0.05, 0, 0],
    [-0.025, 0.043, 0],
    [-0.025, -0.043, 0]
])

normals = np.array([
    [-1, 0, 0],
    [0.5, -0.866, 0],
    [0.5, 0.866, 0]
])

# 품질 계산
quality = compute_grasp_quality(contact_points, normals, friction_coeff=0.6)
print(f"Grasp Quality: {quality:.4f}")

# 마찰 계수 변화에 따른 품질
for mu in [0.3, 0.5, 0.7, 1.0]:
    q = compute_grasp_quality(contact_points, normals, friction_coeff=mu)
    print(f"μ = {mu:.1f} → Quality = {q:.4f}")`}</code>
            </pre>
          </div>
        </section>

        {/* 5. Visual Servoing */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            5. 비주얼 서보잉 (Visual Servoing)
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              정의
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              카메라에서 획득한 시각 정보를 피드백으로 사용하여 로봇의 움직임을 제어하는 기법입니다.
              물체의 위치/자세를 실시간으로 추적하며 정밀한 조작을 수행할 수 있습니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            5.1 Image-Based Visual Servoing (IBVS)
          </h3>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 mb-6 border-l-4 border-purple-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              이미지 평면에서 직접 피처(점, 선, 영역 등)의 위치 오차를 최소화하는 방식입니다.
            </p>

            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-3">
              제어 법칙
            </h4>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-4">
              <p className="text-center text-lg font-semibold text-purple-700 dark:text-purple-300 mb-2">
                v = -λ L<sup>+</sup> (s - s*)
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
                • v: 카메라(엔드 이펙터) 속도 (6×1)<br />
                • λ: 제어 이득<br />
                • L<sup>+</sup>: Image Jacobian의 의사역행렬<br />
                • s: 현재 이미지 피처<br />
                • s*: 목표 이미지 피처
              </p>
            </div>

            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-3">
              장점
            </h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 mb-3">
              <li>3D 재구성 불필요</li>
              <li>카메라 캘리브레이션 오차에 덜 민감</li>
            </ul>

            <h4 className="font-semibold text-purple-700 dark:text-purple-300 mb-3">
              단점
            </h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
              <li>피처가 이미지 밖으로 벗어날 수 있음</li>
              <li>경로가 직관적이지 않음 (이미지 공간에서만 최적)</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            5.2 Position-Based Visual Servoing (PBVS)
          </h3>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 mb-6 border-l-4 border-green-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              이미지에서 물체의 3D 위치/자세를 추정한 후, 작업 공간에서 오차를 최소화하는 방식입니다.
            </p>

            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3">
              제어 법칙
            </h4>
            <div className="bg-white dark:bg-gray-800 rounded p-4 mb-4">
              <p className="text-center text-lg font-semibold text-green-700 dark:text-green-300 mb-2">
                v = -λ (x - x*)
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-3">
                • v: 엔드 이펙터 속도<br />
                • x: 현재 3D 위치/자세<br />
                • x*: 목표 3D 위치/자세
              </p>
            </div>

            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3">
              장점
            </h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300 mb-3">
              <li>직관적인 직선 경로</li>
              <li>충돌 회피 계획 용이</li>
            </ul>

            <h4 className="font-semibold text-green-700 dark:text-green-300 mb-3">
              단점
            </h4>
            <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
              <li>정확한 3D 재구성 필요</li>
              <li>카메라 캘리브레이션 오차에 민감</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            5.3 Hybrid Visual Servoing
          </h3>

          <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-6 border-l-4 border-orange-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              IBVS와 PBVS의 장점을 결합한 방법으로, 일부 자유도는 이미지 공간에서,
              나머지는 작업 공간에서 제어합니다.
            </p>

            <h4 className="font-semibold text-orange-700 dark:text-orange-300 mb-3">
              예시: 2½D Visual Servoing
            </h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>X, Y 이동</strong>: 이미지 피처 기반 제어 (IBVS)</li>
              <li><strong>Z 이동 및 회전</strong>: 3D 포즈 기반 제어 (PBVS)</li>
            </ul>
          </div>
        </section>

        {/* 6. Contact Modeling */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            6. 접촉 모델링 (Contact Modeling)
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6 mb-6 border-l-4 border-blue-500">
            <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300 mb-3">
              중요성
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              로봇과 물체 간의 접촉을 정확히 모델링하는 것은 안정적인 그리핑과 조작에 필수적입니다.
              접촉 모델은 시뮬레이션, 제어, 그래스프 계획 모두에 사용됩니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            6.1 Point Contact (PC)
          </h3>

          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-6 mb-6">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              가장 단순한 모델로, 접촉점에서 법선 방향의 힘만 전달됩니다 (마찰 없음).
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
              <li><strong>자유도 제약</strong>: 1 DOF (법선 방향 이동 제한)</li>
              <li><strong>가능한 힘</strong>: 법선 방향 압력만</li>
              <li><strong>응용</strong>: 이론적 분석, 초기 그래스프 계획</li>
            </ul>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            6.2 Point Contact with Friction (PCwF)
          </h3>

          <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-6 mb-6 border-l-4 border-purple-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              마찰을 고려한 모델로, 법선 방향과 접선 방향의 힘이 모두 전달됩니다.
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li><strong>자유도 제약</strong>: 3 DOF (2D: 2 DOF)</li>
              <li><strong>쿨롱 마찰 법칙</strong>: |f<sub>t</sub>| ≤ μ f<sub>n</sub></li>
              <li><strong>마찰 원뿔</strong>: 가능한 힘의 범위 정의</li>
            </ul>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              대부분의 실제 그리핑 시나리오에서 사용되는 표준 모델입니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            6.3 Soft Finger Contact (SFC)
          </h3>

          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6 mb-6 border-l-4 border-green-500">
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              접촉면의 변형을 고려한 모델로, 법선 방향 힘, 접선 방향 힘, 그리고 법선 축 주위의 토크를 전달합니다.
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300 mb-4">
              <li><strong>자유도 제약</strong>: 4 DOF (2D: 3 DOF)</li>
              <li><strong>가능한 렌치</strong>: [f<sub>n</sub>, f<sub>t1</sub>, f<sub>t2</sub>, τ<sub>n</sub>]</li>
              <li><strong>응용</strong>: 소프트 그리퍼, 탄성 재질 접촉</li>
            </ul>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              소프트 로보틱스 및 인간-로봇 상호작용에서 중요한 모델입니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4 mt-8">
            접촉 모델 비교
          </h3>

          <table className="min-w-full border border-blue-300 dark:border-blue-700 mt-3">
            <thead className="bg-blue-100 dark:bg-blue-900/50">
              <tr>
                <th className="px-4 py-2 border-b text-left">모델</th>
                <th className="px-4 py-2 border-b text-left">자유도 제약 (3D)</th>
                <th className="px-4 py-2 border-b text-left">전달 가능한 렌치</th>
                <th className="px-4 py-2 border-b text-left">응용</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">PC</td>
                <td className="px-4 py-2">1 DOF</td>
                <td className="px-4 py-2">법선 방향 힘</td>
                <td className="px-4 py-2">이론적 분석</td>
              </tr>
              <tr className="border-b">
                <td className="px-4 py-2 font-semibold">PCwF</td>
                <td className="px-4 py-2">3 DOF</td>
                <td className="px-4 py-2">법선 + 접선 힘</td>
                <td className="px-4 py-2">대부분의 그리핑</td>
              </tr>
              <tr>
                <td className="px-4 py-2 font-semibold">SFC</td>
                <td className="px-4 py-2">4 DOF</td>
                <td className="px-4 py-2">힘 + 법선 토크</td>
                <td className="px-4 py-2">소프트 그리퍼</td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* 7. Summary */}
        <section className="mb-12 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-8 border border-orange-200 dark:border-orange-800">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            핵심 요약
          </h2>

          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                1. 그리퍼 종류
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>평행 그리퍼</strong>: 간단하고 정밀, 규칙적 물체에 적합</li>
                <li><strong>진공 그리퍼</strong>: 평평한 표면에 효과적, 빠른 동작</li>
                <li><strong>소프트 그리퍼</strong>: 불규칙 물체, 충격 흡수</li>
                <li><strong>다지 그리퍼</strong>: 복잡한 조작, In-Hand Manipulation</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                2. Force Closure
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li>모든 방향의 외력을 상쇄할 수 있는 안정적 파지 상태</li>
                <li>2D: 최소 3개 접촉점, 3D: 최소 4개 접촉점</li>
                <li>마찰 원뿔 내에서 힘을 가해야 미끄러지지 않음</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                3. 그래스프 계획
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>분석적 방법</strong>: 접촉점 샘플링 → Force Closure 검증 → 품질 평가</li>
                <li><strong>데이터 기반</strong>: GraspNet, Dex-Net 등 딥러닝 활용</li>
                <li>Ferrari-Canny, GWS 등의 품질 메트릭 사용</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                4. Visual Servoing
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>IBVS</strong>: 이미지 피처 기반, 3D 재구성 불필요</li>
                <li><strong>PBVS</strong>: 3D 포즈 기반, 직관적 경로</li>
                <li><strong>Hybrid</strong>: 두 방법의 장점 결합</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-orange-600 dark:text-orange-400 mb-3">
                5. 접촉 모델
              </h3>
              <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                <li><strong>PC</strong>: 법선 힘만 (이론적)</li>
                <li><strong>PCwF</strong>: 마찰 고려 (실용적)</li>
                <li><strong>SFC</strong>: 변형 고려 (소프트 로보틱스)</li>
              </ul>
            </div>
          </div>

          <div className="mt-8 bg-orange-100 dark:bg-orange-900/30 rounded-lg p-6 border-l-4 border-orange-500">
            <h3 className="text-xl font-semibold text-orange-700 dark:text-orange-300 mb-3">
              다음 단계: ROS2 프로그래밍
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              다음 챕터에서는 이론을 실전에 적용하는 <strong>ROS2 (Robot Operating System 2)</strong>를
              배웁니다. MoveIt2를 이용한 모션 플래닝, Gazebo 시뮬레이션, 그리고 실제 로봇 제어까지
              전체 워크플로우를 경험하게 됩니다.
            </p>
          </div>
        </section>

        {/* Chapter Navigation */}
        <ChapterNavigation
          currentChapter={6}
          totalChapters={8}
          moduleSlug="robotics-manipulation"
        />
      </div>
    </div>
  )
}
