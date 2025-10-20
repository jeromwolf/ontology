'use client'

import React from 'react'
import ChapterNavigation from '../ChapterNavigation'

export default function Chapter1() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-gradient-to-r from-orange-600 to-red-600 rounded-2xl p-8 mb-8 text-white">
        <h1 className="text-4xl font-bold mb-4">Chapter 1: 로봇 공학 기초</h1>
        <p className="text-xl text-white/90">
          로봇의 구조, 좌표계, 자유도 이해
        </p>
      </div>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        {/* Section 1: 로봇의 정의와 분류 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            1.1 로봇의 정의와 분류
          </h2>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">로봇이란?</h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              로봇(Robot)은 외부 환경을 감지하고, 자율적으로 또는 반자율적으로 작업을 수행할 수 있는 프로그래밍 가능한 기계 시스템입니다.
              1920년 체코 작가 카렐 차페크의 희곡 "R.U.R"에서 처음 사용된 "로봇"이라는 단어는 체코어 "robota(강제 노동)"에서 유래했습니다.
            </p>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">로봇의 3대 구성 요소</h3>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg border border-orange-200 dark:border-orange-800">
                <div className="text-3xl mb-3">🤖</div>
                <h4 className="text-xl font-bold text-orange-600 dark:text-orange-400 mb-2">
                  1. 센서 (Sensing)
                </h4>
                <p className="text-gray-700 dark:text-gray-300 text-sm">
                  환경 인식: 카메라, LiDAR, 힘/토크 센서, 인코더 등
                </p>
              </div>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg border border-orange-200 dark:border-orange-800">
                <div className="text-3xl mb-3">🧠</div>
                <h4 className="text-xl font-bold text-orange-600 dark:text-orange-400 mb-2">
                  2. 제어 (Control)
                </h4>
                <p className="text-gray-700 dark:text-gray-300 text-sm">
                  의사결정: CPU, 제어 알고리즘, AI/ML 모델
                </p>
              </div>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-6 rounded-lg border border-orange-200 dark:border-orange-800">
                <div className="text-3xl mb-3">⚙️</div>
                <h4 className="text-xl font-bold text-orange-600 dark:text-orange-400 mb-2">
                  3. 액추에이터 (Actuation)
                </h4>
                <p className="text-gray-700 dark:text-gray-300 text-sm">
                  동작 실행: 모터, 그리퍼, 바퀴, 유압/공압 시스템
                </p>
              </div>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">로봇 분류</h3>
            <div className="space-y-4">
              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
                <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                  📦 산업용 로봇 (Industrial Robots)
                </h4>
                <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                  <li><strong>용도</strong>: 제조업 생산 라인 자동화</li>
                  <li><strong>특징</strong>: 고속, 고정밀, 반복 작업 특화</li>
                  <li><strong>예시</strong>: KUKA, ABB, FANUC 로봇 팔</li>
                  <li><strong>시장 점유율</strong>: 전체 로봇 시장의 약 70%</li>
                </ul>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
                <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                  👥 협동 로봇 (Collaborative Robots, Cobots)
                </h4>
                <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                  <li><strong>용도</strong>: 인간과 안전하게 협업</li>
                  <li><strong>특징</strong>: 힘 제한, 충돌 감지, 안전 기능 내장</li>
                  <li><strong>예시</strong>: Universal Robots (UR 시리즈), Franka Emika</li>
                  <li><strong>성장률</strong>: 연평균 40% 이상 급성장 중</li>
                </ul>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
                <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                  🏠 서비스 로봇 (Service Robots)
                </h4>
                <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                  <li><strong>용도</strong>: 가정, 의료, 물류, 청소 등</li>
                  <li><strong>특징</strong>: 자율 이동, 인간-로봇 상호작용</li>
                  <li><strong>예시</strong>: 룸바, Boston Dynamics Spot, 배달 로봇</li>
                  <li><strong>트렌드</strong>: AI와 결합하여 급속 발전</li>
                </ul>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
                <h4 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                  🚗 이동 로봇 (Mobile Robots)
                </h4>
                <ul className="list-disc list-inside space-y-2 text-gray-700 dark:text-gray-300">
                  <li><strong>용도</strong>: 자율주행, AGV, 드론</li>
                  <li><strong>특징</strong>: SLAM, 경로 계획, 장애물 회피</li>
                  <li><strong>예시</strong>: AMR (자율 이동 로봇), 자율주행차</li>
                  <li><strong>응용</strong>: 물류 창고, 병원, 호텔 등</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: 로봇 좌표계 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            1.2 로봇 좌표계와 변환
          </h2>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">좌표계의 중요성</h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              로봇 공학에서는 여러 좌표계가 사용되며, 이들 간의 정확한 변환이 필수적입니다.
              로봇의 위치와 방향을 정의하고, 작업 공간을 설명하기 위해 좌표계 변환 행렬을 사용합니다.
            </p>

            <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-6 rounded">
              <h4 className="text-lg font-bold text-blue-900 dark:text-blue-300 mb-3">
                주요 좌표계
              </h4>
              <ul className="space-y-3 text-gray-700 dark:text-gray-300">
                <li className="flex items-start gap-3">
                  <span className="text-blue-600 font-bold">1.</span>
                  <div>
                    <strong>World/Base Frame</strong>: 작업 공간의 절대 좌표계 (고정)
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-600 font-bold">2.</span>
                  <div>
                    <strong>Robot Base Frame</strong>: 로봇 베이스 중심의 좌표계
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-600 font-bold">3.</span>
                  <div>
                    <strong>Joint Frames</strong>: 각 관절에 부착된 좌표계
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-600 font-bold">4.</span>
                  <div>
                    <strong>End-Effector Frame</strong>: 로봇 끝단 (그리퍼, 도구)의 좌표계
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-600 font-bold">5.</span>
                  <div>
                    <strong>Tool/Object Frame</strong>: 작업 대상 물체의 좌표계
                  </div>
                </li>
              </ul>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">동차 변환 행렬 (Homogeneous Transformation)</h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              좌표계 간 변환을 표현하는 4×4 행렬로, 회전(Rotation)과 이동(Translation)을 하나의 행렬로 표현합니다.
            </p>

            <div className="bg-gray-100 dark:bg-gray-900 p-6 rounded-lg font-mono text-sm overflow-x-auto">
              <pre className="text-gray-800 dark:text-gray-200">
{`T = [ R  p ]
    [ 0  1 ]

여기서:
- R: 3×3 회전 행렬 (Rotation Matrix)
- p: 3×1 위치 벡터 (Position Vector)
- 0: [0 0 0] 행 벡터
- 1: 스칼라

예시 (Z축 회전 θ + x축 이동 d):
    [ cos(θ)  -sin(θ)  0  d ]
T = [ sin(θ)   cos(θ)  0  0 ]
    [   0        0      1  0 ]
    [   0        0      0  1 ]`}
              </pre>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">Denavit-Hartenberg (DH) 파라미터</h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              로봇 관절 간 좌표계 변환을 체계적으로 표현하는 표준 방법입니다. 4개의 파라미터로 구성됩니다:
            </p>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border border-orange-200 dark:border-orange-800">
                <strong className="text-orange-600 dark:text-orange-400">θ (theta)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                  관절 각도: z축 주위 회전
                </p>
              </div>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border border-orange-200 dark:border-orange-800">
                <strong className="text-orange-600 dark:text-orange-400">d (distance)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                  링크 오프셋: z축 방향 이동
                </p>
              </div>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border border-orange-200 dark:border-orange-800">
                <strong className="text-orange-600 dark:text-orange-400">a (link length)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                  링크 길이: x축 방향 이동
                </p>
              </div>
              <div className="bg-orange-50 dark:bg-orange-900/20 p-4 rounded-lg border border-orange-200 dark:border-orange-800">
                <strong className="text-orange-600 dark:text-orange-400">α (alpha)</strong>
                <p className="text-sm text-gray-700 dark:text-gray-300 mt-2">
                  링크 비틀림: x축 주위 회전
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: 자유도 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            1.3 자유도 (Degrees of Freedom, DOF)
          </h2>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">자유도란?</h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
              자유도(DOF)는 로봇이 독립적으로 움직일 수 있는 방향의 수를 의미합니다.
              3차원 공간에서 물체를 완전히 제어하려면 최소 6-DOF가 필요합니다:
            </p>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border border-blue-200 dark:border-blue-800">
                <h4 className="text-xl font-bold text-blue-600 dark:text-blue-400 mb-3">
                  3 위치 자유도 (Position)
                </h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>• X축 이동 (좌우)</li>
                  <li>• Y축 이동 (전후)</li>
                  <li>• Z축 이동 (상하)</li>
                </ul>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-lg border border-purple-200 dark:border-purple-800">
                <h4 className="text-xl font-bold text-purple-600 dark:text-purple-400 mb-3">
                  3 방향 자유도 (Orientation)
                </h4>
                <ul className="space-y-2 text-gray-700 dark:text-gray-300">
                  <li>• Roll (X축 회전)</li>
                  <li>• Pitch (Y축 회전)</li>
                  <li>• Yaw (Z축 회전)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-semibold mb-4">DOF에 따른 로봇 분류</h3>
            <div className="space-y-4">
              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
                <h4 className="text-lg font-bold mb-2">3-DOF 로봇</h4>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  간단한 Pick-and-Place 작업 (위치만 제어)
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  예: SCARA 로봇, Delta 로봇
                </p>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
                <h4 className="text-lg font-bold mb-2">6-DOF 로봇</h4>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  완전한 위치 및 방향 제어 (가장 일반적)
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  예: UR5, KUKA KR, ABB IRB 6축 로봇 팔
                </p>
              </div>

              <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
                <h4 className="text-lg font-bold mb-2">7-DOF+ 로봇</h4>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  중복 자유도(Redundant DOF) - 특이점 회피 및 장애물 회피 가능
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  예: Franka Emika Panda (7-DOF), KUKA LBR iiwa (7-DOF)
                </p>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-6 rounded">
            <h4 className="text-lg font-bold text-yellow-900 dark:text-yellow-300 mb-2">
              🔑 핵심 개념
            </h4>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>작업 공간 (Workspace)</strong>: 로봇 엔드이펙터가 도달할 수 있는 모든 점들의 집합입니다.
              DOF가 높을수록 작업 공간이 넓고 유연하지만, 제어가 복잡해집니다.
            </p>
          </div>
        </section>

        {/* Section 4: 로봇 관절 종류 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            1.4 로봇 관절 종류
          </h2>

          <div className="grid md:grid-cols-3 gap-6 mb-6">
            <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-6 rounded-xl border border-red-200 dark:border-red-800">
              <div className="text-4xl mb-4">🔄</div>
              <h3 className="text-xl font-bold text-red-600 dark:text-red-400 mb-3">
                Revolute Joint (R)
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm mb-3">
                회전 관절 - 한 축을 중심으로 회전
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• DOF: 1 (회전)</li>
                <li>• 범위: 0° ~ 360° (또는 제한)</li>
                <li>• 가장 일반적인 관절</li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-800">
              <div className="text-4xl mb-4">↕️</div>
              <h3 className="text-xl font-bold text-blue-600 dark:text-blue-400 mb-3">
                Prismatic Joint (P)
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm mb-3">
                직선 관절 - 한 축 방향으로 직선 이동
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• DOF: 1 (이동)</li>
                <li>• 범위: 최소 ~ 최대 거리</li>
                <li>• 예: 리니어 액추에이터</li>
              </ul>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-800">
              <div className="text-4xl mb-4">⚽</div>
              <h3 className="text-xl font-bold text-purple-600 dark:text-purple-400 mb-3">
                Spherical Joint (S)
              </h3>
              <p className="text-gray-700 dark:text-gray-300 text-sm mb-3">
                구형 관절 - 3축 회전 가능
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• DOF: 3 (Roll, Pitch, Yaw)</li>
                <li>• 예: 사람 어깨 관절</li>
                <li>• 복잡한 제어 필요</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Section 5: 로봇 아키텍처 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            1.5 로봇 매니퓰레이터 아키텍처
          </h2>

          <div className="space-y-6">
            <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-3">🔲 직교 좌표 로봇 (Cartesian Robot)</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                X, Y, Z 축을 따라 직선 이동하는 3개의 Prismatic 관절
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400 ml-4">
                <li>• 장점: 프로그래밍 간단, 높은 정밀도, 큰 작업 공간</li>
                <li>• 단점: 큰 설치 공간 필요, 무거움</li>
                <li>• 응용: 3D 프린터, CNC, 갠트리 로봇</li>
              </ul>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-3">🏹 SCARA 로봇</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                Selective Compliance Assembly Robot Arm - 수평면 이동 + 수직 이동
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400 ml-4">
                <li>• 구조: R-R-P (2 회전 + 1 직선)</li>
                <li>• 장점: 빠른 속도, 높은 반복 정밀도</li>
                <li>• 응용: 전자 부품 조립, PCB 작업</li>
              </ul>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-3">🦾 다관절 로봇 (Articulated Robot)</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                6개의 Revolute 관절로 구성된 가장 일반적인 산업용 로봇
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400 ml-4">
                <li>• 구조: R-R-R-R-R-R (6축)</li>
                <li>• 장점: 높은 유연성, 넓은 작업 범위, 장애물 회피</li>
                <li>• 단점: 복잡한 역기구학, 특이점 존재</li>
                <li>• 응용: 용접, 도장, 조립, 핸들링</li>
              </ul>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700/50 p-6 rounded-lg">
              <h3 className="text-xl font-bold mb-3">🔺 Delta/Parallel 로봇</h3>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                병렬 구조로 초고속 작업 가능
              </p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400 ml-4">
                <li>• 장점: 매우 빠른 속도 (사이클 타임 0.5초 이하)</li>
                <li>• 단점: 작은 작업 공간, 방향 제어 제한</li>
                <li>• 응용: 식품 포장, 약품 분류, 빠른 픽앤플레이스</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Summary */}
        <section className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-8 border border-orange-200 dark:border-orange-800">
          <h2 className="text-2xl font-bold text-orange-600 dark:text-orange-400 mb-4">
            📌 Chapter 1 요약
          </h2>
          <ul className="space-y-3 text-gray-700 dark:text-gray-300">
            <li className="flex items-start gap-3">
              <span className="text-orange-500 font-bold">✓</span>
              <span>로봇은 센서, 제어기, 액추에이터로 구성되며 산업용, 협동, 서비스, 이동 로봇으로 분류됩니다.</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-orange-500 font-bold">✓</span>
              <span>동차 변환 행렬과 DH 파라미터는 로봇 좌표계 간 변환을 표현하는 핵심 도구입니다.</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-orange-500 font-bold">✓</span>
              <span>자유도(DOF)는 로봇의 유연성을 결정하며, 6-DOF가 3D 공간 완전 제어의 기준입니다.</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="text-orange-500 font-bold">✓</span>
              <span>Revolute, Prismatic, Spherical 관절 조합으로 다양한 로봇 아키텍처를 구성합니다.</span>
            </li>
          </ul>
        </section>

        {/* Chapter Navigation */}
        <ChapterNavigation
          currentChapter={1}
          totalChapters={8}
          moduleSlug="robotics-manipulation"
        />
      </div>
    </div>
  )
}
