'use client'

import React from 'react'
import ChapterNavigation from '../ChapterNavigation'

export default function Chapter4() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-gradient-to-r from-orange-600 to-red-600 rounded-2xl p-8 mb-8 text-white">
        <h1 className="text-4xl font-bold mb-4">Chapter 4: 경로 계획 (Path Planning)</h1>
        <p className="text-xl text-white/90">
          로봇이 장애물을 피하며 목표 지점까지 안전하게 이동하는 경로를 찾는 알고리즘
        </p>
      </div>

      <div className="prose prose-lg dark:prose-invert max-w-none">
        {/* 1. 경로 계획 개요 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            1. 경로 계획이란?
          </h2>

          <div className="bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-600 p-6 mb-6">
            <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-300 mb-3">
              정의
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              <strong>경로 계획(Path Planning)</strong>은 로봇이 시작 위치에서 목표 위치까지 이동할 때,
              장애물과의 충돌을 피하면서 최적의 경로를 찾는 과정입니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            경로 계획의 핵심 요소
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border border-green-300 dark:border-green-700">
              <h4 className="font-bold text-lg text-green-900 dark:text-green-300 mb-2">
                구성 공간 (Configuration Space, C-space)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                로봇의 모든 가능한 상태(위치+방향)를 표현하는 추상적 공간.
                n-DOF 로봇은 n차원 C-space를 가짐
              </p>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border border-purple-300 dark:border-purple-700">
              <h4 className="font-bold text-lg text-purple-900 dark:text-purple-300 mb-2">
                장애물 공간 (C-obstacle)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                C-space에서 로봇과 장애물이 충돌하는 모든 구성의 집합.
                자유 공간(C-free)은 충돌 없는 영역
              </p>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-5 rounded-lg border border-orange-300 dark:border-orange-700">
              <h4 className="font-bold text-lg text-orange-900 dark:text-orange-300 mb-2">
                최적성 기준 (Optimality Criteria)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                경로 길이, 이동 시간, 에너지 소비, 부드러움(smoothness) 등
                목적에 따라 다양한 최적화 목표 설정
              </p>
            </div>

            <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg border border-blue-300 dark:border-blue-700">
              <h4 className="font-bold text-lg text-blue-900 dark:text-blue-300 mb-2">
                완전성 (Completeness)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                해가 존재하면 반드시 찾는가?<br/>
                - Resolution Complete: 해상도 내에서 완전<br/>
                - Probabilistically Complete: 확률적 완전
              </p>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            경로 계획 알고리즘 분류
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full border border-gray-300 dark:border-gray-600">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="px-4 py-3 border-b text-left">분류</th>
                  <th className="px-4 py-3 border-b text-left">알고리즘</th>
                  <th className="px-4 py-3 border-b text-left">특징</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">Graph Search</td>
                  <td className="px-4 py-3">A*, Dijkstra, D*</td>
                  <td className="px-4 py-3">격자 기반, 최적성 보장, 고차원 어려움</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">Sampling-based</td>
                  <td className="px-4 py-3">RRT, RRT*, PRM</td>
                  <td className="px-4 py-3">고차원 효율적, 확률적 완전성</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">Optimization</td>
                  <td className="px-4 py-3">CHOMP, TrajOpt</td>
                  <td className="px-4 py-3">부드러운 경로, 비용 함수 최소화</td>
                </tr>
                <tr>
                  <td className="px-4 py-3 font-semibold">Reactive</td>
                  <td className="px-4 py-3">DWA, VFH, APF</td>
                  <td className="px-4 py-3">실시간, 동적 환경, 지역 최소값 문제</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* 2. RRT (Rapidly-exploring Random Tree) */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            2. RRT (Rapidly-exploring Random Tree)
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            RRT는 샘플링 기반 알고리즘의 대표주자로, 고차원 구성 공간에서 빠르게 탐색하는 트리 구조를 생성합니다.
            Steven LaValle이 1998년 제안했으며, 로봇 공학에서 가장 널리 사용되는 경로 계획 알고리즘입니다.
          </p>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            RRT 알고리즘 핵심 아이디어
          </h3>
          <ol className="list-decimal list-inside space-y-3 text-gray-700 dark:text-gray-300 mb-6">
            <li>
              <strong>랜덤 샘플링</strong>: C-space에서 무작위로 점 q_rand 선택
            </li>
            <li>
              <strong>가장 가까운 노드 찾기</strong>: 트리에서 q_rand에 가장 가까운 q_near 찾기
            </li>
            <li>
              <strong>확장 (Extend)</strong>: q_near에서 q_rand 방향으로 step_size만큼 이동하여 q_new 생성
            </li>
            <li>
              <strong>충돌 검사</strong>: q_near → q_new 경로가 장애물과 충돌하지 않는지 확인
            </li>
            <li>
              <strong>트리 추가</strong>: 충돌 없으면 q_new를 트리에 추가하고 q_near와 연결
            </li>
            <li>
              <strong>목표 확인</strong>: q_new가 목표에 충분히 가까우면 종료
            </li>
          </ol>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            RRT 의사 코드 (Pseudocode)
          </h3>
          <div className="bg-gray-900 text-gray-100 p-6 rounded-lg overflow-x-auto mb-6">
            <pre className="text-sm font-mono">
{`Algorithm: RRT(q_init, q_goal, K, Δq)
----------------------------------------------
Input:
  q_init    : 시작 위치
  q_goal    : 목표 위치
  K         : 최대 반복 횟수
  Δq        : step_size (확장 거리)

Output:
  경로 또는 실패

1. T ← InitializeTree(q_init)

2. for i = 1 to K do
3.     q_rand ← SampleRandomPoint(C-space)

4.     // Goal bias: 10% 확률로 목표를 샘플링
5.     if Random(0, 1) < 0.1 then
6.         q_rand ← q_goal

7.     q_near ← NearestNeighbor(T, q_rand)

8.     q_new ← Steer(q_near, q_rand, Δq)

9.     if CollisionFree(q_near, q_new) then
10.        T.AddVertex(q_new)
11.        T.AddEdge(q_near, q_new)

12.        if Distance(q_new, q_goal) < threshold then
13.            return ExtractPath(T, q_init, q_new)
14.    end if
15. end for

16. return FAILURE

----------------------------------------------
Function Steer(q_near, q_rand, Δq):
    direction ← Normalize(q_rand - q_near)
    distance ← min(Distance(q_near, q_rand), Δq)
    return q_near + direction × distance`}
            </pre>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            RRT 시각화 예제
          </h3>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border border-blue-300 dark:border-blue-700 mb-6">
            <h4 className="font-semibold text-blue-900 dark:text-blue-300 mb-3">
              2D 평면에서 RRT 탐색 과정
            </h4>
            <div className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
              <p><strong>반복 1</strong>: q_rand(7, 8) 샘플링 → q_near(0, 0) → q_new(0.5, 0.6) 추가</p>
              <p><strong>반복 2</strong>: q_rand(3, 9) 샘플링 → q_near(0.5, 0.6) → q_new(0.9, 1.2) 추가</p>
              <p><strong>반복 3</strong>: q_rand(6, 2) 샘플링 → q_near(0.5, 0.6) → q_new(1.1, 0.5) 추가</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-3">
                step_size = 1.0으로 가정. 트리가 점진적으로 C-space를 탐색하며 성장함
              </p>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-600 p-6">
            <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-300 mb-3">
              ⚡ RRT의 장단점
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-green-700 dark:text-green-400 mb-2">✅ 장점</h4>
                <ul className="list-disc list-inside space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>고차원 공간에서 효율적</li>
                  <li>확률적으로 완전(Probabilistically Complete)</li>
                  <li>구현이 단순</li>
                  <li>동역학 제약 통합 용이</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-red-700 dark:text-red-400 mb-2">❌ 단점</h4>
                <ul className="list-disc list-inside space-y-1 text-sm text-gray-700 dark:text-gray-300">
                  <li>경로가 최적이 아님 (지그재그)</li>
                  <li>좁은 통로에서 비효율적</li>
                  <li>수렴 속도가 느림</li>
                  <li>step_size 튜닝 필요</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 3. RRT* (RRT Star) */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            3. RRT* (RRT Star) - 최적 경로 보장
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            RRT*는 RRT의 확장 버전으로, 점근적 최적성(Asymptotic Optimality)을 보장합니다.
            반복 횟수가 무한대로 갈 때 경로가 최적해로 수렴하는 특성을 가집니다.
          </p>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            RRT와 RRT*의 핵심 차이점
          </h3>
          <div className="overflow-x-auto mb-6">
            <table className="min-w-full border border-gray-300 dark:border-gray-600">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="px-4 py-3 border-b text-left">단계</th>
                  <th className="px-4 py-3 border-b text-left">RRT</th>
                  <th className="px-4 py-3 border-b text-left">RRT*</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">1. 노드 연결</td>
                  <td className="px-4 py-3">q_near에 직접 연결</td>
                  <td className="px-4 py-3 text-blue-600 dark:text-blue-400">
                    <strong>반경 내 여러 노드 고려</strong><br/>
                    비용 최소화 부모 선택
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">2. 트리 추가 후</td>
                  <td className="px-4 py-3">종료</td>
                  <td className="px-4 py-3 text-blue-600 dark:text-blue-400">
                    <strong>Rewire 수행</strong><br/>
                    주변 노드 재연결로 비용 감소
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">3. 최적성</td>
                  <td className="px-4 py-3">보장 없음</td>
                  <td className="px-4 py-3 text-green-600 dark:text-green-400">
                    <strong>점근적 최적</strong>
                  </td>
                </tr>
                <tr>
                  <td className="px-4 py-3 font-semibold">4. 계산 복잡도</td>
                  <td className="px-4 py-3">O(log n) per iteration</td>
                  <td className="px-4 py-3">O(log n × k) per iteration</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            RRT* 핵심 개선 사항
          </h3>

          <div className="space-y-4 mb-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg border border-blue-300 dark:border-blue-700">
              <h4 className="font-bold text-lg text-blue-900 dark:text-blue-300 mb-2">
                1. ChooseParent (최적 부모 선택)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                q_new 주변 반경 r 내의 모든 노드를 확인하여 시작점으로부터의 누적 비용이 최소가 되는 부모를 선택
              </p>
              <div className="bg-white dark:bg-gray-900 p-3 rounded font-mono text-xs">
                <p>Q_near ← Near(T, q_new, r)</p>
                <p>q_min ← argmin(Cost(q) + Distance(q, q_new)) for q in Q_near</p>
                <p>T.AddEdge(q_min, q_new)</p>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border border-green-300 dark:border-green-700">
              <h4 className="font-bold text-lg text-green-900 dark:text-green-300 mb-2">
                2. Rewire (트리 재구성)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                q_new를 통해 더 짧은 경로를 얻을 수 있는 주변 노드들의 부모를 q_new로 변경
              </p>
              <div className="bg-white dark:bg-gray-900 p-3 rounded font-mono text-xs">
                <p>for each q_near in Q_near do</p>
                <p className="ml-4">if Cost(q_new) + Distance(q_new, q_near) &lt; Cost(q_near) then</p>
                <p className="ml-8">T.ChangeParent(q_near, q_new)</p>
                <p className="ml-4">end if</p>
                <p>end for</p>
              </div>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            반경(Radius) 계산
          </h3>
          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-300 dark:border-gray-600 mb-6">
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              RRT*의 핵심 파라미터인 검색 반경은 트리 크기에 따라 동적으로 조정됩니다:
            </p>
            <div className="font-mono text-sm bg-white dark:bg-gray-800 p-4 rounded">
              <p className="mb-3">r(n) = γ · (log(n) / n)^(1/d)</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                n: 현재 트리 노드 수<br/>
                d: C-space 차원<br/>
                γ: 조정 상수 (보통 2배 × (1 + 1/d)^(1/d) × measure(C-free)^(1/d))
              </p>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-600 p-6">
            <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-300 mb-3">
              📊 RRT* 성능 특성
            </h3>
            <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li><strong>수렴 속도</strong>: 최적해에 가까워지는 속도는 O(n^(-1/d))</li>
              <li><strong>메모리 사용</strong>: 모든 노드와 엣지를 저장하므로 RRT보다 많음</li>
              <li><strong>실시간 적용</strong>: Informed RRT*, Anytime RRT* 등 변형으로 개선</li>
              <li><strong>실무 활용</strong>: MoveIt(ROS), OMPL 라이브러리에 표준 구현</li>
            </ul>
          </div>
        </section>

        {/* 4. A* 알고리즘 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            4. A* (A Star) 알고리즘
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            A*는 그래프 탐색 알고리즘으로, 휴리스틱 함수를 활용하여 최단 경로를 효율적으로 찾습니다.
            1968년 Peter Hart, Nils Nilsson, Bertram Raphael이 개발했으며, 격자 기반 경로 계획의 표준입니다.
          </p>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            A* 비용 함수
          </h3>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border border-blue-300 dark:border-blue-700 mb-6">
            <div className="font-mono text-lg mb-4 text-center">
              <strong className="text-blue-900 dark:text-blue-300">f(n) = g(n) + h(n)</strong>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="bg-white dark:bg-gray-900 p-4 rounded">
                <p className="font-semibold mb-2 text-green-700 dark:text-green-400">g(n) - 실제 비용</p>
                <p className="text-gray-700 dark:text-gray-300">
                  시작 노드에서 현재 노드 n까지의 <strong>실제 이동 비용</strong>
                </p>
              </div>
              <div className="bg-white dark:bg-gray-900 p-4 rounded">
                <p className="font-semibold mb-2 text-purple-700 dark:text-purple-400">h(n) - 휴리스틱 추정</p>
                <p className="text-gray-700 dark:text-gray-300">
                  현재 노드 n에서 목표까지의 <strong>예상 비용</strong> (추정치)
                </p>
              </div>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            휴리스틱 함수의 종류
          </h3>
          <div className="space-y-4 mb-6">
            <div className="bg-gray-50 dark:bg-gray-900 p-5 rounded-lg border border-gray-300 dark:border-gray-600">
              <h4 className="font-bold text-lg mb-2">1. 맨해튼 거리 (Manhattan Distance)</h4>
              <p className="font-mono text-sm mb-2">h(n) = |x_goal - x_n| + |y_goal - y_n|</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                4방향 이동(상하좌우)만 가능한 격자에서 사용. 계산 빠름
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-5 rounded-lg border border-gray-300 dark:border-gray-600">
              <h4 className="font-bold text-lg mb-2">2. 유클리드 거리 (Euclidean Distance)</h4>
              <p className="font-mono text-sm mb-2">h(n) = √[(x_goal - x_n)² + (y_goal - y_n)²]</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                대각선 이동 가능 시 적합. 실제 거리에 더 가까움
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900 p-5 rounded-lg border border-gray-300 dark:border-gray-600">
              <h4 className="font-bold text-lg mb-2">3. 체비셰프 거리 (Chebyshev Distance)</h4>
              <p className="font-mono text-sm mb-2">h(n) = max(|x_goal - x_n|, |y_goal - y_n|)</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                8방향 이동(대각선 포함) 가능 시 사용
              </p>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            A* 알고리즘 의사 코드
          </h3>
          <div className="bg-gray-900 text-gray-100 p-6 rounded-lg overflow-x-auto mb-6">
            <pre className="text-sm font-mono">
{`Algorithm: A*(start, goal)
----------------------------------------------
1. openSet ← {start}
2. closedSet ← {}
3. g[start] ← 0
4. f[start] ← g[start] + h(start)

5. while openSet is not empty do
6.     current ← node in openSet with lowest f score

7.     if current == goal then
8.         return ReconstructPath(current)

9.     openSet.Remove(current)
10.    closedSet.Add(current)

11.    for each neighbor of current do
12.        if neighbor in closedSet then
13.            continue

14.        tentative_g ← g[current] + Distance(current, neighbor)

15.        if neighbor not in openSet then
16.            openSet.Add(neighbor)
17.        else if tentative_g >= g[neighbor] then
18.            continue  // 더 나은 경로가 아님

19.        // 이 경로가 최선임
20.        parent[neighbor] ← current
21.        g[neighbor] ← tentative_g
22.        f[neighbor] ← g[neighbor] + h(neighbor)
23.    end for
24. end while

25. return FAILURE`}
            </pre>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            최적성 보장 조건: Admissible Heuristic
          </h3>
          <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-600 p-6 mb-6">
            <p className="font-semibold text-red-900 dark:text-red-300 mb-3">
              A*가 최적 경로를 보장하려면 휴리스틱이 <strong>허용 가능(Admissible)</strong>해야 합니다:
            </p>
            <div className="font-mono text-sm bg-white dark:bg-gray-900 p-4 rounded mb-3">
              h(n) ≤ h*(n)
            </div>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              여기서 h*(n)은 n에서 목표까지의 <strong>실제 최소 비용</strong>입니다.
              즉, 휴리스틱은 절대 실제 비용을 과대평가하면 안 됩니다.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            A* 예제: 5×5 격자 탐색
          </h3>
          <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-lg border border-green-300 dark:border-green-700">
            <p className="font-semibold mb-3">초기 상태:</p>
            <pre className="text-xs font-mono bg-white dark:bg-gray-900 p-3 rounded mb-4">
{`S: 시작(0,0)   G: 목표(4,4)   X: 장애물
┌─┬─┬─┬─┬─┐
│S│ │ │X│ │
├─┼─┼─┼─┼─┤
│ │X│ │X│ │
├─┼─┼─┼─┼─┤
│ │ │ │ │ │
├─┼─┼─┼─┼─┤
│ │X│X│ │ │
├─┼─┼─┼─┼─┤
│ │ │ │ │G│
└─┴─┴─┴─┴─┘`}
            </pre>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              <strong>휴리스틱</strong>: 맨해튼 거리 사용<br/>
              <strong>결과</strong>: S(0,0) → (1,0) → (2,0) → (2,1) → (2,2) → (3,2) → (4,2) → (4,3) → G(4,4)<br/>
              <strong>총 비용</strong>: 8 (최적 경로)
            </p>
          </div>
        </section>

        {/* 5. DWA (Dynamic Window Approach) */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            5. DWA (Dynamic Window Approach)
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            DWA는 동적 환경에서 실시간으로 로봇의 속도를 조절하여 장애물을 회피하는 <strong>반응형(Reactive)</strong> 알고리즘입니다.
            Dieter Fox, Wolfram Burgard, Sebastian Thrun이 1997년 개발했으며, 이동 로봇에 널리 사용됩니다.
          </p>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            DWA 핵심 개념: Velocity Space
          </h3>
          <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border border-blue-300 dark:border-blue-700 mb-6">
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              로봇의 현재 상태에서 <strong>물리적으로 도달 가능한 속도 집합</strong>을 계산하고,
              그 중 최적의 선속도(v)와 각속도(ω)를 선택합니다.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
              <div className="bg-white dark:bg-gray-900 p-3 rounded">
                <p className="font-semibold mb-1">1. Admissible Window</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">장애물 회피 가능한 속도</p>
              </div>
              <div className="bg-white dark:bg-gray-900 p-3 rounded">
                <p className="font-semibold mb-1">2. Dynamic Window</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">가감속 제약 내 도달 속도</p>
              </div>
              <div className="bg-white dark:bg-gray-900 p-3 rounded">
                <p className="font-semibold mb-1">3. Search Space</p>
                <p className="text-xs text-gray-600 dark:text-gray-400">두 윈도우의 교집합</p>
              </div>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            DWA 목적 함수
          </h3>
          <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg border border-gray-300 dark:border-gray-600 mb-6">
            <div className="font-mono text-sm mb-4">
              G(v, ω) = α·heading(v, ω) + β·dist(v, ω) + γ·velocity(v, ω)
            </div>
            <div className="space-y-3 text-sm text-gray-700 dark:text-gray-300">
              <div className="flex items-start gap-3">
                <span className="font-semibold min-w-[120px]">heading(v, ω)</span>
                <span>목표 방향과의 정렬도 (높을수록 좋음)</span>
              </div>
              <div className="flex items-start gap-3">
                <span className="font-semibold min-w-[120px]">dist(v, ω)</span>
                <span>가장 가까운 장애물까지의 거리 (멀수록 좋음)</span>
              </div>
              <div className="flex items-start gap-3">
                <span className="font-semibold min-w-[120px]">velocity(v, ω)</span>
                <span>전진 속도 (빠를수록 좋음)</span>
              </div>
              <div className="flex items-start gap-3">
                <span className="font-semibold min-w-[120px]">α, β, γ</span>
                <span>가중치 (α + β + γ = 1)</span>
              </div>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            DWA 알고리즘 단계
          </h3>
          <ol className="list-decimal list-inside space-y-3 text-gray-700 dark:text-gray-300 mb-6">
            <li>
              <strong>Dynamic Window 계산</strong>: 현재 속도와 가감속 한계로 다음 time step에 도달 가능한 (v, ω) 범위 산출
            </li>
            <li>
              <strong>Admissible Velocities 필터링</strong>: 각 (v, ω) 조합으로 원호 궤적 시뮬레이션 후 충돌 여부 확인
            </li>
            <li>
              <strong>목적 함수 평가</strong>: 모든 허용 가능한 속도 조합에 대해 G(v, ω) 계산
            </li>
            <li>
              <strong>최적 속도 선택</strong>: G(v, ω)를 최대화하는 (v*, ω*) 선택
            </li>
            <li>
              <strong>명령 전송</strong>: 선택한 속도로 짧은 시간(예: 0.1초) 동안 로봇 제어
            </li>
            <li>
              <strong>반복</strong>: 센서 데이터 업데이트 후 1단계부터 재실행 (10Hz 주기)
            </li>
          </ol>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            DWA 수식 상세
          </h3>
          <div className="space-y-4 mb-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border border-green-300 dark:border-green-700">
              <h4 className="font-bold mb-2">Dynamic Window (가감속 제약)</h4>
              <pre className="text-xs font-mono bg-white dark:bg-gray-900 p-3 rounded">
{`V_d = {
  (v, ω) |
  v ∈ [v_current - v̇_max·Δt, v_current + v̇_max·Δt],
  ω ∈ [ω_current - ω̇_max·Δt, ω_current + ω̇_max·Δt]
}

v̇_max: 최대 선가속도
ω̇_max: 최대 각가속도
Δt: 시간 간격 (예: 0.1s)`}
              </pre>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border border-purple-300 dark:border-purple-700">
              <h4 className="font-bold mb-2">Admissible Velocities (충돌 회피)</h4>
              <pre className="text-xs font-mono bg-white dark:bg-gray-900 p-3 rounded">
{`V_a = {
  (v, ω) |
  v ≤ √(2·dist(v,ω)·v̇_brake)
}

dist(v, ω): 해당 속도로 이동 시 가장 가까운 장애물 거리
v̇_brake: 제동 감속도`}
              </pre>
            </div>

            <div className="bg-orange-50 dark:bg-orange-900/20 p-5 rounded-lg border border-orange-300 dark:border-orange-700">
              <h4 className="font-bold mb-2">Search Space (최종 후보)</h4>
              <pre className="text-xs font-mono bg-white dark:bg-gray-900 p-3 rounded">
{`V_r = V_s ∩ V_d ∩ V_a

V_s: 로봇 하드웨어 속도 한계
      v ∈ [0, v_max], ω ∈ [-ω_max, ω_max]`}
              </pre>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-600 p-6">
            <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-300 mb-3">
              ⚡ DWA 특징 요약
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <h4 className="font-semibold text-green-700 dark:text-green-400 mb-2">✅ 강점</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>실시간 처리 (10-20Hz)</li>
                  <li>동역학 제약 자동 반영</li>
                  <li>동적 장애물 대응 가능</li>
                  <li>구현 단순, 튜닝 용이</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-red-700 dark:text-red-400 mb-2">❌ 약점</h4>
                <ul className="list-disc list-inside space-y-1 text-gray-700 dark:text-gray-300">
                  <li>지역 최소값(Local minima) 문제</li>
                  <li>좁은 통로에서 진동 가능</li>
                  <li>전역 경로 필요 (단독 사용 어려움)</li>
                  <li>가중치 α, β, γ 튜닝 필요</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* 6. 경로 평활화 */}
        <section className="mb-12 bg-white dark:bg-gray-800 rounded-xl p-8 border border-gray-200 dark:border-gray-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            6. 경로 평활화 (Path Smoothing)
          </h2>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-6">
            RRT나 A* 같은 알고리즘으로 생성된 경로는 종종 지그재그 형태로 비효율적입니다.
            경로 평활화는 원래 경로의 충돌 없음을 유지하면서 부드럽고 짧은 경로로 개선합니다.
          </p>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            주요 평활화 기법
          </h3>

          <div className="space-y-4 mb-6">
            <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-lg border border-blue-300 dark:border-blue-700">
              <h4 className="font-bold text-lg text-blue-900 dark:text-blue-300 mb-2">
                1. Shortcutting (지름길 찾기)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                경로 상의 두 임의 점을 직선으로 연결 시도. 충돌 없으면 중간 경로 제거
              </p>
              <div className="bg-white dark:bg-gray-900 p-3 rounded text-xs font-mono">
                <p>for iteration = 1 to max_iterations do</p>
                <p className="ml-4">p1, p2 ← 경로에서 랜덤 선택 (p1 &lt; p2)</p>
                <p className="ml-4">if CollisionFree(p1, p2) then</p>
                <p className="ml-8">경로[p1+1:p2-1] 삭제</p>
                <p className="ml-4">end if</p>
                <p>end for</p>
              </div>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-5 rounded-lg border border-green-300 dark:border-green-700">
              <h4 className="font-bold text-lg text-green-900 dark:text-green-300 mb-2">
                2. Gradient Descent Smoothing
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                각 경로 점을 주변 점들의 평균 방향으로 이동하여 부드럽게 만듦
              </p>
              <div className="bg-white dark:bg-gray-900 p-3 rounded text-xs font-mono">
                <p>for each waypoint p_i (except start/goal) do</p>
                <p className="ml-4">p_i_new ← p_i + α·(p_i-1 + p_i+1 - 2·p_i)</p>
                <p className="ml-4">if CollisionFree(p_i_new) then</p>
                <p className="ml-8">p_i ← p_i_new</p>
                <p className="ml-4">end if</p>
                <p>end for</p>
                <p className="text-gray-500">α: 학습률 (0.1 ~ 0.5)</p>
              </div>
            </div>

            <div className="bg-purple-50 dark:bg-purple-900/20 p-5 rounded-lg border border-purple-300 dark:border-purple-700">
              <h4 className="font-bold text-lg text-purple-900 dark:text-purple-300 mb-2">
                3. Spline Interpolation (스플라인 보간)
              </h4>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
                경로 포인트를 통과하는 곡선(Cubic Spline, B-Spline)을 생성하여 부드러운 궤적 생성
              </p>
              <ul className="list-disc list-inside space-y-1 text-xs text-gray-700 dark:text-gray-300">
                <li>Cubic Spline: 2차 미분 연속성 보장 (가속도 부드러움)</li>
                <li>B-Spline: 제어점 변경 시 지역적 영향만 (전역 재계산 불필요)</li>
                <li>Hermite Spline: 접선 방향 명시 가능 (방향 제약)</li>
              </ul>
            </div>
          </div>

          <h3 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
            실전 예제: Shortcutting 비포/애프터
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-300 dark:border-red-700">
              <p className="font-semibold text-red-900 dark:text-red-300 mb-2">원본 경로 (RRT 결과)</p>
              <ul className="text-xs font-mono space-y-1">
                <li>(0, 0) → (1, 2) → (2, 1) →</li>
                <li>(3, 3) → (4, 2) → (5, 4) →</li>
                <li>(6, 3) → (7, 5) → (8, 4) →</li>
                <li>(10, 10)</li>
              </ul>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">총 길이: ~18.5</p>
            </div>

            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-300 dark:border-green-700">
              <p className="font-semibold text-green-900 dark:text-green-300 mb-2">평활화 후</p>
              <ul className="text-xs font-mono space-y-1">
                <li>(0, 0) → (3, 3) →</li>
                <li>(6, 3) → (10, 10)</li>
              </ul>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">총 길이: ~14.8 (20% 단축)</p>
            </div>
          </div>

          <div className="bg-orange-50 dark:bg-orange-900/20 border-l-4 border-orange-600 p-6">
            <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-300 mb-3">
              ⚠️ 평활화 주의사항
            </h3>
            <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li><strong>충돌 재검사 필수</strong>: 평활화 후 모든 선분에 대해 충돌 검사</li>
              <li><strong>동역학 제약 확인</strong>: 곡률이 로봇 회전 반경 내인지 검증</li>
              <li><strong>반복 횟수 제한</strong>: 무한 루프 방지 (100~1000회)</li>
              <li><strong>수렴 조건 설정</strong>: 개선폭이 미미하면 조기 종료</li>
            </ul>
          </div>
        </section>

        {/* 7. 요약 */}
        <section className="mb-12 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-8 border border-orange-300 dark:border-orange-700">
          <h2 className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-6">
            📌 핵심 요약
          </h2>

          <div className="overflow-x-auto mb-6">
            <table className="min-w-full border border-gray-300 dark:border-gray-600">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="px-4 py-3 border-b text-left">알고리즘</th>
                  <th className="px-4 py-3 border-b text-left">장점</th>
                  <th className="px-4 py-3 border-b text-left">단점</th>
                  <th className="px-4 py-3 border-b text-left">적합한 상황</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">RRT</td>
                  <td className="px-4 py-3">고차원 빠름, 구현 쉬움</td>
                  <td className="px-4 py-3">비최적 경로</td>
                  <td className="px-4 py-3">복잡한 C-space, 6-DOF+</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">RRT*</td>
                  <td className="px-4 py-3">점근적 최적, 고차원</td>
                  <td className="px-4 py-3">수렴 느림, 메모리</td>
                  <td className="px-4 py-3">최적성 중요, 계산 시간 여유</td>
                </tr>
                <tr className="border-b">
                  <td className="px-4 py-3 font-semibold">A*</td>
                  <td className="px-4 py-3">최적 보장, 예측 가능</td>
                  <td className="px-4 py-3">고차원 비효율</td>
                  <td className="px-4 py-3">2D/3D 격자, 정적 환경</td>
                </tr>
                <tr>
                  <td className="px-4 py-3 font-semibold">DWA</td>
                  <td className="px-4 py-3">실시간, 동적 환경</td>
                  <td className="px-4 py-3">지역 최소값</td>
                  <td className="px-4 py-3">이동 로봇, 장애물 회피</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                🔑 실무 선택 가이드
              </h3>
              <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li><strong>산업용 로봇 팔(6-DOF)</strong>: RRT* + MoveIt</li>
                <li><strong>창고 이동 로봇</strong>: A* (전역) + DWA (지역)</li>
                <li><strong>자율주행차</strong>: Hybrid A* + MPC</li>
                <li><strong>드론 경로</strong>: RRT-Connect + Trajectory Opt</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                🛠️ 구현 라이브러리
              </h3>
              <ul className="list-disc list-inside space-y-2 text-sm text-gray-700 dark:text-gray-300">
                <li><strong>OMPL</strong> (C++): RRT, RRT*, PRM 등 50+ 알고리즘</li>
                <li><strong>MoveIt</strong> (ROS): 로봇 팔 경로 계획 통합</li>
                <li><strong>Navigation Stack</strong> (ROS): A*, DWA, costmap</li>
                <li><strong>Python Robotics</strong>: 교육용 시각화 구현</li>
              </ul>
            </div>
          </div>

          <div className="mt-6 bg-orange-100 dark:bg-orange-900/30 p-5 rounded-lg border border-orange-400 dark:border-orange-600">
            <p className="text-sm text-gray-800 dark:text-gray-200">
              <strong>다음 장 미리보기:</strong> Chapter 5에서는 궤적 생성(Trajectory Generation)을 학습합니다.
              경로 계획으로 얻은 웨이포인트를 시간 변수를 포함한 부드러운 궤적으로 변환하는 방법을 다룹니다.
            </p>
          </div>
        </section>

        {/* Chapter Navigation */}
        <ChapterNavigation
          currentChapter={4}
          totalChapters={8}
          moduleSlug="robotics-manipulation"
        />
      </div>
    </div>
  )
}
