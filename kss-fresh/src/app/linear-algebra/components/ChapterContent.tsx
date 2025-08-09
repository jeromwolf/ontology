'use client'

import { 
  Move, 
  Grid3x3, 
  Maximize2, 
  GitBranch, 
  Axis3d,
  Layers,
  Binary,
  Sparkles,
  ChevronRight,
  ArrowRight,
  Info,
  Lightbulb,
  Code,
  BookOpen,
  Target,
  Zap
} from 'lucide-react'

const chaptersContent: { [key: string]: React.ReactNode } = {
  'vectors': (
    <div className="space-y-8">
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          벡터와 벡터공간
        </h2>
        
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-8">
          <div className="flex items-start gap-3">
            <Move className="w-6 h-6 text-indigo-600 dark:text-indigo-400 mt-1" />
            <div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                학습 목표
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-indigo-500 mt-0.5" />
                  <span>벡터의 기하학적/대수적 의미 이해</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-indigo-500 mt-0.5" />
                  <span>벡터 연산과 그 응용 마스터</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-indigo-500 mt-0.5" />
                  <span>벡터공간의 개념과 부분공간 이해</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
              <BookOpen className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              벡터의 정의
            </h3>
            <div className="space-y-3 text-gray-600 dark:text-gray-300">
              <p>
                벡터는 <strong>크기(magnitude)</strong>와 <strong>방향(direction)</strong>을 가진 수학적 객체입니다.
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm">
                <div>v = [v₁, v₂, ..., vₙ]</div>
                <div className="mt-2 text-gray-500 dark:text-gray-400">
                  n차원 벡터 표현
                </div>
              </div>
              <ul className="space-y-2 text-sm">
                <li>• 위치 벡터: 원점에서 시작하는 벡터</li>
                <li>• 자유 벡터: 시작점이 자유로운 벡터</li>
                <li>• 단위 벡터: 크기가 1인 벡터</li>
              </ul>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
              <Zap className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
              벡터 연산
            </h3>
            <div className="space-y-3 text-gray-600 dark:text-gray-300">
              <div className="space-y-2">
                <h4 className="font-medium text-gray-900 dark:text-white">덧셈과 스칼라 곱</h4>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3 font-mono text-sm">
                  <div>u + v = [u₁+v₁, u₂+v₂, ...]</div>
                  <div>αv = [αv₁, αv₂, ...]</div>
                </div>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-gray-900 dark:text-white">내적 (Dot Product)</h4>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3 font-mono text-sm">
                  <div>u·v = Σ(uᵢ × vᵢ)</div>
                  <div className="text-gray-500 dark:text-gray-400 mt-1">
                    = |u||v|cos(θ)
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
            <Target className="w-5 h-5 text-green-600 dark:text-green-400" />
            벡터공간과 부분공간
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">벡터공간의 공리</h4>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                <li>1. 덧셈의 결합법칙: (u+v)+w = u+(v+w)</li>
                <li>2. 덧셈의 교환법칙: u+v = v+u</li>
                <li>3. 영벡터의 존재: v+0 = v</li>
                <li>4. 역원소의 존재: v+(-v) = 0</li>
                <li>5. 스칼라 곱의 분배법칙</li>
                <li>6. 스칼라 곱의 결합법칙</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">중요한 부분공간</h4>
              <div className="space-y-3">
                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <h5 className="font-medium text-blue-700 dark:text-blue-300">Null Space</h5>
                  <p className="text-sm text-blue-600 dark:text-blue-400">
                    Ax = 0을 만족하는 모든 x의 집합
                  </p>
                </div>
                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                  <h5 className="font-medium text-green-700 dark:text-green-300">Column Space</h5>
                  <p className="text-sm text-green-600 dark:text-green-400">
                    A의 열벡터들의 선형결합으로 만들어지는 공간
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
            <Code className="w-5 h-5 text-purple-600 dark:text-purple-400" />
            Python 구현 예제
          </h3>
          <pre className="bg-white dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <code className="text-sm text-gray-700 dark:text-gray-300">{`import numpy as np

# 벡터 생성
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 벡터 연산
addition = v1 + v2  # 벡터 덧셈
scalar_mult = 3 * v1  # 스칼라 곱
dot_product = np.dot(v1, v2)  # 내적

# 벡터 크기와 정규화
magnitude = np.linalg.norm(v1)  # 크기
normalized = v1 / magnitude  # 정규화

# 두 벡터 사이의 각도
cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
angle = np.arccos(cos_angle) * 180 / np.pi

print(f"벡터 덧셈: {addition}")
print(f"내적: {dot_product}")
print(f"벡터 크기: {magnitude:.2f}")
print(f"각도: {angle:.2f}°")`}</code>
          </pre>
        </div>
      </section>
    </div>
  ),

  'matrices': (
    <div className="space-y-8">
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          행렬과 행렬연산
        </h2>
        
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-xl p-6 mb-8">
          <div className="flex items-start gap-3">
            <Grid3x3 className="w-6 h-6 text-purple-600 dark:text-purple-400 mt-1" />
            <div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                학습 목표
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-purple-500 mt-0.5" />
                  <span>행렬의 기본 연산과 성질 이해</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-purple-500 mt-0.5" />
                  <span>역행렬과 행렬식의 의미와 계산</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-purple-500 mt-0.5" />
                  <span>특수 행렬의 종류와 응용</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              행렬의 기본 개념
            </h3>
            <div className="space-y-3 text-gray-600 dark:text-gray-300">
              <p>
                행렬은 수를 직사각형 배열로 나열한 수학적 객체입니다.
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <div className="font-mono text-sm">
                  A = [a₁₁  a₁₂  ...  a₁ₙ]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;[a₂₁  a₂₂  ...  a₂ₙ]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;[...  ...  ...  ...]<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;[aₘ₁  aₘ₂  ...  aₘₙ]
                </div>
                <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                  m × n 행렬 (m행 n열)
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              행렬 곱셈
            </h3>
            <div className="space-y-3 text-gray-600 dark:text-gray-300">
              <p className="text-sm">
                행렬 곱셈은 첫 번째 행렬의 행과 두 번째 행렬의 열을 내적하여 계산합니다.
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-sm">
                <div>(AB)ᵢⱼ = Σₖ aᵢₖbₖⱼ</div>
              </div>
              <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                <p className="text-sm text-yellow-700 dark:text-yellow-300">
                  <strong>주의:</strong> AB ≠ BA (교환법칙 성립 안함)
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            역행렬과 행렬식
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">역행렬 (Inverse Matrix)</h4>
              <div className="space-y-3 text-sm text-gray-600 dark:text-gray-300">
                <p>AA⁻¹ = A⁻¹A = I를 만족하는 행렬 A⁻¹</p>
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
                  <p className="text-blue-700 dark:text-blue-300">
                    2×2 행렬의 역행렬:
                  </p>
                  <div className="font-mono text-xs mt-2">
                    A⁻¹ = (1/det(A)) × [d  -b]<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[-c  a]
                  </div>
                </div>
              </div>
            </div>
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">행렬식 (Determinant)</h4>
              <div className="space-y-3 text-sm text-gray-600 dark:text-gray-300">
                <p>정사각행렬에 대해 정의되는 스칼라 값</p>
                <ul className="space-y-1">
                  <li>• det(A) ≠ 0 ⟺ A는 가역행렬</li>
                  <li>• det(AB) = det(A) × det(B)</li>
                  <li>• det(A^T) = det(A)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            특수 행렬
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg">
              <h4 className="font-medium mb-2 text-blue-700 dark:text-blue-300">대칭 행렬</h4>
              <p className="text-sm text-blue-600 dark:text-blue-400">
                A = A^T<br/>
                모든 고유값이 실수
              </p>
            </div>
            <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
              <h4 className="font-medium mb-2 text-green-700 dark:text-green-300">직교 행렬</h4>
              <p className="text-sm text-green-600 dark:text-green-400">
                Q^T Q = QQ^T = I<br/>
                회전과 반사 변환
              </p>
            </div>
            <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg">
              <h4 className="font-medium mb-2 text-purple-700 dark:text-purple-300">대각 행렬</h4>
              <p className="text-sm text-purple-600 dark:text-purple-400">
                비대각 원소가 0<br/>
                계산이 매우 효율적
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
            <Code className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            NumPy로 행렬 연산하기
          </h3>
          <pre className="bg-white dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <code className="text-sm text-gray-700 dark:text-gray-300">{`import numpy as np

# 행렬 생성
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 기본 연산
C = A + B  # 행렬 덧셈
D = A @ B  # 행렬 곱셈 (Python 3.5+)
E = A * B  # 원소별 곱셈

# 역행렬과 행렬식
det_A = np.linalg.det(A)  # 행렬식
inv_A = np.linalg.inv(A)  # 역행렬

# 전치 행렬
A_transpose = A.T

# 고유값과 고유벡터
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"행렬식: {det_A:.2f}")
print(f"역행렬:\\n{inv_A}")
print(f"고유값: {eigenvalues}")`}</code>
          </pre>
        </div>
      </section>
    </div>
  ),

  'linear-transformations': (
    <div className="space-y-8">
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          선형변환
        </h2>
        
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 mb-8">
          <div className="flex items-start gap-3">
            <Maximize2 className="w-6 h-6 text-green-600 dark:text-green-400 mt-1" />
            <div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                학습 목표
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-green-500 mt-0.5" />
                  <span>선형변환의 정의와 성질 이해</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-green-500 mt-0.5" />
                  <span>변환 행렬과 기하학적 해석</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-green-500 mt-0.5" />
                  <span>좌표계 변환과 기저 변환</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            선형변환의 정의
          </h3>
          <div className="space-y-4 text-gray-600 dark:text-gray-300">
            <p>
              함수 T: V → W가 다음 두 조건을 만족할 때 <strong>선형변환</strong>이라 합니다:
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-2">가법성</h4>
                <p className="font-mono text-sm">T(u + v) = T(u) + T(v)</p>
              </div>
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <h4 className="font-medium text-green-700 dark:text-green-300 mb-2">동차성</h4>
                <p className="font-mono text-sm">T(αv) = αT(v)</p>
              </div>
            </div>
            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <p className="text-sm text-yellow-700 dark:text-yellow-300">
                <strong>핵심:</strong> 모든 선형변환은 행렬로 표현 가능합니다!
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            2D 기하학적 변환
          </h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 rounded-lg">
              <h4 className="font-medium text-red-700 dark:text-red-300 mb-2">회전</h4>
              <div className="font-mono text-xs">
                [cos θ  -sin θ]<br/>
                [sin θ   cos θ]
              </div>
            </div>
            <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg">
              <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-2">스케일링</h4>
              <div className="font-mono text-xs">
                [sx  0 ]<br/>
                [0   sy]
              </div>
            </div>
            <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
              <h4 className="font-medium text-green-700 dark:text-green-300 mb-2">반사</h4>
              <div className="font-mono text-xs">
                [1   0]  (x축)<br/>
                [0  -1]
              </div>
            </div>
            <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg">
              <h4 className="font-medium text-purple-700 dark:text-purple-300 mb-2">전단</h4>
              <div className="font-mono text-xs">
                [1  k]<br/>
                [0  1]
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            핵과 상 (Kernel and Image)
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">핵 (Kernel/Null Space)</h4>
              <div className="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                <p className="text-sm text-orange-700 dark:text-orange-300 mb-2">
                  {`ker(T) = {v ∈ V : T(v) = 0}`}
                </p>
                <p className="text-sm text-orange-600 dark:text-orange-400">
                  변환에 의해 0으로 매핑되는 벡터들의 집합
                </p>
              </div>
            </div>
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">상 (Image/Range)</h4>
              <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
                <p className="text-sm text-indigo-700 dark:text-indigo-300 mb-2">
                  {`Im(T) = {T(v) : v ∈ V}`}
                </p>
                <p className="text-sm text-indigo-600 dark:text-indigo-400">
                  변환의 출력으로 가능한 모든 벡터들의 집합
                </p>
              </div>
            </div>
          </div>
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <strong>차원 정리:</strong> dim(V) = dim(ker(T)) + dim(Im(T))
            </p>
          </div>
        </div>

        <div className="bg-gradient-to-br from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
            <Code className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
            변환 시각화 코드
          </h3>
          <pre className="bg-white dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <code className="text-sm text-gray-700 dark:text-gray-300">{`import numpy as np
import matplotlib.pyplot as plt

def visualize_transformation(A, vectors):
    """선형변환 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 원본 벡터
    ax1.set_title("Original")
    for v in vectors:
        ax1.arrow(0, 0, v[0], v[1], head_width=0.1, 
                 head_length=0.1, fc='blue', ec='blue')
    
    # 변환된 벡터
    ax2.set_title("Transformed")
    for v in vectors:
        transformed = A @ v
        ax2.arrow(0, 0, transformed[0], transformed[1], 
                 head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    # 축 설정
    for ax in [ax1, ax2]:
        ax.grid(True)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.show()

# 회전 변환 예제
theta = np.pi/4  # 45도 회전
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# 기저 벡터
basis_vectors = [np.array([1, 0]), np.array([0, 1])]
visualize_transformation(rotation_matrix, basis_vectors)`}</code>
          </pre>
        </div>
      </section>
    </div>
  ),

  'eigenvalues': (
    <div className="space-y-8">
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          고유값과 고유벡터
        </h2>
        
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-xl p-6 mb-8">
          <div className="flex items-start gap-3">
            <GitBranch className="w-6 h-6 text-yellow-600 dark:text-yellow-400 mt-1" />
            <div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                학습 목표
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-yellow-500 mt-0.5" />
                  <span>고유값과 고유벡터의 기하학적 의미</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-yellow-500 mt-0.5" />
                  <span>특성방정식과 고유값 계산</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-yellow-500 mt-0.5" />
                  <span>대각화와 그 응용</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            고유값과 고유벡터란?
          </h3>
          <div className="space-y-4 text-gray-600 dark:text-gray-300">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-blue-700 dark:text-blue-300 mb-2">
                <strong>정의:</strong> Av = λv를 만족하는 0이 아닌 벡터 v와 스칼라 λ
              </p>
              <ul className="text-sm space-y-1 text-blue-600 dark:text-blue-400">
                <li>• λ: 고유값 (eigenvalue)</li>
                <li>• v: 고유벡터 (eigenvector)</li>
              </ul>
            </div>
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <p className="text-green-700 dark:text-green-300">
                <strong>기하학적 의미:</strong> 행렬 A에 의한 변환에서 방향이 바뀌지 않는 벡터
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            고유값 계산 과정
          </h3>
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center font-bold">
                1
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">특성방정식 설정</h4>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3 font-mono text-sm">
                  det(A - λI) = 0
                </div>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center font-bold">
                2
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">특성다항식 전개</h4>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  n차 다항식을 얻고 근을 구함
                </p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-indigo-600 text-white rounded-full flex items-center justify-center font-bold">
                3
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">고유벡터 계산</h4>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3 font-mono text-sm">
                  (A - λI)v = 0 해결
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            행렬의 대각화
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">대각화 조건</h4>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-gray-400 mt-0.5" />
                  <span>n개의 선형독립인 고유벡터 존재</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-gray-400 mt-0.5" />
                  <span>대칭행렬은 항상 대각화 가능</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-gray-400 mt-0.5" />
                  <span>서로 다른 고유값 → 대각화 가능</span>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">대각화 공식</h4>
              <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                <div className="font-mono text-sm text-purple-700 dark:text-purple-300">
                  A = PDP⁻¹
                </div>
                <ul className="mt-2 text-sm space-y-1 text-purple-600 dark:text-purple-400">
                  <li>• P: 고유벡터 행렬</li>
                  <li>• D: 고유값 대각행렬</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            응용 분야
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 rounded-lg">
              <h4 className="font-medium text-red-700 dark:text-red-300 mb-2">PCA</h4>
              <p className="text-sm text-red-600 dark:text-red-400">
                주성분 분석에서 공분산 행렬의 고유벡터 활용
              </p>
            </div>
            <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg">
              <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-2">PageRank</h4>
              <p className="text-sm text-blue-600 dark:text-blue-400">
                구글의 페이지 순위 알고리즘
              </p>
            </div>
            <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
              <h4 className="font-medium text-green-700 dark:text-green-300 mb-2">진동 분석</h4>
              <p className="text-sm text-green-600 dark:text-green-400">
                고유진동수와 모드 해석
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
            <Code className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            고유값 분해 실습
          </h3>
          <pre className="bg-white dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <code className="text-sm text-gray-700 dark:text-gray-300">{`import numpy as np
from numpy.linalg import eig, inv

# 행렬 정의
A = np.array([[4, -2], 
              [1,  1]])

# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = eig(A)

print("고유값:", eigenvalues)
print("고유벡터:\\n", eigenvectors)

# 대각화 검증
P = eigenvectors
D = np.diag(eigenvalues)
P_inv = inv(P)

# A = PDP^(-1) 검증
A_reconstructed = P @ D @ P_inv
print("\\n원본 행렬:\\n", A)
print("재구성된 행렬:\\n", A_reconstructed)

# 거듭제곱 계산 (대각화 활용)
def matrix_power(A, n):
    """대각화를 이용한 행렬 거듭제곱"""
    eigenvalues, P = eig(A)
    D_n = np.diag(eigenvalues ** n)
    return P @ D_n @ inv(P)

A_10 = matrix_power(A, 10)
print("\\nA^10:\\n", A_10)`}</code>
          </pre>
        </div>
      </section>
    </div>
  ),

  'orthogonality': (
    <div className="space-y-8">
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          직교성과 정규화
        </h2>
        
        <div className="bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 rounded-xl p-6 mb-8">
          <div className="flex items-start gap-3">
            <Axis3d className="w-6 h-6 text-cyan-600 dark:text-cyan-400 mt-1" />
            <div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                학습 목표
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-cyan-500 mt-0.5" />
                  <span>직교와 정규직교의 개념 이해</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-cyan-500 mt-0.5" />
                  <span>그람-슈미트 과정 마스터</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-cyan-500 mt-0.5" />
                  <span>QR 분해와 최소제곱법 응용</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            직교성의 개념
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">직교 벡터</h4>
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-blue-700 dark:text-blue-300 mb-2">
                  u·v = 0 ⟺ u ⊥ v
                </p>
                <p className="text-sm text-blue-600 dark:text-blue-400">
                  두 벡터의 내적이 0일 때 직교
                </p>
              </div>
            </div>
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">정규직교 벡터</h4>
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <p className="text-green-700 dark:text-green-300 mb-2">
                  u·v = δᵢⱼ (크로네커 델타)
                </p>
                <p className="text-sm text-green-600 dark:text-green-400">
                  직교하면서 크기가 1인 벡터들
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            그람-슈미트 과정
          </h3>
          <div className="space-y-4">
            <p className="text-gray-600 dark:text-gray-300">
              선형독립인 벡터들을 정규직교 벡터로 변환하는 알고리즘
            </p>
            <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg p-4">
              <ol className="space-y-3 text-sm">
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-indigo-600 text-white rounded-full flex items-center justify-center text-xs font-bold">1</span>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">첫 번째 벡터 정규화</p>
                    <p className="font-mono text-xs mt-1">u₁ = v₁ / ||v₁||</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-indigo-600 text-white rounded-full flex items-center justify-center text-xs font-bold">2</span>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">직교 성분 추출</p>
                    <p className="font-mono text-xs mt-1">w₂ = v₂ - (v₂·u₁)u₁</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-indigo-600 text-white rounded-full flex items-center justify-center text-xs font-bold">3</span>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">정규화</p>
                    <p className="font-mono text-xs mt-1">u₂ = w₂ / ||w₂||</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 bg-indigo-600 text-white rounded-full flex items-center justify-center text-xs font-bold">4</span>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">반복</p>
                    <p className="text-gray-600 dark:text-gray-300">모든 벡터에 대해 과정 반복</p>
                  </div>
                </li>
              </ol>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            QR 분해
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <p className="text-yellow-700 dark:text-yellow-300 mb-2">
                <strong>A = QR</strong>
              </p>
              <ul className="text-sm space-y-1 text-yellow-600 dark:text-yellow-400">
                <li>• Q: 정규직교 행렬 (직교 열벡터)</li>
                <li>• R: 상삼각 행렬</li>
              </ul>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">응용 1: 선형시스템</h4>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  Ax = b → QRx = b → Rx = Q^Tb
                </p>
              </div>
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">응용 2: 고유값</h4>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  QR 알고리즘으로 고유값 계산
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            최소제곱법 (Least Squares)
          </h3>
          <div className="space-y-4">
            <p className="text-gray-600 dark:text-gray-300">
              Ax = b가 해가 없을 때, ||Ax - b||²를 최소화하는 x 찾기
            </p>
            <div className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg p-4">
              <h4 className="font-medium text-green-700 dark:text-green-300 mb-2">정규방정식</h4>
              <p className="font-mono text-sm text-green-600 dark:text-green-400">
                A^TAx = A^Tb
              </p>
              <p className="text-sm text-green-600 dark:text-green-400 mt-2">
                해: x = (A^TA)⁻¹A^Tb
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
            <Code className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            Python 구현
          </h3>
          <pre className="bg-white dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <code className="text-sm text-gray-700 dark:text-gray-300">{`import numpy as np

def gram_schmidt(V):
    """그람-슈미트 과정 구현"""
    n = V.shape[1]
    U = np.zeros_like(V, dtype=float)
    
    for i in range(n):
        # i번째 벡터 가져오기
        u = V[:, i].copy()
        
        # 이전 직교 벡터들에 대한 투영 제거
        for j in range(i):
            u -= np.dot(V[:, i], U[:, j]) * U[:, j]
        
        # 정규화
        U[:, i] = u / np.linalg.norm(u)
    
    return U

# QR 분해
def qr_decomposition(A):
    """QR 분해 구현"""
    Q = gram_schmidt(A)
    R = Q.T @ A
    return Q, R

# 최소제곱법
def least_squares(A, b):
    """최소제곱 해 계산"""
    # 방법 1: 정규방정식
    x1 = np.linalg.inv(A.T @ A) @ A.T @ b
    
    # 방법 2: QR 분해 사용
    Q, R = np.linalg.qr(A)
    x2 = np.linalg.solve(R, Q.T @ b)
    
    return x1, x2

# 예제
A = np.array([[1, 0], [1, 1], [1, 2]])
b = np.array([1, 2, 2])

x1, x2 = least_squares(A, b)
print(f"정규방정식 해: {x1}")
print(f"QR 분해 해: {x2}")
print(f"잔차: {np.linalg.norm(A @ x1 - b):.4f}")`}</code>
          </pre>
        </div>
      </section>
    </div>
  ),

  'svd': (
    <div className="space-y-8">
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          SVD와 차원축소
        </h2>
        
        <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-xl p-6 mb-8">
          <div className="flex items-start gap-3">
            <Layers className="w-6 h-6 text-orange-600 dark:text-orange-400 mt-1" />
            <div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                학습 목표
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-orange-500 mt-0.5" />
                  <span>특이값 분해(SVD)의 원리 이해</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-orange-500 mt-0.5" />
                  <span>저계수 근사와 데이터 압축</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-orange-500 mt-0.5" />
                  <span>추천시스템과 이미지 압축 응용</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            SVD 분해란?
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
              <p className="text-indigo-700 dark:text-indigo-300 font-mono text-lg mb-2">
                A = UΣV^T
              </p>
              <ul className="space-y-2 text-sm text-indigo-600 dark:text-indigo-400">
                <li>• U: 왼쪽 특이벡터 (m×m 직교행렬)</li>
                <li>• Σ: 특이값 대각행렬 (m×n)</li>
                <li>• V^T: 오른쪽 특이벡터 전치 (n×n 직교행렬)</li>
              </ul>
            </div>
            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <p className="text-yellow-700 dark:text-yellow-300">
                <strong>핵심:</strong> 모든 행렬은 SVD 분해 가능 (정사각행렬이 아니어도 OK!)
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            특이값의 의미
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">기하학적 해석</h4>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-gray-400 mt-0.5" />
                  <span>특이값 = 변환의 스케일링 팩터</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-gray-400 mt-0.5" />
                  <span>큰 특이값 = 중요한 정보</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-4 h-4 text-gray-400 mt-0.5" />
                  <span>작은 특이값 = 노이즈나 세부사항</span>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">정보 압축</h4>
              <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg">
                <p className="text-sm text-purple-700 dark:text-purple-300 mb-2">
                  상위 k개 특이값만 사용:
                </p>
                <p className="font-mono text-xs text-purple-600 dark:text-purple-400">
                  A_k = U_k Σ_k V_k^T
                </p>
                <p className="text-xs text-purple-600 dark:text-purple-400 mt-2">
                  압축률: k(m+n+1) / (mn)
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            SVD 응용 분야
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg">
              <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-2">이미지 압축</h4>
              <p className="text-sm text-blue-600 dark:text-blue-400">
                JPEG 압축, 노이즈 제거, 특징 추출
              </p>
            </div>
            <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
              <h4 className="font-medium text-green-700 dark:text-green-300 mb-2">추천 시스템</h4>
              <p className="text-sm text-green-600 dark:text-green-400">
                Netflix, Amazon의 협업 필터링
              </p>
            </div>
            <div className="p-4 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg">
              <h4 className="font-medium text-orange-700 dark:text-orange-300 mb-2">NLP</h4>
              <p className="text-sm text-orange-600 dark:text-orange-400">
                LSA(잠재 의미 분석), 토픽 모델링
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            Truncated SVD (차원 축소)
          </h3>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="flex-1">
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-red-500 via-orange-500 to-yellow-500" style={{width: '100%'}}></div>
                </div>
                <p className="text-xs text-gray-500 mt-1">전체 특이값</p>
              </div>
              <ArrowRight className="w-5 h-5 text-gray-400" />
              <div className="flex-1">
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-red-500 to-orange-500" style={{width: '30%'}}></div>
                </div>
                <p className="text-xs text-gray-500 mt-1">상위 k개만 선택</p>
              </div>
            </div>
            <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-300">
                <strong>에너지 보존:</strong> 상위 k개 특이값이 전체 정보의 90% 이상 포함
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
            <Code className="w-5 h-5 text-red-600 dark:text-red-400" />
            SVD 실습 코드
          </h3>
          <pre className="bg-white dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <code className="text-sm text-gray-700 dark:text-gray-300">{`import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# SVD 분해
def perform_svd(A):
    """행렬 A의 SVD 분해"""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U, s, Vt

# 저계수 근사
def low_rank_approx(A, k):
    """상위 k개 특이값만 사용한 근사"""
    U, s, Vt = perform_svd(A)
    
    # k개만 선택
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    # 재구성
    A_k = U_k @ np.diag(s_k) @ Vt_k
    
    # 압축률 계산
    original_size = A.shape[0] * A.shape[1]
    compressed_size = k * (A.shape[0] + A.shape[1] + 1)
    compression_ratio = compressed_size / original_size
    
    return A_k, compression_ratio

# 이미지 압축 예제
def compress_image(image_path, k):
    """이미지 SVD 압축"""
    # 이미지 로드
    img = Image.open(image_path).convert('L')  # 그레이스케일
    img_array = np.array(img)
    
    # SVD 압축
    compressed, ratio = low_rank_approx(img_array, k)
    
    # 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title(f'Original ({img_array.shape[0]}×{img_array.shape[1]})')
    axes[0].axis('off')
    
    axes[1].imshow(compressed, cmap='gray')
    axes[1].set_title(f'Compressed (k={k}, ratio={ratio:.1%})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return compressed

# 추천 시스템 예제
def recommendation_svd(ratings_matrix, k=10):
    """SVD 기반 추천 시스템"""
    # 평균 중심화
    mean_ratings = np.mean(ratings_matrix, axis=1, keepdims=True)
    centered = ratings_matrix - mean_ratings
    
    # SVD 분해
    U, s, Vt = perform_svd(centered)
    
    # k차원으로 축소
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    # 예측 행렬 생성
    predictions = U_k @ np.diag(s_k) @ Vt_k + mean_ratings
    
    return predictions

# 특이값 스펙트럼 분석
def analyze_singular_values(A):
    """특이값 분포 분석"""
    _, s, _ = perform_svd(A)
    
    # 누적 에너지
    energy = np.cumsum(s**2) / np.sum(s**2)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(s, 'o-')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Value Spectrum')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(energy, 'o-')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% energy')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Energy')
    plt.title('Energy Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 90% 에너지를 위한 컴포넌트 수
    k_90 = np.argmax(energy >= 0.9) + 1
    print(f"90% 에너지 보존을 위한 컴포넌트 수: {k_90}/{len(s)}")

# 예제 실행
A = np.random.randn(100, 50)
analyze_singular_values(A)`}</code>
          </pre>
        </div>
      </section>
    </div>
  ),

  'linear-systems': (
    <div className="space-y-8">
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          선형시스템
        </h2>
        
        <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-8">
          <div className="flex items-start gap-3">
            <Binary className="w-6 h-6 text-violet-600 dark:text-violet-400 mt-1" />
            <div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                학습 목표
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-violet-500 mt-0.5" />
                  <span>연립방정식의 다양한 해법 마스터</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-violet-500 mt-0.5" />
                  <span>LU 분해와 효율적인 계산</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-violet-500 mt-0.5" />
                  <span>반복법과 수치적 안정성</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            선형시스템 Ax = b
          </h3>
          <div className="space-y-4">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                <h4 className="font-medium text-green-700 dark:text-green-300 mb-2">유일해</h4>
                <p className="text-sm text-green-600 dark:text-green-400">
                  det(A) ≠ 0<br/>
                  rank(A) = n
                </p>
              </div>
              <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                <h4 className="font-medium text-yellow-700 dark:text-yellow-300 mb-2">무수히 많은 해</h4>
                <p className="text-sm text-yellow-600 dark:text-yellow-400">
                  rank(A) = rank([A|b]) &lt; n
                </p>
              </div>
              <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                <h4 className="font-medium text-red-700 dark:text-red-300 mb-2">해 없음</h4>
                <p className="text-sm text-red-600 dark:text-red-400">
                  rank(A) &lt; rank([A|b])
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            가우스 소거법
          </h3>
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-violet-600 text-white rounded-full flex items-center justify-center font-bold">
                1
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">전진 소거</h4>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  상삼각 행렬로 변환
                </p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-violet-600 text-white rounded-full flex items-center justify-center font-bold">
                2
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">후진 대입</h4>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  아래에서 위로 해 계산
                </p>
              </div>
            </div>
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm text-blue-700 dark:text-blue-300">
                <strong>계산 복잡도:</strong> O(n³)
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            LU 분해
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
              <p className="font-mono text-indigo-700 dark:text-indigo-300 mb-2">
                PA = LU
              </p>
              <ul className="text-sm space-y-1 text-indigo-600 dark:text-indigo-400">
                <li>• P: 순열 행렬 (피벗팅)</li>
                <li>• L: 하삼각 행렬</li>
                <li>• U: 상삼각 행렬</li>
              </ul>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2 text-gray-900 dark:text-white">장점</h4>
                <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-300">
                  <li>• 여러 b에 대해 효율적</li>
                  <li>• 행렬식 계산 용이</li>
                  <li>• 역행렬 계산 효율적</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium mb-2 text-gray-900 dark:text-white">해법 과정</h4>
                <ol className="text-sm space-y-1 text-gray-600 dark:text-gray-300">
                  <li>1. Ly = Pb (전진 대입)</li>
                  <li>2. Ux = y (후진 대입)</li>
                </ol>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            반복법
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
              <h4 className="font-medium text-green-700 dark:text-green-300 mb-2">야코비 방법</h4>
              <p className="text-xs font-mono text-green-600 dark:text-green-400">
                x_i^(k+1) = (b_i - Σ a_ij x_j^(k)) / a_ii
              </p>
            </div>
            <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg">
              <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-2">가우스-자이델</h4>
              <p className="text-xs text-blue-600 dark:text-blue-400">
                새 값을 즉시 사용<br/>
                더 빠른 수렴
              </p>
            </div>
            <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg">
              <h4 className="font-medium text-purple-700 dark:text-purple-300 mb-2">SOR 방법</h4>
              <p className="text-xs text-purple-600 dark:text-purple-400">
                과이완 계수 ω 사용<br/>
                최적 수렴
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
            <Code className="w-5 h-5 text-violet-600 dark:text-violet-400" />
            선형시스템 솔버 구현
          </h3>
          <pre className="bg-white dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <code className="text-sm text-gray-700 dark:text-gray-300">{`import numpy as np
from scipy.linalg import lu, solve_triangular

# 가우스 소거법
def gaussian_elimination(A, b):
    """가우스 소거법으로 Ax = b 해결"""
    n = len(A)
    Ab = np.column_stack([A.copy(), b.copy()])
    
    # 전진 소거
    for i in range(n):
        # 피벗팅
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # 소거
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]
    
    # 후진 대입
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]
    
    return x

# LU 분해 솔버
def lu_solver(A, b):
    """LU 분해를 이용한 선형시스템 해결"""
    P, L, U = lu(A)
    
    # Ly = Pb
    y = solve_triangular(L, P @ b, lower=True)
    
    # Ux = y
    x = solve_triangular(U, y)
    
    return x

# 야코비 반복법
def jacobi_method(A, b, x0=None, tol=1e-10, max_iter=1000):
    """야코비 반복법"""
    n = len(A)
    x = x0 if x0 is not None else np.zeros(n)
    
    for iteration in range(max_iter):
        x_new = np.zeros(n)
        
        for i in range(n):
            sum_ax = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_ax) / A[i, i]
        
        # 수렴 확인
        if np.linalg.norm(x_new - x) < tol:
            print(f"수렴 (반복: {iteration+1})")
            return x_new
        
        x = x_new
    
    print("최대 반복 횟수 도달")
    return x

# 가우스-자이델 방법
def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    """가우스-자이델 반복법"""
    n = len(A)
    x = x0 if x0 is not None else np.zeros(n)
    
    for iteration in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sum1 = sum(A[i, j] * x[j] for j in range(i))
            sum2 = sum(A[i, j] * x_old[j] for j in range(i+1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        # 수렴 확인
        if np.linalg.norm(x - x_old) < tol:
            print(f"수렴 (반복: {iteration+1})")
            return x
    
    print("최대 반복 횟수 도달")
    return x

# 조건수 분석
def condition_analysis(A):
    """행렬의 조건수 분석"""
    cond = np.linalg.cond(A)
    
    print(f"조건수: {cond:.2e}")
    
    if cond < 10:
        print("매우 안정적")
    elif cond < 100:
        print("안정적")
    elif cond < 1000:
        print("주의 필요")
    else:
        print("불안정 (ill-conditioned)")
    
    return cond

# 예제
A = np.array([[10, -1, 2], 
              [-1, 11, -1], 
              [2, -1, 10]])
b = np.array([6, 25, -11])

print("가우스 소거법:", gaussian_elimination(A, b))
print("LU 분해:", lu_solver(A, b))
print("야코비 방법:", jacobi_method(A, b))
print("가우스-자이델:", gauss_seidel(A, b))
print("\\n조건수 분석:")
condition_analysis(A)`}</code>
          </pre>
        </div>
      </section>
    </div>
  ),

  'ml-applications': (
    <div className="space-y-8">
      <section>
        <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
          AI/ML 응용
        </h2>
        
        <div className="bg-gradient-to-r from-pink-50 to-purple-50 dark:from-pink-900/20 dark:to-purple-900/20 rounded-xl p-6 mb-8">
          <div className="flex items-start gap-3">
            <Sparkles className="w-6 h-6 text-pink-600 dark:text-pink-400 mt-1" />
            <div>
              <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
                학습 목표
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-pink-500 mt-0.5" />
                  <span>신경망에서의 행렬 연산 이해</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-pink-500 mt-0.5" />
                  <span>역전파 알고리즘의 수학적 기초</span>
                </li>
                <li className="flex items-start gap-2">
                  <ChevronRight className="w-5 h-5 text-pink-500 mt-0.5" />
                  <span>최적화와 텐서 연산</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            신경망과 행렬 연산
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg">
              <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-2">순전파 (Forward Pass)</h4>
              <p className="font-mono text-sm text-blue-600 dark:text-blue-400">
                z = Wx + b<br/>
                a = σ(z)
              </p>
              <p className="text-xs text-blue-600 dark:text-blue-400 mt-2">
                W: 가중치 행렬, x: 입력, b: 편향, σ: 활성화 함수
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">배치 처리</h4>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  X: (배치크기 × 특징수)<br/>
                  병렬 처리로 효율성 극대화
                </p>
              </div>
              <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">GPU 최적화</h4>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  행렬 곱셈의 병렬화<br/>
                  CUDA/cuBLAS 활용
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            역전파와 경사하강법
          </h3>
          <div className="space-y-4">
            <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
              <h4 className="font-medium text-green-700 dark:text-green-300 mb-2">체인 룰</h4>
              <p className="font-mono text-sm text-green-600 dark:text-green-400">
                ∂L/∂W = ∂L/∂z · ∂z/∂W = δ · x^T
              </p>
            </div>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded">
                <h5 className="text-sm font-medium text-red-700 dark:text-red-300">SGD</h5>
                <p className="text-xs text-red-600 dark:text-red-400">
                  W = W - η∇W
                </p>
              </div>
              <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                <h5 className="text-sm font-medium text-blue-700 dark:text-blue-300">Adam</h5>
                <p className="text-xs text-blue-600 dark:text-blue-400">
                  적응적 학습률
                </p>
              </div>
              <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                <h5 className="text-sm font-medium text-purple-700 dark:text-purple-300">RMSprop</h5>
                <p className="text-xs text-purple-600 dark:text-purple-400">
                  기울기 제곱 평균
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            주요 ML 알고리즘의 선형대수
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">PCA (주성분 분석)</h4>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                <li>• 공분산 행렬 계산</li>
                <li>• 고유값 분해</li>
                <li>• 차원 축소</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">SVM (서포트 벡터 머신)</h4>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                <li>• 커널 트릭 (내적 계산)</li>
                <li>• 이차 계획법</li>
                <li>• 라그랑주 승수</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">선형 회귀</h4>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                <li>• 정규방정식: (X^TX)^(-1)X^Ty</li>
                <li>• QR 분해 활용</li>
                <li>• Ridge/Lasso 정규화</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-3 text-gray-900 dark:text-white">Word2Vec</h4>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                <li>• 단어 임베딩 행렬</li>
                <li>• 내적을 통한 유사도</li>
                <li>• SVD 기반 차원 축소</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
            텐서와 딥러닝
          </h3>
          <div className="space-y-4">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="p-4 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg">
                <h4 className="font-medium text-orange-700 dark:text-orange-300 mb-2">CNN</h4>
                <p className="text-sm text-orange-600 dark:text-orange-400">
                  4D 텐서: (N, C, H, W)<br/>
                  컨볼루션 = 행렬 곱셈
                </p>
              </div>
              <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg">
                <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-2">RNN/LSTM</h4>
                <p className="text-sm text-blue-600 dark:text-blue-400">
                  3D 텐서: (배치, 시퀀스, 특징)<br/>
                  게이트 = 행렬 연산
                </p>
              </div>
              <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg">
                <h4 className="font-medium text-purple-700 dark:text-purple-300 mb-2">Transformer</h4>
                <p className="text-sm text-purple-600 dark:text-purple-400">
                  어텐션 = Q·K^T / √d<br/>
                  다중 헤드 어텐션
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-br from-pink-50 to-purple-50 dark:from-pink-900/20 dark:to-purple-900/20 rounded-xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-900 dark:text-white">
            <Code className="w-5 h-5 text-pink-600 dark:text-pink-400" />
            딥러닝 구현 예제
          </h3>
          <pre className="bg-white dark:bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <code className="text-sm text-gray-700 dark:text-gray-300">{`import numpy as np

class NeuralNetwork:
    """간단한 신경망 구현"""
    
    def __init__(self, layers):
        """
        layers: [입력층, 은닉층1, ..., 출력층]
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # 가중치 초기화 (Xavier)
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU 활성화 함수"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU 도함수"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax 활성화 함수"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """순전파"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            # 마지막 층은 softmax, 나머지는 ReLU
            if i == len(self.weights) - 1:
                a = self.softmax(z)
            else:
                a = self.relu(z)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.01):
        """역전파 및 가중치 업데이트"""
        m = X.shape[0]
        
        # 출력층 그래디언트
        delta = self.activations[-1] - y
        
        # 역전파
        for i in range(len(self.weights) - 1, -1, -1):
            # 가중치 그래디언트
            dW = (self.activations[i].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # 가중치 업데이트
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # 다음 층을 위한 델타 계산
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
    
    def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=32):
        """미니배치 SGD 훈련"""
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # 데이터 셔플
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # 미니배치 훈련
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 순전파
                output = self.forward(X_batch)
                
                # 역전파
                self.backward(X_batch, y_batch, learning_rate)
            
            # 손실 계산 (크로스 엔트로피)
            if epoch % 100 == 0:
                output = self.forward(X)
                loss = -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# PCA 구현
def pca(X, n_components):
    """주성분 분석"""
    # 1. 데이터 중심화
    X_centered = X - np.mean(X, axis=0)
    
    # 2. 공분산 행렬
    cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    
    # 3. 고유값 분해
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 4. 고유값 기준 정렬
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. 상위 n개 주성분 선택
    components = eigenvectors[:, :n_components]
    
    # 6. 변환
    X_transformed = X_centered @ components
    
    # 설명된 분산 비율
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_transformed, components, explained_variance_ratio

# 사용 예제
if __name__ == "__main__":
    # 신경망 테스트
    nn = NeuralNetwork([784, 128, 64, 10])  # MNIST용
    
    # 더미 데이터
    X = np.random.randn(100, 784)
    y = np.eye(10)[np.random.randint(0, 10, 100)]  # one-hot
    
    nn.train(X, y, epochs=100, learning_rate=0.01)
    
    # PCA 테스트
    X_pca, components, var_ratio = pca(X, n_components=50)
    print(f"\\nPCA: {var_ratio.sum():.2%} 분산 설명")`}</code>
          </pre>
        </div>
      </section>
    </div>
  )
}

export default function ChapterContent({ chapterId }: { chapterId: string }) {
  const content = chaptersContent[chapterId]
  
  if (!content) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-12">
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-8 text-center">
          <h2 className="text-2xl font-bold text-yellow-700 dark:text-yellow-300 mb-4">
            챕터를 찾을 수 없습니다
          </h2>
          <p className="text-yellow-600 dark:text-yellow-400">
            요청하신 챕터 &apos;{chapterId}&apos;는 존재하지 않습니다.
          </p>
        </div>
      </div>
    )
  }
  
  return <>{content}</>
}