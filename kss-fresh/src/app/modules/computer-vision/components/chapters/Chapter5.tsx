'use client';

import { useState } from 'react';
import {
  Lightbulb,
  PlayCircle,
  Terminal,
  Copy,
  Check,
  CheckCircle,
  Zap
} from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter5() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">스테레오 비전의 원리</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          스테레오 비전은 두 개 이상의 카메라를 사용하여 깊이 정보를 추출하는 기술입니다.
          인간의 양안 시차와 같은 원리로, 두 이미지에서 동일 점의 위치 차이(disparity)를 이용해 3D 정보를 복원합니다.
        </p>

        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-3">
            <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
            <div>
              <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">깊이 계산 수식</h3>
              <div className="space-y-3 text-blue-800 dark:text-blue-200">
                <p><strong>disparity (d)</strong>: 왼쪽 이미지와 오른쪽 이미지에서 동일 점의 x 좌표 차이</p>
                <p className="font-mono bg-white dark:bg-gray-800 p-3 rounded">
                  depth (Z) = (focal_length × baseline) / disparity
                </p>
                <ul className="space-y-2 text-sm">
                  <li>• <strong>focal_length</strong>: 카메라의 초점 거리 (픽셀 단위)</li>
                  <li>• <strong>baseline</strong>: 두 카메라 사이의 거리</li>
                  <li>• <strong>disparity</strong>: 디스패리티 (클수록 가까운 물체)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4">스테레오 비전 기하학 (Epipolar Geometry)</h3>
          <svg viewBox="0 0 800 500" className="w-full h-auto">
            {/* 왼쪽 카메라 */}
            <g transform="translate(100, 250)">
              <rect x="-20" y="-15" width="40" height="30" className="fill-blue-500" opacity="0.8" />
              <polygon points="20,0 50,20 50,-20" className="fill-blue-500" opacity="0.6" />
              <text x="0" y="40" textAnchor="middle" className="fill-blue-600 dark:fill-blue-400 text-sm font-semibold">
                왼쪽 카메라 (C_L)
              </text>
            </g>

            {/* 오른쪽 카메라 */}
            <g transform="translate(300, 250)">
              <rect x="-20" y="-15" width="40" height="30" className="fill-green-500" opacity="0.8" />
              <polygon points="20,0 50,20 50,-20" className="fill-green-500" opacity="0.6" />
              <text x="0" y="40" textAnchor="middle" className="fill-green-600 dark:fill-green-400 text-sm font-semibold">
                오른쪽 카메라 (C_R)
              </text>
            </g>

            {/* Baseline */}
            <line x1="100" y1="250" x2="300" y2="250" className="stroke-purple-500" strokeWidth="3" />
            <text x="200" y="240" textAnchor="middle" className="fill-purple-600 dark:fill-purple-400 text-sm font-semibold">
              Baseline (b)
            </text>

            {/* 3D 점 */}
            <circle cx="500" cy="150" r="10" className="fill-red-500" />
            <text x="500" y="130" textAnchor="middle" className="fill-red-600 dark:fill-red-400 text-sm font-semibold">
              3D 점 P (X, Y, Z)
            </text>

            {/* 투영선 - 왼쪽 카메라에서 */}
            <line x1="100" y1="250" x2="500" y2="150" className="stroke-blue-400" strokeWidth="2" strokeDasharray="5,5" />
            <text x="250" y="190" className="fill-blue-600 dark:fill-blue-400 text-xs">왼쪽 투영선</text>

            {/* 투영선 - 오른쪽 카메라에서 */}
            <line x1="300" y1="250" x2="500" y2="150" className="stroke-green-400" strokeWidth="2" strokeDasharray="5,5" />
            <text x="360" y="190" className="fill-green-600 dark:fill-green-400 text-xs">오른쪽 투영선</text>

            {/* 왼쪽 이미지 평면 */}
            <line x1="120" y1="180" x2="120" y2="320" className="stroke-blue-600" strokeWidth="3" />
            <circle cx="120" cy="220" r="5" className="fill-blue-600" />
            <text x="140" y="225" className="fill-blue-600 dark:fill-blue-400 text-xs">P_L (x_L, y)</text>

            {/* 오른쪽 이미지 평면 */}
            <line x1="320" y1="180" x2="320" y2="320" className="stroke-green-600" strokeWidth="3" />
            <circle cx="320" cy="235" r="5" className="fill-green-600" />
            <text x="340" y="240" className="fill-green-600 dark:fill-green-400 text-xs">P_R (x_R, y)</text>

            {/* Disparity 표시 */}
            <path d="M 120 210 Q 220 200 320 225" className="fill-none stroke-orange-500" strokeWidth="2" markerEnd="url(#arroworange)" />
            <defs>
              <marker id="arroworange" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#f97316" />
              </marker>
            </defs>
            <text x="220" y="190" textAnchor="middle" className="fill-orange-600 dark:fill-orange-400 text-sm font-semibold">
              Disparity (d = x_L - x_R)
            </text>

            {/* 깊이 Z */}
            <line x1="500" y1="150" x2="500" y2="400" className="stroke-gray-500" strokeWidth="2" strokeDasharray="5,5" />
            <text x="510" y="270" className="fill-gray-600 dark:fill-gray-400 text-sm font-semibold">
              Depth (Z)
            </text>

            {/* 수식 박스 */}
            <g transform="translate(520, 300)">
              <rect x="0" y="0" width="260" height="120" className="fill-yellow-50 dark:fill-yellow-900/20 stroke-yellow-500" strokeWidth="2" rx="8" />
              <text x="130" y="25" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm font-bold">
                깊이 계산 공식
              </text>
              <text x="130" y="55" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-mono">
                Z = (f × b) / d
              </text>
              <text x="10" y="80" className="fill-gray-600 dark:fill-gray-400 text-xs">
                • f: 초점거리
              </text>
              <text x="10" y="95" className="fill-gray-600 dark:fill-gray-400 text-xs">
                • b: Baseline
              </text>
              <text x="10" y="110" className="fill-gray-600 dark:fill-gray-400 text-xs">
                • d: Disparity (x_L - x_R)
              </text>
            </g>

            {/* 제목 */}
            <text x="400" y="30" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-lg font-bold">
              스테레오 비전 원리
            </text>
            <text x="400" y="50" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
              동일한 3D 점이 두 이미지에서 다른 위치에 투영됨 → Disparity → Depth
            </text>
          </svg>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">1. 카메라 캘리브레이션</h3>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              내부/외부 파라미터 추정 - 체스보드 패턴 활용
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">2. 스테레오 매칭</h3>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              대응점 찾기 - 블록 매칭, 세미-글로벌 매칭(SGM)
            </p>
          </div>
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">3. 3D 복원</h3>
            <p className="text-sm text-gray-600 dark:text-gray-300">
              디스패리티 → 깊이 → Point Cloud 생성
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">OpenCV 스테레오 매칭</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          OpenCV는 StereoBM (Block Matching)과 StereoSGBM (Semi-Global Block Matching) 알고리즘을 제공합니다.
          SGBM이 더 정확하지만 계산량이 많습니다.
        </p>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - SGBM 디스패리티 맵 생성</span>
            </div>
            <button
              onClick={() => copyToClipboard(`import cv2
import numpy as np

# 스테레오 이미지 쌍 읽기
img_left = cv2.imread('left.jpg', 0)   # 왼쪽 이미지
img_right = cv2.imread('right.jpg', 0)  # 오른쪽 이미지

# StereoSGBM 매처 생성
window_size = 5
min_disp = 0
num_disp = 16 * 6  # 96 (16의 배수여야 함)

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,    # 작은 디스패리티 변화 페널티
    P2=32 * 3 * window_size ** 2,   # 큰 디스패리티 변화 페널티
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# 디스패리티 맵 계산
disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

# 시각화를 위한 정규화
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# 컬러 맵 적용
disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

cv2.imshow('Disparity Map', disparity_color)
cv2.waitKey(0)`, 'stereo-1')}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {copiedCode === 'stereo-1' ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
            <code>{`import cv2
import numpy as np

# 스테레오 이미지 쌍 읽기
img_left = cv2.imread('left.jpg', 0)   # 왼쪽 이미지
img_right = cv2.imread('right.jpg', 0)  # 오른쪽 이미지

# StereoSGBM 매처 생성
window_size = 5
min_disp = 0
num_disp = 16 * 6  # 96 (16의 배수여야 함)

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,    # 작은 디스패리티 변화 페널티
    P2=32 * 3 * window_size ** 2,   # 큰 디스패리티 변화 페널티
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# 디스패리티 맵 계산
disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

# 시각화를 위한 정규화
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# 컬러 맵 적용
disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

cv2.imshow('Disparity Map', disparity_color)
cv2.waitKey(0)`}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">3D Point Cloud 생성</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          디스패리티 맵을 이용하여 3D 포인트 클라우드를 생성할 수 있습니다.
          각 픽셀의 (x, y) 좌표와 깊이(Z)를 결합하여 3D 공간상의 점으로 변환합니다.
        </p>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - Point Cloud 생성</span>
            </div>
            <button
              onClick={() => copyToClipboard(`import cv2
import numpy as np

# 디스패리티 맵 (위에서 계산)
# ... disparity 얻기 ...

# 카메라 매트릭스 (캘리브레이션으로 얻음)
focal_length = 700  # 픽셀 단위
baseline = 0.1      # 미터 단위 (10cm)

# Q 행렬 생성 (Reprojection Matrix)
Q = np.float32([[1, 0, 0, -img_left.shape[1]/2],
                [0, 1, 0, -img_left.shape[0]/2],
                [0, 0, 0, focal_length],
                [0, 0, 1/baseline, 0]])

# 3D 좌표 계산
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# 유효한 점만 필터링 (디스패리티 > 0)
mask = disparity > disparity.min()
points_3D = points_3D[mask]
colors = img_left[mask]  # 왼쪽 이미지의 색상 정보

# PLY 파일로 저장
def write_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write('ply\\n')
        f.write('format ascii 1.0\\n')
        f.write(f'element vertex {len(points)}\\n')
        f.write('property float x\\n')
        f.write('property float y\\n')
        f.write('property float z\\n')
        f.write('property uchar red\\n')
        f.write('property uchar green\\n')
        f.write('property uchar blue\\n')
        f.write('end_header\\n')
        for p, c in zip(points, colors):
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c} {c} {c}\\n')

write_ply('output.ply', points_3D, colors)
print("Point Cloud saved to output.ply")`, 'pointcloud-1')}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {copiedCode === 'pointcloud-1' ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
            <code>{`import cv2
import numpy as np

# 디스패리티 맵 (위에서 계산)
# ... disparity 얻기 ...

# 카메라 매트릭스 (캘리브레이션으로 얻음)
focal_length = 700  # 픽셀 단위
baseline = 0.1      # 미터 단위 (10cm)

# Q 행렬 생성 (Reprojection Matrix)
Q = np.float32([[1, 0, 0, -img_left.shape[1]/2],
                [0, 1, 0, -img_left.shape[0]/2],
                [0, 0, 0, focal_length],
                [0, 0, 1/baseline, 0]])

# 3D 좌표 계산
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# 유효한 점만 필터링 (디스패리티 > 0)
mask = disparity > disparity.min()
points_3D = points_3D[mask]
colors = img_left[mask]  # 왼쪽 이미지의 색상 정보

# PLY 파일로 저장
def write_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write('ply\\n')
        f.write('format ascii 1.0\\n')
        f.write(f'element vertex {len(points)}\\n')
        f.write('property float x\\n')
        f.write('property float y\\n')
        f.write('property float z\\n')
        f.write('property uchar red\\n')
        f.write('property uchar green\\n')
        f.write('property uchar blue\\n')
        f.write('end_header\\n')
        for p, c in zip(points, colors):
            f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c} {c} {c}\\n')

write_ply('output.ply', points_3D, colors)
print("Point Cloud saved to output.ply")`}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">단일 이미지 깊이 추정 - MiDaS</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          최신 딥러닝 기술을 활용하면 단일 이미지에서도 깊이 정보를 추정할 수 있습니다.
          MiDaS는 다양한 데이터셋으로 학습되어 제로샷 일반화 성능이 뛰어납니다.
        </p>

        <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-3">
            <Zap className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5" />
            <div>
              <h3 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">MiDaS 특징</h3>
              <ul className="space-y-2 text-purple-800 dark:text-purple-200 text-sm">
                <li>• <strong>제로샷 일반화</strong>: 어떤 이미지에도 적용 가능</li>
                <li>• <strong>다양한 모델</strong>: Small, Large, ViT 기반 버전</li>
                <li>• <strong>실시간 처리</strong>: Small 모델은 모바일에서도 가능</li>
                <li>• <strong>상대적 깊이</strong>: 절대 거리가 아닌 상대적 깊이 출력</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - MiDaS 깊이 추정</span>
            </div>
            <button
              onClick={() => copyToClipboard(`import cv2
import torch
import numpy as np
from torchvision.transforms import Compose

# MiDaS 모델 로드
model_type = "DPT_Large"  # 또는 "MiDaS_small", "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Transform 준비
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# 이미지 읽기
img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 전처리 및 추론
input_batch = transform(img_rgb).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

# 시각화
output_normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
output_normalized = np.uint8(output_normalized)
depth_color = cv2.applyColorMap(output_normalized, cv2.COLORMAP_MAGMA)

cv2.imshow('MiDaS Depth', depth_color)
cv2.waitKey(0)`, 'midas-1')}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {copiedCode === 'midas-1' ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
            <code>{`import cv2
import torch
import numpy as np
from torchvision.transforms import Compose

# MiDaS 모델 로드
model_type = "DPT_Large"  # 또는 "MiDaS_small", "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Transform 준비
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# 이미지 읽기
img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 전처리 및 추론
input_batch = transform(img_rgb).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

# 시각화
output_normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
output_normalized = np.uint8(output_normalized)
depth_color = cv2.applyColorMap(output_normalized, cv2.COLORMAP_MAGMA)

cv2.imshow('MiDaS Depth', depth_color)
cv2.waitKey(0)`}</code>
          </pre>
        </div>

        <div className="flex items-center gap-3 p-4 bg-teal-50 dark:bg-teal-900/20 rounded-lg mb-6">
          <PlayCircle className="w-6 h-6 text-teal-600 dark:text-teal-400" />
          <p className="text-teal-800 dark:text-teal-200">
            2D to 3D Converter 시뮬레이터에서 실시간으로 깊이 추정을 체험해보세요!
          </p>
        </div>
      </section>

      <section className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
        <div className="flex items-start gap-3">
          <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
          <div>
            <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">실습 과제</h3>
            <ul className="space-y-2 text-green-800 dark:text-green-200">
              <li>1. 스테레오 이미지 쌍에서 디스패리티 맵을 생성하고 저장해보세요</li>
              <li>2. numDisparities, blockSize 파라미터를 조정하며 결과 변화를 관찰하세요</li>
              <li>3. MiDaS로 단일 이미지 깊이 추정을 수행하고 결과를 비교해보세요</li>
              <li>4. Point Cloud를 생성하고 MeshLab 등의 도구로 3D 시각화 해보세요</li>
              <li>5. 깊이 맵을 이용하여 배경 흐림(Bokeh) 효과를 구현해보세요</li>
            </ul>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Stereo Vision & Multi-View Geometry',
            icon: 'paper' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'Multiple View Geometry in Computer Vision',
                authors: 'Richard Hartley, Andrew Zisserman',
                year: '2004',
                description: '스테레오 비전의 바이블 - Epipolar Geometry, Triangulation',
                link: 'https://www.robots.ox.ac.uk/~vgg/hzbook/'
              },
              {
                title: 'Computing Rectifying Homographies for Stereo Vision',
                authors: 'Charles Loop, Zhengyou Zhang',
                year: '1999',
                description: 'Stereo Rectification - 에피폴라 선을 수평으로 정렬',
                link: 'https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-99-21.pdf'
              },
              {
                title: 'A Taxonomy of Stereo Matching',
                authors: 'Daniel Scharstein, Richard Szeliski',
                year: '2002',
                description: 'Stereo Matching 알고리즘 분류 - Middlebury 데이터셋',
                link: 'https://link.springer.com/article/10.1023/A:1014573219977'
              }
            ]
          },
          {
            title: 'Single Image Depth Estimation',
            icon: 'paper' as const,
            color: 'border-cyan-500',
            items: [
              {
                title: 'Towards Robust Monocular Depth Estimation (MiDaS)',
                authors: 'Rene Ranftl, et al.',
                year: '2020',
                description: 'MiDaS - 단일 이미지 깊이 추정의 표준, 제로샷 일반화',
                link: 'https://arxiv.org/abs/1907.01341'
              },
              {
                title: 'Vision Transformers for Dense Prediction (DPT)',
                authors: 'Rene Ranftl, et al.',
                year: '2021',
                description: 'ViT 기반 깊이 추정 - MiDaS v3.0의 핵심 기술',
                link: 'https://arxiv.org/abs/2103.13413'
              },
              {
                title: 'Depth Map Prediction from Single Image (Eigen et al.)',
                authors: 'David Eigen, et al.',
                year: '2014',
                description: 'CNN으로 깊이 추정 - 단일 이미지 깊이의 시작',
                link: 'https://arxiv.org/abs/1406.2283'
              }
            ]
          },
          {
            title: '3D Reconstruction',
            icon: 'paper' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Structure from Motion (SfM)',
                authors: 'Noah Snavely, et al.',
                year: '2006',
                description: 'Photo Tourism - 인터넷 사진으로 3D 복원',
                link: 'https://www.cs.cornell.edu/~snavely/bundler/'
              },
              {
                title: 'COLMAP: Structure-from-Motion Revisited',
                authors: 'Johannes L. Schönberger, Jan-Michael Frahm',
                year: '2016',
                description: 'SfM/MVS의 표준 도구 - 오픈소스 3D 재구성',
                link: 'https://colmap.github.io/'
              },
              {
                title: 'NeRF: Neural Radiance Fields',
                authors: 'Ben Mildenhall, et al.',
                year: '2020',
                description: '뷰 합성의 혁명 - 암시적 3D 표현 학습',
                link: 'https://arxiv.org/abs/2003.08934'
              }
            ]
          },
          {
            title: 'SLAM & Real-time 3D',
            icon: 'paper' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'ORB-SLAM: Real-Time SLAM',
                authors: 'Raul Mur-Artal, et al.',
                year: '2015',
                description: 'ORB 특징점 기반 실시간 SLAM - 로봇/AR 핵심 기술',
                link: 'https://arxiv.org/abs/1502.00956'
              },
              {
                title: 'Instant Neural Graphics Primitives (Instant-NGP)',
                authors: 'Thomas Müller, et al.',
                year: '2022',
                description: 'NeRF 훈련 10초로 단축 - 해시 인코딩의 마법',
                link: 'https://nvlabs.github.io/instant-ngp/'
              }
            ]
          },
          {
            title: 'Tools & Datasets',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'MiDaS v3.1',
                authors: 'Intel ISL',
                year: '2024',
                description: '단일 이미지 깊이 추정 - PyTorch, ONNX, TFLite',
                link: 'https://github.com/isl-org/MiDaS'
              },
              {
                title: 'Depth Anything',
                authors: 'TikTok',
                year: '2024',
                description: '1.5B 이미지로 학습한 깊이 추정 - SOTA 성능',
                link: 'https://github.com/LiheYoung/Depth-Anything'
              },
              {
                title: 'OpenCV Stereo Vision',
                authors: 'OpenCV Team',
                year: '2024',
                description: 'Stereo Calibration, Disparity Map, 3D 재구성',
                link: 'https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html'
              },
              {
                title: 'KITTI Stereo Dataset',
                authors: 'Karlsruhe Institute',
                year: '2024',
                description: '자율주행 스테레오 벤치마크 - 200개 이미지 쌍',
                link: 'http://www.cvlibs.net/datasets/kitti/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}