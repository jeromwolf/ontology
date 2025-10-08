'use client';

import { useState } from 'react';
import {
  Terminal,
  Copy,
  Check
} from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter4() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">CNN 아키텍처의 진화</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Convolutional Neural Networks(CNN)는 컴퓨터 비전 분야에 혁명을 일으켰습니다.
          LeNet부터 최신 EfficientNet까지, CNN 아키텍처는 계속 발전하고 있습니다.
        </p>

        <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4 text-teal-900 dark:text-teal-100">주요 CNN 아키텍처</h3>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
              <span className="font-medium">AlexNet (2012)</span> - ImageNet 대회 우승, 딥러닝 시대 개막
            </div>
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
              <span className="font-medium">VGGNet (2014)</span> - 단순하고 깊은 구조의 효과 증명
            </div>
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
              <span className="font-medium">ResNet (2015)</span> - Skip connection으로 매우 깊은 네트워크 학습
            </div>
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-teal-500 rounded-full"></div>
              <span className="font-medium">EfficientNet (2019)</span> - 효율적인 스케일링으로 높은 성능
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Convolution 연산의 원리</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Convolution은 CNN의 핵심 연산입니다. 작은 필터(커널)가 입력 이미지를 슬라이딩하며
          특징을 추출합니다. 이를 통해 엣지, 텍스처, 패턴 등을 감지할 수 있습니다.
        </p>

        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4">Convolution 연산 시각화</h3>
          <svg viewBox="0 0 900 600" className="w-full h-auto">
            {/* 제목 */}
            <text x="450" y="30" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-lg font-bold">
              3×3 필터로 5×5 입력 이미지 컨볼루션
            </text>

            {/* 입력 이미지 (5x5) */}
            <g transform="translate(100, 80)">
              <text x="60" y="-10" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">
                입력 이미지 (5×5)
              </text>
              {/* 5x5 그리드 */}
              {[0, 1, 2, 3, 4].map((row) =>
                [0, 1, 2, 3, 4].map((col) => {
                  const isHighlight = row <= 2 && col <= 2; // 왼쪽 상단 3x3 강조
                  return (
                    <g key={`input-${row}-${col}`}>
                      <rect
                        x={col * 30}
                        y={row * 30}
                        width="28"
                        height="28"
                        className={isHighlight ? "fill-blue-100 dark:fill-blue-900 stroke-blue-500" : "fill-gray-100 dark:fill-gray-700 stroke-gray-400"}
                        strokeWidth="1"
                      />
                      <text
                        x={col * 30 + 14}
                        y={row * 30 + 18}
                        textAnchor="middle"
                        className="fill-gray-700 dark:fill-gray-300 text-xs"
                      >
                        {Math.floor(Math.random() * 9)}
                      </text>
                    </g>
                  );
                })
              )}
            </g>

            {/* 필터/커널 (3x3) */}
            <g transform="translate(350, 120)">
              <text x="45" y="-10" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">
                필터 (3×3)
              </text>
              {/* Edge Detection 필터 예시 */}
              {[
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
              ].map((row, rowIdx) =>
                row.map((val, colIdx) => (
                  <g key={`filter-${rowIdx}-${colIdx}`}>
                    <rect
                      x={colIdx * 30}
                      y={rowIdx * 30}
                      width="28"
                      height="28"
                      className="fill-green-100 dark:fill-green-900 stroke-green-500"
                      strokeWidth="2"
                    />
                    <text
                      x={colIdx * 30 + 14}
                      y={rowIdx * 30 + 18}
                      textAnchor="middle"
                      className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold"
                    >
                      {val}
                    </text>
                  </g>
                ))
              )}
              <text x="45" y="110" textAnchor="middle" className="fill-green-600 dark:fill-green-400 text-xs">
                (엣지 검출 필터)
              </text>
            </g>

            {/* 곱셈 기호 */}
            <text x="520" y="180" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-2xl font-bold">
              ⊗
            </text>

            {/* 출력 특징 맵 (3x3) */}
            <g transform="translate(600, 120)">
              <text x="45" y="-10" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">
                출력 (3×3)
              </text>
              {[0, 1, 2].map((row) =>
                [0, 1, 2].map((col) => (
                  <g key={`output-${row}-${col}`}>
                    <rect
                      x={col * 30}
                      y={row * 30}
                      width="28"
                      height="28"
                      className="fill-orange-100 dark:fill-orange-900 stroke-orange-500"
                      strokeWidth="2"
                    />
                    <text
                      x={col * 30 + 14}
                      y={row * 30 + 18}
                      textAnchor="middle"
                      className="fill-gray-700 dark:fill-gray-300 text-xs"
                    >
                      {Math.floor(Math.random() * 20)}
                    </text>
                  </g>
                ))
              )}
            </g>

            {/* 연산 과정 설명 */}
            <g transform="translate(100, 280)">
              <rect x="0" y="0" width="700" height="280" className="fill-gray-50 dark:fill-gray-800 stroke-gray-300 dark:stroke-gray-600" strokeWidth="1" rx="8" />

              <text x="350" y="30" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm font-bold">
                연산 과정 (Element-wise 곱셈 후 합산)
              </text>

              {/* Step 1 */}
              <g transform="translate(50, 50)">
                <text x="0" y="0" className="fill-teal-600 dark:fill-teal-400 text-xs font-semibold">Step 1: 왼쪽 상단</text>
                <text x="0" y="20" className="fill-gray-600 dark:fill-gray-400 text-xs font-mono">
                  입력[0:3, 0:3] ⊙ 필터 = 결과[0,0]
                </text>
              </g>

              {/* Step 2 */}
              <g transform="translate(250, 50)">
                <text x="0" y="0" className="fill-teal-600 dark:fill-teal-400 text-xs font-semibold">Step 2: 1칸 오른쪽</text>
                <text x="0" y="20" className="fill-gray-600 dark:fill-gray-400 text-xs font-mono">
                  입력[0:3, 1:4] ⊙ 필터 = 결과[0,1]
                </text>
              </g>

              {/* Step 3 */}
              <g transform="translate(450, 50)">
                <text x="0" y="0" className="fill-teal-600 dark:fill-teal-400 text-xs font-semibold">Step 3: 2칸 오른쪽</text>
                <text x="0" y="20" className="fill-gray-600 dark:fill-gray-400 text-xs font-mono">
                  입력[0:3, 2:5] ⊙ 필터 = 결과[0,2]
                </text>
              </g>

              {/* 슬라이딩 윈도우 시각화 */}
              <g transform="translate(50, 100)">
                {/* 3개의 위치 표시 */}
                {[0, 1, 2].map((pos) => (
                  <g key={`slide-${pos}`} transform={`translate(${pos * 200}, 0)`}>
                    <rect x="0" y="0" width="90" height="90" className="fill-blue-50 dark:fill-blue-900/20 stroke-blue-400" strokeWidth="2" strokeDasharray="4" />
                    <text x="45" y="110" textAnchor="middle" className="fill-blue-600 dark:fill-blue-400 text-xs">
                      위치 {pos + 1}
                    </text>
                  </g>
                ))}
              </g>

              {/* 수식 */}
              <g transform="translate(50, 220)">
                <text x="0" y="0" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">출력 크기 계산:</text>
                <text x="0" y="20" className="fill-gray-600 dark:fill-gray-400 text-xs font-mono">
                  출력 = (입력크기 - 필터크기) / 스트라이드 + 1 = (5 - 3) / 1 + 1 = 3
                </text>
              </g>
            </g>

            {/* 핵심 개념 박스 */}
            <g transform="translate(100, 570)">
              <text x="0" y="0" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">
                ✓ Stride (보폭): 필터가 이동하는 간격 (여기서는 1)
              </text>
              <text x="300" y="0" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">
                ✓ Padding: 가장자리 보존을 위한 패딩 (여기서는 0)
              </text>
            </g>
          </svg>
        </div>

        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4">CNN 전체 아키텍처 흐름</h3>
          <svg viewBox="0 0 1200 400" className="w-full h-auto">
            {/* 제목 */}
            <text x="600" y="30" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-lg font-bold">
              전형적인 CNN 구조: 입력 → 특징 추출 → 분류
            </text>

            {/* Input Image */}
            <g transform="translate(50, 100)">
              <rect x="0" y="0" width="80" height="80" className="fill-blue-100 dark:fill-blue-900 stroke-blue-500" strokeWidth="2" />
              <text x="40" y="95" textAnchor="middle" className="fill-blue-600 dark:fill-blue-400 text-xs font-semibold">
                입력 이미지
              </text>
              <text x="40" y="110" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                224×224×3
              </text>
            </g>

            {/* Arrow 1 */}
            <path d="M 135 140 L 175 140" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrowgray)" />
            <defs>
              <marker id="arrowgray" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#9ca3af" />
              </marker>
            </defs>

            {/* Conv1 + ReLU */}
            <g transform="translate(180, 90)">
              <rect x="0" y="0" width="70" height="100" className="fill-green-100 dark:fill-green-900 stroke-green-500" strokeWidth="2" rx="4" />
              <text x="35" y="40" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">
                Conv2D
              </text>
              <text x="35" y="55" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                64 filters
              </text>
              <text x="35" y="70" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                3×3
              </text>
              <text x="35" y="85" textAnchor="middle" className="fill-green-600 dark:fill-green-400 text-xs">
                +ReLU
              </text>
              <text x="35" y="115" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                224×224×64
              </text>
            </g>

            {/* Arrow 2 */}
            <path d="M 255 140 L 295 140" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrowgray)" />

            {/* MaxPool1 */}
            <g transform="translate(300, 100)">
              <rect x="0" y="0" width="70" height="80" className="fill-purple-100 dark:fill-purple-900 stroke-purple-500" strokeWidth="2" rx="4" />
              <text x="35" y="35" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">
                MaxPool
              </text>
              <text x="35" y="50" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                2×2
              </text>
              <text x="35" y="95" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                112×112×64
              </text>
            </g>

            {/* Arrow 3 */}
            <path d="M 375 140 L 415 140" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrowgray)" />

            {/* Conv2 + ReLU */}
            <g transform="translate(420, 80)">
              <rect x="0" y="0" width="70" height="120" className="fill-green-100 dark:fill-green-900 stroke-green-500" strokeWidth="2" rx="4" />
              <text x="35" y="45" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">
                Conv2D
              </text>
              <text x="35" y="60" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                128 filters
              </text>
              <text x="35" y="75" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                3×3
              </text>
              <text x="35" y="90" textAnchor="middle" className="fill-green-600 dark:fill-green-400 text-xs">
                +ReLU
              </text>
              <text x="35" y="135" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                112×112×128
              </text>
            </g>

            {/* Arrow 4 */}
            <path d="M 495 140 L 535 140" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrowgray)" />

            {/* MaxPool2 */}
            <g transform="translate(540, 90)">
              <rect x="0" y="0" width="70" height="100" className="fill-purple-100 dark:fill-purple-900 stroke-purple-500" strokeWidth="2" rx="4" />
              <text x="35" y="40" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">
                MaxPool
              </text>
              <text x="35" y="55" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                2×2
              </text>
              <text x="35" y="115" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                56×56×128
              </text>
            </g>

            {/* Arrow 5 */}
            <path d="M 615 140 L 655 140" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrowgray)" />

            {/* Flatten */}
            <g transform="translate(660, 110)">
              <rect x="0" y="0" width="70" height="60" className="fill-yellow-100 dark:fill-yellow-900 stroke-yellow-500" strokeWidth="2" rx="4" />
              <text x="35" y="25" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">
                Flatten
              </text>
              <text x="35" y="75" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                1D Vector
              </text>
              <text x="35" y="90" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                (401,408)
              </text>
            </g>

            {/* Arrow 6 */}
            <path d="M 735 140 L 775 140" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrowgray)" />

            {/* FC1 + ReLU */}
            <g transform="translate(780, 100)">
              <rect x="0" y="0" width="70" height="80" className="fill-orange-100 dark:fill-orange-900 stroke-orange-500" strokeWidth="2" rx="4" />
              <text x="35" y="30" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">
                Dense
              </text>
              <text x="35" y="45" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                512 units
              </text>
              <text x="35" y="60" textAnchor="middle" className="fill-orange-600 dark:fill-orange-400 text-xs">
                +ReLU
              </text>
              <text x="35" y="95" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                (512)
              </text>
            </g>

            {/* Arrow 7 */}
            <path d="M 855 140 L 895 140" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrowgray)" />

            {/* Dropout */}
            <g transform="translate(900, 110)">
              <rect x="0" y="0" width="70" height="60" className="fill-pink-100 dark:fill-pink-900 stroke-pink-500" strokeWidth="2" rx="4" />
              <text x="35" y="25" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">
                Dropout
              </text>
              <text x="35" y="40" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                p=0.5
              </text>
              <text x="35" y="75" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                (512)
              </text>
            </g>

            {/* Arrow 8 */}
            <path d="M 975 140 L 1015 140" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrowgray)" />

            {/* Output */}
            <g transform="translate(1020, 100)">
              <rect x="0" y="0" width="70" height="80" className="fill-red-100 dark:fill-red-900 stroke-red-500" strokeWidth="2" rx="4" />
              <text x="35" y="30" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">
                Dense
              </text>
              <text x="35" y="45" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                10 classes
              </text>
              <text x="35" y="60" textAnchor="middle" className="fill-red-600 dark:fill-red-400 text-xs">
                Softmax
              </text>
              <text x="35" y="95" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                (10)
              </text>
            </g>

            {/* 하단 설명 */}
            <g transform="translate(50, 250)">
              <text x="0" y="0" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">특징 추출 (Feature Extraction)</text>
              <rect x="0" y="10" width="550" height="3" className="fill-green-500" />
            </g>

            <g transform="translate(650, 250)">
              <text x="0" y="0" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">분류 (Classification)</text>
              <rect x="0" y="10" width="440" height="3" className="fill-orange-500" />
            </g>

            {/* 주요 특징 박스 */}
            <g transform="translate(50, 290)">
              <rect x="0" y="0" width="1100" height="90" className="fill-gray-50 dark:fill-gray-800 stroke-gray-300 dark:stroke-gray-600" strokeWidth="1" rx="8" />

              <text x="550" y="25" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm font-bold">
                핵심 레이어 역할
              </text>

              <g transform="translate(50, 40)">
                <circle cx="5" cy="0" r="3" className="fill-green-500" />
                <text x="15" y="4" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  <tspan className="font-semibold">Convolution:</tspan> 공간적 특징 추출 (엣지, 텍스처, 패턴)
                </text>
              </g>

              <g transform="translate(450, 40)">
                <circle cx="5" cy="0" r="3" className="fill-purple-500" />
                <text x="15" y="4" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  <tspan className="font-semibold">Pooling:</tspan> 공간 크기 축소, 주요 특징 보존
                </text>
              </g>

              <g transform="translate(50, 65)">
                <circle cx="5" cy="0" r="3" className="fill-orange-500" />
                <text x="15" y="4" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  <tspan className="font-semibold">Dense (FC):</tspan> 추출된 특징을 기반으로 분류 결정
                </text>
              </g>

              <g transform="translate(450, 65)">
                <circle cx="5" cy="0" r="3" className="fill-pink-500" />
                <text x="15" y="4" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  <tspan className="font-semibold">Dropout:</tspan> 과적합 방지 (학습 시에만 활성화)
                </text>
              </g>
            </g>
          </svg>
        </div>

        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4">Pooling 연산 비교: Max vs Average</h3>
          <svg viewBox="0 0 900 500" className="w-full h-auto">
            {/* 제목 */}
            <text x="450" y="30" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-lg font-bold">
              Pooling 레이어의 역할과 종류
            </text>

            {/* 입력 특징 맵 */}
            <g transform="translate(50, 80)">
              <text x="80" y="-10" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">
                입력 특징 맵 (4×4)
              </text>
              {/* 4x4 그리드 with 데이터 */}
              {[
                [8, 3, 4, 6],
                [2, 9, 7, 1],
                [5, 4, 8, 3],
                [1, 6, 2, 5]
              ].map((row, rowIdx) =>
                row.map((val, colIdx) => {
                  // 2x2 영역별로 색상 구분
                  const isTopLeft = rowIdx < 2 && colIdx < 2;
                  const isTopRight = rowIdx < 2 && colIdx >= 2;
                  const isBottomLeft = rowIdx >= 2 && colIdx < 2;
                  const isBottomRight = rowIdx >= 2 && colIdx >= 2;

                  let fillClass = "fill-gray-100 dark:fill-gray-700";
                  if (isTopLeft) fillClass = "fill-blue-50 dark:fill-blue-900/30";
                  if (isTopRight) fillClass = "fill-green-50 dark:fill-green-900/30";
                  if (isBottomLeft) fillClass = "fill-yellow-50 dark:fill-yellow-900/30";
                  if (isBottomRight) fillClass = "fill-red-50 dark:fill-red-900/30";

                  return (
                    <g key={`input-pool-${rowIdx}-${colIdx}`}>
                      <rect
                        x={colIdx * 40}
                        y={rowIdx * 40}
                        width="38"
                        height="38"
                        className={`${fillClass} stroke-gray-400`}
                        strokeWidth="1"
                      />
                      <text
                        x={colIdx * 40 + 19}
                        y={rowIdx * 40 + 24}
                        textAnchor="middle"
                        className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold"
                      >
                        {val}
                      </text>
                    </g>
                  );
                })
              )}
              {/* 2x2 풀링 윈도우 표시 */}
              <rect x="0" y="0" width="78" height="78" className="fill-none stroke-blue-500" strokeWidth="3" strokeDasharray="5,5" />
              <rect x="80" y="0" width="78" height="78" className="fill-none stroke-green-500" strokeWidth="3" strokeDasharray="5,5" />
              <rect x="0" y="80" width="78" height="78" className="fill-none stroke-yellow-500" strokeWidth="3" strokeDasharray="5,5" />
              <rect x="80" y="80" width="78" height="78" className="fill-none stroke-red-500" strokeWidth="3" strokeDasharray="5,5" />
            </g>

            {/* Max Pooling 결과 */}
            <g transform="translate(350, 110)">
              <text x="40" y="-20" textAnchor="middle" className="fill-purple-600 dark:fill-purple-400 text-sm font-bold">
                Max Pooling (2×2)
              </text>
              <text x="40" y="-5" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                각 영역의 최대값 선택
              </text>
              {/* 2x2 출력 */}
              {[
                [9, 7],  // max(8,3,2,9), max(4,6,7,1)
                [6, 8]   // max(5,4,1,6), max(8,3,2,5)
              ].map((row, rowIdx) =>
                row.map((val, colIdx) => {
                  const colors = [
                    ["blue", "green"],
                    ["yellow", "red"]
                  ];
                  const color = colors[rowIdx][colIdx];

                  return (
                    <g key={`max-${rowIdx}-${colIdx}`}>
                      <rect
                        x={colIdx * 40}
                        y={rowIdx * 40}
                        width="38"
                        height="38"
                        className={`fill-${color}-100 dark:fill-${color}-900 stroke-${color}-500`}
                        strokeWidth="2"
                      />
                      <text
                        x={colIdx * 40 + 19}
                        y={rowIdx * 40 + 24}
                        textAnchor="middle"
                        className="fill-gray-700 dark:fill-gray-300 text-sm font-bold"
                      >
                        {val}
                      </text>
                    </g>
                  );
                })
              )}
              <text x="40" y="100" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                출력: 2×2
              </text>
            </g>

            {/* Average Pooling 결과 */}
            <g transform="translate(550, 110)">
              <text x="40" y="-20" textAnchor="middle" className="fill-orange-600 dark:fill-orange-400 text-sm font-bold">
                Average Pooling (2×2)
              </text>
              <text x="40" y="-5" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                각 영역의 평균값 계산
              </text>
              {/* 2x2 출력 */}
              {[
                [5.5, 4.5],  // avg(8,3,2,9), avg(4,6,7,1)
                [4, 4.5]     // avg(5,4,1,6), avg(8,3,2,5)
              ].map((row, rowIdx) =>
                row.map((val, colIdx) => {
                  const colors = [
                    ["blue", "green"],
                    ["yellow", "red"]
                  ];
                  const color = colors[rowIdx][colIdx];

                  return (
                    <g key={`avg-${rowIdx}-${colIdx}`}>
                      <rect
                        x={colIdx * 40}
                        y={rowIdx * 40}
                        width="38"
                        height="38"
                        className={`fill-${color}-100 dark:fill-${color}-900 stroke-${color}-500`}
                        strokeWidth="2"
                      />
                      <text
                        x={colIdx * 40 + 19}
                        y={rowIdx * 40 + 24}
                        textAnchor="middle"
                        className="fill-gray-700 dark:fill-gray-300 text-xs font-bold"
                      >
                        {val}
                      </text>
                    </g>
                  );
                })
              )}
              <text x="40" y="100" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">
                출력: 2×2
              </text>
            </g>

            {/* 비교 설명 박스 */}
            <g transform="translate(50, 280)">
              <rect x="0" y="0" width="800" height="180" className="fill-gray-50 dark:fill-gray-800 stroke-gray-300 dark:stroke-gray-600" strokeWidth="1" rx="8" />

              <text x="400" y="25" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm font-bold">
                Pooling 방식 비교 및 특징
              </text>

              {/* Max Pooling */}
              <g transform="translate(50, 50)">
                <circle cx="5" cy="0" r="4" className="fill-purple-500" />
                <text x="20" y="4" className="fill-gray-700 dark:fill-gray-300 text-xs font-bold">
                  Max Pooling
                </text>
                <text x="20" y="25" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  • 가장 강한 특징(최대값)을 보존
                </text>
                <text x="20" y="40" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  • 엣지, 텍스처 등 중요한 특징 강조
                </text>
                <text x="20" y="55" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  • CNN에서 가장 널리 사용됨
                </text>
                <text x="20" y="70" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  • 공간 해상도 절반으로 축소 (4×4 → 2×2)
                </text>
              </g>

              {/* Average Pooling */}
              <g transform="translate(420, 50)">
                <circle cx="5" cy="0" r="4" className="fill-orange-500" />
                <text x="20" y="4" className="fill-gray-700 dark:fill-gray-300 text-xs font-bold">
                  Average Pooling
                </text>
                <text x="20" y="25" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  • 영역 전체의 평균값 계산
                </text>
                <text x="20" y="40" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  • 더 부드러운 특징 맵 생성
                </text>
                <text x="20" y="55" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  • Global Average Pooling (GAP)에 활용
                </text>
                <text x="20" y="70" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  • 최종 분류층 전에 자주 사용
                </text>
              </g>

              {/* 공통 특징 */}
              <g transform="translate(50, 135)">
                <text x="0" y="0" className="fill-teal-600 dark:fill-teal-400 text-xs font-bold">
                  ✓ 공통 역할:
                </text>
                <text x="80" y="0" className="fill-gray-600 dark:fill-gray-400 text-xs">
                  파라미터 없이 특징 맵 크기 축소 (다운샘플링) | 계산량 감소 | 위치 불변성 (Translation Invariance) 향상 | 과적합 방지
                </text>
              </g>
            </g>
          </svg>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">Vision Transformer</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Transformer 아키텍처를 컴퓨터 비전에 적용한 Vision Transformer(ViT)는 
          CNN의 대안으로 주목받고 있습니다. 이미지를 패치로 나누어 시퀀스로 처리합니다.
        </p>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - Vision Transformer 사용</span>
            </div>
            <button
              onClick={() => copyToClipboard(`from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# 사전 학습된 ViT 모델 로드
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 이미지 전처리
image = Image.open('sample.jpg')
inputs = processor(images=image, return_tensors="pt")

# 예측
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print(f"예측된 클래스: {model.config.id2label[predicted_class]}")`, 'dl-vision-1')}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {copiedCode === 'dl-vision-1' ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
            <code>{`from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# 사전 학습된 ViT 모델 로드
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 이미지 전처리
image = Image.open('sample.jpg')
inputs = processor(images=image, return_tensors="pt")

# 예측
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print(f"예측된 클래스: {model.config.id2label[predicted_class]}")`}</code>
          </pre>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Landmark CNN Architectures',
            icon: 'paper' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'ImageNet Classification with Deep CNNs',
                authors: 'Alex Krizhevsky, et al.',
                year: '2012',
                description: 'AlexNet - ImageNet 우승으로 딥러닝 시대 개막 (100,000+ 인용)',
                link: 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html'
              },
              {
                title: 'Very Deep Convolutional Networks (VGG)',
                authors: 'Karen Simonyan, Andrew Zisserman',
                year: '2014',
                description: 'VGGNet - 3x3 컨볼루션의 힘 증명, Transfer Learning의 기준',
                link: 'https://arxiv.org/abs/1409.1556'
              },
              {
                title: 'Deep Residual Learning (ResNet)',
                authors: 'Kaiming He, et al.',
                year: '2015',
                description: 'ResNet - Skip Connection으로 152층 학습 성공 (150,000+ 인용)',
                link: 'https://arxiv.org/abs/1512.03385'
              },
              {
                title: 'EfficientNet: Rethinking Model Scaling',
                authors: 'Mingxing Tan, Quoc V. Le',
                year: '2019',
                description: 'Compound Scaling - 깊이, 너비, 해상도 동시 최적화',
                link: 'https://arxiv.org/abs/1905.11946'
              }
            ]
          },
          {
            title: 'Vision Transformers',
            icon: 'paper' as const,
            color: 'border-cyan-500',
            items: [
              {
                title: 'An Image is Worth 16x16 Words (ViT)',
                authors: 'Alexey Dosovitskiy, et al.',
                year: '2020',
                description: 'Vision Transformer - CNN 없이 Transformer만으로 ImageNet SOTA',
                link: 'https://arxiv.org/abs/2010.11929'
              },
              {
                title: 'Swin Transformer',
                authors: 'Ze Liu, et al.',
                year: '2021',
                description: 'Hierarchical ViT - Object Detection, Segmentation에도 탁월',
                link: 'https://arxiv.org/abs/2103.14030'
              },
              {
                title: 'BEiT: BERT Pre-Training of Image Transformers',
                authors: 'Hangbo Bao, et al.',
                year: '2021',
                description: 'Masked Image Modeling - Self-Supervised Vision 학습',
                link: 'https://arxiv.org/abs/2106.08254'
              }
            ]
          },
          {
            title: 'Training Techniques',
            icon: 'paper' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Batch Normalization',
                authors: 'Sergey Ioffe, Christian Szegedy',
                year: '2015',
                description: 'Internal Covariate Shift 해결 - 학습 속도 대폭 향상',
                link: 'https://arxiv.org/abs/1502.03167'
              },
              {
                title: 'Dropout: A Simple Way to Prevent Overfitting',
                authors: 'Geoffrey Hinton, et al.',
                year: '2014',
                description: '랜덤 뉴런 제거로 Regularization - 과적합 방지',
                link: 'https://jmlr.org/papers/v15/srivastava14a.html'
              },
              {
                title: 'Data Augmentation',
                authors: 'Connor Shorten, Taghi M. Khoshgoftaar',
                year: '2019',
                description: '이미지 증강 기법 총정리 - Flip, Rotation, Mixup, CutMix',
                link: 'https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0'
              }
            ]
          },
          {
            title: 'Transfer Learning & Pre-training',
            icon: 'paper' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'How transferable are features in deep neural networks?',
                authors: 'Jason Yosinski, et al.',
                year: '2014',
                description: 'Transfer Learning의 과학적 근거 - 층별 특징 분석',
                link: 'https://arxiv.org/abs/1411.1792'
              },
              {
                title: 'Self-Supervised Learning (SimCLR)',
                authors: 'Ting Chen, et al.',
                year: '2020',
                description: 'Contrastive Learning - 라벨 없이 강력한 표현 학습',
                link: 'https://arxiv.org/abs/2002.05709'
              }
            ]
          },
          {
            title: 'Tools & Frameworks',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'PyTorch Vision (torchvision)',
                authors: 'PyTorch Team',
                year: '2024',
                description: '사전학습 모델 라이브러리 - ResNet, VGG, ViT 등',
                link: 'https://pytorch.org/vision/stable/index.html'
              },
              {
                title: 'TensorFlow Hub',
                authors: 'TensorFlow Team',
                year: '2024',
                description: '재사용 가능한 ML 모델 - EfficientNet, MobileNet 등',
                link: 'https://tfhub.dev/'
              },
              {
                title: 'Hugging Face Transformers',
                authors: 'Hugging Face',
                year: '2024',
                description: 'ViT, DeiT, Swin 등 Vision Transformer 모델',
                link: 'https://huggingface.co/docs/transformers/model_doc/vit'
              }
            ]
          }
        ]}
      />
    </div>
  );
}