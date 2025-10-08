'use client';

import { useState } from 'react';
import {
  Lightbulb,
  Terminal,
  Copy,
  Check,
  CheckCircle,
  AlertTriangle
} from 'lucide-react';
import References from '@/components/common/References';

export default function Chapter3() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">특징점 검출 알고리즘</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          특징점(feature points)은 이미지에서 독특하고 식별 가능한 지점들입니다.
          이러한 특징점들은 이미지 매칭, 객체 인식, 3D 재구성, 파노라마 스티칭 등에 핵심적으로 활용됩니다.
          좋은 특징점은 <strong>반복 가능(repeatable)</strong>하고, <strong>독특(distinctive)</strong>하며, <strong>불변성(invariance)</strong>을 가져야 합니다.
        </p>

        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-3">
            <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
            <div>
              <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">좋은 특징점의 조건</h3>
              <ul className="space-y-2 text-blue-800 dark:text-blue-200">
                <li>• <strong>반복 가능성</strong>: 다른 시점/조명에서도 동일한 위치에서 검출</li>
                <li>• <strong>독특성</strong>: 다른 특징점과 명확히 구별 가능</li>
                <li>• <strong>불변성</strong>: 크기, 회전, 조명 변화에 강건</li>
                <li>• <strong>효율성</strong>: 빠른 검출 및 매칭 속도</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">SIFT</h3>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
              Scale-Invariant Feature Transform. 크기와 회전에 불변한 특징점 검출
            </p>
            <ul className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
              <li>✓ 128차원 디스크립터</li>
              <li>✓ 매우 높은 정확도</li>
              <li>✗ 특허 제한 (2020년 만료)</li>
              <li>✗ 상대적으로 느림</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">SURF</h3>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
              Speeded-Up Robust Features. SIFT보다 3배 빠른 특징점 검출
            </p>
            <ul className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
              <li>✓ 64/128차원 디스크립터</li>
              <li>✓ 적분 이미지로 가속</li>
              <li>✗ 특허 제한 있음</li>
              <li>✓ SIFT 대비 빠름</li>
            </ul>
          </div>
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-teal-600 dark:text-teal-400 mb-2">ORB</h3>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
              Oriented FAST and Rotated BRIEF. 실시간 처리에 적합한 빠른 알고리즘
            </p>
            <ul className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
              <li>✓ 256비트 바이너리</li>
              <li>✓ 매우 빠름 (SIFT의 100배)</li>
              <li>✓ 특허 없음 (무료)</li>
              <li>✓ OpenCV 기본 제공</li>
            </ul>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">코너 검출 - Harris & Shi-Tomasi</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          코너는 2개 이상의 엣지가 만나는 지점으로, 특징점 검출의 기초가 됩니다.
          Harris Corner Detector는 이미지 그래디언트의 자기상관 행렬(autocorrelation matrix)을 분석하여 코너를 찾습니다.
        </p>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - Harris Corner Detection</span>
            </div>
            <button
              onClick={() => copyToClipboard(`import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris Corner Detection
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 결과 확대하여 시각화
dst = cv2.dilate(dst, None)

# 임계값 이상의 코너 표시 (빨간색)
img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('Harris Corners', img)
cv2.waitKey(0)`, 'harris-1')}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {copiedCode === 'harris-1' ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
            <code>{`import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris Corner Detection
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 결과 확대하여 시각화
dst = cv2.dilate(dst, None)

# 임계값 이상의 코너 표시 (빨간색)
img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('Harris Corners', img)
cv2.waitKey(0)`}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">ORB 특징점 검출 및 매칭</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          ORB는 FAST 코너 검출기와 BRIEF 디스크립터를 결합한 알고리즘입니다.
          실시간 애플리케이션에 적합하며, OpenCV에서 기본으로 제공되어 활용도가 높습니다.
        </p>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - ORB 특징점 매칭</span>
            </div>
            <button
              onClick={() => copyToClipboard(`import cv2
import numpy as np

# 두 이미지 읽기
img1 = cv2.imread('image1.jpg', 0)  # 쿼리 이미지
img2 = cv2.imread('image2.jpg', 0)  # 훈련 이미지

# ORB 검출기 생성
orb = cv2.ORB_create(nfeatures=2000)

# 특징점 및 디스크립터 검출
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# BFMatcher로 매칭 (Hamming 거리)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 거리 기준 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 상위 50개 매칭 시각화
img_matches = cv2.drawMatches(img1, kp1, img2, kp2,
                                matches[:50], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('ORB Matches', img_matches)
cv2.waitKey(0)`, 'orb-1')}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {copiedCode === 'orb-1' ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
            <code>{`import cv2
import numpy as np

# 두 이미지 읽기
img1 = cv2.imread('image1.jpg', 0)  # 쿼리 이미지
img2 = cv2.imread('image2.jpg', 0)  # 훈련 이미지

# ORB 검출기 생성
orb = cv2.ORB_create(nfeatures=2000)

# 특징점 및 디스크립터 검출
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# BFMatcher로 매칭 (Hamming 거리)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 거리 기준 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 상위 50개 매칭 시각화
img_matches = cv2.drawMatches(img1, kp1, img2, kp2,
                                matches[:50], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('ORB Matches', img_matches)
cv2.waitKey(0)`}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">RANSAC을 이용한 이상치 제거</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          특징점 매칭에는 항상 잘못된 매칭(outliers)이 포함됩니다.
          RANSAC(Random Sample Consensus)은 이러한 이상치를 제거하고 정확한 기하학적 변환을 추정합니다.
        </p>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
            <div>
              <h3 className="font-semibold text-yellow-900 dark:text-yellow-100 mb-2">RANSAC 알고리즘</h3>
              <ol className="space-y-2 text-yellow-800 dark:text-yellow-200 text-sm">
                <li>1. 최소한의 점 집합을 무작위로 선택 (4개 점)</li>
                <li>2. 이 점들로 기하학적 모델 (Homography) 추정</li>
                <li>3. 모든 점에 대해 모델과의 일치도(inlier) 계산</li>
                <li>4. 가장 많은 inlier를 가진 모델 선택</li>
                <li>5. 충분히 반복하여 최적의 모델 찾기</li>
              </ol>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - RANSAC으로 Homography 추정</span>
            </div>
            <button
              onClick={() => copyToClipboard(`import cv2
import numpy as np

# ORB로 특징점 매칭 (위 코드 활용)
# ... kp1, kp2, matches 얻기 ...

# 매칭 결과에서 좋은 매칭만 선택
good_matches = matches[:100]

# 매칭된 점들의 좌표 추출
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# RANSAC으로 Homography 추정
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

print(f"총 매칭: {len(good_matches)}")
print(f"Inliers: {sum(matchesMask)}")
print(f"Outliers: {len(matchesMask) - sum(matchesMask)}")

# Inlier만 시각화
draw_params = dict(matchColor=(0, 255, 0),  # 녹색
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)

img_result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
cv2.imshow('RANSAC Inliers', img_result)
cv2.waitKey(0)`, 'ransac-1')}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {copiedCode === 'ransac-1' ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
            <code>{`import cv2
import numpy as np

# ORB로 특징점 매칭 (위 코드 활용)
# ... kp1, kp2, matches 얻기 ...

# 매칭 결과에서 좋은 매칭만 선택
good_matches = matches[:100]

# 매칭된 점들의 좌표 추출
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# RANSAC으로 Homography 추정
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

print(f"총 매칭: {len(good_matches)}")
print(f"Inliers: {sum(matchesMask)}")
print(f"Outliers: {len(matchesMask) - sum(matchesMask)}")

# Inlier만 시각화
draw_params = dict(matchColor=(0, 255, 0),  # 녹색
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)

img_result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
cv2.imshow('RANSAC Inliers', img_result)
cv2.waitKey(0)`}</code>
          </pre>
        </div>
      </section>

      <section className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
        <div className="flex items-start gap-3">
          <CheckCircle className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5" />
          <div>
            <h3 className="font-semibold text-green-900 dark:text-green-100 mb-2">실습 과제</h3>
            <ul className="space-y-2 text-green-800 dark:text-green-200">
              <li>1. 두 장의 사진에서 ORB 특징점을 검출하고 매칭해보세요</li>
              <li>2. SIFT, SURF, ORB의 검출 속도를 비교 측정해보세요 (time 모듈 사용)</li>
              <li>3. RANSAC을 사용하여 파노라마 이미지를 만들어보세요</li>
              <li>4. 회전/크기 변화된 이미지에서도 매칭이 잘 되는지 테스트해보세요</li>
            </ul>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Foundational Papers',
            icon: 'paper' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'Distinctive Image Features from Scale-Invariant Keypoints',
                authors: 'David G. Lowe',
                year: '2004',
                description: 'SIFT 알고리즘 - CV 역사상 가장 영향력 있는 논문 (90,000+ 인용)',
                link: 'https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf'
              },
              {
                title: 'SURF: Speeded Up Robust Features',
                authors: 'Herbert Bay, et al.',
                year: '2008',
                description: 'SIFT보다 3배 빠른 특징점 검출 - 적분 이미지 활용',
                link: 'https://www.vision.ee.ethz.ch/~surf/eccv06.pdf'
              },
              {
                title: 'ORB: An Efficient Alternative to SIFT or SURF',
                authors: 'Ethan Rublee, et al.',
                year: '2011',
                description: 'OpenCV의 기본 특징점 알고리즘 - 특허 없는 오픈소스',
                link: 'https://ieeexplore.ieee.org/document/6126544'
              }
            ]
          },
          {
            title: 'Corner & Keypoint Detection',
            icon: 'paper' as const,
            color: 'border-cyan-500',
            items: [
              {
                title: 'Good Features to Track',
                authors: 'Jianbo Shi, Carlo Tomasi',
                year: '1994',
                description: 'Shi-Tomasi Corner Detector - 광학 흐름의 기초',
                link: 'https://ieeexplore.ieee.org/document/323794'
              },
              {
                title: 'FAST Corner Detection',
                authors: 'Edward Rosten, Tom Drummond',
                year: '2006',
                description: '실시간 코너 검출 - ORB의 핵심 구성 요소',
                link: 'https://link.springer.com/chapter/10.1007/11744023_34'
              },
              {
                title: 'Harris Corner Detector',
                authors: 'Chris Harris, Mike Stephens',
                year: '1988',
                description: '최초의 실용적 코너 검출 - CV의 고전',
                link: 'https://www.bmva.org/bmvc/1988/avc-88-023.pdf'
              }
            ]
          },
          {
            title: 'Feature Descriptors',
            icon: 'paper' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'BRIEF: Binary Robust Independent Elementary Features',
                authors: 'Michael Calonder, et al.',
                year: '2010',
                description: '바이너리 디스크립터 - 메모리 효율적인 특징 표현',
                link: 'https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf'
              },
              {
                title: 'Local Binary Patterns (LBP)',
                authors: 'Timo Ojala, et al.',
                year: '2002',
                description: '텍스처 분류의 표준 - 얼굴 인식에 널리 활용',
                link: 'https://ieeexplore.ieee.org/document/990477'
              }
            ]
          },
          {
            title: 'Modern Deep Learning Features',
            icon: 'paper' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'SuperPoint: Self-Supervised Interest Point Detection',
                authors: 'Daniel DeTone, et al.',
                year: '2018',
                description: 'CNN 기반 특징점 검출 - 딥러닝 시대의 SIFT',
                link: 'https://arxiv.org/abs/1712.07629'
              },
              {
                title: 'D2-Net: A Trainable CNN for Joint Detection and Description',
                authors: 'Mihail Dusmanu, et al.',
                year: '2019',
                description: '검출과 기술자 통합 학습 - 3D 재구성에 탁월',
                link: 'https://arxiv.org/abs/1905.03561'
              }
            ]
          },
          {
            title: 'Tools & Implementation',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'OpenCV Feature Detection',
                authors: 'OpenCV Team',
                year: '2024',
                description: 'SIFT, SURF, ORB, AKAZE 구현 및 튜토리얼',
                link: 'https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html'
              },
              {
                title: 'VLFeat Library',
                authors: 'Andrea Vedaldi, Brian Fulkerson',
                year: '2024',
                description: 'SIFT, HOG, MSER 등 CV 알고리즘 라이브러리',
                link: 'https://www.vlfeat.org/'
              },
              {
                title: 'Feature Matching Tutorial',
                authors: 'PyImageSearch',
                year: '2023',
                description: '특징점 매칭 실전 가이드 - Homography, RANSAC',
                link: 'https://pyimagesearch.com/category/feature-detection-description/'
              }
            ]
          }
        ]}
      />
    </div>
  );
}