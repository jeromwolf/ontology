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

export default function Chapter7() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4">얼굴 인식 파이프라인</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          얼굴 인식은 여러 단계로 구성된 복잡한 프로세스입니다.
          검출, 정렬, 특징 추출, 매칭의 단계를 거칩니다.
        </p>

        <div className="bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4 text-teal-900 dark:text-teal-100">처리 단계</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">1</div>
              <p className="font-medium">얼굴 검출</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">이미지에서 얼굴 영역 찾기</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">2</div>
              <p className="font-medium">얼굴 정렬</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">랜드마크 기반 정규화</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">3</div>
              <p className="font-medium">특징 추출</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">얼굴 임베딩 벡터 생성</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">4</div>
              <p className="font-medium">신원 확인</p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">벡터 유사도 비교</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4">68개 얼굴 랜드마크 위치</h3>
          <svg viewBox="0 0 600 700" className="w-full h-auto max-w-md mx-auto">
            {/* 얼굴 윤곽 */}
            <ellipse cx="300" cy="350" rx="150" ry="200" className="fill-none stroke-gray-400 dark:stroke-gray-500" strokeWidth="2" />

            {/* 턱선 (0-16) */}
            <path d="M 180 250 Q 170 350 180 450 Q 200 500 250 540 Q 300 560 350 540 Q 400 500 420 450 Q 430 350 420 250"
                  className="fill-none stroke-purple-500" strokeWidth="2" />

            {/* 왼쪽 눈썹 (17-21) */}
            <path d="M 210 260 Q 230 250 250 255 Q 270 260 280 265"
                  className="fill-none stroke-blue-500" strokeWidth="3" />

            {/* 오른쪽 눈썹 (22-26) */}
            <path d="M 320 265 Q 330 260 350 255 Q 370 250 390 260"
                  className="fill-none stroke-blue-500" strokeWidth="3" />

            {/* 코 (27-35) */}
            <path d="M 300 280 L 300 380" className="fill-none stroke-green-500" strokeWidth="2" />
            <path d="M 270 380 Q 285 395 300 395 Q 315 395 330 380" className="fill-none stroke-green-500" strokeWidth="2" />

            {/* 왼쪽 눈 (36-41) */}
            <ellipse cx="250" cy="310" rx="25" ry="15" className="fill-none stroke-cyan-500" strokeWidth="2" />
            <circle cx="250" cy="310" r="8" className="fill-gray-800 dark:fill-gray-300" />

            {/* 오른쪽 눈 (42-47) */}
            <ellipse cx="350" cy="310" rx="25" ry="15" className="fill-none stroke-cyan-500" strokeWidth="2" />
            <circle cx="350" cy="310" r="8" className="fill-gray-800 dark:fill-gray-300" />

            {/* 입술 외곽 (48-59) */}
            <path d="M 250 450 Q 275 465 300 465 Q 325 465 350 450" className="fill-none stroke-red-500" strokeWidth="2" />
            <path d="M 250 450 Q 275 445 300 445 Q 325 445 350 450" className="fill-none stroke-red-500" strokeWidth="2" />

            {/* 입술 내부 (60-67) */}
            <path d="M 265 450 Q 282 457 300 457 Q 318 457 335 450" className="fill-none stroke-pink-500" strokeWidth="1.5" />

            {/* 랜드마크 포인트들 */}
            <circle cx="180" cy="250" r="3" className="fill-purple-500" />
            <circle cx="420" cy="250" r="3" className="fill-purple-500" />
            <circle cx="210" cy="260" r="3" className="fill-blue-500" />
            <circle cx="390" cy="260" r="3" className="fill-blue-500" />
            <circle cx="300" cy="280" r="3" className="fill-green-500" />
            <circle cx="225" cy="310" r="3" className="fill-cyan-500" />
            <circle cx="275" cy="310" r="3" className="fill-cyan-500" />
            <circle cx="325" cy="310" r="3" className="fill-cyan-500" />
            <circle cx="375" cy="310" r="3" className="fill-cyan-500" />
            <circle cx="250" cy="450" r="3" className="fill-red-500" />
            <circle cx="350" cy="450" r="3" className="fill-red-500" />

            {/* 범례 */}
            <g transform="translate(50, 600)">
              <text x="0" y="0" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">랜드마크 그룹:</text>
              <circle cx="10" cy="20" r="4" className="fill-purple-500" />
              <text x="20" y="24" className="fill-gray-600 dark:fill-gray-400 text-xs">턱선 (0-16)</text>

              <circle cx="10" cy="40" r="4" className="fill-blue-500" />
              <text x="20" y="44" className="fill-gray-600 dark:fill-gray-400 text-xs">눈썹 (17-26)</text>

              <circle cx="120" cy="20" r="4" className="fill-green-500" />
              <text x="130" y="24" className="fill-gray-600 dark:fill-gray-400 text-xs">코 (27-35)</text>

              <circle cx="120" cy="40" r="4" className="fill-cyan-500" />
              <text x="130" y="44" className="fill-gray-600 dark:fill-gray-400 text-xs">눈 (36-47)</text>

              <circle cx="230" cy="20" r="4" className="fill-red-500" />
              <text x="240" y="24" className="fill-gray-600 dark:fill-gray-400 text-xs">입술 외곽 (48-59)</text>

              <circle cx="230" cy="40" r="4" className="fill-pink-500" />
              <text x="240" y="44" className="fill-gray-600 dark:fill-gray-400 text-xs">입술 내부 (60-67)</text>
            </g>
          </svg>
        </div>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - dlib 얼굴 검출 및 랜드마크</span>
            </div>
            <button
              onClick={() => copyToClipboard(`import dlib
import cv2

# 얼굴 검출기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 이미지 로드 및 그레이스케일 변환
img = cv2.imread('face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = detector(gray)

for face in faces:
    # 68개 랜드마크 추출
    landmarks = predictor(gray, face)

    # 랜드마크 그리기
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # 바운딩 박스 그리기
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow('Face Landmarks', img)
cv2.waitKey(0)`, 'face-det-1')}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {copiedCode === 'face-det-1' ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
            <code>{`import dlib
import cv2

# 얼굴 검출기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 이미지 로드 및 그레이스케일 변환
img = cv2.imread('face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = detector(gray)

for face in faces:
    # 68개 랜드마크 추출
    landmarks = predictor(gray, face)

    # 랜드마크 그리기
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # 바운딩 박스 그리기
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow('Face Landmarks', img)
cv2.waitKey(0)`}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">얼굴 인식 - FaceNet 임베딩</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          FaceNet은 얼굴을 128차원 벡터로 변환하여 유사도 비교를 통해 신원을 확인합니다.
          같은 사람의 얼굴 벡터는 가까운 거리에, 다른 사람은 먼 거리에 배치됩니다.
        </p>

        <div className="bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-800 rounded-lg p-6 mb-6">
          <h3 className="font-semibold text-teal-900 dark:text-teal-100 mb-3">Triplet Loss 수식</h3>
          <p className="font-mono bg-white dark:bg-gray-800 p-3 rounded text-sm mb-2">
            L = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + α, 0)
          </p>
          <ul className="text-sm text-teal-800 dark:text-teal-200 space-y-1">
            <li>• A (Anchor): 기준 얼굴</li>
            <li>• P (Positive): 같은 사람의 다른 얼굴</li>
            <li>• N (Negative): 다른 사람의 얼굴</li>
            <li>• α (margin): 최소 분리 거리 (보통 0.2)</li>
          </ul>
        </div>

        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4">Triplet Loss 시각화</h3>
          <svg viewBox="0 0 800 400" className="w-full h-auto">
            {/* 배경 */}
            <rect width="800" height="400" fill="none" />

            {/* 임베딩 공간 제목 */}
            <text x="400" y="30" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-lg font-semibold">
              128차원 임베딩 공간 (2D로 투영)
            </text>

            {/* Anchor 얼굴 */}
            <circle cx="400" cy="200" r="40" className="fill-blue-500" opacity="0.8" />
            <text x="400" y="205" textAnchor="middle" className="fill-white font-bold text-xl">A</text>
            <text x="400" y="260" textAnchor="middle" className="fill-blue-600 dark:fill-blue-400 text-sm font-semibold">
              Anchor (기준)
            </text>

            {/* Positive 얼굴 (가까이) */}
            <circle cx="500" cy="180" r="40" className="fill-green-500" opacity="0.8" />
            <text x="500" y="185" textAnchor="middle" className="fill-white font-bold text-xl">P</text>
            <text x="500" y="240" textAnchor="middle" className="fill-green-600 dark:fill-green-400 text-sm font-semibold">
              Positive (같은 사람)
            </text>

            {/* Negative 얼굴 (멀리) */}
            <circle cx="200" cy="220" r="40" className="fill-red-500" opacity="0.8" />
            <text x="200" y="225" textAnchor="middle" className="fill-white font-bold text-xl">N</text>
            <text x="200" y="280" textAnchor="middle" className="fill-red-600 dark:fill-red-400 text-sm font-semibold">
              Negative (다른 사람)
            </text>

            {/* 화살표: A → P (가까운 거리) */}
            <defs>
              <marker id="arrowgreen" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#22c55e" />
              </marker>
              <marker id="arrowred" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#ef4444" />
              </marker>
            </defs>

            <line x1="440" y1="190" x2="460" y2="185" stroke="#22c55e" strokeWidth="3" markerEnd="url(#arrowgreen)" />
            <text x="450" y="175" textAnchor="middle" className="fill-green-600 dark:fill-green-400 text-xs font-medium">
              작은 거리
            </text>

            {/* 화살표: A → N (먼 거리) */}
            <line x1="360" y1="205" x2="240" y2="215" stroke="#ef4444" strokeWidth="3" markerEnd="url(#arrowred)" />
            <text x="300" y="200" textAnchor="middle" className="fill-red-600 dark:fill-red-400 text-xs font-medium">
              큰 거리 (+ margin α)
            </text>

            {/* 설명 박스 */}
            <rect x="550" y="280" width="230" height="100" className="fill-gray-100 dark:fill-gray-700" rx="8" />
            <text x="665" y="305" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">
              목표:
            </text>
            <text x="565" y="330" className="fill-gray-600 dark:fill-gray-400 text-xs">
              • A-P 거리 최소화
            </text>
            <text x="565" y="350" className="fill-gray-600 dark:fill-gray-400 text-xs">
              • A-N 거리 최대화
            </text>
            <text x="565" y="370" className="fill-gray-600 dark:fill-gray-400 text-xs">
              • 거리 차이 ≥ margin (α)
            </text>
          </svg>
        </div>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - FaceNet 얼굴 인식</span>
            </div>
            <button
              onClick={() => copyToClipboard(`from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import numpy as np

# 모델 로드
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_face_embedding(img_path):
    """얼굴 이미지로부터 128차원 임베딩 추출"""
    img = Image.open(img_path)

    # 얼굴 검출 및 정렬
    img_cropped = mtcnn(img)

    if img_cropped is not None:
        # 임베딩 추출
        with torch.no_grad():
            embedding = resnet(img_cropped.unsqueeze(0))
        return embedding.numpy()[0]
    return None

# 데이터베이스에 등록된 얼굴들
db_embeddings = {
    'person1': get_face_embedding('person1.jpg'),
    'person2': get_face_embedding('person2.jpg'),
    'person3': get_face_embedding('person3.jpg')
}

# 인식할 얼굴
query_embedding = get_face_embedding('unknown.jpg')

# 유사도 계산 (Euclidean distance)
def recognize_face(query_emb, db_embs, threshold=0.6):
    min_distance = float('inf')
    identity = 'Unknown'

    for name, db_emb in db_embs.items():
        distance = np.linalg.norm(query_emb - db_emb)
        if distance < min_distance and distance < threshold:
            min_distance = distance
            identity = name

    return identity, min_distance

result, distance = recognize_face(query_embedding, db_embeddings)
print(f"인식 결과: {result} (거리: {distance:.3f})")`, 'face-rec-1')}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {copiedCode === 'face-rec-1' ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
            <code>{`from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import numpy as np

# 모델 로드
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_face_embedding(img_path):
    """얼굴 이미지로부터 128차원 임베딩 추출"""
    img = Image.open(img_path)

    # 얼굴 검출 및 정렬
    img_cropped = mtcnn(img)

    if img_cropped is not None:
        # 임베딩 추출
        with torch.no_grad():
            embedding = resnet(img_cropped.unsqueeze(0))
        return embedding.numpy()[0]
    return None

# 데이터베이스에 등록된 얼굴들
db_embeddings = {
    'person1': get_face_embedding('person1.jpg'),
    'person2': get_face_embedding('person2.jpg'),
    'person3': get_face_embedding('person3.jpg')
}

# 인식할 얼굴
query_embedding = get_face_embedding('unknown.jpg')

# 유사도 계산 (Euclidean distance)
def recognize_face(query_emb, db_embs, threshold=0.6):
    min_distance = float('inf')
    identity = 'Unknown'

    for name, db_emb in db_embs.items():
        distance = np.linalg.norm(query_emb - db_emb)
        if distance < min_distance and distance < threshold:
            min_distance = distance
            identity = name

    return identity, min_distance

result, distance = recognize_face(query_embedding, db_embeddings)
print(f"인식 결과: {result} (거리: {distance:.3f})")`}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">감정 인식</h2>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          얼굴 표정으로부터 감정을 인식하는 기술은 HCI, 마케팅, 의료 등 다양한 분야에서 활용됩니다.
        </p>

        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-3">7가지 기본 감정</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😊 행복</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😢 슬픔</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😠 분노</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😱 두려움</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😲 놀람</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">🤢 혐오</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">😐 중립</div>
            <div className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">🤔 기타</div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 mb-6">
          <h3 className="font-semibold mb-4">CNN 감정 인식 아키텍처</h3>
          <svg viewBox="0 0 900 500" className="w-full h-auto">
            {/* 입력 이미지 */}
            <rect x="20" y="180" width="80" height="80" className="fill-blue-100 dark:fill-blue-900 stroke-blue-500" strokeWidth="2" />
            <text x="60" y="170" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">입력 이미지</text>
            <text x="60" y="280" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">48×48×1</text>

            {/* Conv1 + MaxPool */}
            <rect x="140" y="160" width="70" height="100" className="fill-green-100 dark:fill-green-900 stroke-green-500" strokeWidth="2" rx="4" />
            <text x="175" y="150" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">Conv2D</text>
            <text x="175" y="205" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">32 filters</text>
            <text x="175" y="220" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">3×3</text>
            <text x="175" y="235" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">ReLU</text>
            <rect x="140" y="270" width="70" height="30" className="fill-purple-100 dark:fill-purple-900 stroke-purple-500" strokeWidth="2" rx="4" />
            <text x="175" y="290" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">MaxPool 2×2</text>

            {/* 화살표 */}
            <path d="M 100 220 L 135 220" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrow)" />
            <defs>
              <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L0,6 L9,3 z" fill="#9ca3af" />
              </marker>
            </defs>

            {/* Conv2 + MaxPool */}
            <rect x="250" y="140" width="70" height="120" className="fill-green-100 dark:fill-green-900 stroke-green-500" strokeWidth="2" rx="4" />
            <text x="285" y="130" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">Conv2D</text>
            <text x="285" y="195" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">64 filters</text>
            <text x="285" y="210" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">3×3</text>
            <text x="285" y="225" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">ReLU</text>
            <rect x="250" y="270" width="70" height="30" className="fill-purple-100 dark:fill-purple-900 stroke-purple-500" strokeWidth="2" rx="4" />
            <text x="285" y="290" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">MaxPool 2×2</text>

            <path d="M 210 220 L 245 220" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrow)" />

            {/* Conv3 + MaxPool */}
            <rect x="360" y="120" width="70" height="140" className="fill-green-100 dark:fill-green-900 stroke-green-500" strokeWidth="2" rx="4" />
            <text x="395" y="110" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">Conv2D</text>
            <text x="395" y="185" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">128 filters</text>
            <text x="395" y="200" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">3×3</text>
            <text x="395" y="215" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">ReLU</text>
            <rect x="360" y="270" width="70" height="30" className="fill-purple-100 dark:fill-purple-900 stroke-purple-500" strokeWidth="2" rx="4" />
            <text x="395" y="290" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">MaxPool 2×2</text>

            <path d="M 320 220 L 355 220" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrow)" />

            {/* Flatten */}
            <rect x="470" y="190" width="70" height="40" className="fill-yellow-100 dark:fill-yellow-900 stroke-yellow-500" strokeWidth="2" rx="4" />
            <text x="505" y="180" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">Flatten</text>
            <text x="505" y="215" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">1D 벡터</text>

            <path d="M 430 220 L 465 220" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrow)" />

            {/* Dense 128 */}
            <rect x="580" y="170" width="70" height="80" className="fill-orange-100 dark:fill-orange-900 stroke-orange-500" strokeWidth="2" rx="4" />
            <text x="615" y="160" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">Dense</text>
            <text x="615" y="205" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">128 units</text>
            <text x="615" y="220" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">ReLU</text>
            <text x="615" y="235" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">Dropout 0.5</text>

            <path d="M 540 215 L 575 215" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrow)" />

            {/* Output Dense 7 */}
            <rect x="690" y="170" width="70" height="80" className="fill-red-100 dark:fill-red-900 stroke-red-500" strokeWidth="2" rx="4" />
            <text x="725" y="160" textAnchor="middle" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">Dense</text>
            <text x="725" y="205" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">7 units</text>
            <text x="725" y="220" textAnchor="middle" className="fill-gray-600 dark:fill-gray-400 text-xs">Softmax</text>

            <path d="M 650 215 L 685 215" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrow)" />

            {/* 출력 감정 */}
            <g transform="translate(800, 140)">
              <text x="0" y="0" className="fill-gray-700 dark:fill-gray-300 text-xs font-semibold">감정 확률</text>
              <text x="0" y="20" className="fill-gray-600 dark:fill-gray-400 text-xs">😊 0.75</text>
              <text x="0" y="35" className="fill-gray-600 dark:fill-gray-400 text-xs">😢 0.05</text>
              <text x="0" y="50" className="fill-gray-600 dark:fill-gray-400 text-xs">😠 0.10</text>
              <text x="0" y="65" className="fill-gray-600 dark:fill-gray-400 text-xs">😱 0.03</text>
              <text x="0" y="80" className="fill-gray-600 dark:fill-gray-400 text-xs">😲 0.02</text>
              <text x="0" y="95" className="fill-gray-600 dark:fill-gray-400 text-xs">🤢 0.01</text>
              <text x="0" y="110" className="fill-gray-600 dark:fill-gray-400 text-xs">😐 0.04</text>
            </g>

            <path d="M 760 215 L 795 215" className="stroke-gray-400" strokeWidth="2" markerEnd="url(#arrow)" />

            {/* 하단 설명 */}
            <g transform="translate(100, 350)">
              <text x="0" y="0" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">특징 추출 (Convolution)</text>
              <rect x="-10" y="10" width="250" height="2" className="fill-green-500" />
            </g>

            <g transform="translate(500, 350)">
              <text x="0" y="0" className="fill-gray-700 dark:fill-gray-300 text-sm font-semibold">분류 (Classification)</text>
              <rect x="-10" y="10" width="200" height="2" className="fill-orange-500" />
            </g>

            {/* 파라미터 정보 */}
            <g transform="translate(50, 420)">
              <text x="0" y="0" className="fill-gray-600 dark:fill-gray-400 text-xs font-semibold">✓ 총 파라미터: ~1.5M</text>
              <text x="200" y="0" className="fill-gray-600 dark:fill-gray-400 text-xs font-semibold">✓ 학습 데이터: FER2013 (35K 이미지)</text>
              <text x="500" y="0" className="fill-gray-600 dark:fill-gray-400 text-xs font-semibold">✓ 정확도: ~65-70%</text>
            </g>
          </svg>
        </div>

        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Python - CNN 감정 인식 (Keras)</span>
            </div>
            <button
              onClick={() => copyToClipboard(`import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np

# 감정 클래스
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 간단한 CNN 모델 정의
def create_emotion_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')  # 7개 감정 클래스
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 모델 로드 (미리 학습된 가중치)
model = create_emotion_model()
model.load_weights('emotion_model.h5')

# 실시간 감정 인식
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # 얼굴 영역 추출 및 전처리
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # 감정 예측
        prediction = model.predict(face_roi)
        emotion_idx = np.argmax(prediction)
        emotion = emotions[emotion_idx]
        confidence = prediction[0][emotion_idx]

        # 결과 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{emotion}: {confidence:.2f}',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()`, 'emotion-rec-1')}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
            >
              {copiedCode === 'emotion-rec-1' ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              )}
            </button>
          </div>
          <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
            <code>{`import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np

# 감정 클래스
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 간단한 CNN 모델 정의
def create_emotion_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')  # 7개 감정 클래스
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 모델 로드 (미리 학습된 가중치)
model = create_emotion_model()
model.load_weights('emotion_model.h5')

# 실시간 감정 인식
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # 얼굴 영역 추출 및 전처리
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # 감정 예측
        prediction = model.predict(face_roi)
        emotion_idx = np.argmax(prediction)
        emotion = emotions[emotion_idx]
        confidence = prediction[0][emotion_idx]

        # 결과 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{emotion}: {confidence:.2f}',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()`}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실습 과제</h2>
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-6">
          <div className="flex items-start gap-3 mb-4">
            <Lightbulb className="w-6 h-6 text-purple-600 dark:text-purple-400 flex-shrink-0 mt-1" />
            <div className="space-y-3">
              <div>
                <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">1. 얼굴 검출 비교 실습</h4>
                <p className="text-sm text-purple-800 dark:text-purple-200">
                  Haar Cascade, dlib, MTCNN 세 가지 방법으로 같은 이미지에서 얼굴을 검출하고 성능을 비교해보세요.
                  검출 속도와 정확도를 측정하세요.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">2. 얼굴 인식 시스템 구축</h4>
                <p className="text-sm text-purple-800 dark:text-purple-200">
                  5명의 얼굴 데이터베이스를 구축하고, FaceNet을 사용하여 새로운 얼굴 이미지를 인식하는 시스템을 만들어보세요.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">3. 실시간 감정 인식 앱</h4>
                <p className="text-sm text-purple-800 dark:text-purple-200">
                  웹캠을 사용하여 실시간으로 감정을 인식하고, 감정별 통계를 시각화하는 애플리케이션을 개발해보세요.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">4. ArcFace Loss 구현</h4>
                <p className="text-sm text-purple-800 dark:text-purple-200">
                  PyTorch로 ArcFace Loss를 구현하고, 간단한 얼굴 인식 모델을 학습시켜보세요.
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">5. 얼굴 랜드마크 응용</h4>
                <p className="text-sm text-purple-800 dark:text-purple-200">
                  68개 얼굴 랜드마크를 사용하여 가상 안경이나 마스크를 씌우는 AR 필터를 만들어보세요.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <References
        sections={[
          {
            title: 'Face Detection',
            icon: 'paper' as const,
            color: 'border-teal-500',
            items: [
              {
                title: 'Rapid Object Detection (Viola-Jones)',
                authors: 'Paul Viola, Michael Jones',
                year: '2001',
                description: 'Haar Cascade - 최초의 실시간 얼굴 검출 (50,000+ 인용)',
                link: 'https://ieeexplore.ieee.org/document/990517'
              },
              {
                title: 'Joint Face Detection and Alignment (MTCNN)',
                authors: 'Kaipeng Zhang, et al.',
                year: '2016',
                description: 'Multi-task CNN - 검출+랜드마크 동시 수행',
                link: 'https://arxiv.org/abs/1604.02878'
              },
              {
                title: 'RetinaFace: Single-stage Dense Face Localisation',
                authors: 'Jiankang Deng, et al.',
                year: '2019',
                description: '다중 스케일 얼굴 검출 - WIDER FACE 1위',
                link: 'https://arxiv.org/abs/1905.00641'
              }
            ]
          },
          {
            title: 'Face Recognition',
            icon: 'paper' as const,
            color: 'border-cyan-500',
            items: [
              {
                title: 'DeepFace: Closing the Gap to Human-Level Performance',
                authors: 'Yaniv Taigman, et al.',
                year: '2014',
                description: 'Facebook - 인간 수준 97.35% 정확도 달성',
                link: 'https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf'
              },
              {
                title: 'FaceNet: A Unified Embedding',
                authors: 'Florian Schroff, et al.',
                year: '2015',
                description: 'Triplet Loss - 128차원 얼굴 임베딩 (30,000+ 인용)',
                link: 'https://arxiv.org/abs/1503.03832'
              },
              {
                title: 'ArcFace: Additive Angular Margin Loss',
                authors: 'Jiankang Deng, et al.',
                year: '2019',
                description: 'Angular Softmax - SOTA 얼굴 인식 정확도',
                link: 'https://arxiv.org/abs/1801.07698'
              },
              {
                title: 'CosFace: Large Margin Cosine Loss',
                authors: 'Hao Wang, et al.',
                year: '2018',
                description: 'Cosine Margin - 클래스 간 분리 극대화',
                link: 'https://arxiv.org/abs/1801.09414'
              }
            ]
          },
          {
            title: 'Emotion Recognition',
            icon: 'paper' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'Facial Action Coding System (FACS)',
                authors: 'Paul Ekman, Wallace V. Friesen',
                year: '1978',
                description: '얼굴 동작 코딩 시스템 - 감정 인식의 과학적 기초',
                link: 'https://www.paulekman.com/facial-action-coding-system/'
              },
              {
                title: 'EmotionNet: Deep Learning for Facial Emotion',
                authors: 'Gil Levi, Tal Hassner',
                year: '2015',
                description: 'CNN 기반 감정 인식 - FER2013 데이터셋',
                link: 'https://arxiv.org/abs/1512.00567'
              }
            ]
          },
          {
            title: 'Datasets & Benchmarks',
            icon: 'paper' as const,
            color: 'border-indigo-500',
            items: [
              {
                title: 'Labeled Faces in the Wild (LFW)',
                authors: 'Gary B. Huang, et al.',
                year: '2007',
                description: '13,000 얼굴 - 얼굴 인식 벤치마크의 표준',
                link: 'http://vis-www.cs.umass.edu/lfw/'
              },
              {
                title: 'WIDER FACE Dataset',
                authors: 'Shuo Yang, et al.',
                year: '2016',
                description: '32,000 이미지, 400K 얼굴 - 다양한 스케일/포즈',
                link: 'http://shuoyang1213.me/WIDERFACE/'
              },
              {
                title: 'VGGFace2',
                authors: 'Qiong Cao, et al.',
                year: '2018',
                description: '9,000명, 3.3M 이미지 - 대규모 얼굴 인식',
                link: 'https://arxiv.org/abs/1710.08092'
              }
            ]
          },
          {
            title: 'Tools & Libraries',
            icon: 'web' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'dlib Face Recognition',
                authors: 'Davis King',
                year: '2024',
                description: 'C++ 라이브러리 - 68점 랜드마크, 얼굴 인식',
                link: 'http://dlib.net/'
              },
              {
                title: 'face_recognition (Python)',
                authors: 'Adam Geitgey',
                year: '2024',
                description: 'dlib 래퍼 - 가장 쉬운 얼굴 인식 라이브러리',
                link: 'https://github.com/ageitgey/face_recognition'
              },
              {
                title: 'InsightFace',
                authors: 'InsightFace Team',
                year: '2024',
                description: 'SOTA 얼굴 분석 - ArcFace, RetinaFace 구현',
                link: 'https://github.com/deepinsight/insightface'
              },
              {
                title: 'DeepFace (Python)',
                authors: 'Serengil, Ozpinar',
                year: '2024',
                description: 'VGGFace, Facenet, ArcFace 등 통합 인터페이스',
                link: 'https://github.com/serengil/deepface'
              }
            ]
          }
        ]}
      />
    </div>
  );
}