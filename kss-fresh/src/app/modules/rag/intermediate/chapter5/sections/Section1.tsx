'use client'

import { Image } from 'lucide-react'

export default function Section1() {
  return (
    <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-violet-100 dark:bg-violet-900/20 flex items-center justify-center">
          <Image className="text-violet-600" size={24} />
        </div>
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.1 CLIP 기반 이미지-텍스트 RAG</h2>
          <p className="text-gray-600 dark:text-gray-400">시각적 정보와 텍스트 통합 검색</p>
        </div>
      </div>

      <div className="space-y-6">
        <div className="bg-violet-50 dark:bg-violet-900/20 p-6 rounded-xl border border-violet-200 dark:border-violet-700">
          <h3 className="font-bold text-violet-800 dark:text-violet-200 mb-4">CLIP 기반 멀티모달 검색 시스템</h3>

          <div className="prose prose-sm dark:prose-invert mb-4">
            <p className="text-gray-700 dark:text-gray-300">
              <strong>CLIP(Contrastive Language-Image Pre-training)은 텍스트와 이미지를 동일한 벡터 공간에 매핑하는 혁신적인 기술입니다.</strong>
              이를 통해 텍스트로 이미지를 검색하거나, 이미지로 텍스트를 검색하는 크로스모달 검색이 가능해집니다.
            </p>
            <p className="text-gray-700 dark:text-gray-300">
              <strong>멀티모달 RAG의 핵심 장점:</strong>
            </p>
            <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
              <li><strong>통합 검색</strong>: 하나의 쿼리로 텍스트, 이미지, 비디오 등 모든 타입 검색</li>
              <li><strong>시각적 이해</strong>: 차트, 다이어그램, 스크린샷의 내용까지 이해</li>
              <li><strong>컨텍스트 융합</strong>: 시각적 정보와 텍스트 정보를 결합한 풍부한 답변</li>
              <li><strong>제로샷 성능</strong>: 학습하지 않은 새로운 카테고리도 검색 가능</li>
            </ul>
          </div>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`import torch
import clip
from PIL import Image
import requests
from typing import List, Dict, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class MultimodalRAGSystem:
    def __init__(self, clip_model_name: str = "ViT-B/32"):
        # CLIP 모델 로드
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)

        # 텍스트 전용 임베딩 모델
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

        # 벡터 저장소
        self.image_index = None
        self.text_index = None
        self.image_metadata = []
        self.text_metadata = []

        # 통합 검색을 위한 가중치
        self.image_weight = 0.6
        self.text_weight = 0.4

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """이미지들을 CLIP으로 인코딩"""
        embeddings = []

        for image_path in image_paths:
            try:
                # 이미지 로드 및 전처리
                if image_path.startswith('http'):
                    image = Image.open(requests.get(image_path, stream=True).raw)
                else:
                    image = Image.open(image_path)

                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

                # CLIP으로 인코딩
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                embeddings.append(image_features.cpu().numpy().flatten())

            except Exception as e:
                print(f"이미지 인코딩 실패 {image_path}: {e}")
                # 영벡터로 대체
                embeddings.append(np.zeros(512))

        return np.array(embeddings)

    def encode_text_with_clip(self, texts: List[str]) -> np.ndarray:
        """텍스트를 CLIP으로 인코딩 (이미지와 동일한 공간)"""
        embeddings = []

        for text in texts:
            try:
                text_tokens = clip.tokenize([text]).to(self.device)

                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                embeddings.append(text_features.cpu().numpy().flatten())

            except Exception as e:
                print(f"텍스트 인코딩 실패 {text}: {e}")
                embeddings.append(np.zeros(512))

        return np.array(embeddings)

    def build_multimodal_index(self,
                             image_data: List[Dict],
                             text_data: List[Dict]):
        """멀티모달 인덱스 구축"""
        print("멀티모달 인덱스 구축 시작...")

        # 이미지 데이터 처리
        if image_data:
            image_paths = [item['path'] for item in image_data]
            image_embeddings = self.encode_images(image_paths)

            # FAISS 인덱스 생성 (이미지)
            dimension = image_embeddings.shape[1]
            self.image_index = faiss.IndexFlatIP(dimension)  # 내적 유사도
            self.image_index.add(image_embeddings.astype('float32'))
            self.image_metadata = image_data

        # 텍스트 데이터 처리
        if text_data:
            texts = [item['content'] for item in text_data]
            # CLIP과 전통적 텍스트 임베딩 모두 생성
            clip_text_embeddings = self.encode_text_with_clip(texts)

            # 텍스트 인덱스는 CLIP 임베딩 사용 (이미지와 호환)
            if clip_text_embeddings.size > 0:
                dimension = clip_text_embeddings.shape[1]
                self.text_index = faiss.IndexFlatIP(dimension)
                self.text_index.add(clip_text_embeddings.astype('float32'))
                self.text_metadata = text_data

        print("멀티모달 인덱스 구축 완료")

    def multimodal_search(self,
                         query: Union[str, Image.Image],
                         k: int = 10,
                         search_images: bool = True,
                         search_texts: bool = True) -> Dict:
        """멀티모달 검색"""
        results = {
            'query_type': 'text' if isinstance(query, str) else 'image',
            'image_results': [],
            'text_results': [],
            'combined_results': []
        }

        # 쿼리 인코딩
        if isinstance(query, str):
            # 텍스트 쿼리: CLIP으로 인코딩
            query_embedding = self.encode_text_with_clip([query])[0]
        else:
            # 이미지 쿼리: CLIP으로 인코딩
            query_embedding = self.encode_images([query])[0]

        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # 이미지 검색
        if search_images and self.image_index is not None:
            scores, indices = self.image_index.search(query_embedding, k)

            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # 유효한 인덱스
                    result = {
                        'type': 'image',
                        'score': float(score),
                        'metadata': self.image_metadata[idx],
                        'content_type': 'image'
                    }
                    results['image_results'].append(result)

        # 텍스트 검색
        if search_texts and self.text_index is not None:
            scores, indices = self.text_index.search(query_embedding, k)

            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    result = {
                        'type': 'text',
                        'score': float(score),
                        'metadata': self.text_metadata[idx],
                        'content_type': 'text'
                    }
                    results['text_results'].append(result)

        # 결과 통합 및 정렬
        all_results = []

        # 이미지 결과에 가중치 적용
        for result in results['image_results']:
            result['weighted_score'] = result['score'] * self.image_weight
            all_results.append(result)

        # 텍스트 결과에 가중치 적용
        for result in results['text_results']:
            result['weighted_score'] = result['score'] * self.text_weight
            all_results.append(result)

        # 점수순 정렬
        all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        results['combined_results'] = all_results[:k]

        return results

    def visual_qa(self, image_path: str, question: str, k: int = 5) -> Dict:
        """이미지 기반 질문 답변"""
        # 이미지와 질문을 모두 사용한 컨텍스트 검색
        image_query_results = self.multimodal_search(
            Image.open(image_path), k=k//2
        )

        text_query_results = self.multimodal_search(
            question, k=k//2
        )

        # 컨텍스트 준비
        context_items = []

        # 이미지 관련 컨텍스트
        for result in image_query_results['combined_results']:
            if result['type'] == 'text':
                context_items.append({
                    'content': result['metadata']['content'],
                    'source': 'similar_content',
                    'score': result['score']
                })

        # 질문 관련 컨텍스트
        for result in text_query_results['combined_results']:
            context_items.append({
                'content': result['metadata']['content'],
                'source': 'question_related',
                'score': result['score']
            })

        return {
            'image_path': image_path,
            'question': question,
            'context': context_items[:k],
            'retrieval_method': 'multimodal_rag'
        }

# 사용 예시
multimodal_rag = MultimodalRAGSystem()

# 데이터 준비
image_data = [
    {
        'path': '/path/to/chart.png',
        'caption': '2023년 AI 모델 성능 비교 차트',
        'category': 'data_visualization'
    },
    {
        'path': '/path/to/architecture.png',
        'caption': 'Transformer 아키텍처 다이어그램',
        'category': 'technical_diagram'
    }
]

text_data = [
    {
        'content': 'GPT-4는 대규모 언어모델로 다양한 태스크에서 뛰어난 성능을 보입니다.',
        'title': 'GPT-4 개요',
        'category': 'ai_models'
    },
    {
        'content': 'Transformer는 어텐션 메커니즘을 기반으로 한 신경망 아키텍처입니다.',
        'title': 'Transformer 소개',
        'category': 'deep_learning'
    }
]

# 인덱스 구축
multimodal_rag.build_multimodal_index(image_data, text_data)

# 텍스트로 이미지 검색
text_query_results = multimodal_rag.multimodal_search(
    "AI 모델 성능 비교",
    k=5,
    search_images=True,
    search_texts=True
)

print("텍스트 쿼리 결과:")
for result in text_query_results['combined_results']:
    print(f"타입: {result['type']}, 점수: {result['weighted_score']:.3f}")

# 시각적 질문 답변
# vqa_result = multimodal_rag.visual_qa("chart.png", "이 차트에서 가장 성능이 좋은 모델은?")
# print("\\nVisual QA 결과:")
# print(f"컨텍스트 개수: {len(vqa_result['context'])}")`}
            </pre>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-200 dark:border-blue-700">
          <h3 className="font-bold text-blue-800 dark:text-blue-200 mb-4">이미지 메타데이터 추출</h3>

          <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`class ImageMetadataExtractor:
    def __init__(self):
        # OCR을 위한 라이브러리
        import easyocr
        self.ocr_reader = easyocr.Reader(['ko', 'en'])

        # 객체 감지 모델 (YOLO)
        from ultralytics import YOLO
        self.yolo_model = YOLO('yolov8n.pt')

    def extract_comprehensive_metadata(self, image_path: str) -> Dict:
        """이미지에서 포괄적 메타데이터 추출"""
        metadata = {
            'path': image_path,
            'text_content': '',
            'objects': [],
            'colors': [],
            'technical_info': {},
            'clip_description': ''
        }

        image = Image.open(image_path)

        # 1. OCR로 텍스트 추출
        try:
            ocr_results = self.ocr_reader.readtext(image_path)
            texts = [result[1] for result in ocr_results if result[2] > 0.5]
            metadata['text_content'] = ' '.join(texts)
        except Exception as e:
            print(f"OCR 실패: {e}")

        # 2. 객체 감지
        try:
            results = self.yolo_model(image_path)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.yolo_model.names[class_id]

                        if confidence > 0.5:
                            metadata['objects'].append({
                                'class': class_name,
                                'confidence': confidence
                            })
        except Exception as e:
            print(f"객체 감지 실패: {e}")

        # 3. 색상 분석
        try:
            import cv2
            img_cv = cv2.imread(image_path)
            dominant_colors = self.extract_dominant_colors(img_cv)
            metadata['colors'] = dominant_colors
        except Exception as e:
            print(f"색상 분석 실패: {e}")

        # 4. 기술적 정보
        metadata['technical_info'] = {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': image.format
        }

        return metadata

    def extract_dominant_colors(self, image, k=5):
        """K-means로 주요 색상 추출"""
        import cv2
        from sklearn.cluster import KMeans

        # 이미지를 1D 배열로 변환
        data = image.reshape((-1, 3))
        data = np.float32(data)

        # K-means 클러스터링
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # RGB 값을 색상 이름으로 변환 (간단한 매핑)
        colors = []
        for center in centers:
            rgb = tuple(map(int, center))
            color_name = self.rgb_to_color_name(rgb)
            colors.append({
                'rgb': rgb,
                'name': color_name
            })

        return colors

    def rgb_to_color_name(self, rgb):
        """RGB 값을 색상 이름으로 변환 (간단한 휴리스틱)"""
        r, g, b = rgb

        # 기본 색상 구분
        if r > 200 and g > 200 and b > 200:
            return '흰색'
        elif r < 50 and g < 50 and b < 50:
            return '검은색'
        elif r > g and r > b:
            return '빨간색'
        elif g > r and g > b:
            return '초록색'
        elif b > r and b > g:
            return '파란색'
        elif r > 150 and g > 150:
            return '노란색'
        else:
            return '기타색상'

# 사용 예시
extractor = ImageMetadataExtractor()
metadata = extractor.extract_comprehensive_metadata('/path/to/image.jpg')

print("추출된 메타데이터:")
print(f"텍스트: {metadata['text_content']}")
print(f"감지된 객체: {len(metadata['objects'])}개")
print(f"주요 색상: {[c['name'] for c in metadata['colors']]}")`}
            </pre>
          </div>
        </div>
      </div>
    </section>
  )
}
