'use client'

import Link from 'next/link'
import { ArrowLeft, ArrowRight, Image, Video, AudioLines, Table, Layout, Shuffle } from 'lucide-react'
import References from '@/components/common/References'

export default function Chapter5Page() {
  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/modules/rag/intermediate"
          className="inline-flex items-center gap-2 text-emerald-600 hover:text-emerald-700 mb-4 transition-colors"
        >
          <ArrowLeft size={20} />
          중급 과정으로 돌아가기
        </Link>
        
        <div className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-2xl p-8 text-white">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-xl bg-white/20 flex items-center justify-center">
              <Image size={32} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Chapter 5: 멀티모달 RAG</h1>
              <p className="text-violet-100 text-lg">이미지, 비디오, 오디오 데이터를 활용한 고급 RAG</p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {/* Section 1: Image-Text RAG with CLIP */}
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

        {/* Section 2: Video Search and Summarization */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
              <Video className="text-red-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.2 비디오 검색 및 요약</h2>
              <p className="text-gray-600 dark:text-gray-400">시간 기반 멀티미디어 컨텐츠 처리</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-xl border border-red-200 dark:border-red-700">
              <h3 className="font-bold text-red-800 dark:text-red-200 mb-4">비디오 RAG 시스템</h3>
              
              <div className="prose prose-sm dark:prose-invert mb-4">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>비디오 RAG는 시간 순서가 있는 멀티미디어 콘텐츠를 처리하는 고급 기술입니다.</strong> 
                  기존 텍스트 RAG와 달리 프레임 시퀀스, 오디오 트랙, 자막 등 다층적 정보를 통합 처리해야 합니다.
                </p>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>비디오 RAG 핵심 과정:</strong>
                </p>
                <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
                  <li><strong>키프레임 추출</strong>: 10초 간격으로 대표 프레임 샘플링 (1시간→360프레임)</li>
                  <li><strong>이미지 캡셔닝</strong>: BLIP 모델로 각 프레임을 텍스트로 변환</li>
                  <li><strong>시맨틱 세그먼테이션</strong>: 의미 단위로 비디오를 60초 구간 분할</li>
                  <li><strong>시간 기반 검색</strong>: 쿼리와 관련된 정확한 타임스탬프 반환</li>
                </ul>
                
                <div className="bg-amber-50 dark:bg-amber-900/20 p-4 rounded-lg border border-amber-200 dark:border-amber-700 mt-4">
                  <h4 className="font-bold text-amber-800 dark:text-amber-200 mb-2">💡 실무 활용 사례</h4>
                  <div className="grid md:grid-cols-2 gap-3 text-sm">
                    <div>
                      <strong>온라인 교육</strong>
                      <ul className="list-disc list-inside ml-2 text-amber-700 dark:text-amber-300">
                        <li>강의 내용 자동 인덱싱</li>
                        <li>"경사하강법 설명" → 정확한 구간 점프</li>
                        <li>자동 퀴즈 및 요약 생성</li>
                      </ul>
                    </div>
                    <div>
                      <strong>미디어 검색</strong>
                      <ul className="list-disc list-inside ml-2 text-amber-700 dark:text-amber-300">
                        <li>뉴스 아카이브 검색</li>
                        <li>"코로나 관련 보도" → 관련 영상 클립</li>
                        <li>광고 콘텐츠 자동 태깅</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
                <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Dict, Tuple
import numpy as np
from datetime import timedelta

class VideoRAGSystem:
    def __init__(self):
        # 이미지 캡셔닝 모델 (비디오 프레임 설명용)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # 비디오 세그먼트 저장소
        self.video_segments = []
        self.segment_embeddings = []
        
    def extract_keyframes(self, video_path: str, interval_seconds: int = 10) -> List[Dict]:
        """비디오에서 키프레임 추출"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_seconds)
        
        keyframes = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # 프레임을 PIL 이미지로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # 타임스탬프 계산
                timestamp_seconds = frame_count / fps
                timestamp = str(timedelta(seconds=int(timestamp_seconds)))
                
                keyframes.append({
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'timestamp_seconds': timestamp_seconds,
                    'image': pil_image
                })
            
            frame_count += 1
        
        cap.release()
        return keyframes
    
    def generate_frame_captions(self, keyframes: List[Dict]) -> List[Dict]:
        """키프레임에 대한 캡션 생성"""
        captioned_frames = []
        
        for frame_data in keyframes:
            try:
                # BLIP으로 이미지 캡션 생성
                inputs = self.processor(frame_data['image'], return_tensors="pt")
                
                with torch.no_grad():
                    out = self.captioning_model.generate(**inputs, max_length=50)
                
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                
                captioned_frames.append({
                    'timestamp': frame_data['timestamp'],
                    'timestamp_seconds': frame_data['timestamp_seconds'],
                    'caption': caption,
                    'frame_number': frame_data['frame_number']
                })
                
            except Exception as e:
                print(f"캡션 생성 실패 (프레임 {frame_data['frame_number']}): {e}")
                captioned_frames.append({
                    'timestamp': frame_data['timestamp'],
                    'timestamp_seconds': frame_data['timestamp_seconds'],
                    'caption': '캡션 생성 실패',
                    'frame_number': frame_data['frame_number']
                })
        
        return captioned_frames
    
    def create_video_segments(self, video_path: str, 
                            captioned_frames: List[Dict],
                            segment_duration: int = 60) -> List[Dict]:
        """비디오를 의미있는 세그먼트로 분할"""
        segments = []
        current_segment = {
            'start_time': 0,
            'end_time': segment_duration,
            'captions': [],
            'key_topics': []
        }
        
        for frame in captioned_frames:
            timestamp = frame['timestamp_seconds']
            
            # 현재 세그먼트에 속하는지 확인
            if current_segment['start_time'] <= timestamp < current_segment['end_time']:
                current_segment['captions'].append(frame['caption'])
            else:
                # 현재 세그먼트 완료 및 저장
                if current_segment['captions']:
                    current_segment['summary'] = self.summarize_segment_captions(
                        current_segment['captions']
                    )
                    segments.append(current_segment.copy())
                
                # 새 세그먼트 시작
                current_segment = {
                    'start_time': int(timestamp // segment_duration) * segment_duration,
                    'end_time': int(timestamp // segment_duration + 1) * segment_duration,
                    'captions': [frame['caption']],
                    'key_topics': []
                }
        
        # 마지막 세그먼트 처리
        if current_segment['captions']:
            current_segment['summary'] = self.summarize_segment_captions(
                current_segment['captions']
            )
            segments.append(current_segment)
        
        return segments
    
    def summarize_segment_captions(self, captions: List[str]) -> str:
        """세그먼트 캡션들을 요약"""
        if not captions:
            return "내용 없음"
        
        # 간단한 요약 로직 (실제로는 더 고급 요약 모델 사용)
        caption_text = ' '.join(captions)
        
        # 키워드 추출 (빈도 기반)
        words = caption_text.lower().split()
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # 3글자 이상 단어만
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 상위 키워드들로 요약 구성
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = [word for word, freq in top_keywords]
        
        summary = f"주요 내용: {', '.join(keywords)}"
        
        # 첫 번째와 마지막 캡션도 포함
        if len(captions) > 1:
            summary += f". 시작: {captions[0][:50]}... 끝: {captions[-1][:50]}..."
        
        return summary
    
    def search_video_content(self, query: str, video_segments: List[Dict]) -> List[Dict]:
        """비디오 컨텐츠에서 쿼리 관련 부분 검색"""
        results = []
        
        for segment in video_segments:
            # 세그먼트 요약과 쿼리의 유사도 계산 (간단한 키워드 매칭)
            segment_text = segment['summary'] + ' ' + ' '.join(segment['captions'])
            
            # 키워드 기반 점수 계산
            query_words = query.lower().split()
            segment_words = segment_text.lower().split()
            
            matches = sum(1 for word in query_words if word in segment_words)
            score = matches / len(query_words) if query_words else 0
            
            if score > 0.3:  # 30% 이상 매칭시 결과에 포함
                results.append({
                    'segment': segment,
                    'score': score,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'relevant_captions': [
                        cap for cap in segment['captions'] 
                        if any(word in cap.lower() for word in query_words)
                    ]
                })
        
        # 점수순 정렬
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def process_video(self, video_path: str) -> Dict:
        """비디오 전체 처리 파이프라인"""
        print("1. 키프레임 추출 중...")
        keyframes = self.extract_keyframes(video_path, interval_seconds=15)
        
        print("2. 프레임 캡션 생성 중...")
        captioned_frames = self.generate_frame_captions(keyframes)
        
        print("3. 비디오 세그먼트 생성 중...")
        video_segments = self.create_video_segments(video_path, captioned_frames)
        
        return {
            'video_path': video_path,
            'total_keyframes': len(keyframes),
            'total_segments': len(video_segments),
            'segments': video_segments,
            'processing_complete': True
        }

# 사용 예시
video_rag = VideoRAGSystem()

# 비디오 처리
# video_data = video_rag.process_video('/path/to/lecture_video.mp4')

# print(f"처리 완료: {video_data['total_segments']}개 세그먼트")

# 비디오에서 검색
# search_results = video_rag.search_video_content(
#     "machine learning", 
#     video_data['segments']
# )

# print("\\n검색 결과:")
# for result in search_results[:3]:
#     segment = result['segment']
#     print(f"시간: {segment['start_time']}s - {segment['end_time']}s")
#     print(f"점수: {result['score']:.3f}")
#     print(f"요약: {segment['summary'][:100]}...")
#     print()`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Audio/Speech Processing */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-green-100 dark:bg-green-900/20 flex items-center justify-center">
              <AudioLines className="text-green-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.3 오디오/음성 데이터 처리</h2>
              <p className="text-gray-600 dark:text-gray-400">음성 인식 및 오디오 기반 검색</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-green-50 dark:bg-green-900/20 p-6 rounded-xl border border-green-200 dark:border-green-700">
              <h3 className="font-bold text-green-800 dark:text-green-200 mb-4">오디오 RAG 시스템</h3>
              
              <div className="prose prose-sm dark:prose-invert mb-4">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>오디오 RAG는 음성 인식과 자연어 처리를 결합한 음향 기반 정보 검색 시스템입니다.</strong> 
                  OpenAI의 Whisper 모델을 활용하여 99% 정확도의 다국어 음성 인식을 구현하며, 
                  실시간 스트리밍과 배치 처리 모두를 지원합니다.
                </p>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>오디오 RAG 핵심 기술:</strong>
                </p>
                <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
                  <li><strong>Whisper ASR</strong>: 680,000시간 다국어 데이터로 훈련된 최고 성능 모델</li>
                  <li><strong>단어별 타임스탬프</strong>: 정확한 시간 동기화로 구간별 검색 가능</li>
                  <li><strong>MFCC 특성 추출</strong>: 13차원 멜-켑스트럼 계수로 음향 특성 분석</li>
                  <li><strong>화자 분리</strong>: 다중 화자 환경에서 개별 발언자 구분</li>
                </ul>
                
                <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg border border-emerald-200 dark:border-emerald-700 mt-4">
                  <h4 className="font-bold text-emerald-800 dark:text-emerald-200 mb-2">🎯 성능 벤치마크</h4>
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr className="border-b border-emerald-300 dark:border-emerald-600">
                          <th className="text-left py-2 text-emerald-800 dark:text-emerald-200">모델</th>
                          <th className="text-left py-2 text-emerald-800 dark:text-emerald-200">WER</th>
                          <th className="text-left py-2 text-emerald-800 dark:text-emerald-200">처리속도</th>
                          <th className="text-left py-2 text-emerald-800 dark:text-emerald-200">언어지원</th>
                        </tr>
                      </thead>
                      <tbody className="text-emerald-700 dark:text-emerald-300">
                        <tr>
                          <td className="py-1">Whisper-base</td>
                          <td className="py-1">5.1%</td>
                          <td className="py-1">~6x실시간</td>
                          <td className="py-1">99개 언어</td>
                        </tr>
                        <tr>
                          <td className="py-1">Whisper-large</td>
                          <td className="py-1">2.8%</td>
                          <td className="py-1">~2x실시간</td>
                          <td className="py-1">99개 언어</td>
                        </tr>
                        <tr>
                          <td className="py-1">Google STT</td>
                          <td className="py-1">4.2%</td>
                          <td className="py-1">실시간</td>
                          <td className="py-1">125개 언어</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-700 mt-4">
                  <h4 className="font-bold text-blue-800 dark:text-blue-200 mb-2">🔊 실제 적용 분야</h4>
                  <div className="grid md:grid-cols-3 gap-3 text-sm">
                    <div>
                      <strong className="text-blue-800 dark:text-blue-200">콜센터 분석</strong>
                      <ul className="list-disc list-inside ml-2 text-blue-700 dark:text-blue-300 mt-1">
                        <li>고객 불만 키워드 검출</li>
                        <li>상담원 스크립트 준수 체크</li>
                        <li>감정 분석 및 품질 관리</li>
                      </ul>
                    </div>
                    <div>
                      <strong className="text-blue-800 dark:text-blue-200">회의 스마트 요약</strong>
                      <ul className="list-disc list-inside ml-2 text-blue-700 dark:text-blue-300 mt-1">
                        <li>핵심 결정사항 자동 추출</li>
                        <li>액션 아이템 할당 추적</li>
                        <li>다음 회의 어젠다 생성</li>
                      </ul>
                    </div>
                    <div>
                      <strong className="text-blue-800 dark:text-blue-200">방송 아카이빙</strong>
                      <ul className="list-disc list-inside ml-2 text-blue-700 dark:text-blue-300 mt-1">
                        <li>뉴스 자동 색인 생성</li>
                        <li>인물 발언 검색 시스템</li>
                        <li>실시간 자막 및 번역</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
                <pre className="text-sm text-slate-800 dark:text-slate-200 overflow-x-auto max-h-96 overflow-y-auto font-mono">
{`import whisper
import torch
import librosa
import numpy as np
from typing import List, Dict
from datetime import timedelta
import re

class AudioRAGSystem:
    def __init__(self, whisper_model_size: str = "base"):
        # Whisper 모델 로드 (음성 인식용)
        self.whisper_model = whisper.load_model(whisper_model_size)
        
        # 오디오 세그먼트 저장
        self.audio_segments = []
        self.transcripts = []
        
    def transcribe_audio(self, audio_path: str, language: str = "ko") -> Dict:
        """오디오 파일 전사"""
        print(f"오디오 전사 시작: {audio_path}")
        
        try:
            # Whisper로 전사
            result = self.whisper_model.transcribe(
                audio_path, 
                language=language,
                word_timestamps=True  # 단어별 타임스탬프
            )
            
            return {
                'text': result['text'],
                'language': result['language'],
                'segments': result['segments'],
                'audio_path': audio_path,
                'success': True
            }
            
        except Exception as e:
            print(f"전사 실패: {e}")
            return {
                'text': '',
                'error': str(e),
                'success': False
            }
    
    def extract_audio_features(self, audio_path: str) -> Dict:
        """오디오 특성 추출"""
        try:
            # 오디오 로드
            y, sr = librosa.load(audio_path)
            
            # 기본 특성들
            features = {
                'duration': librosa.get_duration(y=y, sr=sr),
                'sample_rate': sr,
                'rms_energy': float(np.mean(librosa.feature.rms(y=y))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0])
            }
            
            # MFCC 특성 (음성 인식에 중요)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
            
            return features
            
        except Exception as e:
            print(f"특성 추출 실패: {e}")
            return {}
    
    def segment_transcript(self, transcript_data: Dict, segment_duration: int = 30) -> List[Dict]:
        """전사 텍스트를 의미있는 세그먼트로 분할"""
        segments = []
        
        if not transcript_data.get('segments'):
            return segments
        
        current_segment = {
            'start_time': 0,
            'end_time': segment_duration,
            'text': '',
            'words': [],
            'speaker_changes': 0
        }
        
        for segment in transcript_data['segments']:
            segment_start = segment['start']
            segment_end = segment['end']
            segment_text = segment['text'].strip()
            
            # 세그먼트 시간 범위 확인
            if segment_start < current_segment['end_time']:
                current_segment['text'] += ' ' + segment_text
                current_segment['words'].extend(segment.get('words', []))
            else:
                # 현재 세그먼트 저장
                if current_segment['text'].strip():
                    current_segment['word_count'] = len(current_segment['text'].split())
                    current_segment['summary'] = self.summarize_audio_segment(current_segment['text'])
                    segments.append(current_segment.copy())
                
                # 새 세그먼트 시작
                segment_num = int(segment_start // segment_duration)
                current_segment = {
                    'start_time': segment_num * segment_duration,
                    'end_time': (segment_num + 1) * segment_duration,
                    'text': segment_text,
                    'words': segment.get('words', []),
                    'speaker_changes': 0
                }
        
        # 마지막 세그먼트
        if current_segment['text'].strip():
            current_segment['word_count'] = len(current_segment['text'].split())
            current_segment['summary'] = self.summarize_audio_segment(current_segment['text'])
            segments.append(current_segment)
        
        return segments
    
    def summarize_audio_segment(self, text: str) -> str:
        """오디오 세그먼트 요약"""
        if len(text) < 50:
            return text
        
        # 문장 분리
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= 2:
            return text
        
        # 첫 문장과 마지막 문장으로 요약 구성
        summary = f"{sentences[0]}... {sentences[-1]}"
        return summary[:150]  # 150자로 제한
    
    def search_audio_content(self, query: str, audio_segments: List[Dict]) -> List[Dict]:
        """오디오 내용에서 검색"""
        results = []
        query_words = query.lower().split()
        
        for segment in audio_segments:
            segment_text = segment.get('text', '').lower()
            
            # 키워드 매칭 점수
            matches = sum(1 for word in query_words if word in segment_text)
            keyword_score = matches / len(query_words) if query_words else 0
            
            # 텍스트 포함 점수 (부분 문자열)
            substring_score = 0.5 if query.lower() in segment_text else 0
            
            # 최종 점수
            final_score = max(keyword_score, substring_score)
            
            if final_score > 0.2:  # 20% 이상 매칭시 포함
                # 관련 구문 추출
                relevant_phrases = self.extract_relevant_phrases(
                    segment_text, query_words
                )
                
                results.append({
                    'segment': segment,
                    'score': final_score,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'relevant_text': relevant_phrases,
                    'context': segment.get('summary', '')
                })
        
        # 점수순 정렬
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def extract_relevant_phrases(self, text: str, query_words: List[str], 
                                context_window: int = 50) -> List[str]:
        """쿼리 관련 구문 추출"""
        phrases = []
        text_lower = text.lower()
        
        for word in query_words:
            if word in text_lower:
                # 단어 위치 찾기
                start_pos = text_lower.find(word)
                
                # 컨텍스트 윈도우 적용
                start = max(0, start_pos - context_window)
                end = min(len(text), start_pos + len(word) + context_window)
                
                phrase = text[start:end].strip()
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
        
        return phrases[:3]  # 최대 3개 구문
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """오디오 파일 전체 처리"""
        print("1. 오디오 전사 중...")
        transcript = self.transcribe_audio(audio_path)
        
        if not transcript['success']:
            return {'error': '전사 실패', 'transcript': transcript}
        
        print("2. 오디오 특성 추출 중...")
        features = self.extract_audio_features(audio_path)
        
        print("3. 세그먼트 생성 중...")
        segments = self.segment_transcript(transcript, segment_duration=30)
        
        return {
            'audio_path': audio_path,
            'transcript': transcript,
            'features': features,
            'segments': segments,
            'total_duration': features.get('duration', 0),
            'segment_count': len(segments)
        }

# 사용 예시
audio_rag = AudioRAGSystem(whisper_model_size="small")

# 오디오 파일 처리
# audio_data = audio_rag.process_audio_file('/path/to/lecture.mp3')

# if 'error' not in audio_data:
#     print(f"처리 완료:")
#     print(f"  총 길이: {audio_data['total_duration']:.1f}초")
#     print(f"  세그먼트 수: {audio_data['segment_count']}개")
#     print(f"  전체 텍스트: {len(audio_data['transcript']['text'])}자")
#     
#     # 검색 테스트
#     results = audio_rag.search_audio_content("머신러닝", audio_data['segments'])
#     
#     print(f"\\n검색 결과: {len(results)}건")
#     for i, result in enumerate(results[:3]):
#         print(f"{i+1}. 시간: {result['start_time']}s-{result['end_time']}s")
#         print(f"   점수: {result['score']:.3f}")
#         print(f"   내용: {result['context'][:100]}...")
# else:
#     print(f"처리 실패: {audio_data['error']}")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: Table and Structured Data */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center">
              <Table className="text-indigo-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.4 테이블 및 구조화된 데이터 RAG</h2>
              <p className="text-gray-600 dark:text-gray-400">정형 데이터와 테이블 기반 검색</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-indigo-50 dark:bg-indigo-900/20 p-6 rounded-xl border border-indigo-200 dark:border-indigo-700">
              <h3 className="font-bold text-indigo-800 dark:text-indigo-200 mb-4">테이블 이해 및 검색 시스템</h3>
              
              <div className="prose prose-sm dark:prose-invert mb-4">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>테이블 RAG는 정형화된 데이터의 구조적 관계를 이해하고 복잡한 분석 질의를 처리합니다.</strong> 
                  기존 텍스트 기반 RAG가 처리하기 어려운 수치적 추론, 비교 분석, 트렌드 파악 등을 테이블의 행/열 구조를 활용해 정확히 수행합니다.
                </p>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>핵심 기술 구성요소:</strong>
                </p>
                <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
                  <li><strong>테이블 구조 파싱</strong>: 헤더 계층, 병합 셀, 서브 테이블 인식</li>
                  <li><strong>스키마 이해</strong>: 컬럼 타입 자동 추론 (숫자, 날짜, 카테고리)</li>
                  <li><strong>관계형 추론</strong>: 행 간 비교, 집계, 그룹핑 연산 지원</li>
                  <li><strong>자연어→SQL 변환</strong>: "가장 높은 매출을 기록한 월은?" → 구조화 쿼리</li>
                </ul>
                
                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-200 dark:border-purple-700 mt-4">
                  <h4 className="font-bold text-purple-800 dark:text-purple-200 mb-2">📈 성능 비교</h4>
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr className="border-b border-purple-300 dark:border-purple-600">
                          <th className="text-left py-2 text-purple-800 dark:text-purple-200">질의 타입</th>
                          <th className="text-left py-2 text-purple-800 dark:text-purple-200">텍스트 RAG</th>
                          <th className="text-left py-2 text-purple-800 dark:text-purple-200">테이블 RAG</th>
                          <th className="text-left py-2 text-purple-800 dark:text-purple-200">향상률</th>
                        </tr>
                      </thead>
                      <tbody className="text-purple-700 dark:text-purple-300">
                        <tr>
                          <td className="py-1">단순 사실 검색</td>
                          <td className="py-1">95%</td>
                          <td className="py-1">97%</td>
                          <td className="py-1">+2%</td>
                        </tr>
                        <tr>
                          <td className="py-1">수치 비교</td>
                          <td className="py-1">72%</td>
                          <td className="py-1">94%</td>
                          <td className="py-1">+31%</td>
                        </tr>
                        <tr>
                          <td className="py-1">집계 연산</td>
                          <td className="py-1">45%</td>
                          <td className="py-1">89%</td>
                          <td className="py-1">+98%</td>
                        </tr>
                        <tr>
                          <td className="py-1">트렌드 분석</td>
                          <td className="py-1">38%</td>
                          <td className="py-1">85%</td>
                          <td className="py-1">+124%</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
              
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">📊 처리 가능한 테이블 타입</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 재무제표 및 수치 데이터</li>
                    <li>• 제품 카탈로그 및 스펙</li>
                    <li>• 연구 결과 및 통계</li>
                    <li>• 일정 및 시간표</li>
                  </ul>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">🔍 검색 방식</h4>
                  <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                    <li>• 셀 값 기반 정확 매칭</li>
                    <li>• 컬럼 헤더 의미적 검색</li>
                    <li>• 수치 범위 및 조건 검색</li>
                    <li>• 행/열 관계 기반 추론</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 5: Layout-aware Processing */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
              <Layout className="text-purple-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.5 레이아웃 인식 문서 처리</h2>
              <p className="text-gray-600 dark:text-gray-400">문서 구조를 이해하는 고급 RAG</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-purple-50 dark:bg-purple-900/20 p-6 rounded-xl border border-purple-200 dark:border-purple-700">
              <h3 className="font-bold text-purple-800 dark:text-purple-200 mb-4">지능형 문서 레이아웃 분석</h3>
              
              <div className="prose prose-sm dark:prose-invert mb-4">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>레이아웃 인식 RAG는 문서의 시각적 구조와 의미적 위계를 이해하여 정보를 추출합니다.</strong> 
                  단순한 텍스트 추출을 넘어 제목-본문 관계, 이미지-캡션 연결, 표-설명 매핑 등 
                  문서 디자이너가 의도한 정보 구조를 AI가 정확히 파악합니다.
                </p>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>고급 레이아웃 분석 기법:</strong>
                </p>
                <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
                  <li><strong>LayoutLMv3</strong>: Microsoft의 2D 위치 정보 통합 언어모델</li>
                  <li><strong>DocFormer</strong>: 문서 이해를 위한 멀티모달 Transformer</li>
                  <li><strong>DETR</strong>: Object Detection으로 문서 요소 탐지</li>
                  <li><strong>OCR + 좌표 매핑</strong>: 텍스트 위치와 의미적 역할 연결</li>
                </ul>
                
                <div className="bg-indigo-50 dark:bg-indigo-900/20 p-4 rounded-lg border border-indigo-200 dark:border-indigo-700 mt-4">
                  <h4 className="font-bold text-indigo-800 dark:text-indigo-200 mb-2">🏗️ 문서 구조 인식 정확도</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-indigo-700 dark:text-indigo-300">제목 추출</span>
                        <span className="font-medium text-indigo-800 dark:text-indigo-200">95.2%</span>
                      </div>
                      <div className="w-full bg-indigo-200 dark:bg-indigo-700 rounded-full h-2">
                        <div className="bg-indigo-500 h-2 rounded-full" style={{width: '95.2%'}}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-indigo-700 dark:text-indigo-300">테이블 경계</span>
                        <span className="font-medium text-indigo-800 dark:text-indigo-200">92.7%</span>
                      </div>
                      <div className="w-full bg-indigo-200 dark:bg-indigo-700 rounded-full h-2">
                        <div className="bg-indigo-500 h-2 rounded-full" style={{width: '92.7%'}}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-indigo-700 dark:text-indigo-300">이미지-캡션</span>
                        <span className="font-medium text-indigo-800 dark:text-indigo-200">89.4%</span>
                      </div>
                      <div className="w-full bg-indigo-200 dark:bg-indigo-700 rounded-full h-2">
                        <div className="bg-indigo-500 h-2 rounded-full" style={{width: '89.4%'}}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-indigo-700 dark:text-indigo-300">리스트 구조</span>
                        <span className="font-medium text-indigo-800 dark:text-indigo-200">91.8%</span>
                      </div>
                      <div className="w-full bg-indigo-200 dark:bg-indigo-700 rounded-full h-2">
                        <div className="bg-indigo-500 h-2 rounded-full" style={{width: '91.8%'}}></div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-rose-50 dark:bg-rose-900/20 p-4 rounded-lg border border-rose-200 dark:border-rose-700 mt-4">
                  <h4 className="font-bold text-rose-800 dark:text-rose-200 mb-2">💼 실무 적용 효과</h4>
                  <div className="grid md:grid-cols-2 gap-3 text-sm">
                    <div>
                      <strong className="text-rose-800 dark:text-rose-200">법률 문서</strong>
                      <ul className="list-disc list-inside ml-2 text-rose-700 dark:text-rose-300 mt-1">
                        <li>조항별 자동 분류 및 인덱싱</li>
                        <li>판례-법조문 연결 구조 파악</li>
                        <li>법률 용어 정의 자동 추출</li>
                      </ul>
                    </div>
                    <div>
                      <strong className="text-rose-800 dark:text-rose-200">연구 논문</strong>
                      <ul className="list-disc list-inside ml-2 text-rose-700 dark:text-rose-300 mt-1">
                        <li>Figure-Table 자동 매핑</li>
                        <li>인용 관계 그래프 구축</li>
                        <li>결과 섹션 핵심 데이터 추출</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">📄 PDF 분석</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    헤더, 푸터, 멀티컬럼 레이아웃 인식
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">🖼️ 이미지 위치</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    이미지와 캡션의 관계 파악
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border text-center">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">📊 차트 해석</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    데이터 시각화 요소 추출
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 6: Cross-modal Retrieval */}
        <section className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-teal-100 dark:bg-teal-900/20 flex items-center justify-center">
              <Shuffle className="text-teal-600" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">5.6 크로스 모달 검색 전략</h2>
              <p className="text-gray-600 dark:text-gray-400">모달리티 간 연관성 활용한 고급 검색</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-teal-50 dark:bg-teal-900/20 p-6 rounded-xl border border-teal-200 dark:border-teal-700">
              <h3 className="font-bold text-teal-800 dark:text-teal-200 mb-4">통합 멀티모달 검색 전략</h3>
              
              <div className="prose prose-sm dark:prose-invert mb-4">
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>크로스모달 검색은 서로 다른 데이터 타입 간의 의미적 연결을 활용하여 통합된 검색 경험을 제공합니다.</strong> 
                  텍스트 쿼리로 이미지를 찾거나, 이미지로 관련 음성을 검색하는 등 
                  기존 단일 모달 검색의 한계를 뛰어넘는 혁신적 접근법입니다.
                </p>
                <p className="text-gray-700 dark:text-gray-300">
                  <strong>핵심 융합 전략:</strong>
                </p>
                <ul className="list-disc list-inside text-gray-700 dark:text-gray-300 space-y-1">
                  <li><strong>Feature-level Fusion</strong>: 임베딩 차원에서 특성 벡터 결합</li>
                  <li><strong>Score-level Fusion</strong>: 각 모달리티 검색 점수의 가중 평균</li>
                  <li><strong>Decision-level Fusion</strong>: 최종 결정 단계에서 결과 통합</li>
                  <li><strong>Adaptive Weighting</strong>: 쿼리 특성에 따른 동적 가중치 조정</li>
                </ul>
                
                <div className="bg-cyan-50 dark:bg-cyan-900/20 p-4 rounded-lg border border-cyan-200 dark:border-cyan-700 mt-4">
                  <h4 className="font-bold text-cyan-800 dark:text-cyan-200 mb-2">🎯 실험 결과 비교</h4>
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr className="border-b border-cyan-300 dark:border-cyan-600">
                          <th className="text-left py-2 text-cyan-800 dark:text-cyan-200">검색 방식</th>
                          <th className="text-left py-2 text-cyan-800 dark:text-cyan-200">텍스트→이미지</th>
                          <th className="text-left py-2 text-cyan-800 dark:text-cyan-200">이미지→텍스트</th>
                          <th className="text-left py-2 text-cyan-800 dark:text-cyan-200">통합 성능</th>
                        </tr>
                      </thead>
                      <tbody className="text-cyan-700 dark:text-cyan-300">
                        <tr>
                          <td className="py-1">단일 모달</td>
                          <td className="py-1">82.4%</td>
                          <td className="py-1">79.1%</td>
                          <td className="py-1">80.8%</td>
                        </tr>
                        <tr>
                          <td className="py-1">Early Fusion</td>
                          <td className="py-1">89.2%</td>
                          <td className="py-1">86.7%</td>
                          <td className="py-1">88.0%</td>
                        </tr>
                        <tr>
                          <td className="py-1">Late Fusion</td>
                          <td className="py-1">91.5%</td>
                          <td className="py-1">88.9%</td>
                          <td className="py-1">90.2%</td>
                        </tr>
                        <tr>
                          <td className="py-1">Adaptive</td>
                          <td className="py-1">93.8%</td>
                          <td className="py-1">91.4%</td>
                          <td className="py-1">92.6%</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                
                <div className="bg-emerald-50 dark:bg-emerald-900/20 p-4 rounded-lg border border-emerald-200 dark:border-emerald-700 mt-4">
                  <h4 className="font-bold text-emerald-800 dark:text-emerald-200 mb-2">🚀 차세대 기술 동향</h4>
                  <div className="grid md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <strong className="text-emerald-800 dark:text-emerald-200">GPT-4V 통합</strong>
                      <p className="text-emerald-700 dark:text-emerald-300 mt-1">
                        OpenAI의 Vision 모델과 RAG 결합으로 이미지 이해도 대폭 향상. 
                        복잡한 차트, 다이어그램도 정확한 텍스트 설명으로 변환.
                      </p>
                    </div>
                    <div>
                      <strong className="text-emerald-800 dark:text-emerald-200">DALL-E 3 역검색</strong>
                      <p className="text-emerald-700 dark:text-emerald-300 mt-1">
                        텍스트 설명으로 유사한 이미지 생성 후, 
                        생성 이미지와 실제 이미지 간 유사도로 검색 정확도 향상.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">🔄 모달리티 융합 방식</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="font-medium text-teal-600 mb-1">Early Fusion</p>
                      <p className="text-gray-600 dark:text-gray-400">
                        임베딩 레벨에서 직접 결합
                      </p>
                    </div>
                    <div>
                      <p className="font-medium text-teal-600 mb-1">Late Fusion</p>
                      <p className="text-gray-600 dark:text-gray-400">
                        검색 결과 수준에서 가중치 결합
                      </p>
                    </div>
                  </div>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">⚖️ 동적 가중치 조정</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    쿼리 타입과 컨텍스트에 따른 모달리티별 중요도 자동 조정
                  </p>
                </div>
                
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">🎯 컨텍스트 인식 검색</h4>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    이전 검색 결과와 사용자 의도를 고려한 개인화된 멀티모달 검색
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Practical Exercise */}
        <section className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-2xl p-8 text-white">
          <h2 className="text-2xl font-bold mb-6">실습 과제</h2>
          
          <div className="bg-white/10 rounded-xl p-6 backdrop-blur">
            <h3 className="font-bold mb-4">멀티모달 RAG 구축 실습</h3>
            
            <div className="space-y-4">
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">🎥 과제 1: 비디오 기반 QA 시스템</h4>
                <ol className="space-y-2 text-sm">
                  <li>1. 교육 비디오에서 키프레임 및 전사 추출</li>
                  <li>2. 시각적 내용과 음성 내용 통합 인덱싱</li>
                  <li>3. "이 부분에서 설명하는 개념은?" 타입 질의 처리</li>
                  <li>4. 정확한 타임스탬프와 함께 답변 제공</li>
                </ol>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">📊 과제 2: 문서 내 차트 분석 RAG</h4>
                <ul className="space-y-1 text-sm">
                  <li>• PDF에서 차트/그래프 자동 추출</li>
                  <li>• 차트 데이터를 텍스트로 변환</li>
                  <li>• "수익이 가장 높은 분기는?" 등 데이터 질의 처리</li>
                  <li>• 시각적 증거와 함께 답변 생성</li>
                </ul>
              </div>
              
              <div className="bg-white/10 p-4 rounded-lg">
                <h4 className="font-medium mb-2">🔄 과제 3: 크로스모달 검색 엔진</h4>
                <ul className="space-y-1 text-sm">
                  <li>• 텍스트 쿼리로 관련 이미지 검색</li>
                  <li>• 이미지 업로드로 관련 텍스트 검색</li>
                  <li>• 오디오 클립으로 관련 문서 검색</li>
                  <li>• 검색 결과의 신뢰도 평가 시스템</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* References */}
        <References
          sections={[
            {
              title: '📚 멀티모달 AI & CLIP',
              icon: 'web' as const,
              color: 'border-teal-500',
              items: [
                {
                  title: 'OpenAI CLIP Documentation',
                  authors: 'OpenAI',
                  year: '2021',
                  description: '이미지-텍스트 통합 임베딩 - 4억 쌍 학습',
                  link: 'https://github.com/openai/CLIP'
                },
                {
                  title: 'Hugging Face Transformers - Vision',
                  authors: 'Hugging Face',
                  year: '2025',
                  description: 'ViT, CLIP, BLIP 등 멀티모달 모델 라이브러리',
                  link: 'https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder'
                },
                {
                  title: 'LangChain Multi-Modal RAG',
                  authors: 'LangChain',
                  year: '2025',
                  description: '이미지/비디오/오디오 처리 - 통합 RAG 파이프라인',
                  link: 'https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector'
                },
                {
                  title: 'GPT-4 Vision API',
                  authors: 'OpenAI',
                  year: '2024',
                  description: '이미지 이해 및 분석 - RAG 응답 생성',
                  link: 'https://platform.openai.com/docs/guides/vision'
                },
                {
                  title: 'Gemini Pro Vision',
                  authors: 'Google DeepMind',
                  year: '2024',
                  description: '네이티브 멀티모달 LLM - 이미지, 비디오, 오디오 통합',
                  link: 'https://ai.google.dev/tutorials/multimodal'
                }
              ]
            },
            {
              title: '📖 멀티모달 학습 & 검색 연구',
              icon: 'research' as const,
              color: 'border-blue-500',
              items: [
                {
                  title: 'CLIP: Learning Transferable Visual Models',
                  authors: 'Radford et al., OpenAI',
                  year: '2021',
                  description: 'Contrastive Learning - 제로샷 이미지 분류',
                  link: 'https://arxiv.org/abs/2103.00020'
                },
                {
                  title: 'BLIP-2: Bootstrapping Vision-Language',
                  authors: 'Li et al., Salesforce',
                  year: '2023',
                  description: 'Q-Former로 효율적인 멀티모달 학습',
                  link: 'https://arxiv.org/abs/2301.12597'
                },
                {
                  title: 'Flamingo: Visual Language Model',
                  authors: 'Alayrac et al., DeepMind',
                  year: '2022',
                  description: '이미지/비디오/텍스트 인터리빙 처리',
                  link: 'https://arxiv.org/abs/2204.14198'
                },
                {
                  title: 'Wav2Vec 2.0: Self-Supervised Audio',
                  authors: 'Baevski et al., Meta',
                  year: '2020',
                  description: '오디오 표현 학습 - 음성 검색 기반',
                  link: 'https://arxiv.org/abs/2006.11477'
                }
              ]
            },
            {
              title: '🛠️ 멀티모달 RAG 도구',
              icon: 'tools' as const,
              color: 'border-purple-500',
              items: [
                {
                  title: 'Unstructured.io',
                  authors: 'Unstructured',
                  year: '2025',
                  description: 'PDF/이미지/표 추출 - RAG용 문서 전처리',
                  link: 'https://unstructured.io/'
                },
                {
                  title: 'Twelve Labs Video Understanding',
                  authors: 'Twelve Labs',
                  year: '2024',
                  description: '비디오 검색 & 분석 API - 장면 기반 검색',
                  link: 'https://docs.twelvelabs.io/'
                },
                {
                  title: 'AssemblyAI Audio Intelligence',
                  authors: 'AssemblyAI',
                  year: '2025',
                  description: '음성-텍스트 변환 - 감정, 화자 분리, 요약',
                  link: 'https://www.assemblyai.com/docs'
                },
                {
                  title: 'Pinecone Namespaces',
                  authors: 'Pinecone',
                  year: '2025',
                  description: '멀티모달 벡터 저장 - 타입별 네임스페이스 분리',
                  link: 'https://docs.pinecone.io/docs/namespaces'
                },
                {
                  title: 'LlamaIndex ImageNode',
                  authors: 'LlamaIndex',
                  year: '2025',
                  description: '이미지 노드 처리 - 텍스트와 이미지 통합 인덱싱',
                  link: 'https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html'
                }
              ]
            }
          ]}
        />
      </div>

      {/* Navigation */}
      <div className="mt-12 bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Link
            href="/modules/rag/intermediate/chapter4"
            className="inline-flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
            이전: RAG 성능 최적화
          </Link>
          
          <Link
            href="/modules/rag/intermediate/chapter6"
            className="inline-flex items-center gap-2 bg-violet-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-violet-600 transition-colors"
          >
            다음: Production RAG Systems
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  )
}