'use client'

import { Video } from 'lucide-react'

export default function Section2() {
  return (
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
  )
}
